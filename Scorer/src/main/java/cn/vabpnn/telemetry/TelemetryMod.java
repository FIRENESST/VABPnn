package cn.vabpnn.telemetry;
import com.google.gson.*;
import net.fabricmc.fabric.api.entity.event.v1.ServerPlayerEvents;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.*;
import net.fabricmc.fabric.api.event.player.*;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.storage.LevelResource;
import org.slf4j.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public final class TelemetryMod implements ModInitializer {
 public static final Logger LOG=LoggerFactory.getLogger("vabpnn-telemetry");
 public static volatile UUID controlledPlayer;
 private static volatile JsonObject visibility=new JsonObject();
 private static final Deque<JsonObject> events=new ArrayDeque<>();
 private static long eventId,sequence;
 private static long lastMiningTick=Long.MIN_VALUE;
 private static String session=UUID.randomUUID().toString(),serverSession=session,world="";
 private static long episodeIndex,episodeStartedTick;private static boolean forceSample;
 private static final Deque<JsonObject> respawns=new ArrayDeque<>();
 public static volatile JsonObject lastPacket=new JsonObject();
 private record TrainingSnapshot(UUID player,String json,boolean alive) {}
 private static final TrainingSnapshot EMPTY_TRAINING=new TrainingSnapshot(null,"{}",true);
 private static volatile TrainingSnapshot trainingSnapshot=EMPTY_TRAINING;
 private static volatile boolean trainingPlaying;
 private static final AreaScanner scanner=new AreaScanner();
 @Override public void onInitialize(){
  TelemetryTransport.start();
  ServerLifecycleEvents.SERVER_STARTED.register(server->{trainingSnapshot=EMPTY_TRAINING;session=UUID.randomUUID().toString();serverSession=session;episodeIndex=episodeStartedTick=0;forceSample=false;respawns.clear();world=worldIdentity(server);events.clear();eventId=sequence=0;lastMiningTick=Long.MIN_VALUE;scanner.reset();Snapshot.reset();});
  ServerLifecycleEvents.SERVER_STOPPED.register(server->{trainingSnapshot=EMPTY_TRAINING;events.clear();controlledPlayer=null;visibility=new JsonObject();lastPacket=new JsonObject();});
  ServerTickEvents.END_SERVER_TICK.register(TelemetryMod::tick);
  ServerPlayerEvents.AFTER_RESPAWN.register(TelemetryMod::respawn);
  AttackBlockCallback.EVENT.register((player,level,hand,pos,direction)->{if(player instanceof ServerPlayer p&&!p.isSpectator())miningAttempt(p,pos);return InteractionResult.PASS;});
  AttackEntityCallback.EVENT.register((player,level,hand,entity,hit)->{if(player instanceof ServerPlayer p)record(p,"attack_attempt",entity.blockPosition(),BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString());return InteractionResult.PASS;});
  UseItemCallback.EVENT.register((player,level,hand)->{if(player instanceof ServerPlayer p)record(p,"use_item",p.blockPosition(),BuiltInRegistries.ITEM.getKey(p.getItemInHand(hand).getItem()).toString());return InteractionResult.PASS;});
 }
 public static void visible(JsonObject value){visibility=value;}
 public static void gameState(JsonObject value){trainingPlaying="playing".equals(text(value,"state"));TelemetryTransport.gameState(value);}
 /** Optional model integration: immutable JSON, published by the server; callers must reject stale samples. */
 public static String trainingContext(){TrainingSnapshot snapshot=trainingSnapshot;return trainingPlaying&&snapshot.player!=null&&snapshot.player.equals(controlledPlayer)?snapshot.json:"{}";}
 /** Death settlement only. The caller must independently enforce focus, pause and training-mode guards. */
 public static String trainingTerminalContext(){TrainingSnapshot snapshot=trainingSnapshot;return !snapshot.alive&&snapshot.player!=null&&snapshot.player.equals(controlledPlayer)?snapshot.json:"{}";}
 public static String status(){return "Scorer 遥测："+TelemetryTransport.status+"；已发送 "+TelemetryTransport.sent+"，覆盖/丢弃 "+TelemetryTransport.dropped+"。采集与覆盖详情见评分器。";}
 private static String worldIdentity(MinecraftServer server){
  Path file=server.getWorldPath(LevelResource.ROOT).resolve("data/vabpnn-scorer-world-id.txt");
  try{if(Files.exists(file))return UUID.fromString(Files.readString(file).trim()).toString();String id=UUID.randomUUID().toString();Files.createDirectories(file.getParent());Files.writeString(file,id,StandardOpenOption.CREATE_NEW);return id;}
  catch(Exception e){LOG.error("Cannot persist Scorer world identity; using path fallback",e);return UUID.nameUUIDFromBytes(server.getWorldPath(LevelResource.ROOT).toAbsolutePath().normalize().toString().getBytes(StandardCharsets.UTF_8)).toString();}
 }
 public static int setting(String key,int fallback,int lo,int hi){return TelemetryTransport.setting(key,fallback,lo,hi);}
 public static boolean controls(ServerPlayer p){return controlledPlayer!=null&&p.getUUID().equals(controlledPlayer);}
 public static void miningAttempt(ServerPlayer p,BlockPos pos){
  if(!controls(p)||p.isCreative()||p.isSpectator()||p.level().getBlockState(pos).isAir())return;
  long tick=p.level().getGameTime();if(lastMiningTick!=Long.MIN_VALUE&&tick-lastMiningTick<20)return;
  lastMiningTick=tick;JsonObject tool=MiningTools.describe(p,pos);record(p,"mining_attempt",pos,tool.get("block").getAsString(),tool);
 }
 public static void record(ServerPlayer p,String type,BlockPos pos,String block){
  record(p,type,pos,block,null);
 }
 public static void record(ServerPlayer p,String type,BlockPos pos,String block,JsonObject tool){
  if(controlledPlayer==null||!p.getUUID().equals(controlledPlayer))return;
  JsonObject e=new JsonObject();e.addProperty("id",++eventId);e.addProperty("tick",p.level().getGameTime());e.addProperty("type",type);e.addProperty("dimension",p.level().dimension().identifier().toString());e.addProperty("pos",pos.getX()+","+pos.getY()+","+pos.getZ());e.addProperty("block",block);if(tool!=null)e.add("tool",tool);events.addLast(e);while(events.size()>256)events.removeFirst();
 }
 public static void recordDetail(ServerPlayer p,String type,BlockPos pos,JsonObject detail){
  if(!controls(p))return;JsonObject e=detail.deepCopy();e.addProperty("id",++eventId);e.addProperty("tick",p.level().getGameTime());e.addProperty("type",type);e.addProperty("dimension",p.level().dimension().identifier().toString());e.addProperty("pos",pos.getX()+","+pos.getY()+","+pos.getZ());events.addLast(e);while(events.size()>256)events.removeFirst();
 }
 private static void respawn(ServerPlayer oldPlayer,ServerPlayer newPlayer,boolean alive){
  if(alive||!controls(newPlayer)||!newPlayer.level().getServer().isSingleplayer())return;
  String previous=session;session=UUID.randomUUID().toString();episodeIndex++;episodeStartedTick=newPlayer.level().getServer().getTickCount();
  JsonObject event=new JsonObject();event.addProperty("type","player_respawn");event.addProperty("event_id",UUID.randomUUID().toString());
  event.addProperty("world_id",world);event.addProperty("player_id",newPlayer.getUUID().toString());event.addProperty("server_session_id",serverSession);
  event.addProperty("previous_session_id",previous);event.addProperty("session_id",session);event.addProperty("episode_index",episodeIndex);
  event.addProperty("tick",episodeStartedTick);event.addProperty("observed_at_ms",System.currentTimeMillis());event.addProperty("reason","death_respawn");
  respawns.addLast(event);while(respawns.size()>128)respawns.removeFirst();
  events.clear();lastMiningTick=Long.MIN_VALUE;scanner.reset();Snapshot.reset();visibility=new JsonObject();forceSample=true;
  publishContext(newPlayer.level().getServer(),newPlayer);
  LOG.info("Player respawn: fresh Scorer episode {}, session {}; immediate baseline queued",episodeIndex,session);
 }
 private static void publishContext(MinecraftServer server,ServerPlayer player){
  boolean alive=player.isAlive();JsonObject context=new JsonObject();context.addProperty("schema",1);context.addProperty("authority","singleplayer_integrated_server");
  context.addProperty("world_id",world);context.addProperty("player_id",player.getUUID().toString());context.addProperty("session_id",session);
  context.addProperty("tick",server.getTickCount());context.addProperty("world_game_time",player.level().getGameTime());context.addProperty("observed_at_ms",System.currentTimeMillis());
  context.addProperty("clock","integrated_server_tick");context.addProperty("dimension",player.level().dimension().identifier().toString());
  context.addProperty("game_mode",player.gameMode.getGameModeForPlayer().getName());context.addProperty("alive",alive);context.addProperty("episode_index",episodeIndex);
  trainingSnapshot=new TrainingSnapshot(player.getUUID(),context.toString(),alive);
 }
 private static void tick(MinecraftServer server){
  if(!server.isSingleplayer()||controlledPlayer==null){trainingSnapshot=EMPTY_TRAINING;return;}
  ServerPlayer player=server.getPlayerList().getPlayer(controlledPlayer);if(player==null){trainingSnapshot=EMPTY_TRAINING;return;}
  try{
   publishContext(server,player);
   long scanStart=System.nanoTime();scanner.tick(player);double scanMs=(System.nanoTime()-scanStart)/1e6;
   if(!forceSample&&server.getTickCount()%setting("sample_ticks",20,10,200)!=0)return;
   forceSample=false;
   long started=System.nanoTime();JsonObject packet=Snapshot.capture(server,player);
   packet.addProperty("schema",1);packet.addProperty("session_id",session);packet.addProperty("world_id",world);packet.addProperty("player_id",player.getUUID().toString());packet.addProperty("seq",++sequence);packet.addProperty("tick",server.getTickCount());
   JsonObject episode=new JsonObject();episode.addProperty("server_session_id",serverSession);episode.addProperty("index",episodeIndex);episode.addProperty("started_tick",episodeStartedTick);packet.add("episode",episode);
   JsonArray lifecycle=new JsonArray();for(JsonObject event:respawns)lifecycle.add(event.deepCopy());packet.add("lifecycle_events",lifecycle);
   JsonArray copied=new JsonArray();for(JsonObject e:events)copied.add(e.deepCopy());packet.add("events",copied);
   JsonObject area=scanner.view();packet.add("area",area);packet.add("structures",area.has("structures")?area.get("structures").deepCopy():new JsonArray());
   JsonObject seen=visibility;if(player.getUUID().toString().equals(text(seen,"player_id"))&&player.level().dimension().identifier().toString().equals(text(seen,"dimension")))packet.add("visibility",seen.deepCopy());
   JsonObject coverage=new JsonObject();coverage.addProperty("authority","singleplayer_integrated_server");coverage.addProperty("entity_list_truncated",packet.get("entities_truncated").getAsBoolean());coverage.addProperty("events_retained",events.size());coverage.addProperty("event_cursor",eventId);coverage.addProperty("lifecycle_retained",respawns.size());coverage.addProperty("oldest_event_id",events.isEmpty()?0:events.getFirst().get("id").getAsLong());coverage.addProperty("transport_overwritten",TelemetryTransport.dropped);coverage.addProperty("loaded_chunks_only",true);coverage.addProperty("worldwide_structure_count_available",false);coverage.addProperty("visibility_exhaustive",false);coverage.addProperty("capture_ms",(System.nanoTime()-started)/1e6);packet.add("coverage",coverage);
   coverage.addProperty("scanner_tick_ms",scanMs);
   JsonArray currentStructures=new JsonArray();for(JsonElement entry:packet.getAsJsonArray("structures")){JsonObject s=entry.getAsJsonObject();JsonArray b=s.getAsJsonArray("bounds");if(player.getX()>=b.get(0).getAsDouble()&&player.getX()<=b.get(3).getAsDouble()+1&&player.getY()>=b.get(1).getAsDouble()&&player.getY()<=b.get(4).getAsDouble()+1&&player.getZ()>=b.get(2).getAsDouble()&&player.getZ()<=b.get(5).getAsDouble()+1)currentStructures.add(s.get("type"));}packet.getAsJsonObject("world").add("structures_at_player",currentStructures);
   lastPacket=packet.deepCopy();TelemetryTransport.submit(packet);
  }catch(Exception e){LOG.warn("Telemetry sample failed (game continues)",e);}
 }
 private static String text(JsonObject o,String k){return o.has(k)?o.get(k).getAsString():"";}
}

