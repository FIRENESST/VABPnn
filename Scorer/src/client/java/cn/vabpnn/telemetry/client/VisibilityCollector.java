package cn.vabpnn.telemetry.client;
import cn.vabpnn.telemetry.TelemetryMod;
import com.google.gson.*;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.command.v2.*;
import net.minecraft.client.Minecraft;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.*;
import net.minecraft.network.chat.Component;
import org.joml.Vector3f;
import java.util.*;

/** Samples the camera frustum, with first-hit occlusion and unloaded chunks stopping each ray. */
public final class VisibilityCollector implements ClientModInitializer {
 private final Map<String,JsonObject> hits=new LinkedHashMap<>();
 private net.minecraft.client.player.LocalPlayer selectedPlayer;
 private int ticks,sampleIndex,rays;private long started;private String dimension="",player="";
 @Override public void onInitializeClient(){
  ClientTickEvents.END_CLIENT_TICK.register(this::tick);
  ClientCommandRegistrationCallback.EVENT.register((dispatcher,access)->dispatcher.register(ClientCommands.literal("scorer").executes(c->{c.getSource().sendFeedback(Component.literal(TelemetryMod.status()));return 1;})));
 }
 private void tick(Minecraft mc){
  JsonObject heartbeat=new JsonObject();heartbeat.addProperty("observed_at_ms",System.currentTimeMillis());
  heartbeat.addProperty("state",mc.player==null||mc.level==null?"menu":mc.getSingleplayerServer()==null?"unsupported_multiplayer":mc.isPaused()?"paused":"playing");
  if(mc.player!=null){heartbeat.addProperty("player_name",mc.player.getName().getString());heartbeat.addProperty("player_id",mc.player.getUUID().toString());}
  if(mc.level!=null)heartbeat.addProperty("dimension",mc.level.dimension().identifier().toString());TelemetryMod.gameState(heartbeat);
  if(mc.player==null||mc.level==null||mc.getSingleplayerServer()==null){TelemetryMod.controlledPlayer=null;selectedPlayer=null;hits.clear();return;}
  TelemetryMod.controlledPlayer=mc.player.getUUID();if(mc.isPaused())return;
  String dim=mc.level.dimension().identifier().toString(),id=mc.player.getUUID().toString();
  if(selectedPlayer!=mc.player||!dim.equals(dimension)||!id.equals(player)){selectedPlayer=mc.player;dimension=dim;player=id;hits.clear();ticks=rays=sampleIndex=0;started=mc.level.getGameTime();}
  if(ticks++%20==0){hits.clear();rays=0;started=mc.level.getGameTime();}
  long start=System.nanoTime();var camera=mc.gameRenderer.getMainCamera();Vec3 origin=camera.position();
  int budget=TelemetryMod.setting("visible_rays_per_tick",24,1,128),range=Math.min(TelemetryMod.setting("visible_range",160,16,160),TelemetryMod.setting("chunk_radius",10,1,10)*16);
  double fov=Math.toRadians(mc.options.fov().get()),aspect=(double)mc.getWindow().getScreenWidth()/Math.max(1,mc.getWindow().getScreenHeight());
  for(int n=0;n<budget;n++){
   int sample=sampleIndex++%(64*36);float u=(float)(((sample%64+.5)/64*2-1)*Math.tan(fov/2)*aspect),v=(float)((1-(sample/64+.5)/36*2)*Math.tan(fov/2));
   Vector3f dir=new Vector3f(u,v,-1).normalize().rotate(camera.rotation());Vec3 end=origin.add(dir.x*range,dir.y*range,dir.z*range);
   // Clip never requests/generates chunks; shorten at the first unloaded section of the path.
   double allowed=range;for(int d=0;d<=range;d+=8){Vec3 point=origin.add(dir.x*d,dir.y*d,dir.z*d);if(!mc.level.hasChunk((int)Math.floor(point.x/16),(int)Math.floor(point.z/16))){allowed=Math.max(0,d-8);break;}}
   end=origin.add(dir.x*allowed,dir.y*allowed,dir.z*allowed);
   BlockHitResult hit=mc.level.clip(new ClipContext(origin,end,ClipContext.Block.OUTLINE,ClipContext.Fluid.ANY,mc.player));rays++;
   if(hit.getType()==HitResult.Type.BLOCK){var pos=hit.getBlockPos();String key=pos.getX()+","+pos.getY()+","+pos.getZ();if(hits.size()<2048){JsonObject b=new JsonObject();b.addProperty("pos",key);b.addProperty("block",BuiltInRegistries.BLOCK.getKey(mc.level.getBlockState(pos).getBlock()).toString());b.addProperty("distance",origin.distanceTo(hit.getLocation()));hits.put(key,b);}}
   if(System.nanoTime()-start>2_000_000)break;
  }
  JsonObject result=new JsonObject();result.addProperty("player_id",player);result.addProperty("dimension",dimension);result.addProperty("sample_start_tick",started);result.addProperty("sample_end_tick",mc.level.getGameTime());result.addProperty("kind","camera_first_hit_ray_samples");result.addProperty("exhaustive",false);result.addProperty("range_blocks",range);result.addProperty("rays_cast",rays);result.addProperty("grid_rays",64*36);result.addProperty("fov_degrees",mc.options.fov().get());JsonArray blocks=new JsonArray();Set<String> registry=new TreeSet<>();for(JsonObject b:hits.values()){blocks.add(b);registry.add(b.get("block").getAsString());}result.add("blocks",blocks);JsonArray ids=new JsonArray();registry.forEach(ids::add);result.add("block_registry",ids);TelemetryMod.visible(result);
 }
}

