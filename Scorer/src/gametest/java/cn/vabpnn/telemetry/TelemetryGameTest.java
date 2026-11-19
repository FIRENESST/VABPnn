package cn.vabpnn.telemetry;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import com.google.gson.*;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;
public final class TelemetryGameTest implements FabricClientGameTest {
 private static void check(boolean ok,String what){if(!ok)throw new AssertionError(what);}
 @Override public void runTest(ClientGameTestContext context){
  logTailTest();
  try(var world=context.worldBuilder().create()){
   world.getClientLevel().waitForChunksDownload();
   world.getServer().runCommand("gamemode survival @p");
   world.getServer().runCommand("give @p minecraft:iron_ingot 7");
   context.waitFor(mc->TelemetryMod.lastPacket.has("inventory")&&TelemetryMod.lastPacket.getAsJsonObject("inventory").has("minecraft:iron_ingot"),400);
   JsonObject t=TelemetryMod.lastPacket,p=t.getAsJsonObject("player");
   check(t.getAsJsonObject("inventory").get("minecraft:iron_ingot").getAsInt()==7,"actual server inventory");
   check(p.has("health")&&p.has("air")&&p.has("food")&&p.has("vx")&&p.has("nbt_snbt"),"state/velocity/NBT");
   check(t.has("stats")&&t.has("advancement_progress")&&t.has("entities")&&t.has("world"),"full snapshot fields");
   context.waitFor(mc->TelemetryMod.lastPacket.has("visibility")&&TelemetryMod.lastPacket.getAsJsonObject("visibility").get("rays_cast").getAsInt()>0,200);
   check(!TelemetryMod.lastPacket.getAsJsonObject("visibility").get("exhaustive").getAsBoolean(),"visibility coverage honest");
   JsonObject training=JsonParser.parseString(TelemetryMod.trainingContext()).getAsJsonObject();
   check(training.get("session_id").equals(TelemetryMod.lastPacket.get("session_id"))&&training.get("world_id").equals(TelemetryMod.lastPacket.get("world_id"))&&training.get("player_id").equals(TelemetryMod.lastPacket.get("player_id")),"model binding shares the telemetry world/player/session");
   check(training.get("tick").getAsLong()>=TelemetryMod.lastPacket.get("tick").getAsLong()&&training.get("clock").getAsString().equals("integrated_server_tick"),"model context publishes current integrated-server clock");
   check(training.get("alive").getAsBoolean()&&TelemetryMod.trainingTerminalContext().equals("{}"),"living player cannot provide a terminal context");
   context.waitFor(mc->TelemetryTransport.sent>0,200);
   check(JsonParser.parseString(TelemetryTransport.reward).getAsJsonObject().has("components"),"real collector to local Scorer reward response");
   context.waitFor(mc->TelemetryMod.lastPacket.getAsJsonObject("area").get("cycle_complete").getAsBoolean(),500);
   check(TelemetryMod.lastPacket.getAsJsonObject("area").get("chunks_requested").getAsInt()==441,"ten-chunk radius scan");
   check(!TelemetryMod.lastPacket.getAsJsonObject("area").getAsJsonArray("block_registry").isEmpty(),"actual loaded block registry");
   ProgressionGameTest.run(context,world.getServer());
   world.getServer().runCommand("tp @p 0.5 -60 0.5");
   world.getServer().runCommand("clear @p");
   world.getServer().runCommand("setblock 2 -60 0 minecraft:stone");
   context.waitFor(mc->mc.player.getMainHandItem().isEmpty()&&mc.player.getX()<1,200);
   BlockPos minePos=new BlockPos(2,-60,0);
   context.runOnClient(mc->mc.gameMode.startDestroyBlock(minePos,Direction.UP));
   context.waitFor(mc->hasTool("mining_attempt","minecraft:air",false),200);
   context.runOnClient(mc->mc.gameMode.stopDestroyBlock());
   world.getServer().runCommand("item replace entity @p weapon.mainhand with minecraft:iron_axe");
   context.waitFor(mc->mc.player.getMainHandItem().is(net.minecraft.world.item.Items.IRON_AXE),200);
   context.runOnClient(mc->mc.gameMode.startDestroyBlock(minePos,Direction.UP));
   context.waitFor(mc->hasTool("mining_attempt","minecraft:iron_axe",false),200);
   context.runOnClient(mc->mc.gameMode.stopDestroyBlock());
   world.getServer().runCommand("setblock 2 -60 0 minecraft:iron_ore");
   world.getServer().runCommand("item replace entity @p weapon.mainhand with minecraft:wooden_pickaxe");
   context.waitFor(mc->mc.player.getMainHandItem().is(net.minecraft.world.item.Items.WOODEN_PICKAXE)&&mc.level.getBlockState(minePos).is(net.minecraft.world.level.block.Blocks.IRON_ORE),200);
   context.runOnClient(mc->mc.gameMode.startDestroyBlock(minePos,Direction.UP));
   context.waitFor(mc->hasTool("mining_attempt","minecraft:wooden_pickaxe",false),200);
   check(findTool("mining_attempt","minecraft:wooden_pickaxe").get("family_matches").getAsBoolean(),"actual low-tier pickaxe distinguished from wrong family");
   context.runOnClient(mc->mc.gameMode.stopDestroyBlock());
   world.getServer().runCommand("setblock 2 -60 0 minecraft:stone");
   world.getServer().runCommand("item replace entity @p weapon.mainhand with minecraft:iron_pickaxe");
   context.waitFor(mc->mc.player.getMainHandItem().is(net.minecraft.world.item.Items.IRON_PICKAXE)&&mc.level.getBlockState(minePos).is(net.minecraft.world.level.block.Blocks.STONE),200);
   context.runOnClient(mc->mc.gameMode.startDestroyBlock(minePos,Direction.UP));
   context.waitFor(mc->{mc.gameMode.continueDestroyBlock(minePos,Direction.UP);return hasTool("block_break","minecraft:iron_pickaxe",true);},300);
   context.runOnClient(mc->mc.gameMode.stopDestroyBlock());
   TelemetryMod.LOG.info("PASS: wrong-tool attempts before break, hand/axe on stone, low-tier pickaxe, successful appropriate-tool mining");
   world.getServer().runCommand("gamemode creative @p");
   world.getServer().runCommand("tp @p 0.5 -60 0.5");
   world.getServer().runCommand("item replace entity @p weapon.mainhand with minecraft:cobblestone 10");
   world.getServer().runCommand("setblock 2 -61 0 minecraft:stone");
   world.getServer().runCommand("setblock 2 -60 0 minecraft:air");
   world.getServer().runCommand("setblock 1 -60 1 minecraft:chest");
   world.getServer().runCommand("summon minecraft:pig 3 -60 2 {NoAI:1b}");
   context.waitFor(mc->mc.player.getMainHandItem().getCount()==10&&mc.player.getX()<1,200);
   context.runOnClient(mc->mc.gameMode.useItemOn(mc.player,InteractionHand.MAIN_HAND,new BlockHitResult(new Vec3(2.5,-60,.5),Direction.UP,new BlockPos(2,-61,0),false)));
   context.waitFor(mc->hasEvent("block_place"),200);
   context.runOnClient(mc->mc.gameMode.startDestroyBlock(new BlockPos(2,-60,0),Direction.UP));
   context.waitFor(mc->hasEvent("block_break"),200);
   context.waitFor(mc->TelemetryMod.lastPacket.getAsJsonArray("entities").asList().stream().anyMatch(a->a.getAsJsonObject().get("type").getAsString().equals("minecraft:pig")),200);
   context.waitFor(mc->TelemetryMod.lastPacket.getAsJsonObject("area").getAsJsonArray("block_entities").asList().stream().anyMatch(a->a.getAsJsonObject().get("type").getAsString().equals("minecraft:chest")&&a.getAsJsonObject().has("nbt_snbt")),300);
   check(TelemetryMod.lastPacket.getAsJsonArray("events").asList().stream().anyMatch(a->a.getAsJsonObject().get("type").getAsString().equals("block_place")&&a.getAsJsonObject().get("pos").getAsString().equals("2,-60,0")),"placement reports successful target position");
   world.getServer().runCommand("advancement grant @p only minecraft:story/smelt_iron");
   context.waitFor(mc->TelemetryMod.lastPacket.getAsJsonArray("advancements").asList().stream().anyMatch(a->a.getAsString().equals("minecraft:story/smelt_iron")),200);
   TelemetryMod.LOG.info("PASS: real integrated-server telemetry: inventory, health/hunger/oxygen, NBT, stats, advancement, client visibility; capture_ms={}",TelemetryMod.lastPacket.getAsJsonObject("coverage").get("capture_ms"));
   world.getServer().runCommand("gamemode survival @p");
   world.getServer().runCommand("kill @p");
   context.waitFor(mc->mc.player!=null&&!mc.player.isAlive()&&!TelemetryMod.trainingTerminalContext().equals("{}"),200);
   JsonObject terminal=JsonParser.parseString(TelemetryMod.trainingTerminalContext()).getAsJsonObject();
   check(!terminal.get("alive").getAsBoolean()&&terminal.get("session_id").equals(training.get("session_id"))&&terminal.get("player_id").equals(training.get("player_id"))&&terminal.get("world_id").equals(training.get("world_id")),"death settlement retains exact authoritative binding");
   context.runOnClient(mc->{JsonObject paused=new JsonObject();paused.addProperty("state","paused");TelemetryMod.gameState(paused);check(TelemetryMod.trainingContext().equals("{}"),"normal training remains hidden while paused");java.util.UUID selected=TelemetryMod.controlledPlayer;TelemetryMod.controlledPlayer=java.util.UUID.randomUUID();check(TelemetryMod.trainingTerminalContext().equals("{}"),"terminal snapshot cannot bind another player");TelemetryMod.controlledPlayer=selected;});
   context.runOnClient(mc->mc.player.respawn());
   context.waitFor(mc->mc.player!=null&&mc.player.isAlive()&&TelemetryMod.trainingTerminalContext().equals("{}"),200);
   context.runOnClient(mc->mc.setScreen(null));
   context.waitFor(mc->{String json=TelemetryMod.trainingContext();return !json.equals("{}")&&JsonParser.parseString(json).getAsJsonObject().get("alive").getAsBoolean();},200);
   TelemetryMod.LOG.info("PASS: training death context, paused normal context, exact player binding and respawn clearing");
   if(Boolean.getBoolean("vabpnn.test.respawns"))RespawnGameTest.run(context,world.getServer());
  }
  check(TelemetryMod.trainingContext().equals("{}")&&TelemetryMod.trainingTerminalContext().equals("{}"),"leaving the world clears both model training contexts");
 }
 private static boolean hasEvent(String type){return TelemetryMod.lastPacket.has("events")&&TelemetryMod.lastPacket.getAsJsonArray("events").asList().stream().anyMatch(a->a.getAsJsonObject().get("type").getAsString().equals(type));}
 private static JsonObject findTool(String type,String held){
  if(TelemetryMod.lastPacket.has("events"))for(JsonElement item:TelemetryMod.lastPacket.getAsJsonArray("events")){JsonObject e=item.getAsJsonObject();if(e.get("type").getAsString().equals(type)&&e.has("tool")&&e.getAsJsonObject("tool").get("held_item").getAsString().equals(held))return e.getAsJsonObject("tool");}return new JsonObject();
 }
 private static boolean hasTool(String type,String held,boolean harvest){JsonObject t=findTool(type,held);return t.has("can_harvest")&&t.get("can_harvest").getAsBoolean()==harvest;}
 private static void logTailTest(){
  try{
   var file=java.nio.file.Files.createTempFile("scorer-log-tail-",".log");GameLogTailer tail=new GameLogTailer(file);tail.poll();byte[] bytes="[Client thread/INFO] 中文日志\n".getBytes(java.nio.charset.StandardCharsets.UTF_8);int split=bytes.length-3;
   java.nio.file.Files.write(file,java.util.Arrays.copyOf(bytes,split));tail.poll();check(tail.snapshot(0).getAsJsonArray("lines").isEmpty(),"partial UTF-8 line waits for newline");java.nio.file.Files.write(file,java.util.Arrays.copyOfRange(bytes,split,bytes.length),java.nio.file.StandardOpenOption.APPEND);tail.poll();check(tail.snapshot(0).getAsJsonArray("lines").get(0).getAsJsonObject().get("text").getAsString().endsWith("中文日志"),"UTF-8 line survives split reads");
   java.nio.file.Files.writeString(file,"x\n");tail.poll();check(tail.snapshot(1).getAsJsonArray("lines").get(0).getAsJsonObject().get("text").getAsString().equals("x"),"log truncation resumes with new data");java.nio.file.Files.delete(file);
  }catch(java.io.IOException e){throw new RuntimeException(e);}
 }
}

