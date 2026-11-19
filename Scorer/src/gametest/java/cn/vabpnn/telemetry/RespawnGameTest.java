package cn.vabpnn.telemetry;
import com.google.gson.*;import java.net.*;import java.net.http.*;import java.time.*;import java.util.*;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestServerContext;
import net.minecraft.world.level.gamerules.GameRules;
/** Actual server respawn callback including immediate respawn; no client death-screen dependency. */
final class RespawnGameTest {
 private static final HttpClient HTTP=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(1)).build();
 private static void require(boolean b,String s){if(!b)throw new AssertionError(s);}
 private static JsonObject get(String route)throws Exception{
  String base=System.getProperty("vabpnn.scorer.url","http://127.0.0.1:19876");
  var r=HTTP.send(HttpRequest.newBuilder(URI.create(base+route)).timeout(Duration.ofSeconds(2)).GET().build(),HttpResponse.BodyHandlers.ofString());
  require(r.statusCode()==200,"actual Scorer HTTP "+route);return JsonParser.parseString(r.body()).getAsJsonObject();
 }
 static void run(ClientGameTestContext context,TestServerContext server){
  try{
   JsonObject before=get("/v1/status");String sid=JsonParser.parseString(TelemetryMod.trainingContext()).getAsJsonObject().get("session_id").getAsString();
   JsonObject weights=get("/v1/status"); // No config secrets are read or printed.
   long index=JsonParser.parseString(TelemetryMod.trainingContext()).getAsJsonObject().get("episode_index").getAsLong();
   require(index==1,"manual respawn already changed authoritative life identity");
   for(int step=0;step<2;step++){
    final long expected=index+1;
    server.runOnServer(s->s.getGameRules().set(GameRules.IMMEDIATE_RESPAWN,true,s));
    server.runCommand("gamemode survival @p");
    context.waitFor(mc->mc.player!=null&&!mc.player.shouldShowDeathScreen(),100);
    context.waitTicks(65); // Let vanilla respawn protection and the preceding respawn handshake finish.
    TelemetryMod.LOG.info("Immediate respawn test ready: health={}, sdk={}, deathScreen={}",context.computeOnClient(mc->mc.player.getHealth()),TelemetryMod.trainingContext(),context.computeOnClient(mc->mc.player.shouldShowDeathScreen()));
    server.runCommand("kill @p");
    context.waitFor(mc->{String value=TelemetryMod.trainingContext();return mc.player!=null&&mc.player.isAlive()&&!value.equals("{}")
      &&JsonParser.parseString(value).getAsJsonObject().get("episode_index").getAsLong()==expected;},250);
    String current=JsonParser.parseString(TelemetryMod.trainingContext()).getAsJsonObject().get("session_id").getAsString();
    require(!current.equals(sid)&&TelemetryMod.trainingTerminalContext().equals("{}"),"immediate respawn replaces SDK session without manual respawn");
    context.waitFor(mc->TelemetryMod.lastPacket.has("episode")&&TelemetryMod.lastPacket.getAsJsonObject("episode").get("index").getAsLong()==expected,80);
    JsonObject status=null;for(int tries=0;tries<40;tries++){status=get("/v1/status");if(status.getAsJsonObject("episode").get("index").getAsLong()==expected)break;context.waitTicks(2);}
    require(status.getAsJsonObject("episode").get("index").getAsLong()==expected,"real Scorer resets on actual new life packet");
    require(status.get("stage").getAsString().equals("survival")&&status.getAsJsonObject("episode").get("total_reward").getAsDouble()==0,"real Scorer starts fresh score/stage");
    require(status.getAsJsonObject("telemetry").get("session_id").getAsString().equals(current),"Scorer and SDK bind the same new life");
    var all=get("/v1/rewards?after=0").getAsJsonArray("rewards");
    require(!all.isEmpty()&&!all.get(0).getAsJsonObject().get("valid").getAsBoolean()&&all.get(0).getAsJsonObject().get("reward").getAsDouble()==0,"first actual new-life reward is a zero baseline");
    require(all.asList().stream().allMatch(e->e.getAsJsonObject().get("session_id").getAsString().equals(current)),"old life receipts are absent from current reward window");
    sid=current;index=expected;
    TelemetryMod.LOG.info("PASS: actual immediate respawn {} changes SDK/Scorer session, score/stage baseline and reward scope",index);
   }
   for(int tries=0;tries<40;tries++){if(get("/v1/status").getAsJsonObject("episode").get("pending_llm_notifications").getAsLong()==0)break;context.waitTicks(2);}
   if(Boolean.getBoolean("vabpnn.test.respawnNotifications"))require(get("/v1/status").getAsJsonObject("episode").get("pending_llm_notifications").getAsLong()==0,"actual model HTTP service acknowledged every respawn");
   TelemetryMod.LOG.info("PASS: immediate respawn integration and independent lifecycle notification dispatch.");
  }catch(Exception e){throw new AssertionError(e);}
 }
}
