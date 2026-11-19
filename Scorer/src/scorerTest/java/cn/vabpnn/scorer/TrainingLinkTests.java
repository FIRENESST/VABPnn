package cn.vabpnn.scorer;

import com.google.gson.*;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import static cn.vabpnn.scorer.ScorerTest.*;

final class TrainingLinkTests {
 static JsonObject observation(long seq,long tick){JsonObject t=sample(seq,tick,0),coverage=new JsonObject();coverage.addProperty("authority","singleplayer_integrated_server");t.add("coverage",coverage);Json.obj(t,"world").addProperty("dimension","minecraft:overworld");Json.obj(t,"world").addProperty("game_time",100000+tick);return t;}
 static void run(Path root)throws Exception{
  Engine e=new Engine(root.resolve("training-link"));check(!Json.bool(e.trainingInfo(),"ready",true),"empty scorer is not ready for training");
  check(Json.bool(e.rewardAfter(9),"gap",false),"nonzero cursor after empty restart is a gap");
  JsonObject t=observation(1,20);e.accept(t);e.receiveGame(FeatureTests.heartbeat("playing",1));JsonObject contract=e.trainingInfo();check(Json.bool(contract,"ready",false)&&Json.str(contract,"clock","").equals("integrated_server_tick"),"fresh integrated-server handshake");
  check(Json.num(contract,"tick",0)==20&&Json.str(Json.obj(contract,"binding"),"session_id","").equals("test-session"),"binding uses exact server tick, not persistent world game time");
  check(Json.num(contract,"config_version",-1)==e.version(),"handshake fixes scoring configuration before actions are collected");
  t=observation(2,40);Json.obj(t,"stats").addProperty("deaths",1);Json.obj(t,"player").addProperty("health",0);JsonObject death=e.accept(t);check(Json.bool(death,"valid",false)&&Json.bool(death,"terminal",false)&&Json.str(death,"terminal_reason","").equals("death")&&Json.num(death,"start_tick",0)==20,"death reward closes the matching interval");
  t=observation(3,60);Json.obj(t,"stats").addProperty("deaths",1);Json.obj(t,"player").addProperty("health",0);check(!Json.bool(e.accept(t),"terminal",true),"remaining dead does not repeatedly terminate");
  t=observation(4,80);Json.obj(t,"stats").addProperty("deaths",1);Json.obj(t,"player").addProperty("health",0);Json.obj(t,"world").addProperty("dimension","minecraft:the_nether");JsonObject dimension=e.accept(t);check(Json.bool(dimension,"terminal",false)&&Json.str(dimension,"start_dimension","").equals("minecraft:overworld"),"dimension transition marked with both endpoints");
  String stream=Json.str(e.trainingInfo(),"stream_id","");check(Json.arr(e.rewardAfter(1,stream),"rewards").size()==3,"matching stream cursor delivers only new rewards");
  JsonObject wrong=e.rewardAfter(1,"previous-process");check(Json.bool(wrong,"gap",false)&&Json.arr(wrong,"rewards").isEmpty(),"old stream cannot deliver a new process reward");
  e.receiveGame(FeatureTests.heartbeat("paused",2));check(!Json.bool(e.trainingInfo(),"ready",true),"paused game not ready");e.receiveGame(FeatureTests.heartbeat("playing",3));
  JsonObject report=new JsonObject();report.addProperty("schema",1);report.addProperty("trainer_id","integration-test");report.addProperty("stream_id",stream);report.add("binding",Json.obj(e.trainingInfo(),"binding").deepCopy());report.addProperty("mode","self_training");report.addProperty("connection","connected");report.addProperty("updates",2);report.addProperty("applied_rewards",1);report.addProperty("last_reward_id",4);
  check(Json.bool(e.receiveTraining(report),"binding_matches",false),"model status bound to exact reward session");check(Json.str(Json.obj(e.status(),"training"),"status_source","").equals("trainer_report"),"UI distinguishes report from verified training evidence");
  Json.obj(report,"binding").addProperty("world_id","other-world");e.receiveTraining(report);check(Json.str(Json.obj(e.status(),"training"),"state","").equals("unbound"),"wrong world cannot appear connected");
  var server=ScorerApp.server(e,0);var client=HttpClient.newBuilder().proxy(new ProxySelector(){public java.util.List<Proxy> select(URI uri){return java.util.List.of(Proxy.NO_PROXY);}public void connectFailed(URI uri,SocketAddress address,java.io.IOException error){}}).build();String base="http://127.0.0.1:"+server.getAddress().getPort();
  try{
   var response=client.send(HttpRequest.newBuilder(URI.create(base+"/v1/training")).GET().build(),HttpResponse.BodyHandlers.ofString());check(response.statusCode()==200&&JsonParser.parseString(response.body()).getAsJsonObject().has("binding"),"HTTP training handshake");
   response=client.send(HttpRequest.newBuilder(URI.create(base+"/v1/rewards?stream_id=obsolete&after=0")).GET().build(),HttpResponse.BodyHandlers.ofString());check(response.statusCode()==200&&JsonParser.parseString(response.body()).getAsJsonObject().getAsJsonArray("rewards").isEmpty(),"stream guard and reordered URL parameters");
   response=client.send(HttpRequest.newBuilder(URI.create(base+"/v1/rewards?after=0&after=1")).GET().build(),HttpResponse.BodyHandlers.ofString());check(response.statusCode()==400,"ambiguous cursor rejected");
   response=client.send(HttpRequest.newBuilder(URI.create(base+"/v1/training-status")).header("Content-Type","application/json").POST(HttpRequest.BodyPublishers.ofString(report.toString())).build(),HttpResponse.BodyHandlers.ofString());check(response.statusCode()==200,"HTTP model training status ingestion");
  }finally{server.stop(0);}
  boolean rejected=false;report.addProperty("updates",-1);try{e.receiveTraining(report);}catch(IllegalArgumentException ex){rejected=true;}check(rejected,"invalid training counters rejected");
  System.out.println("PASS: training handshake, exact session/clock, pause and terminal boundaries, reward stream/cursor guards, model reports and HTTP integration");
 }
}
