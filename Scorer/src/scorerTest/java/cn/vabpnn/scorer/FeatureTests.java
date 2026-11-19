package cn.vabpnn.scorer;

import com.google.gson.*;
import com.sun.net.httpserver.HttpServer;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import static cn.vabpnn.scorer.ScorerTest.*;

final class FeatureTests {
 static JsonObject tool(String held,boolean requires,boolean harvest,boolean matches,boolean isTool,double speed,double best){
  JsonObject o=new JsonObject();o.addProperty("held_item",held);o.addProperty("recommended_family","pickaxe");o.addProperty("requires_correct_tool",requires);o.addProperty("can_harvest",harvest);o.addProperty("family_matches",matches);o.addProperty("held_is_tool",isTool);o.addProperty("speed",speed);o.addProperty("best_speed",best);o.addProperty("best_item","minecraft:iron_pickaxe");o.addProperty("hardness",1.5);return o;
 }
 static JsonObject heartbeat(String state,long id){JsonObject h=new JsonObject();h.addProperty("instance_id","game-fixture");h.addProperty("observed_at_ms",System.currentTimeMillis());h.addProperty("state",state);h.addProperty("player_name","测试角色");h.addProperty("dimension","minecraft:overworld");JsonObject log=new JsonObject(),line=new JsonObject();line.addProperty("id",id);line.addProperty("text","[Client thread/INFO] 游戏日志测试 "+id);line.addProperty("level","INFO");JsonArray rows=new JsonArray();rows.add(line);log.add("lines",rows);log.addProperty("oldest_id",1);h.add("log",log);return h;}
 static void run(Path root)throws Exception{
  Engine e=new Engine(root.resolve("tools"));e.accept(sample(1,20,0));
  JsonObject hand=tool("minecraft:air",true,false,false,false,1,6),axe=tool("minecraft:iron_axe",true,false,false,true,1,6),low=tool("minecraft:wooden_pickaxe",true,false,true,true,2,6),pick=tool("minecraft:iron_pickaxe",true,true,true,true,6,6);
  long seq=2,id=1;
  for(JsonObject facts:new JsonObject[]{hand,axe,low}){JsonObject t=sample(seq,seq*20,0),event=event(id++,"mining_attempt");event.add("tool",facts);Json.arr(t,"events").add(event);JsonObject reward=e.accept(t);check(amount(reward,"tool_mismatch")>0&&Json.num(Json.obj(Json.obj(reward,"components"),"tool_mismatch"),"score",0)<0,"wrong tool penalized before any successful break");check(Json.arr(reward,"tool_evaluations").size()==1,"tool evidence included");seq++;}
  JsonObject repeated=sample(seq,seq*20,0),repeat=event(id-1,"mining_attempt");repeat.add("tool",low);Json.arr(repeated,"events").add(repeat);check(amount(e.accept(repeated),"tool_mismatch")==0,"retained events not charged again");seq++;
  JsonObject success=sample(seq,seq*20,0),broke=event(id++,"block_break");broke.add("tool",pick);Json.arr(success,"events").add(broke);check(amount(e.accept(success),"tool_correct")==1,"correct tool rewards first successful harvest");seq++;
  JsonObject placed=sample(seq,seq*20,0),place=event(id++,"block_place"),breakAgain=event(id++,"block_break");breakAgain.add("tool",pick);Json.arr(placed,"events").add(place);Json.arr(placed,"events").add(breakAgain);check(amount(e.accept(placed),"tool_correct")==0,"place-break cannot farm correct-tool reward");
  JsonObject rules=Json.obj(e.config(),"tool_rules"),wood=tool("minecraft:air",false,true,false,false,1,1);wood.addProperty("recommended_family","axe");check(ToolScoring.evaluate(wood,rules).penalty()==0,"bootstrap hand gathering stays neutral");wood.addProperty("best_speed",6);wood.addProperty("best_item","minecraft:iron_axe");check(ToolScoring.evaluate(wood,rules).kind().equals("better_tool_available"),"known better available tool detected");
  check(ToolScoring.evaluate(low,rules).kind().equals("insufficient_tier"),"tier distinguished from wrong family");rules.addProperty("enabled",false);check(ToolScoring.evaluate(hand,rules).penalty()==0,"tool rules can be disabled");
  JsonObject legacy=Config.defaults();legacy.remove("tool_rules");for(String m:new String[]{"tool_mismatch","tool_correct"}){Json.obj(legacy,"weights").remove(m);Json.obj(legacy,"caps").remove(m);}Json.obj(Json.obj(legacy,"weights"),"mining").addProperty("survival",7.25);Path upgrade=root.resolve("upgrade");Files.createDirectories(upgrade);Json.write(upgrade.resolve("config.json"),legacy);Engine migrated=new Engine(upgrade);check(Json.num(Json.obj(Json.obj(migrated.config(),"weights"),"mining"),"survival",0)==7.25&&Json.obj(migrated.config(),"weights").has("tool_mismatch"),"existing settings preserved by upgrade");
  Engine live=new Engine(root.resolve("live"));JsonObject hb=heartbeat("paused",1);live.receiveGame(hb);long count=(long)Json.num(live.logsAfter(0,""),"latest_id",0);live.receiveGame(hb);check(Json.num(live.logsAfter(0,""),"latest_id",0)==count,"heartbeat retries do not duplicate logs");check(Json.str(Json.obj(live.runtime(),"game"),"state","").equals("paused"),"pause visible without telemetry");
  RuntimeMonitor monitor=new RuntimeMonitor();monitor.receive(hb,row->{});check(Json.str(Json.obj(monitor.status(System.currentTimeMillis()+9000,0,5),"game"),"state","").equals("offline"),"lost heartbeat is not reported live");
  hb.addProperty("observed_at_ms",System.currentTimeMillis()-20000);monitor.receive(hb,row->{});check(Json.str(Json.obj(monitor.status(System.currentTimeMillis(),0,5),"game"),"state","").equals("client_unresponsive"),"HTTP worker heartbeat cannot hide frozen renderer");
  for(int i=0;i<1010;i++)monitor.add("game","INFO","bounded "+i,null);check(Json.arr(monitor.after(0,"game"),"logs").size()==1000&&Json.bool(monitor.after(0,"game"),"gap",false),"live history bounded with cursor gap");
  HttpServer server=ScorerApp.server(live,0);try{HttpClient client=HttpClient.newHttpClient();String base="http://127.0.0.1:"+server.getAddress().getPort();var response=client.send(HttpRequest.newBuilder(URI.create(base+"/v1/game-status")).header("Content-Type","application/json").POST(HttpRequest.BodyPublishers.ofString(Json.COMPACT.toJson(heartbeat("playing",2)))).build(),HttpResponse.BodyHandlers.ofString());check(response.statusCode()==200&&Json.num(JsonParser.parseString(response.body()).getAsJsonObject(),"log_ack",0)==2,"heartbeat log acknowledgment API");response=client.send(HttpRequest.newBuilder(URI.create(base+"/v1/logs?after=0")).GET().build(),HttpResponse.BodyHandlers.ofString());check(response.statusCode()==200&&response.body().contains("游戏日志测试"),"game logs API");}finally{server.stop(0);}
  modelCommunication(root.resolve("model-trace"));System.out.println("PASS: mining attempt penalties, tool tiers, bootstrap exception, success-only reward, migration, pause/offline/frozen status, log dedup/bounds/API, LLM communication/error/timeout/redaction");
 }
 static void modelCommunication(Path root)throws Exception{
  Engine e=new Engine(root);e.accept(sample(1,20,0));HttpServer mock=HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);AtomicInteger mode=new AtomicInteger();CountDownLatch request=new CountDownLatch(1),release=new CountDownLatch(1);String secret="fixture-model-secret";
  mock.createContext("/v1/chat/completions",ex->{ex.getRequestBody().readAllBytes();request.countDown();int current=mode.get();try{if(current==0)release.await(4,TimeUnit.SECONDS);if(current==2)Thread.sleep(1600);String body=current==0?"{\"choices\":[{\"message\":{\"content\":\"{\\\"weights\\\":{\\\"resource_gain\\\":1.1},\\\"reason\\\":\\\"工具与资源状态\\\"}\"}}]}":"{\"error\":\"mock failure "+secret+"\"}";byte[] bytes=body.getBytes(StandardCharsets.UTF_8);ex.sendResponseHeaders(current==1?503:200,bytes.length);ex.getResponseBody().write(bytes);}catch(Exception ignored){}finally{ex.close();}});mock.start();
  JsonObject c=e.config(),l=Json.obj(c,"llm");l.addProperty("base_url","http://127.0.0.1:"+mock.getAddress().getPort());l.addProperty("model","mock");l.addProperty("api_key",secret);l.addProperty("enabled",false);l.addProperty("timeout_seconds",1);e.configure(c,"test");
  try(LlmAdvisor advisor=new LlmAdvisor(e)){
   Thread worker=new Thread(advisor::analyze);worker.start();check(request.await(2,TimeUnit.SECONDS),"mock received request");check(Json.bool(Json.obj(e.runtime(),"llm"),"busy",false),"in-flight request visible before response");check(Json.COMPACT.toJson(e.logsAfter(0,"llm")).contains("/v1/chat/completions"),"request endpoint shown");release.countDown();worker.join(3000);check(!worker.isAlive()&&Json.str(Json.obj(e.runtime(),"llm"),"phase","").equals("applied"),"successful response and application visible");
   long version=e.version();mode.set(1);advisor.analyze();check(e.version()==version&&Json.num(Json.obj(e.runtime(),"llm"),"last_http_status",0)==503,"HTTP failure visible and weights unchanged");check(!Json.COMPACT.toJson(e.logsAfter(0,"llm")).contains(secret)&&!Files.readString(root.resolve("runtime.jsonl")).contains(secret),"API key redacted in live and disk logs");
   mode.set(2);advisor.analyze();check(e.version()==version&&Json.str(Json.obj(e.runtime(),"llm"),"connection","").equals("timeout"),"timeout visible and weights unchanged");
  }finally{release.countDown();mock.stop(0);}
 }
}
