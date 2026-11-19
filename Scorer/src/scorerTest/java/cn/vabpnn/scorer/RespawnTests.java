package cn.vabpnn.scorer;
import com.google.gson.*;import com.sun.net.httpserver.*;import java.net.*;import java.nio.file.*;import java.nio.charset.*;import java.util.*;import java.util.concurrent.*;import java.util.concurrent.atomic.*;
import static cn.vabpnn.scorer.ScorerTest.*;
final class RespawnTests {
 static JsonObject event(long index,String previous,String current){
  var e=new JsonObject();e.addProperty("type","player_respawn");e.addProperty("event_id",UUID.randomUUID().toString());e.addProperty("world_id","test-world");
  e.addProperty("player_id","test-player");e.addProperty("server_session_id","test-server");e.addProperty("previous_session_id",previous);e.addProperty("session_id",current);
  e.addProperty("episode_index",index);e.addProperty("tick",index*40+1);e.addProperty("observed_at_ms",System.currentTimeMillis());return e;
 }
 static JsonObject life(long seq,long tick,long index,String session,int iron,JsonObject...events){
  var t=sample(seq,tick,iron);t.addProperty("session_id",session);var ep=new JsonObject();ep.addProperty("server_session_id","test-server");
  ep.addProperty("index",index);ep.addProperty("started_tick",index==0?0:index*40+1);t.add("episode",ep);
  var a=new JsonArray();for(var e:events)a.add(e.deepCopy());t.add("lifecycle_events",a);return t;
 }
 static void await(java.util.function.BooleanSupplier ok,long millis,String message)throws Exception{
  long deadline=System.currentTimeMillis()+millis;while(!ok.getAsBoolean()&&System.currentTimeMillis()<deadline)Thread.sleep(20);check(ok.getAsBoolean(),message);
 }
 static void run(Path root)throws Exception{
  var e=new Engine(root.resolve("respawn"));var c=e.config();Json.obj(Json.obj(c,"weights"),"resource_gain").addProperty("survival",7.25);
  e.configure(c,"manual");var first=life(1,20,0,"life-0",0);Json.arr(first,"advancements").add("minecraft:story/enter_the_end");e.accept(first);
  var before=life(2,40,0,"life-0",5);Json.arr(before,"advancements").add("minecraft:story/enter_the_end");
  Json.arr(before,"events").add(ScorerTest.event(1,"block_break"));e.accept(before);check(e.stage().equals("end"),"old life can reach end stage");
  long version=e.version();var weights=Json.obj(e.config(),"weights").deepCopy();var resetEvent=event(1,"life-0","life-1");
  var reset=life(3,42,1,"life-1",0,resetEvent);Json.obj(reset,"stats").addProperty("deaths",1);Json.obj(reset,"stats").addProperty("damage_taken",800);
  Json.arr(reset,"advancements").add("minecraft:story/enter_the_end");var baseline=e.accept(reset);
  check(!Json.bool(baseline,"valid",true)&&Json.num(baseline,"reward",999)==0,"new life establishes zero reward baseline despite cumulative statistics");
  check(e.stage().equals("survival")&&e.version()>version&&Json.obj(e.config(),"weights").equals(weights),"reset auto stage and keep every current weight");
  check(Json.num(Json.obj(e.status(),"episode"),"total_reward",999)==0&&e.pendingEpisodeNotices()==1,"score cleared and exactly one durable notification");
  var next=life(4,62,1,"life-1",5,resetEvent);Json.obj(next,"stats").addProperty("deaths",1);Json.obj(next,"stats").addProperty("damage_taken",800);
  Json.arr(next,"advancements").add("minecraft:story/enter_the_end");Json.arr(next,"events").add(ScorerTest.event(2,"block_break"));
  var reward=e.accept(next);check(amount(reward,"resource_gain")>0&&amount(reward,"mining")==1&&amount(reward,"death")==0&&amount(reward,"damage")==0,"fresh high-water/mining memory and no old life deltas");
  check(e.stage().equals("survival")&&e.pendingEpisodeNotices()==1,"old vanilla advancements and retained lifecycle event do not undo/reset again");
  next=life(5,82,1,"life-1",5,resetEvent);Json.arr(next,"advancements").add("minecraft:story/enter_the_end");Json.arr(next,"advancements").add("minecraft:story/mine_stone");e.accept(next);
  check(e.stage().equals("resources"),"new life can progress through a genuinely new advancement");
  boolean stale=false;try{var p=new JsonObject();p.add("weights",new JsonObject());e.applyProposal(p,"end",version);}catch(IllegalArgumentException expected){stale=true;}
  check(stale,"old life LLM advice cannot apply");
  var event2=event(2,"life-1","life-2");var instant=life(6,83,2,"life-2",0,resetEvent,event2);Json.obj(instant,"stats").addProperty("deaths",2);
  check(!Json.bool(e.accept(instant),"valid",true)&&e.pendingEpisodeNotices()==2,"immediate respawn detected while both sampled health values are positive");
  boolean old=false;try{e.accept(life(7,84,1,"life-1",9,resetEvent));}catch(IllegalArgumentException expected){old=true;}check(old,"old episode packet cannot roll current state back");
  e.save();var restarted=new Engine(root.resolve("respawn"));check(restarted.pendingEpisodeNotices()==2,"unacknowledged notices survive scorer restart");
  restarted.accept(life(7,103,2,"life-2",0,resetEvent,event2));check(restarted.pendingEpisodeNotices()==2,"restart cannot enqueue retained notices twice");
  check(Json.num(Json.obj(e.minuteSummary(),"episode"),"index",0)==2&&Json.str(e.minuteSummary(),"session_id","").equals("life-2"),"future advisor summaries carry new episode identity");
  var legacy=new Engine(root.resolve("legacy-respawn"));legacy.accept(sample(1,20,0));var living=sample(2,40,5);legacy.accept(living);
  var auto=sample(3,60,0);Json.obj(auto,"stats").addProperty("deaths",1);check(!Json.bool(legacy.accept(auto),"valid",true),"legacy instant death-stat fallback establishes baseline");
  notification(root.resolve("notify"));staleAdvice(root.resolve("stale-advice"));
  System.out.println("PASS: respawn resets score/stage/high-water/behavior chains, preserves weights, fresh baseline, exact episode replay guards, durable/deduplicated HTTP notices and stale LLM advice rejection.");
 }
 static void configureModel(Engine e,int port)throws Exception{
  var c=e.config();var l=Json.obj(c,"llm");l.addProperty("enabled",true);l.addProperty("base_url","http://127.0.0.1:"+port);l.addProperty("model","respawn-fixture");l.addProperty("timeout_seconds",2);e.configure(c,"fixture");
 }
 static void notification(Path dir)throws Exception{
  var received=new CopyOnWriteArrayList<JsonObject>();var status=new AtomicInteger(503);
  var mock=HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);
  mock.createContext("/v1/chat/completions",x->{var body=JsonParser.parseString(new String(x.getRequestBody().readAllBytes(),StandardCharsets.UTF_8)).getAsJsonObject();received.add(body);
   String reply="{\"choices\":[{\"message\":{\"content\":\"{\\\"weights\\\":{\\\"resource_gain\\\":99},\\\"reason\\\":\\\"ack\\\"}\"}}]}";byte[] bytes=reply.getBytes(StandardCharsets.UTF_8);x.sendResponseHeaders(status.get(),bytes.length);try(var out=x.getResponseBody()){out.write(bytes);}});
  mock.start();var e=new Engine(dir);configureModel(e,mock.getAddress().getPort());var weights=Json.obj(e.config(),"weights").deepCopy();
  e.accept(life(1,20,0,"life-0",0));var one=event(1,"life-0","life-1");var two=event(2,"life-1","life-2");
  try(var advisor=new LlmAdvisor(e)){
   e.accept(life(2,42,1,"life-1",0,one));
   await(()->!received.isEmpty()&&Json.str(Json.obj(Json.obj(e.status(),"episode"),"notification"),"state","").equals("retry_pending"),4000,"failed notification is visible");
   check(e.pendingEpisodeNotices()==1&&Json.obj(e.config(),"weights").equals(weights),"offline service leaves pending event and existing weights");
   e.accept(life(3,82,2,"life-2",0,one,two));status.set(200);
   await(()->e.pendingEpisodeNotices()==0,6500,"retry delivers every accumulated respawn notification");
   var finalRequest=received.getLast();var payload=JsonParser.parseString(finalRequest.getAsJsonArray("messages").get(1).getAsJsonObject().get("content").getAsString()).getAsJsonObject();
   check(Json.str(payload,"event_type","").equals("player_respawn")&&Json.arr(payload,"events").size()==2,"real chat request contains both event identities");
   check(Json.obj(e.config(),"weights").equals(weights),"notification response containing weights is never applied");
   int count=received.size();e.accept(life(4,102,2,"life-2",0,one,two));Thread.sleep(100);check(received.size()==count,"retained events never resend after acknowledgement");
  }finally{mock.stop(0);}
 }
 static void staleAdvice(Path dir)throws Exception{
  var begun=new CountDownLatch(1);var release=new CountDownLatch(1);
  var mock=HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);
  mock.setExecutor(Executors.newCachedThreadPool(r->{var t=new Thread(r,"respawn-model-test");t.setDaemon(true);return t;}));
  mock.createContext("/v1/chat/completions",x->{begun.countDown();try{release.await(4,TimeUnit.SECONDS);}catch(InterruptedException interrupted){Thread.currentThread().interrupt();}
   byte[] b="{\"choices\":[{\"message\":{\"content\":\"{\\\"weights\\\":{\\\"resource_gain\\\":1.1},\\\"reason\\\":\\\"old episode\\\"}\"}}]}".getBytes(StandardCharsets.UTF_8);x.sendResponseHeaders(200,b.length);try(var out=x.getResponseBody()){out.write(b);}});
  mock.start();var e=new Engine(dir);configureModel(e,mock.getAddress().getPort());e.receiveGame(FeatureTests.heartbeat("playing",1));e.accept(life(1,20,0,"life-0",0));
  var weights=Json.obj(e.config(),"weights").deepCopy();
  try(var advisor=new LlmAdvisor(e)){
   var worker=new Thread(advisor::analyze);worker.start();check(begun.await(2,TimeUnit.SECONDS),"old analysis reached actual model HTTP");
   var event=event(1,"life-0","life-1");e.accept(life(2,42,1,"life-1",0,event));release.countDown();worker.join(3000);
   check(!worker.isAlive()&&Json.obj(e.config(),"weights").equals(weights),"in-flight old analysis discarded after respawn");
  }finally{release.countDown();mock.stop(0);}
 }
}
