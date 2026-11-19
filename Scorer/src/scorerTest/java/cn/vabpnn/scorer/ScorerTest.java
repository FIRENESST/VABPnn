package cn.vabpnn.scorer;
import com.google.gson.*;
import com.sun.net.httpserver.HttpServer;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import java.util.*;
import java.awt.image.BufferedImage;
import javax.swing.*;
import javax.imageio.ImageIO;

public final class ScorerTest {
 static void check(boolean ok,String label){if(!ok)throw new AssertionError(label);}
 static JsonObject sample(long seq,long tick,int iron){
  JsonObject t=new JsonObject();t.addProperty("schema",1);t.addProperty("world_id","test-world");t.addProperty("player_id","test-player");t.addProperty("session_id","test-session");t.addProperty("seq",seq);t.addProperty("tick",tick);
  JsonObject p=new JsonObject();for(String k:List.of("x","y","z"))p.addProperty(k,0);p.addProperty("health",20);p.addProperty("max_health",20);p.addProperty("food",20);p.addProperty("air",300);p.addProperty("max_air",300);p.addProperty("game_mode","survival");t.add("player",p);
  JsonObject inv=new JsonObject();inv.addProperty("minecraft:iron_ingot",iron);t.add("inventory",inv);t.add("world",new JsonObject());t.add("events",new JsonArray());t.add("advancements",new JsonArray());t.add("structures",new JsonArray());t.add("stats",new JsonObject());return t;
 }
 static double amount(JsonObject r,String m){return Json.num(Json.obj(Json.obj(r,"components"),m),"amount",-1);}
 static JsonObject event(long id,String type){JsonObject e=new JsonObject();e.addProperty("id",id);e.addProperty("type",type);e.addProperty("pos","0,0,0");e.addProperty("dimension","overworld");e.addProperty("block","minecraft:stone");return e;}
 public static void main(String[] args)throws Exception {
  Path root=Files.createTempDirectory("scorer-test-");Engine e=new Engine(root);
  RespawnTests.run(root);
  FeatureTests.run(root);
  ProgressionTests.run(root);
  TrainingLinkTests.run(root);
  check(!e.accept(sample(1,20,4)).get("valid").getAsBoolean(),"join establishes baseline");
  JsonObject r=e.accept(sample(2,40,5));check(Math.abs(amount(r,"resource_gain")-.3)<1e-8,"only net new resource high-water rewarded");
  check(amount(e.accept(sample(3,60,0)),"resource_gain")==0,"dropping no reward");check(amount(e.accept(sample(4,80,5)),"resource_gain")==0,"pickup no repeat reward");
  boolean rejected=false;try{e.accept(sample(4,80,5));}catch(IllegalArgumentException x){rejected=true;}check(rejected,"duplicates rejected");
  JsonObject block=sample(5,100,5);Json.arr(block,"events").add(event(1,"block_place"));Json.arr(block,"events").add(event(2,"block_break"));r=e.accept(block);check(amount(r,"mining")==0&&amount(r,"repetition")==1,"place/break loop cannot earn mining reward");
  JsonObject hurt=sample(6,120,5);Json.obj(hurt,"stats").addProperty("damage_taken",40);check(amount(e.accept(hurt),"damage")==4,"damage stats use tenths of health");
  JsonObject adv=sample(7,140,5);Json.arr(adv,"advancements").add("minecraft:story/smelt_iron");r=e.accept(adv);check(r.get("stage").getAsString().equals("iron")&&amount(r,"advancement")==1,"stage and first advancement");
  e.save();Engine restored=new Engine(root);JsonObject reconnect=sample(1,20,5);reconnect.addProperty("session_id","reconnected");restored.accept(reconnect);JsonObject next=sample(2,40,5);next.addProperty("session_id","reconnected");Json.arr(next,"advancements").add("minecraft:story/smelt_iron");check(amount(restored.accept(next),"advancement")==0,"persistent advancement dedup");
  JsonObject gap=sample(3,10000,100);gap.addProperty("session_id","reconnected");check(!restored.accept(gap).get("valid").getAsBoolean(),"long gap never pays backlog");
  JsonObject c=e.config();Json.arr(c,"locked_metrics").add("damage");e.configure(c,"test");long ver=e.version();double old=Json.num(Json.obj(Json.obj(c,"weights"),"resource_gain"),"iron",0);
  JsonObject proposal=JsonParser.parseString("{\"weights\":{\"resource_gain\":999,\"damage\":99,\"building\":9},\"reason\":\"test\"}").getAsJsonObject();e.applyProposal(proposal,"iron",ver);
  c=e.config();check(Json.num(Json.obj(Json.obj(c,"weights"),"resource_gain"),"iron",0)<=old*1.2+1e-8,"LLM step limited");check(Json.num(Json.obj(Json.obj(c,"weights"),"damage"),"iron",0)<0,"locked penalty unchanged");check(Json.num(Json.obj(Json.obj(c,"weights"),"building"),"iron",-1)==0,"disabled stays disabled");
  rejected=false;try{e.applyProposal(proposal,"iron",ver);}catch(IllegalArgumentException x){rejected=true;}check(rejected,"stale proposal rejected");
  ver=e.version();rejected=false;try{e.applyProposal(JsonParser.parseString("{\"weights\":{\"resource_gain\":1,\"invented\":2}}").getAsJsonObject(),"iron",ver);}catch(IllegalArgumentException x){rejected=true;}check(rejected&&e.version()==ver,"unknown metric rejected atomically");
  JsonObject bad=e.config();Json.obj(Json.obj(bad,"weights"),"damage").addProperty("iron",Double.NaN);rejected=false;try{e.configure(bad,"test");}catch(IllegalArgumentException x){rejected=true;}check(rejected,"nonfinite configuration rejected");
  Engine modes=new Engine(root.resolve("modes"));JsonObject creative=sample(1,20,1000);Json.obj(creative,"player").addProperty("game_mode","creative");check(!modes.accept(creative).get("valid").getAsBoolean(),"creative not rewarded");modes.accept(sample(2,40,0));check(amount(modes.accept(sample(3,60,1)),"resource_gain")>.29,"creative inventory never poisons survival high-water");
  JsonObject overflow=sample(4,80,2),coverage=new JsonObject();coverage.addProperty("oldest_event_id",400);overflow.add("coverage",coverage);Json.arr(overflow,"events").add(event(401,"block_break"));check(modes.accept(overflow).get("reason").getAsString().equals("event_gap"),"lost event interval rejected");
  Engine gear=new Engine(root.resolve("gear"));JsonObject gearFirst=sample(1,20,0);gearFirst.add("inventory_slots",JsonParser.parseString("[{\"item\":\"pickaxe\",\"stack_identity\":1,\"damage\":5}]"));gear.accept(gearFirst);
  JsonObject gearNext=sample(2,40,0);gearNext.add("inventory_slots",JsonParser.parseString("[{\"item\":\"pickaxe\",\"stack_identity\":1,\"damage\":7},{\"item\":\"pickaxe\",\"stack_identity\":2,\"damage\":60}]"));check(amount(gear.accept(gearNext),"durability_loss")==2,"only actual tracked wear, not picking up damaged equipment");
  JsonObject nbt=sample(3,60,0);Json.obj(nbt,"player").addProperty("nbt_snbt","SECRET_FIXTURE_NBT");gear.accept(nbt);check(!Json.COMPACT.toJson(gear.minuteSummary()).contains("SECRET_FIXTURE_NBT"),"raw NBT excluded from LLM summary");
  Engine httpEngine=new Engine(root.resolve("http"));HttpServer server=ScorerApp.server(httpEngine,0);int port=server.getAddress().getPort();HttpClient client=HttpClient.newHttpClient();
  try{
   var response=client.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+port+"/v1/telemetry")).header("Content-Type","application/json").POST(HttpRequest.BodyPublishers.ofString(Json.COMPACT.toJson(sample(1,20,0)))).build(),HttpResponse.BodyHandlers.ofString());
   check(response.statusCode()==200,"HTTP collector to scorer end-to-end");
   response=client.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+port+"/v1/rewards?after=0")).GET().build(),HttpResponse.BodyHandlers.ofString());check(JsonParser.parseString(response.body()).getAsJsonObject().getAsJsonArray("rewards").size()==1,"reward cursor API");
   response=client.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+port+"/v1/status")).header("Origin","https://example.invalid").GET().build(),HttpResponse.BodyHandlers.ofString());check(response.statusCode()==403,"browser cross-origin request rejected");
  }finally{server.stop(0);}
  HttpServer model=HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);model.createContext("/v1/models",ex->{byte[] b="{\"data\":[{\"id\":\"test-model\"}]}".getBytes();ex.sendResponseHeaders(200,b.length);ex.getResponseBody().write(b);ex.close();});
  model.createContext("/v1/chat/completions",ex->{String request=new String(ex.getRequestBody().readAllBytes());check(request.contains("test-model")&&request.contains("amounts"),"minute summary sent to model");JsonObject response=new JsonObject(),choice=new JsonObject(),message=new JsonObject();message.addProperty("content","{\"weights\":{\"resource_gain\":1.05},\"reason\":\"mock local model\"}");choice.add("message",message);JsonArray a=new JsonArray();a.add(choice);response.add("choices",a);byte[] b=Json.COMPACT.toJson(response).getBytes();ex.sendResponseHeaders(200,b.length);ex.getResponseBody().write(b);ex.close();});model.start();
  c=httpEngine.config();Json.obj(c,"llm").addProperty("base_url","http://127.0.0.1:"+model.getAddress().getPort());Json.obj(c,"llm").addProperty("enabled",false);httpEngine.configure(c,"test");
  try(LlmAdvisor advisor=new LlmAdvisor(httpEngine)){long before=httpEngine.version();advisor.analyze();check(httpEngine.version()==before+1,"OpenAI-compatible discovery/chat/validated update");
   httpEngine.accept(sample(2,40,3));httpEngine.receiveGame(FeatureTests.heartbeat("playing",1));httpEngine.trace("behavior","WARN","minecraft:stone ← minecraft:air：工具类型不适合，无法获得掉落物",FeatureTests.tool("minecraft:air",true,false,false,false,1,6));
   SwingUtilities.invokeAndWait(()->{try{JPanel panel=ScorerApp.panel(httpEngine,advisor);panel.setSize(1200,840);layout(panel);Files.createDirectories(Path.of("build/reports"));for(int tab:new int[]{0,1,2,6,7,8,9,10,11,12,13}){for(var component:panel.getComponents())if(component instanceof JTabbedPane tabs)tabs.setSelectedIndex(tab);layout(panel);BufferedImage image=new BufferedImage(1200,840,BufferedImage.TYPE_INT_RGB);var graphics=image.createGraphics();panel.printAll(graphics);graphics.dispose();ImageIO.write(image,"png",Path.of("build/reports/scorer-ui-"+tab+".png").toFile());}((javax.swing.Timer)panel.getClientProperty("timer")).stop();}catch(Exception x){throw new RuntimeException(x);}});
  }finally{model.stop(0);}
  System.out.println("PASS: delta rewards, persistent dedup, anti-loop, stages, gaps, config bounds, LLM guards, HTTP integration, mock model integration, Swing render");
 }
 static void layout(java.awt.Container c){c.doLayout();for(java.awt.Component x:c.getComponents())if(x instanceof java.awt.Container nested)layout(nested);}
}

