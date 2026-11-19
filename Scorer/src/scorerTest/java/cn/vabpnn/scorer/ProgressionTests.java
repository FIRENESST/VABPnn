package cn.vabpnn.scorer;

import com.google.gson.*;
import java.nio.file.*;
import static cn.vabpnn.scorer.ScorerTest.*;

final class ProgressionTests {
 static JsonObject snapshot(long seq){JsonObject t=sample(seq,seq*20,0);t.add("crafted_by_type",new JsonObject());t.add("broken_by_type",new JsonObject());t.add("equipment",new JsonObject());return t;}
 static JsonObject next(JsonObject last){JsonObject t=last.deepCopy();t.addProperty("seq",Json.num(last,"seq",0)+1);t.addProperty("tick",Json.num(last,"tick",0)+20);t.add("events",new JsonArray());return t;}
 static JsonObject harvest(long id,String source,String item,int count){JsonObject e=event(id,"block_break");e.addProperty("pos",id+",0,0");e.addProperty("tick",40);JsonObject tool=FeatureTests.tool("minecraft:wooden_axe",false,true,true,true,2,2),h=new JsonObject(),items=new JsonObject();h.addProperty("source_id",source);items.addProperty(item,count);h.add("items",items);tool.add("harvest",h);e.add("tool",tool);return e;}
 static JsonObject pickup(long id,String item,int count,JsonObject origins){JsonObject e=event(id,"item_pickup");e.addProperty("item",item);e.addProperty("count",count);e.add("origins",origins);return e;}
 static JsonObject origin(String key,int count){JsonObject o=new JsonObject();o.addProperty(key,count);return o;}
 static void run(Path root)throws Exception{
  Engine e=new Engine(root.resolve("craft"));JsonObject t=snapshot(1);e.accept(t);
  t=next(t);Json.obj(t,"crafted_by_type").addProperty("minecraft:wooden_axe",1);Json.obj(t,"inventory").addProperty("minecraft:wooden_axe",1);check(amount(e.accept(t),"crafting_milestone")==1,"first wooden axe milestone");
  t=next(t);Json.obj(t,"crafted_by_type").addProperty("minecraft:wooden_axe",4);Json.obj(t,"inventory").addProperty("minecraft:wooden_axe",4);check(amount(e.accept(t),"crafting_milestone")==0,"batch duplicate crafting does not farm milestones");
  t=next(t);Json.obj(t,"crafted_by_type").addProperty("minecraft:iron_pickaxe",1);Json.obj(t,"inventory").addProperty("minecraft:iron_pickaxe",1);check(amount(e.accept(t),"crafting_milestone")==1,"first iron pickaxe independent of inventory resource values");
  t=next(t);Json.obj(t,"crafted_by_type").addProperty("minecraft:iron_axe",1);Json.obj(t,"inventory").addProperty("minecraft:iron_axe",1);JsonObject r=e.accept(t);check(amount(r,"crafting_milestone")==1.5,"first and upgrade take the larger value, not both");
  t=next(t);Json.obj(t,"crafted_by_type").addProperty("minecraft:stone_axe",1);check(amount(e.accept(t),"crafting_milestone")==0,"obsolete lower tier is not a milestone");e.save();
  Engine restored=new Engine(root.resolve("craft"));t=next(t);t.addProperty("session_id","reconnected");restored.accept(t);t=next(t);Json.obj(t,"crafted_by_type").addProperty("minecraft:iron_axe",2);check(amount(restored.accept(t),"crafting_milestone")==0,"milestone dedup survives restart");
  replacements(root);pickups(root);
  JsonObject legacy=Config.defaults();legacy.remove("crafting_rules");legacy.remove("pickup_rules");for(String metric:new String[]{"crafting_milestone","item_pickup"}){Json.obj(legacy,"weights").remove(metric);Json.obj(legacy,"caps").remove(metric);}Json.obj(Json.obj(legacy,"weights"),"tool_mismatch").addProperty("survival",-.8);Path old=root.resolve("legacy02");Files.createDirectories(old);Json.write(old.resolve("config.json"),legacy);Engine upgraded=new Engine(old);check(Json.obj(upgraded.config(),"weights").has("item_pickup")&&Json.num(Json.obj(Json.obj(upgraded.config(),"weights"),"tool_mismatch"),"survival",0)==-.8,"0.2 configuration migration preserves custom weights");
  System.out.println("PASS: first crafts, upgrades, duplicate/downgrade rejection, depletion/replacement/use chain, backups, pickup provenance/counts/replays/gaps, restart persistence and config migration");
 }
 private static void replacements(Path root)throws Exception{
  for(boolean backup:new boolean[]{false,true}){
   Engine e=new Engine(root.resolve(backup?"backup":"replace"));JsonObject t=snapshot(1);Json.obj(t,"inventory").addProperty("minecraft:wooden_axe",backup?2:1);Json.obj(t,"crafted_by_type").addProperty("minecraft:wooden_axe",1);e.accept(t);
   t=next(t);for(int i=1;i<=3;i++)Json.arr(t,"events").add(harvest(i,"r"+i,"minecraft:oak_log",1));e.accept(t);
   t=next(t);Json.obj(t,"broken_by_type").addProperty("minecraft:wooden_axe",1);Json.obj(t,"inventory").addProperty("minecraft:wooden_axe",backup?1:0);e.accept(t);
   t=next(t);Json.obj(t,"crafted_by_type").addProperty("minecraft:wooden_axe",2);Json.obj(t,"inventory").addProperty("minecraft:wooden_axe",backup?2:1);check(amount(e.accept(t),"crafting_milestone")==0,"replacement waits for effective use");
   t=next(t);JsonObject action=harvest(4,"r4","minecraft:oak_log",1);action.addProperty("tick",100);Json.arr(t,"events").add(action);check(amount(e.accept(t),"crafting_milestone")==(backup?0:.5),backup?"spare equipment prevents unnecessary replacement reward":"depleted crafted tool earns replacement reward upon use");
   t=next(t);action=harvest(5,"r5","minecraft:oak_log",1);action.addProperty("tick",120);Json.arr(t,"events").add(action);check(amount(e.accept(t),"crafting_milestone")==0,"replacement token consumed exactly once");
   t=next(t);Json.obj(t,"crafted_by_type").addProperty("minecraft:wooden_axe",3);check(amount(e.accept(t),"crafting_milestone")==0,"inventory loss or repeat crafting alone cannot fake depletion");
  }
  Engine e=new Engine(root.resolve("replacement-gap"));JsonObject t=snapshot(1);Json.obj(t,"crafted_by_type").addProperty("minecraft:wooden_axe",1);e.accept(t);t=next(t);Json.obj(t,"broken_by_type").addProperty("minecraft:wooden_axe",1);e.accept(t);t=next(t);Json.obj(t,"crafted_by_type").addProperty("minecraft:wooden_axe",2);check(amount(e.accept(t),"crafting_milestone")==0,"breaking an unused tool creates no replacement credit");
 }
 private static void pickups(Path root)throws Exception{
  Engine e=new Engine(root.resolve("pickup"));JsonObject t=snapshot(1);e.accept(t);t=next(t);Json.arr(t,"events").add(harvest(1,"source-a","minecraft:oak_log",4));Json.arr(t,"events").add(harvest(2,"source-b","minecraft:cobblestone",2));e.accept(t);
  t=next(t);Json.arr(t,"events").add(pickup(3,"minecraft:oak_log",1,origin("source-a",1)));JsonObject r=e.accept(t);check(amount(r,"item_pickup")==1&&Json.num(Json.obj(Json.obj(r,"components"),"item_pickup"),"score",0)==.01,"partial pickup rewards only actual quantity");
  t=next(t);Json.arr(t,"events").add(pickup(4,"minecraft:oak_log",3,origin("source-a",3)));check(amount(e.accept(t),"item_pickup")==3,"remaining pickup consumes only remaining source credit");
  t=next(t);Json.arr(t,"events").add(pickup(5,"minecraft:oak_log",4,new JsonObject()));check(amount(e.accept(t),"item_pickup")==0,"redropped or unknown-origin items get no reward");
  t=next(t);Json.arr(t,"events").add(pickup(6,"minecraft:oak_log",4,origin("source-a",4)));check(amount(e.accept(t),"item_pickup")==0,"exhausted source cannot be replayed with a new event id");
  e.save();Engine saved=new Engine(root.resolve("pickup"));t=next(t);t.addProperty("session_id","reloaded");saved.accept(t);t=next(t);Json.arr(t,"events").add(pickup(7,"minecraft:cobblestone",2,origin("source-b",2)));check(amount(saved.accept(t),"item_pickup")==2,"uncollected known source persists across restart");
  t=next(t);JsonObject place=event(8,"block_place");place.addProperty("pos","9,0,0");Json.arr(t,"events").add(place);Json.arr(t,"events").add(harvest(9,"placed-source","minecraft:oak_log",1));Json.arr(t,"events").add(pickup(10,"minecraft:oak_log",1,origin("placed-source",1)));check(amount(saved.accept(t),"item_pickup")==0,"place-break-pickup loops excluded");
  t=next(t);t.addProperty("tick",20000);Json.arr(t,"events").add(harvest(11,"gap-source","minecraft:oak_log",1));check(!Json.bool(saved.accept(t),"valid",true),"long gap establishes baseline");t=next(t);Json.arr(t,"events").add(pickup(12,"minecraft:oak_log",1,origin("gap-source",1)));check(amount(saved.accept(t),"item_pickup")==0,"unobserved source during gap not invented");
 }
}
