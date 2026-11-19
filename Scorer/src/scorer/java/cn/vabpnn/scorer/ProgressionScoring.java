package cn.vabpnn.scorer;

import com.google.gson.*;
import java.util.*;

/** Craft milestones and actual pickups have separate evidence and bounded durable state. */
final class ProgressionScoring {
 private final JsonObject state,craftRules,pickupRules,catalog;
 final JsonArray crafting=new JsonArray(),pickups=new JsonArray();
 double craftAmount,pickupAmount;
 private final long now;
 ProgressionScoring(JsonObject state,JsonObject config,JsonObject telemetry){
  this.state=state;craftRules=Json.obj(config,"crafting_rules");pickupRules=Json.obj(config,"pickup_rules");catalog=Json.obj(craftRules,"equipment");now=time(telemetry);
  expire(map("sources"),"created_at",Json.num(pickupRules,"source_window_seconds",600));
  expire(map("broken"),"created_at",Json.num(craftRules,"replacement_window_seconds",600));
  expire(map("pending"),"created_at",Json.num(craftRules,"replacement_use_seconds",120));
 }
 static long time(JsonObject telemetry){return (long)Json.num(Json.obj(telemetry,"world"),"game_time",Json.num(telemetry,"tick",0));}
 private JsonObject map(String key){if(!state.has(key))state.add(key,new JsonObject());return Json.obj(state,key);}
 private void expire(JsonObject entries,String key,double seconds){entries.entrySet().removeIf(e->{double age=now-Json.num(e.getValue().getAsJsonObject(),key,now);return age<0||age>seconds*20;});}
 private JsonObject spec(String item){return Json.obj(catalog,item);}
 private String family(String item){return Json.str(spec(item),"family","");}
 private double rank(String item){return Json.num(spec(item),"rank",0);}
 void baseline(JsonObject t){
  for(var e:Json.obj(t,"crafted_by_type").entrySet())if(catalog.has(e.getKey())&&e.getValue().getAsDouble()>0)map("seen_crafts").addProperty(e.getKey(),true);
  for(String item:Json.obj(t,"crafted_by_type").keySet())observeEquipment(item);
  observeInventory(t);map("pending").entrySet().clear();map("broken").entrySet().clear();map("uses").entrySet().clear();
 }
 private void observeEquipment(String item){String f=family(item);if(!f.isEmpty())map("best_rank").addProperty(f,Math.max(rank(item),Json.num(map("best_rank"),f,0)));}
 private void observeInventory(JsonObject t){for(var e:Json.obj(t,"inventory").entrySet())if(e.getValue().getAsDouble()>0)observeEquipment(e.getKey());for(var e:Json.obj(t,"equipment").entrySet())observeEquipment(e.getValue().getAsString());}
 void harvested(JsonObject event){
  JsonObject harvest=Json.obj(Json.obj(event,"tool"),"harvest");String id=Json.str(harvest,"source_id","");if(id.isEmpty()||id.length()>100||map("sources").has(id)||map("sources").size()>=2048)return;
  JsonObject items=Json.obj(harvest,"items");if(items.isEmpty()||items.size()>128)return;
  JsonObject source=new JsonObject();source.addProperty("created_at",now);source.add("items",items.deepCopy());map("sources").add(id,source);
 }
 void pickedUp(JsonObject event){
  String item=Json.str(event,"item","");double picked=Math.max(0,Json.num(event,"count",0)),remaining=picked,eligible=0;
  for(var entry:Json.obj(event,"origins").entrySet()){
   double claim=Math.max(0,Math.min(remaining,entry.getValue().getAsDouble()));remaining-=claim;JsonObject available=Json.obj(Json.obj(map("sources"),entry.getKey()),"items");double allowed=Math.min(claim,Math.max(0,Json.num(available,item,0)));
   if(allowed>0){eligible+=allowed;available.addProperty(item,Json.num(available,item,0)-allowed);}
  }
  double amount=Json.bool(pickupRules,"enabled",true)?eligible*Json.num(pickupRules,"per_item_amount",1):0;pickupAmount+=amount;
  JsonObject evidence=new JsonObject();evidence.addProperty("event_id",Json.num(event,"id",0));evidence.addProperty("item",item);evidence.addProperty("picked_count",picked);evidence.addProperty("eligible_count",eligible);evidence.addProperty("raw_amount",amount);evidence.addProperty("reason",eligible>0?"实际拾取有效采集的掉落物":"没有未消费的有效采集来源；玩家重丢、死亡掉落或未知来源不奖励");pickups.add(evidence);
 }
 void used(String item,long tick,String evidence){
  if(!catalog.has(item))return;map("uses").addProperty(item,Math.min(10000,Json.num(map("uses"),item,0)+1));String f=family(item);JsonObject pending=Json.obj(map("pending"),f);
  if(item.equals(Json.str(pending,"item",""))&&tick>Json.num(pending,"created_at",now)){
   map("pending").remove(f);double amount=Json.bool(craftRules,"enabled",true)?Json.num(craftRules,"replacement_amount",.5):0;craftAmount+=amount;
   JsonObject e=craftEvidence(item,"replacement_used","耐久耗尽后补造，并已用于"+evidence,amount);e.addProperty("broken_item",Json.str(pending,"broken_item",""));e.addProperty("crafted_at_tick",Json.num(pending,"created_at",0));crafting.add(e);
  }
 }
 void finish(JsonObject t,JsonObject before){
  observeInventory(before);
  for(var entry:Json.obj(before,"crafted_by_type").entrySet())if(entry.getValue().getAsDouble()>0)observeEquipment(entry.getKey());
  JsonObject stats=Json.obj(t,"stats"),oldStats=Json.obj(before,"stats");String held=Json.str(Json.obj(t,"equipment"),"mainhand","");
  if(Json.num(stats,"mob_kills",0)>Json.num(oldStats,"mob_kills",0))used(held,now,"实际击杀");
  if(Json.num(stats,"damage_blocked_by_shield",0)>Json.num(oldStats,"damage_blocked_by_shield",0)){String shield=Json.str(Json.obj(t,"equipment"),"offhand",held);used(shield,now,"有效格挡");}
  if(Json.num(stats,"fish_caught",0)>Json.num(oldStats,"fish_caught",0))used(held,now,"钓获物品");
  if(Json.num(stats,"damage_taken",0)>Json.num(oldStats,"damage_taken",0))for(String slot:List.of("head","chest","legs","feet")){String item=Json.str(Json.obj(t,"equipment"),slot,"");if(armorWorn(item,t,before))used(item,now,"承受伤害并消耗护甲耐久");}
  JsonObject broken=Json.obj(t,"broken_by_type"),oldBroken=Json.obj(before,"broken_by_type");
  for(String item:broken.keySet()){
   if(!catalog.has(item)||Json.num(broken,item,0)<=Json.num(oldBroken,item,0))continue;
   double uses=Json.num(map("uses"),item,0);map("uses").remove(item);
   if(uses>=Json.num(craftRules,"minimum_effective_uses",3)){JsonObject token=new JsonObject();token.addProperty("item",item);token.addProperty("rank",rank(item));token.addProperty("created_at",now);map("broken").add(family(item),token);}
  }
  JsonObject crafted=Json.obj(t,"crafted_by_type"),oldCrafted=Json.obj(before,"crafted_by_type");
  for(String item:crafted.keySet()){
   double delta=Json.num(crafted,item,0)-Json.num(oldCrafted,item,0);if(delta<=0||!catalog.has(item))continue;
   String f=family(item);double rank=rank(item),best=Json.num(map("best_rank"),f,0);boolean first=!map("seen_crafts").has(item)&&Json.num(oldCrafted,item,0)==0,upgrade=best>0&&rank>best;
   map("seen_crafts").addProperty(item,true);double firstValue=first&&rank>=best?Json.num(craftRules,"first_amount",1):0,upgradeValue=upgrade?Json.num(craftRules,"upgrade_amount",1.5):0;
   double amount=Json.bool(craftRules,"enabled",true)?Math.max(firstValue,upgradeValue):0;
   if(amount>0)map("pending").remove(f);
   JsonObject token=Json.obj(map("broken"),f);boolean replacement=!token.isEmpty()&&rank>=Json.num(token,"rank",0);
   String kind=upgrade?"upgrade":firstValue>0?"first_craft":"repeat",reason=upgrade?"主动合成高于历史已拥有等级的装备":firstValue>0?"首次合成这类关键物品":"重复合成或已拥有更高等级装备，不重复奖励";
   if(replacement){
    map("broken").remove(f);
    if(amount==0&&usableBackups(t,item,delta,Json.num(token,"rank",0))==0&&Json.bool(craftRules,"enabled",true)){
     JsonObject pending=new JsonObject();pending.addProperty("item",item);pending.addProperty("broken_item",Json.str(token,"item",""));pending.addProperty("created_at",now);map("pending").add(f,pending);kind="replacement_pending";reason="有效使用后耐久耗尽，已补造；等待后续有效使用确认用途";
    }
   }
   craftAmount+=amount;JsonObject e=craftEvidence(item,kind,reason,amount);e.addProperty("crafted_count",delta);e.addProperty("first_craft",first);e.addProperty("upgrade",upgrade);e.addProperty("previous_best_rank",best);e.addProperty("rank",rank);crafting.add(e);observeEquipment(item);
  }
  observeInventory(t);
 }
 private double usableBackups(JsonObject t,String crafted,double created,double requiredRank){
  double count=0;for(var e:Json.obj(t,"inventory").entrySet())if(family(e.getKey()).equals(family(crafted))&&rank(e.getKey())>=requiredRank)count+=e.getValue().getAsDouble();
  return Math.max(0,count-created);
 }
 private static boolean armorWorn(String item,JsonObject t,JsonObject before){
  if(item.isEmpty())return false;for(JsonElement v:Json.arr(t,"inventory_slots")){JsonObject current=v.getAsJsonObject();if(!item.equals(Json.str(current,"item","")))continue;for(JsonElement b:Json.arr(before,"inventory_slots")){JsonObject old=b.getAsJsonObject();if(item.equals(Json.str(old,"item",""))&&Json.str(current,"stack_identity","a").equals(Json.str(old,"stack_identity","b"))&&Json.num(current,"damage",0)>Json.num(old,"damage",0))return true;}}return false;
 }
 private static JsonObject craftEvidence(String item,String kind,String reason,double amount){JsonObject e=new JsonObject();e.addProperty("item",item);e.addProperty("kind",kind);e.addProperty("reason",reason);e.addProperty("raw_amount",amount);return e;}
}
