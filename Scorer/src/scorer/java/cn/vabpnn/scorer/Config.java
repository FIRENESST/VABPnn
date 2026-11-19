package cn.vabpnn.scorer;
import com.google.gson.*;
import java.util.*;
final class Config {
 static final String[] STAGES={"survival","resources","iron","diamond","nether","end"};
 static final String[] METRICS={"resource_gain","advancement","exploration","structure_discovery","mining","building","crafting","combat","shielding","eating","damage","death","air_risk","hunger_risk","lava_risk","durability_loss","idle","repetition","resource_search","preparation","creative_strategy","tool_mismatch","tool_correct","crafting_milestone","item_pickup"};
 static final String[] LABELS={"资源库存新高","新进度","探索新区域","发现结构","首次有效挖掘","有效放置","合成","完成击杀","盾牌格挡","按需进食","受到伤害","死亡","缺氧危险","饥饿危险","岩浆危险","装备耐久损耗","无进展停滞","重复放挖","寻找资源（推断）","行前准备（推断）","有效新策略（推断）","挖掘工具不适配","正确工具有效采集","关键合成与装备升级","有效采集物拾取"};
 static JsonObject defaults(){
  JsonObject c=new JsonObject();c.addProperty("schema",1);c.addProperty("stage_override","auto");c.addProperty("reward_clip",20);c.addProperty("max_gap_seconds",10);c.addProperty("idle_seconds",45);c.addProperty("novelty_cooldown_seconds",120);c.addProperty("history_limit",50000);c.addProperty("journal_limit_mb",32);
  JsonObject co=new JsonObject();co.addProperty("sample_ticks",20);co.addProperty("chunk_radius",10);co.addProperty("chunks_per_tick",2);co.addProperty("entity_radius",64);co.addProperty("max_entities",64);co.addProperty("nbt_chars",4096);co.addProperty("nbt_entities",8);co.addProperty("nbt_block_entities",16);co.addProperty("visible_rays_per_tick",24);co.addProperty("visible_range",160);c.add("collector",co);
  JsonObject llm=new JsonObject();llm.addProperty("enabled",true);llm.addProperty("base_url","http://127.0.0.1:8080");llm.addProperty("model","");llm.addProperty("api_key","");llm.addProperty("interval_seconds",60);llm.addProperty("timeout_seconds",20);llm.addProperty("max_relative_change",.2);llm.addProperty("max_absolute_change",.25);llm.addProperty("max_weight",20);c.add("llm",llm);
  JsonObject weights=new JsonObject(),caps=new JsonObject();double[] base={1,8,.3,2,.03,0,.15,.04,.03,.2,-.15,-8,-.1,-.03,-.3,-.01,-.03,-.2,.4,.3,.5,-.35,.02,1,.01};
  for(int i=0;i<METRICS.length;i++){JsonObject row=new JsonObject();for(String stage:STAGES){double v=base[i];if(stage.equals("survival")&&(i==14||i==10||i==11))v*=1.5;if(stage.equals("resources")&&(i==0||i==4))v*=1.5;if(stage.equals("nether")&&(i==1||i==3))v*=1.25;row.addProperty(stage,Math.rint(v*1000000)/1000000);}weights.add(METRICS[i],row);caps.addProperty(METRICS[i],i==1||i==11?4:20);}
  c.add("weights",weights);c.add("caps",caps);c.add("locked_metrics",new JsonArray());
  caps.addProperty("crafting_milestone",6);caps.addProperty("item_pickup",128);
  JsonArray modes=new JsonArray();modes.add("survival");modes.add("adventure");c.add("allowed_game_modes",modes);
  JsonObject rules=new JsonObject();rules.addProperty("resource_search_seconds",120);rules.addProperty("prepared_food",16);rules.addProperty("prepared_health_fraction",.8);rules.addProperty("eating_food_threshold",18);rules.addProperty("hunger_food_threshold",6);c.add("inference_rules",rules);
  JsonObject tool=new JsonObject();tool.addProperty("enabled",true);tool.addProperty("wrong_tool_amount",1);tool.addProperty("insufficient_tier_amount",1.5);tool.addProperty("inefficient_amount",.25);tool.addProperty("minimum_speed_ratio",.5);c.add("tool_rules",tool);
  JsonObject craft=new JsonObject();craft.addProperty("enabled",true);craft.addProperty("first_amount",1);craft.addProperty("upgrade_amount",1.5);craft.addProperty("replacement_amount",.5);craft.addProperty("minimum_effective_uses",3);craft.addProperty("replacement_window_seconds",600);craft.addProperty("replacement_use_seconds",120);
  JsonObject gear=new JsonObject();String[] materials={"wooden","golden","stone","copper","iron","diamond","netherite"};double[] ranks={1,1,2,2.5,3,4,5};
  for(int i=0;i<materials.length;i++)for(String family:List.of("pickaxe","axe","shovel","hoe","sword"))equipment(gear,materials[i]+"_"+family,family,ranks[i]);
  String[] armor={"leather","golden","chainmail","copper","iron","diamond","netherite"};double[] armorRanks={1,1.5,2,2.5,3,4,5};for(int i=0;i<armor.length;i++)for(String family:List.of("helmet","chestplate","leggings","boots"))equipment(gear,armor[i]+"_"+family,family,armorRanks[i]);
  for(String id:List.of("shield","bow","crossbow","fishing_rod","shears","bucket","flint_and_steel","crafting_table","furnace"))equipment(gear,id,id,1);craft.add("equipment",gear);c.add("crafting_rules",craft);
  JsonObject pickup=new JsonObject();pickup.addProperty("enabled",true);pickup.addProperty("per_item_amount",1);pickup.addProperty("source_window_seconds",600);c.add("pickup_rules",pickup);
  JsonObject values=new JsonObject();String[] ids={"oak_log","birch_log","spruce_log","coal","raw_iron","iron_ingot","diamond","obsidian","blaze_rod","ender_pearl","gold_ingot","cobblestone","bread","cooked_beef"};double[] worth={.05,.05,.05,.1,.2,.3,2,.4,2,1,.2,.01,.1,.15};for(int i=0;i<ids.length;i++)values.addProperty("minecraft:"+ids[i],worth[i]);c.add("resource_values",values);
  JsonObject stages=new JsonObject();stages.addProperty("resources","minecraft:story/mine_stone");stages.addProperty("iron","minecraft:story/smelt_iron");stages.addProperty("diamond","minecraft:story/mine_diamond");stages.addProperty("nether","minecraft:story/enter_the_nether");stages.addProperty("end","minecraft:story/enter_the_end");c.add("stage_advancements",stages);return c;
 }
 static void validate(JsonObject c){
  for(String k:defaults().keySet())if(!c.has(k))throw new IllegalArgumentException("缺少配置: "+k);
  String stage=Json.str(c,"stage_override","auto");if(!stage.equals("auto")&&!Arrays.asList(STAGES).contains(stage))throw new IllegalArgumentException("未知阶段");
  bounded(c,"reward_clip",.01,1000);bounded(c,"max_gap_seconds",1,120);bounded(c,"idle_seconds",1,3600);bounded(c,"novelty_cooldown_seconds",1,86400);bounded(c,"history_limit",100,1000000);bounded(c,"journal_limit_mb",1,256);
  for(String m:METRICS){for(String s:STAGES)bounded(Json.obj(Json.obj(c,"weights"),m),s,-100,100);bounded(Json.obj(c,"caps"),m,0,100000);}
  for(JsonElement m:Json.arr(c,"locked_metrics"))if(!Arrays.asList(METRICS).contains(m.getAsString()))throw new IllegalArgumentException("未知锁定项");
  for(String id:Json.obj(c,"resource_values").keySet())bounded(Json.obj(c,"resource_values"),id,0,1000);
  JsonObject rules=Json.obj(c,"inference_rules");bounded(rules,"resource_search_seconds",1,3600);bounded(rules,"prepared_food",0,20);bounded(rules,"prepared_health_fraction",0,1);bounded(rules,"eating_food_threshold",0,20);bounded(rules,"hunger_food_threshold",1,20);
  for(JsonElement mode:Json.arr(c,"allowed_game_modes"))if(!Set.of("survival","adventure","creative","spectator").contains(mode.getAsString()))throw new IllegalArgumentException("无效游戏模式");
  JsonObject tool=Json.obj(c,"tool_rules");for(String key:List.of("wrong_tool_amount","insufficient_tier_amount","inefficient_amount"))bounded(tool,key,0,10);bounded(tool,"minimum_speed_ratio",0,1);
  JsonObject craft=Json.obj(c,"crafting_rules");for(String key:List.of("first_amount","upgrade_amount","replacement_amount"))bounded(craft,key,0,100);bounded(craft,"minimum_effective_uses",1,10000);bounded(craft,"replacement_window_seconds",1,3600);bounded(craft,"replacement_use_seconds",1,3600);
  JsonObject gear=Json.obj(craft,"equipment");if(gear.size()>512)throw new IllegalArgumentException("关键合成物品配置最多 512 项");for(var entry:gear.entrySet()){if(!entry.getKey().matches("[a-z0-9_.-]+:[a-z0-9_./-]+"))throw new IllegalArgumentException("无效物品注册名");JsonObject spec=entry.getValue().getAsJsonObject();bounded(spec,"rank",.01,100);String family=Json.str(spec,"family","");if(family.isEmpty()||family.length()>64)throw new IllegalArgumentException("装备类别不能为空且不能超过 64 字符");}
  JsonObject pickup=Json.obj(c,"pickup_rules");bounded(pickup,"per_item_amount",0,100);bounded(pickup,"source_window_seconds",1,3600);
  JsonObject co=Json.obj(c,"collector");bounded(co,"sample_ticks",10,200);bounded(co,"chunk_radius",1,10);bounded(co,"chunks_per_tick",1,8);bounded(co,"entity_radius",8,160);bounded(co,"max_entities",1,256);bounded(co,"nbt_chars",0,16384);bounded(co,"nbt_entities",0,32);bounded(co,"nbt_block_entities",0,64);bounded(co,"visible_rays_per_tick",1,128);bounded(co,"visible_range",16,160);
  JsonObject llm=Json.obj(c,"llm");bounded(llm,"interval_seconds",60,3600);bounded(llm,"timeout_seconds",1,55);bounded(llm,"max_relative_change",0,1);bounded(llm,"max_absolute_change",0,10);bounded(llm,"max_weight",0,100);
  java.net.URI u=java.net.URI.create(Json.str(llm,"base_url",""));if(!Set.of("127.0.0.1","localhost","::1","[::1]").contains(u.getHost())||!Set.of("http","https").contains(u.getScheme()))throw new IllegalArgumentException("大模型地址须为本地回环 HTTP(S) 地址");
 }
 static JsonObject upgrade(JsonObject old){
  JsonObject c=old.deepCopy(),defaults=defaults();for(String key:List.of("tool_rules","crafting_rules","pickup_rules"))if(!c.has(key))c.add(key,defaults.get(key).deepCopy());
  for(String m:List.of("tool_mismatch","tool_correct","crafting_milestone","item_pickup")){
   if(!Json.obj(c,"weights").has(m))Json.obj(c,"weights").add(m,Json.obj(defaults,"weights").get(m).deepCopy());
   if(!Json.obj(c,"caps").has(m))Json.obj(c,"caps").add(m,Json.obj(defaults,"caps").get(m).deepCopy());
  }
  return c;
 }
 private static void equipment(JsonObject map,String item,String family,double rank){JsonObject entry=new JsonObject();entry.addProperty("family",family);entry.addProperty("rank",rank);map.add("minecraft:"+item,entry);}
 static void bounded(JsonObject o,String k,double lo,double hi){double n=Json.num(o,k,Double.NaN);if(!Double.isFinite(n)||n<lo||n>hi)throw new IllegalArgumentException(k+" 必须在 "+lo+" 到 "+hi+" 之间");}
}

