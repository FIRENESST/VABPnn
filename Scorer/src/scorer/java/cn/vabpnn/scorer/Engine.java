package cn.vabpnn.scorer;
import com.google.gson.*;
import java.nio.file.*;
import java.io.*;
import java.util.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/** Scores deltas only. Durable novelty state is separate from transient transport sessions. */
final class Engine {
 final Path dir;
 private JsonObject config,latest=new JsonObject(),lastReward=new JsonObject(),memory=new JsonObject();
 private JsonObject previous;
 private final Deque<JsonObject> rewards=new ArrayDeque<>(),minute=new ArrayDeque<>();
 private String identity="",session="",stage="survival";
 private final String streamId=UUID.randomUUID().toString();
 private long sequence=-1,eventSequence=0,rewardSequence=0,version=1,lastReceived=0,lastProgress=0,lastNovelty=0;
 private long savedAt=0,windowStart=0;
 private final Map<String,Double> minuteAmounts=new HashMap<>();
 private String llmStatus="等待游戏遥测";
 private final EpisodeNotices episodeNotices;
 private Runnable episodeListener=()->{};
 private String serverSession="";private long episodeIndex=-1,episodeStartTick;private double episodeTotal;
 private JsonObject noticeStatus=new JsonObject();
 private final RuntimeMonitor monitor=new RuntimeMonitor();
 private final TrainingLink training=new TrainingLink();
 Engine(Path dir)throws IOException {this.dir=dir;Files.createDirectories(dir);Path p=dir.resolve("config.json");config=Files.exists(p)?Config.upgrade(Json.read(p)):Config.defaults();Config.validate(config);Json.write(p,config);episodeNotices=new EpisodeNotices(dir);}
 synchronized void onEpisode(Runnable listener){episodeListener=listener;}
 synchronized long pendingEpisodeNotices(){return episodeNotices.pending();}
 synchronized List<JsonObject> episodeNoticeBatch()throws IOException{return episodeNotices.batch();}
 synchronized void episodeNoticeSent(JsonObject event)throws IOException{episodeNotices.acknowledge(event);noticeStatus.addProperty("state","sent");noticeStatus.addProperty("last_sent_id",Json.str(event,"event_id",""));noticeStatus.addProperty("last_sent_ms",System.currentTimeMillis());}
 synchronized void episodeNoticeState(String state,String error){noticeStatus.addProperty("state",state);noticeStatus.addProperty("error",error);}
 synchronized JsonObject episodeStatus(){var e=new JsonObject();e.addProperty("session_id",session);e.addProperty("server_session_id",serverSession);e.addProperty("index",Math.max(0,episodeIndex));e.addProperty("start_tick",episodeStartTick);e.addProperty("total_reward",episodeTotal);e.addProperty("stage",stage);e.addProperty("pending_llm_notifications",episodeNotices.pending());e.add("notification",noticeStatus.deepCopy());return e;}
 synchronized JsonObject config(){return config.deepCopy();}
 synchronized void configure(JsonObject c,String source)throws IOException {Config.validate(c);Json.write(dir.resolve("config.json"),c);config=c.deepCopy();version++;JsonObject audit=new JsonObject();audit.addProperty("source",source);audit.addProperty("version",version);audit.add("config",redact(c));journal("weights",audit);}
 private JsonObject redact(JsonObject c){JsonObject copy=c.deepCopy();Json.obj(copy,"llm").addProperty("api_key","[redacted]");return copy;}
 synchronized void llmStatus(String s){llmStatus=s;}
 synchronized void modelState(JsonObject state){monitor.model(state);}
 synchronized void trace(String channel,String level,String message,JsonElement details){persistTrace(monitor.add(channel,level,message,details));}
 private void persistTrace(JsonObject row){try{journal("runtime",row);}catch(IOException e){System.err.println("Runtime log could not be saved: "+e.getMessage());}}
 synchronized JsonObject receiveGame(JsonObject game){return monitor.receive(game,this::persistTrace);}
 synchronized JsonObject logsAfter(long after,String channel){return monitor.after(after,channel);}
 synchronized JsonObject runtime(){return monitor.status(System.currentTimeMillis(),lastReceived,Math.max(5,Json.num(Json.obj(config,"collector"),"sample_ticks",20)/20*3));}
 synchronized long lastReceived(){return lastReceived;}
 synchronized String stage(){return stage;}
 synchronized long version(){return version;}
 synchronized JsonObject status(){
  JsonObject s=new JsonObject();s.addProperty("stage",stage);s.addProperty("config_version",version);s.addProperty("last_received_ms",lastReceived);s.addProperty("age_seconds",lastReceived==0?-1:(System.currentTimeMillis()-lastReceived)/1000.0);s.addProperty("llm_status",llmStatus);s.add("reward",lastReward.deepCopy());s.add("episode",episodeStatus());s.add("telemetry",latest.deepCopy());s.add("runtime",runtime());s.add("training",training.status(trainingInfo(),System.currentTimeMillis()));return s;
 }
 synchronized JsonObject rewardAfter(long after){
  return rewardAfter(after,"");
 }
 synchronized JsonObject rewardAfter(long after,String expectedStream){
  if(after<0)throw new IllegalArgumentException("奖励游标不能为负数");
  boolean changed=!expectedStream.isEmpty()&&!expectedStream.equals(streamId),gap=changed||after>rewardSequence||!rewards.isEmpty()&&after<rewards.getFirst().get("reward_id").getAsLong()-1;
  JsonObject r=new JsonObject();JsonArray a=new JsonArray();if(!changed)for(JsonObject item:rewards)if(item.get("reward_id").getAsLong()>after)a.add(item.deepCopy());
  r.add("rewards",a);r.addProperty("stream_id",streamId);r.addProperty("latest_id",rewardSequence);r.addProperty("oldest_id",rewards.isEmpty()?0:Json.num(rewards.getFirst(),"reward_id",0));r.addProperty("gap",gap);r.addProperty("reset_reason",changed?"stream_changed":gap?"cursor_gap":"");return r;
 }
 synchronized JsonObject trainingInfo(){
  JsonObject result=new JsonObject(),binding=new JsonObject(),game=Json.obj(runtime(),"game");double age=lastReceived==0?-1:Math.max(0,(System.currentTimeMillis()-lastReceived)/1000.0);
  String gameState=Json.str(game,"state","waiting"),reason=latest.isEmpty()?"waiting_telemetry":Json.bool(game,"telemetry_stale",true)?"stale_telemetry":!gameState.equals("playing")?"game_"+gameState:!Json.str(Json.obj(latest,"coverage"),"authority","").equals("singleplayer_integrated_server")?"unsupported_authority":!strings(Json.arr(config,"allowed_game_modes")).contains(Json.str(Json.obj(latest,"player"),"game_mode",""))?"excluded_game_mode":"ready";
  for(String key:List.of("world_id","player_id","session_id"))binding.addProperty(key,Json.str(latest,key,""));result.add("binding",binding);
  result.addProperty("schema",1);result.addProperty("protocol","vabpnn-reward-v1");result.addProperty("stream_id",streamId);result.addProperty("latest_id",rewardSequence);result.addProperty("oldest_id",rewards.isEmpty()?0:Json.num(rewards.getFirst(),"reward_id",0));result.addProperty("clock","integrated_server_tick");result.addProperty("interval","(start_tick,tick]");result.addProperty("ready",reason.equals("ready"));result.addProperty("reason",reason);result.addProperty("game_state",gameState);result.addProperty("age_seconds",age);result.addProperty("telemetry_seq",Json.num(latest,"seq",0));result.addProperty("tick",Json.num(latest,"tick",0));result.addProperty("dimension",Json.str(Json.obj(latest,"world"),"dimension",""));result.addProperty("config_version",version);result.addProperty("max_gap_seconds",Json.num(config,"max_gap_seconds",10));result.addProperty("reward_clip",Json.num(config,"reward_clip",20));result.add("episode",episodeStatus());return result;
 }
 synchronized JsonObject receiveTraining(JsonObject report){return training.receive(report,trainingInfo(),r->trace("training","INFO","模型自训练状态："+Json.str(r,"mode","unknown")+" / "+Json.str(r,"reason",""),r));}
 private JsonObject map(String key){if(!memory.has(key))memory.add(key,new JsonObject());return memory.getAsJsonObject(key);}
 private boolean remember(String group,String key){JsonObject a=map(group);if(a.has(key))return false;if(a.size()>=(int)Json.num(config,"history_limit",50000))return false;a.addProperty(key,true);return true;}
 private static String hash(String key){try{return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(key.getBytes(StandardCharsets.UTF_8))).substring(0,32);}catch(Exception e){throw new IllegalStateException(e);}}
 synchronized void save()throws IOException{if(!identity.isEmpty()){memory.addProperty("stage",stage);Json.write(dir.resolve("state-"+hash(identity)+".json"),memory);}savedAt=System.currentTimeMillis();}
 private static Set<String> strings(JsonArray a){Set<String>s=new HashSet<>();for(JsonElement e:a)if(e.isJsonPrimitive())s.add(e.getAsString());return s;}
 private double delta(JsonObject cur,JsonObject old,String key){return Math.max(0,Json.num(cur,key,0)-Json.num(old,key,0));}
 private long maxEvent(JsonObject t){long n=0;for(JsonElement e:Json.arr(t,"events"))n=Math.max(n,(long)Json.num(e.getAsJsonObject(),"id",0));return n;}
 private void baseline(JsonObject t){
  new ProgressionScoring(map("progression"),config,t).baseline(t);
  for(var e:Json.obj(t,"inventory").entrySet())map("inventory_max").addProperty(e.getKey(),Math.max(Json.num(map("inventory_max"),e.getKey(),0),e.getValue().getAsDouble()));
  for(String a:strings(Json.arr(t,"advancements")))remember("advancements",a);
  remember("chunks",chunk(t));
  for(JsonElement e:Json.arr(t,"structures"))remember("structures",Json.str(e.getAsJsonObject(),"key","unknown"));
  eventSequence=Math.max(maxEvent(t),(long)Json.num(Json.obj(t,"coverage"),"event_cursor",0));lastProgress=(long)Json.num(t,"tick",0);
 }
 private String chunk(JsonObject t){JsonObject p=Json.obj(t,"player");return Json.str(Json.obj(t,"world"),"dimension","")+"|"+(int)Math.floor(Json.num(p,"x",0)/16)+","+ (int)Math.floor(Json.num(p,"z",0)/16);}
 private String chooseStage(JsonObject t){
  String manual=Json.str(config,"stage_override","auto");if(!manual.equals("auto"))return manual;
  Set<String> done=strings(Json.arr(t,"advancements"));boolean freshLife=memory.has("episode_start_advancements");
  if(freshLife)done.removeAll(strings(Json.arr(memory,"episode_start_advancements")));
  String answer=freshLife?stage:"survival";int rank=java.util.Arrays.asList(Config.STAGES).indexOf(answer);
  for(int i=0;i<Config.STAGES.length;i++){String s=Config.STAGES[i];if(i>=rank&&done.contains(Json.str(Json.obj(config,"stage_advancements"),s,"!")))answer=s;}return answer;
 }
 private static void add(Map<String,Double> m,String k,double v){if(Double.isFinite(v)&&v>0)m.merge(k,v,Double::sum);}
 synchronized JsonObject accept(JsonObject t)throws IOException {
  validateTelemetry(t);
  String ident=Json.str(t,"world_id","")+"|"+Json.str(t,"player_id",""),sid=Json.str(t,"session_id","");long seq=t.get("seq").getAsLong(),tick=t.get("tick").getAsLong();
  if(ident.equals(identity)&&sid.equals(session)&&seq<=sequence)throw new IllegalArgumentException("重复或倒序遥测");
  boolean worldChanged=!ident.equals(identity),newSession=worldChanged||!sid.equals(session);
  JsonObject life=Json.obj(t,"episode");String incomingServer=Json.str(life,"server_session_id","");long incomingIndex=(long)Json.num(life,"index",-1);
  if(!worldChanged&&!incomingServer.isEmpty()&&incomingServer.equals(serverSession)&&incomingIndex<episodeIndex)throw new IllegalArgumentException("丢弃旧生命轮次遥测");
  JsonObject endedSummary=worldChanged?new JsonObject():minuteSummary();
  boolean legacyRespawn=!worldChanged&&previous!=null&&life.isEmpty()&&Json.num(Json.obj(t,"player"),"health",20)>0
    &&(Json.num(Json.obj(previous,"player"),"health",20)<=0||delta(Json.obj(t,"stats"),Json.obj(previous,"stats"),"deaths")>0);
  if(worldChanged){save();identity=ident;Path file=dir.resolve("state-"+hash(identity)+".json");memory=Files.exists(file)?Json.read(file):new JsonObject();stage=Json.str(memory,"stage","survival");lastNovelty=0;}
  boolean respawn=legacyRespawn||!life.isEmpty()&&incomingIndex>0&&!sid.equals(Json.str(memory,"episode_session_id",""));
  if(respawn){
   JsonObject archived=new JsonObject();archived.addProperty("world_player",identity);archived.addProperty("ended_session_id",session);
   archived.addProperty("new_session_id",sid);archived.addProperty("at_tick",tick);archived.add("state",memory.deepCopy());journal("ended-episodes",archived);
   memory=new JsonObject();memory.add("episode_start_advancements",Json.arr(t,"advancements").deepCopy());previous=null;minute.clear();minuteAmounts.clear();rewards.clear();lastReward=new JsonObject();
   stage="survival";eventSequence=0;lastProgress=tick;lastNovelty=tick;episodeTotal=0;episodeStartTick=tick;version++;
   trace("behavior","INFO","玩家重生：清空本轮评分/去重/合成与掉落物链，建立新基线；评分权重保留",null);
  }
  if(newSession||respawn){previous=null;session=sid;minute.clear();windowStart=tick;minuteAmounts.clear();if(!respawn)version++;memory.remove("last_explore_tick");episodeStartTick=tick;episodeTotal=0;}
  if(!life.isEmpty()){serverSession=incomingServer;episodeIndex=incomingIndex;episodeStartTick=(long)Json.num(life,"started_tick",tick);}
  else if(legacyRespawn){serverSession="legacy-"+hash(ident);episodeIndex=Math.max(0,episodeIndex)+1;}
  memory.addProperty("episode_session_id",sid);
  boolean notify=false;
  for(JsonElement value:Json.arr(t,"lifecycle_events")){
   JsonObject event=value.getAsJsonObject().deepCopy();if(!Json.str(event,"type","").equals("player_respawn"))continue;
   event.addProperty("schema",1);event.addProperty("reset_complete",true);event.addProperty("weights_preserved",true);
   boolean observed=!worldChanged&&Json.str(event,"previous_session_id","").equals(Json.str(endedSummary,"session_id",""));
   event.addProperty("previous_episode_observed",observed);
   if(observed)event.add("previous_episode_summary",endedSummary.deepCopy());
   if(episodeNotices.enqueue(event)){journal("episodes",event);notify=true;}
  }
  if(legacyRespawn){
   JsonObject event=new JsonObject();event.addProperty("schema",1);event.addProperty("type","player_respawn");event.addProperty("event_id",UUID.randomUUID().toString());
   event.addProperty("world_id",Json.str(t,"world_id",""));event.addProperty("player_id",Json.str(t,"player_id",""));event.addProperty("server_session_id",serverSession);
   event.addProperty("session_id",sid);event.addProperty("episode_index",episodeIndex);event.addProperty("tick",tick);event.addProperty("observed_at_ms",System.currentTimeMillis());
   event.addProperty("detection","health_or_death_stats");event.addProperty("reset_complete",true);event.addProperty("weights_preserved",true);event.add("previous_episode_summary",endedSummary);
   if(episodeNotices.enqueue(event)){journal("episodes",event);notify=true;}
  }
  if(notify){noticeStatus.addProperty("state","queued");noticeStatus.addProperty("error","");episodeListener.run();}

  double dt=previous==null?0:(tick-Json.num(previous,"tick",tick))/20.0;
  if(previous!=null&&dt<=0)throw new IllegalArgumentException("游戏 tick 未递增");
  Set<String> modes=strings(Json.arr(config,"allowed_game_modes"));
  boolean modeAllowed=modes.contains(Json.str(Json.obj(t,"player"),"game_mode","unknown"));
  long oldestEvent=(long)Json.num(Json.obj(t,"coverage"),"oldest_event_id",0);
  boolean eventGap=previous!=null&&oldestEvent>eventSequence+1;
  boolean valid=previous!=null&&!eventGap&&dt<=Json.num(config,"max_gap_seconds",10)&&modeAllowed&&modes.contains(Json.str(Json.obj(previous,"player"),"game_mode","unknown"));
  sequence=seq;lastReceived=System.currentTimeMillis();latest=t.deepCopy();stage=chooseStage(t);
  Map<String,Double> amounts=new LinkedHashMap<>();JsonArray evidence=new JsonArray(),toolEvidence=new JsonArray();for(String key:Config.METRICS)amounts.put(key,0.0);
  ProgressionScoring progression=new ProgressionScoring(map("progression"),config,t);
  if(!valid){if(modeAllowed)baseline(t);}
  else {
   JsonObject p=Json.obj(t,"player"),oldP=Json.obj(previous,"player"),stats=Json.obj(t,"stats"),oldStats=Json.obj(previous,"stats");
   for(var e:Json.obj(t,"inventory").entrySet()){String id=e.getKey();double count=e.getValue().getAsDouble(),prior=Json.num(map("inventory_max"),id,0);add(amounts,"resource_gain",Math.max(0,count-prior)*Json.num(Json.obj(config,"resource_values"),id,0));if(count>prior)map("inventory_max").addProperty(id,count);}
   for(String a:strings(Json.arr(t,"advancements")))if(remember("advancements",a)){add(amounts,"advancement",1);evidence.add("完成进度: "+a);}
   if(remember("chunks",chunk(t)))add(amounts,"exploration",1);
   for(JsonElement e:Json.arr(t,"structures"))if(remember("structures",Json.str(e.getAsJsonObject(),"key","unknown")))add(amounts,"structure_discovery",1);
   for(JsonElement e:Json.arr(t,"events")){
    JsonObject ev=e.getAsJsonObject();long id=(long)Json.num(ev,"id",0);if(id<=eventSequence)continue;eventSequence=id;
    String type=Json.str(ev,"type",""),pos=Json.str(ev,"dimension","")+"|"+Json.str(ev,"pos",""),block=Json.str(ev,"block","");
    if(type.equals("item_pickup"))progression.pickedUp(ev);
    if(type.equals("block_place")){if(remember("placed",pos))add(amounts,"building",1);}
    ToolScoring.Result tool=ToolScoring.evaluate(Json.obj(ev,"tool"),Json.obj(config,"tool_rules"));
    double toolAmount=0;String metric="";
    if(type.equals("mining_attempt")&&tool.penalty()>0){toolAmount=tool.penalty();metric="tool_mismatch";add(amounts,metric,toolAmount);}
    if(type.equals("block_break")){
     if(map("placed").has(pos))add(amounts,"repetition",1);
     else if(remember("mined",pos+"|"+block)){
      if(tool.penalty()==0){add(amounts,"mining",1);progression.harvested(ev);progression.used(Json.str(Json.obj(ev,"tool"),"held_item",""),(long)Json.num(ev,"tick",ProgressionScoring.time(t)),"有效采集");}
      if(tool.appropriate()){toolAmount=1;metric="tool_correct";add(amounts,metric,toolAmount);}
     }
    }
    if(type.equals("mining_attempt")||type.equals("block_break")){
     JsonObject detail=new JsonObject();detail.addProperty("event_id",id);detail.addProperty("action",type);detail.addProperty("block",block);detail.addProperty("pos",pos);detail.addProperty("kind",tool.kind());detail.addProperty("reason",tool.reason());detail.addProperty("metric",metric);detail.addProperty("raw_amount",toolAmount);detail.add("tool",Json.obj(ev,"tool").deepCopy());toolEvidence.add(detail);
     if(toolAmount>0)trace("behavior",tool.penalty()>0?"WARN":"INFO",block+" ← "+Json.str(Json.obj(ev,"tool"),"held_item","未知")+"："+tool.reason(),detail);
    }
   }
   progression.finish(t,previous);add(amounts,"crafting_milestone",progression.craftAmount);add(amounts,"item_pickup",progression.pickupAmount);
   double progress=amounts.get("resource_gain")+amounts.get("advancement")+amounts.get("crafting_milestone");
   add(amounts,"crafting",progress>0?delta(stats,oldStats,"items_crafted"):0);
   // Completed kills carry reward; damage dealt remains visible telemetry, not an infinite damage farm.
   add(amounts,"combat",delta(stats,oldStats,"mob_kills"));
   add(amounts,"shielding",delta(stats,oldStats,"damage_blocked_by_shield")/10);
   JsonObject rules=Json.obj(config,"inference_rules");
   add(amounts,"eating",Json.num(oldP,"food",20)<Json.num(rules,"eating_food_threshold",18)?delta(stats,oldStats,"items_eaten"):0);
   add(amounts,"damage",delta(stats,oldStats,"damage_taken")/10);add(amounts,"death",delta(stats,oldStats,"deaths"));
   add(amounts,"air_risk",Json.bool(p,"underwater",false)?Math.max(0,1-Json.num(p,"air",300)/Math.max(1,Json.num(p,"max_air",300)))*dt:0);
   double hunger=Json.num(rules,"hunger_food_threshold",6);
   add(amounts,"hunger_risk",Math.max(0,(hunger-Json.num(p,"food",20))/hunger)*dt);add(amounts,"lava_risk",Json.bool(p,"in_lava",false)?dt:0);
   add(amounts,"durability_loss",wear(t,previous));
   if(progress>0||amounts.get("exploration")>0||amounts.get("structure_discovery")>0)lastProgress=tick;
   add(amounts,"idle",(tick-lastProgress)/20.0>Json.num(config,"idle_seconds",45)?dt:0);
   if(amounts.get("exploration")>0)memory.addProperty("last_explore_tick",tick);
   if(progress>0&&tick-Json.num(memory,"last_explore_tick",-100000)<Json.num(rules,"resource_search_seconds",120)*20){add(amounts,"resource_search",1);evidence.add("推断：探索新区域后获得资源/进度；不等同于读出模型意图");}
   boolean ready=Json.num(p,"food",0)>=Json.num(rules,"prepared_food",16)&&Json.num(p,"health",0)>=Json.num(p,"max_health",20)*Json.num(rules,"prepared_health_fraction",.8)&&Json.bool(p,"has_tool",false);
   if(ready&&!Json.bool(oldP,"prepared",false)&&progress>0){add(amounts,"preparation",1);evidence.add("推断：获得资源的同时达到食物、健康和工具准备条件");}
   p.addProperty("prepared",ready);
   String signature=stage+"|"+String.join(",",amounts.entrySet().stream().filter(e->e.getValue()>0&&!e.getKey().endsWith("risk")).map(Map.Entry::getKey).toList());
   if(progress>0&&amounts.get("repetition")==0&&amounts.get("tool_mismatch")==0&&tick-lastNovelty>=Json.num(config,"novelty_cooldown_seconds",120)*20&&remember("strategies",signature)){add(amounts,"creative_strategy",1);lastNovelty=tick;evidence.add("推断：此前未见的行为组合产生了实际进展；置信度 0.5");}
  }
  if(tick-windowStart>=1200){windowStart=tick;minuteAmounts.clear();}
  JsonObject components=new JsonObject();double total=0;
  for(String m:Config.METRICS){
   double cap=Json.num(Json.obj(config,"caps"),m,20);double amount=Math.min(amounts.get(m),cap);
   // One minute budget is independent of packet rate; negative health/death signals remain bounded per sample.
   if(!Set.of("damage","death","air_risk","hunger_risk","lava_risk","durability_loss","idle","repetition","tool_mismatch").contains(m)){amount=Math.min(amount,Math.max(0,cap-minuteAmounts.getOrDefault(m,0.0)));minuteAmounts.merge(m,amount,Double::sum);}
   double weight=Json.num(Json.obj(Json.obj(config,"weights"),m),stage,0),value=amount==0?0:amount*weight;total+=value;
   JsonObject c=new JsonObject();c.addProperty("amount",amount);c.addProperty("weight",weight);c.addProperty("score",value);components.add(m,c);
  }
  JsonObject r=new JsonObject();r.addProperty("schema",1);r.addProperty("reward_id",++rewardSequence);r.addProperty("world_id",Json.str(t,"world_id",""));r.addProperty("player_id",Json.str(t,"player_id",""));r.addProperty("session_id",session);r.addProperty("telemetry_seq",seq);r.addProperty("tick",tick);r.addProperty("interval_seconds",dt);r.addProperty("valid",valid);r.addProperty("reason",valid?"scored":"baseline_or_gap");r.addProperty("stage",stage);r.addProperty("config_version",version);r.addProperty("unclipped_total",total);r.addProperty("reward",Json.clamp(total,-Json.num(config,"reward_clip",20),Json.num(config,"reward_clip",20)));r.add("components",components);r.add("inferences",evidence);r.add("coverage",Json.obj(t,"coverage").deepCopy());
  r.addProperty("stream_id",streamId);r.addProperty("start_tick",previous==null?tick:Json.num(previous,"tick",tick));r.addProperty("reason",valid?"scored":!modeAllowed?"excluded_game_mode":eventGap?"event_gap":previous==null?"baseline":"telemetry_gap");
  String dimension=Json.str(Json.obj(t,"world"),"dimension",""),startDimension=previous==null?dimension:Json.str(Json.obj(previous,"world"),"dimension","");
  boolean died=previous!=null&&(Json.num(Json.obj(t,"stats"),"deaths",0)>Json.num(Json.obj(previous,"stats"),"deaths",0)||Json.num(Json.obj(t,"player"),"health",0)<=0&&Json.num(Json.obj(previous,"player"),"health",0)>0),changedDimension=!dimension.equals(startDimension);
  r.addProperty("clock","integrated_server_tick");r.addProperty("received_at_ms",lastReceived);r.addProperty("dimension",dimension);r.addProperty("start_dimension",startDimension);r.addProperty("terminal",died||changedDimension);r.addProperty("terminal_reason",died?"death":changedDimension?"dimension_changed":"");
  r.add("tool_evaluations",toolEvidence);
  r.add("crafting_evaluations",progression.crafting);r.add("pickup_evaluations",progression.pickups);
  for(JsonElement item:progression.crafting){JsonObject detail=item.getAsJsonObject();trace("behavior","INFO",Json.str(detail,"item","")+"："+Json.str(detail,"reason",""),detail);}
  for(JsonElement item:progression.pickups){JsonObject detail=item.getAsJsonObject();trace("behavior","INFO",Json.str(detail,"item","")+" 拾取 "+Json.num(detail,"picked_count",0)+"，有效 "+Json.num(detail,"eligible_count",0),detail);}
  if(progression.craftAmount>0||progression.pickupAmount>0){JsonObject detail=new JsonObject();detail.add("crafting_milestone",components.get("crafting_milestone").deepCopy());detail.add("item_pickup",components.get("item_pickup").deepCopy());trace("behavior","INFO",String.format(Locale.ROOT,"新增评分：关键合成 %.4f，拾取 %.4f（总奖励截断前）",Json.num(Json.obj(components,"crafting_milestone"),"score",0),Json.num(Json.obj(components,"item_pickup"),"score",0)),detail);}
  if(amounts.get("tool_mismatch")>0||amounts.get("tool_correct")>0){JsonObject detail=new JsonObject();for(String m:List.of("tool_mismatch","tool_correct"))detail.add(m,components.get(m).deepCopy());trace("behavior",amounts.get("tool_mismatch")>0?"WARN":"INFO",String.format(Locale.ROOT,"工具评分：不适配 %.4f，正确采集 %.4f（总奖励截断前）",Json.num(Json.obj(components,"tool_mismatch"),"score",0),Json.num(Json.obj(components,"tool_correct"),"score",0)),detail);}
  if(!valid)trace("behavior","INFO","建立评分基线："+Json.str(r,"reason",""),null);
  previous=t.deepCopy();lastReward=r;rewards.addLast(r);while(rewards.size()>512)rewards.removeFirst();minute.addLast(r);while(!minute.isEmpty()&&tick-minute.getFirst().get("tick").getAsLong()>1200)minute.removeFirst();
  if(valid)episodeTotal+=r.get("reward").getAsDouble();
  journal("rewards",r);journal("telemetry",t);if(respawn||System.currentTimeMillis()-savedAt>5000)save();return r.deepCopy();
 }
 synchronized JsonObject minuteSummary(){
  JsonObject result=new JsonObject();result.addProperty("stage",stage);result.addProperty("config_version",version);result.addProperty("samples",minute.size());
  result.addProperty("world_id",Json.str(latest,"world_id",""));result.addProperty("player_id",Json.str(latest,"player_id",""));result.addProperty("session_id",session);result.add("episode",episodeStatus());
  result.add("last_respawn",episodeNotices.last());JsonObject sums=new JsonObject();
  for(String m:Config.METRICS){double v=0;for(JsonObject r:minute)v+=Json.num(Json.obj(Json.obj(r,"components"),m),"amount",0);sums.addProperty(m,v);}result.add("amounts",sums);double windowSeconds=0;for(JsonObject r:minute)if(Json.bool(r,"valid",false))windowSeconds+=Json.num(r,"interval_seconds",0);result.addProperty("window_game_seconds",windowSeconds);
  JsonObject playerSummary=Json.obj(latest,"player").deepCopy();playerSummary.remove("nbt_snbt");playerSummary.remove("tags");result.add("player",playerSummary);result.add("world",Json.obj(latest,"world").deepCopy());result.add("inventory",Json.obj(latest,"inventory").deepCopy());result.add("coverage",Json.obj(latest,"coverage").deepCopy());
  result.add("visible_block_registry",take(Json.arr(Json.obj(latest,"visibility"),"block_registry"),128));
  result.add("loaded_block_registry",take(Json.arr(Json.obj(latest,"area"),"block_registry"),128));
  result.add("structures",take(Json.arr(latest,"structures"),24));result.add("advancements",take(Json.arr(latest,"advancements"),64));result.add("recent_events",tail(Json.arr(latest,"events"),24));
  JsonArray entities=new JsonArray();for(JsonElement entry:take(Json.arr(latest,"entities"),16)){JsonObject entity=entry.getAsJsonObject().deepCopy();entity.remove("nbt_snbt");entity.remove("tags");entities.add(entity);}result.add("nearby_entities",entities);
  JsonObject current=new JsonObject();for(String m:Config.METRICS)current.addProperty(m,Json.num(Json.obj(Json.obj(config,"weights"),m),stage,0));result.add("weights",current);result.add("locked_metrics",Json.arr(config,"locked_metrics").deepCopy());return result;
 }
 private static JsonArray take(JsonArray source,int count){JsonArray a=new JsonArray();for(int i=0;i<Math.min(count,source.size());i++)a.add(source.get(i).deepCopy());return a;}
 private static JsonArray tail(JsonArray source,int count){JsonArray a=new JsonArray();for(int i=Math.max(0,source.size()-count);i<source.size();i++)a.add(source.get(i).deepCopy());return a;}
 private static double wear(JsonObject now,JsonObject before){
  Map<String,Double> old=new HashMap<>();for(JsonElement value:Json.arr(before,"inventory_slots")){JsonObject item=value.getAsJsonObject();if(item.has("stack_identity"))old.put(Json.str(item,"item","")+"|"+Json.str(item,"stack_identity",""),Json.num(item,"damage",0));}
  double damage=0;for(JsonElement value:Json.arr(now,"inventory_slots")){JsonObject item=value.getAsJsonObject();String key=Json.str(item,"item","")+"|"+Json.str(item,"stack_identity","");if(old.containsKey(key))damage+=Math.max(0,Json.num(item,"damage",0)-old.get(key));}return damage;
 }
 synchronized void applyProposal(JsonObject proposal,String expectedStage,long expectedVersion)throws IOException{
  if(!stage.equals(expectedStage)||version!=expectedVersion)throw new IllegalArgumentException("请求期间阶段或人工配置已改变，丢弃旧建议");
  JsonObject weights=Json.obj(proposal,"weights");if(weights.isEmpty())throw new IllegalArgumentException("模型未返回 weights 对象");JsonObject next=config.deepCopy(),llm=Json.obj(config,"llm");Set<String> locked=strings(Json.arr(config,"locked_metrics"));JsonObject changes=new JsonObject();
  for(String m:weights.keySet()){
   if(!Arrays.asList(Config.METRICS).contains(m))throw new IllegalArgumentException("模型返回未知评分项: "+m);
   double requested=Json.num(weights,m,Double.NaN);if(!Double.isFinite(requested))throw new IllegalArgumentException("权重必须为有限数值");if(locked.contains(m))continue;
   double old=Json.num(Json.obj(Json.obj(config,"weights"),m),stage,0),limit=Math.min(Json.num(llm,"max_absolute_change",.25),Math.abs(old)*Json.num(llm,"max_relative_change",.2));
   // A zero manually disabled weight stays off; LLM cannot reverse rewards and penalties.
   double value=Json.clamp(requested,old-limit,old+limit),max=Json.num(llm,"max_weight",20);value=Math.abs(old)>max?old:Json.clamp(value,-max,max);if(old*value<0||old==0)value=old;
   Json.obj(Json.obj(next,"weights"),m).addProperty(stage,value);JsonObject row=new JsonObject();row.addProperty("before",old);row.addProperty("requested",requested);row.addProperty("after",value);changes.add(m,row);
  }
  configure(next,"llm");JsonObject a=new JsonObject();a.addProperty("stage",stage);a.addProperty("reason",Json.str(proposal,"reason","未提供理由"));a.add("changes",changes);journal("llm",a);trace("llm","INFO","已应用调权 v"+version+"："+Json.str(proposal,"reason",""),a);llmStatus="已调权 v"+version+"："+Json.str(proposal,"reason","");
 }
 synchronized void journal(String name,JsonObject value)throws IOException {
  Path f=dir.resolve(name+".jsonl");long limit=(long)Json.num(config,"journal_limit_mb",32)*1024*1024;
  if(Files.exists(f)&&Files.size(f)>limit){for(int i=3;i>=1;i--){Path src=i==1?f:dir.resolve(name+"."+(i-1)+".jsonl");if(Files.exists(src))Files.move(src,dir.resolve(name+"."+i+".jsonl"),StandardCopyOption.REPLACE_EXISTING);}}
  JsonObject row=value.deepCopy();row.addProperty("recorded_at",java.time.Instant.now().toString());Files.writeString(f,Json.COMPACT.toJson(row)+"\n",StandardOpenOption.CREATE,StandardOpenOption.APPEND);
 }
 static void validateTelemetry(JsonObject t){
  if(Json.num(t,"schema",0)!=1)throw new IllegalArgumentException("需要 schema=1");
  for(String s:List.of("world_id","player_id","session_id")){String v=Json.str(t,s,"");if(v.isEmpty()||v.length()>160)throw new IllegalArgumentException("无效 "+s);}
  Config.bounded(t,"seq",1,9e15);Config.bounded(t,"tick",0,9e15);JsonObject p=Json.obj(t,"player");
  for(String k:List.of("x","y","z"))Config.bounded(p,k,-3.1e7,3.1e7);
  if(Json.obj(t,"inventory").size()>4096||Json.arr(t,"events").size()>512||Json.arr(t,"entities").size()>256)throw new IllegalArgumentException("遥测超出上限");
  JsonObject life=Json.obj(t,"episode");
  if(!life.isEmpty()){
   String id=Json.str(life,"server_session_id","");if(id.isBlank()||id.length()>160)throw new IllegalArgumentException("无效服务器生命轮次");
   Config.bounded(life,"index",0,9e15);Config.bounded(life,"started_tick",0,t.get("tick").getAsDouble());
  }
  if(Json.arr(t,"lifecycle_events").size()>128)throw new IllegalArgumentException("生命事件超出上限");
  for(JsonElement value:Json.arr(t,"lifecycle_events")){
   JsonObject event=value.getAsJsonObject();UUID.fromString(Json.str(event,"event_id",""));
   if(!Json.str(event,"type","").equals("player_respawn")||!Json.str(event,"world_id","").equals(Json.str(t,"world_id",""))
     ||!Json.str(event,"player_id","").equals(Json.str(t,"player_id",""))||!Json.str(event,"server_session_id","").equals(Json.str(life,"server_session_id","")))throw new IllegalArgumentException("生命事件身份不匹配");
   Config.bounded(event,"episode_index",1,Json.num(life,"index",0));Config.bounded(event,"observed_at_ms",0,9e15);
  }
  for(String k:Json.obj(t,"inventory").keySet())Config.bounded(Json.obj(t,"inventory"),k,0,1000000);
  for(String group:List.of("crafted_by_type","broken_by_type")){JsonObject counts=Json.obj(t,group);if(counts.size()>4096)throw new IllegalArgumentException("统计种类过多");for(String key:counts.keySet())Config.bounded(counts,key,0,2.2e9);}
  for(JsonElement value:Json.arr(t,"events")){JsonObject event=value.getAsJsonObject();if(Json.str(event,"type","").equals("item_pickup")){Config.bounded(event,"count",1,1000000);JsonObject origins=Json.obj(event,"origins");if(origins.size()>128)throw new IllegalArgumentException("掉落物来源过多");for(String key:origins.keySet())Config.bounded(origins,key,0,1000000);}
   JsonObject items=Json.obj(Json.obj(Json.obj(event,"tool"),"harvest"),"items");if(items.size()>128)throw new IllegalArgumentException("掉落物种类过多");for(String key:items.keySet())Config.bounded(items,key,0,1000000);
  }
 }
}

