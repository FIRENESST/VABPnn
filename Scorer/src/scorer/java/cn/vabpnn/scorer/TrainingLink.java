package cn.vabpnn.scorer;

import com.google.gson.*;
import java.util.*;

/** Model reports describe consumption; Scorer never treats a report as proof of a weight update. */
final class TrainingLink {
 private JsonObject report=new JsonObject();
 private long received;
 JsonObject receive(JsonObject input,JsonObject contract,java.util.function.Consumer<JsonObject> log){
  if(Json.num(input,"schema",0)!=1)throw new IllegalArgumentException("训练状态 schema 必须为 1");
  String trainer=Json.str(input,"trainer_id","");if(trainer.isBlank()||trainer.length()>100)throw new IllegalArgumentException("无效 trainer_id");
  if(Json.COMPACT.toJson(input).length()>16000)throw new IllegalArgumentException("训练状态过长");
  for(String key:List.of("last_reward_id","updates","applied_rewards","dropped_rewards","pending_actions"))if(input.has(key))Config.bounded(input,key,0,9e15);
  for(String key:List.of("mode","connection","reason","last_error","model_version","algorithm"))if(Json.str(input,key,"").length()>1000)throw new IllegalArgumentException("训练状态字段过长："+key);
  boolean transition=report.isEmpty()||!trainer.equals(Json.str(report,"trainer_id",""))||!Json.str(input,"mode","").equals(Json.str(report,"mode",""))||!Json.str(input,"connection","").equals(Json.str(report,"connection",""))||!Json.str(input,"reason","").equals(Json.str(report,"reason",""));
  report=input.deepCopy();received=System.currentTimeMillis();
  JsonObject result=status(contract,received);if(transition)log.accept(result);
  JsonObject ack=new JsonObject();ack.addProperty("accepted",true);ack.addProperty("binding_matches",Json.bool(result,"binding_matches",false));ack.addProperty("stream_id",Json.str(contract,"stream_id",""));ack.addProperty("latest_id",Json.num(contract,"latest_id",0));return ack;
 }
 JsonObject status(JsonObject contract,long now){
  JsonObject out=report.deepCopy();double age=received==0?-1:Math.max(0,(now-received)/1000.0);boolean same=Json.str(report,"stream_id","!").equals(Json.str(contract,"stream_id",""));
  for(String key:List.of("world_id","player_id","session_id"))same&=Json.str(Json.obj(report,"binding"),key,"!").equals(Json.str(Json.obj(contract,"binding"),key,""));
  out.addProperty("age_seconds",age);out.addProperty("binding_matches",same);out.addProperty("status_source","trainer_report");
  out.addProperty("state",received==0?"waiting":age>8?"offline":!same?"unbound":"connected");return out;
 }
}
