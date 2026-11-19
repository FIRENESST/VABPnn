package cn.vabpnn.scorer;

import com.google.gson.*;
import java.time.Instant;
import java.util.*;

/** Bounded live history. All access is through Engine's lock. */
final class RuntimeMonitor {
 private final Deque<JsonObject> logs=new ArrayDeque<>();
 private final String stream=UUID.randomUUID().toString();
 private JsonObject game=new JsonObject(),llm=new JsonObject();
 private long serial=0,gameReceived=0,logAck=0;
 private String instance="";
 JsonObject add(String channel,String level,String message,JsonElement details){
  JsonObject row=new JsonObject();row.addProperty("id",++serial);row.addProperty("time",Instant.now().toString());row.addProperty("channel",channel);row.addProperty("level",level);row.addProperty("message",message);
  if(details!=null)row.add("details",details.deepCopy());logs.addLast(row);while(logs.size()>1000)logs.removeFirst();return row;
 }
 JsonObject receive(JsonObject packet,java.util.function.Consumer<JsonObject> persist){
  String incoming=Json.str(packet,"instance_id","");if(incoming.isBlank()||incoming.length()>100)throw new IllegalArgumentException("无效游戏 instance_id");
  JsonArray lines=Json.arr(Json.obj(packet,"log"),"lines");if(lines.size()>100)throw new IllegalArgumentException("游戏日志批次过大");
  for(JsonElement entry:lines){JsonObject line=entry.getAsJsonObject();Config.bounded(line,"id",1,9e15);if(Json.str(line,"text","").length()>9000)throw new IllegalArgumentException("游戏日志行过长");}
  boolean changed=!incoming.equals(instance);if(changed){logAck=0;instance=incoming;}
  if(changed||!Json.str(game,"state","").equals(Json.str(packet,"state","")))persist.accept(add("game","INFO","游戏状态："+gameLabel(Json.str(packet,"state","unknown")),null));
  long oldest=(long)Json.num(Json.obj(packet,"log"),"oldest_id",0);
  if(oldest>logAck+1){persist.accept(add("game","WARN","游戏日志缓冲区已覆盖部分内容",null));logAck=oldest-1;}
  for(JsonElement value:lines){JsonObject line=value.getAsJsonObject();long id=(long)Json.num(line,"id",0);if(id<=logAck)continue;logAck=id;persist.accept(add("game",Json.str(line,"level","INFO"),Json.str(line,"text",""),null));}
  game=packet.deepCopy();Json.obj(game,"log").remove("lines");gameReceived=System.currentTimeMillis();
  JsonObject ack=new JsonObject();ack.addProperty("log_ack",logAck);ack.addProperty("stream_id",stream);return ack;
 }
 void model(JsonObject state){for(var entry:state.entrySet())llm.add(entry.getKey(),entry.getValue().deepCopy());}
 JsonObject status(long now,long telemetryAt,double staleSeconds){
  JsonObject s=new JsonObject(),g=game.deepCopy();double age=gameReceived==0?-1:(now-gameReceived)/1000.0;
  String state=gameReceived==0?telemetryAt>0?(now-telemetryAt)/1000.0>staleSeconds?"offline":"telemetry_only":"waiting":age>8?"offline":now-Json.num(game,"observed_at_ms",now)>10000?"client_unresponsive":Json.str(game,"state","unknown");
  g.addProperty("state",state);g.addProperty("label",gameLabel(state));g.addProperty("heartbeat_age_seconds",age);g.addProperty("telemetry_stale",telemetryAt==0||(now-telemetryAt)/1000.0>staleSeconds);s.add("game",g);
  JsonObject l=llm.deepCopy();l.addProperty("request_elapsed_seconds",Json.bool(l,"busy",false)?Math.max(0,(now-Json.num(l,"request_started_ms",now))/1000.0):0);s.add("llm",l);s.addProperty("log_latest_id",serial);return s;
 }
 JsonObject after(long after,String channel){
  JsonObject r=new JsonObject();JsonArray entries=new JsonArray();for(JsonObject log:logs)if(Json.num(log,"id",0)>after&&(channel.isBlank()||channel.equals(Json.str(log,"channel",""))))entries.add(log.deepCopy());
  r.add("logs",entries);r.addProperty("stream_id",stream);r.addProperty("latest_id",serial);r.addProperty("gap",after>serial||!logs.isEmpty()&&after<Json.num(logs.getFirst(),"id",0)-1);return r;
 }
 static String gameLabel(String state){return switch(state){case "playing"->"单人世界运行中";case "paused"->"游戏已暂停";case "menu"->"游戏已启动，当前在菜单";case "unsupported_multiplayer"->"多人游戏（本版不评分）";case "offline"->"游戏失联或已关闭";case "client_unresponsive"->"客户端未更新状态，可能卡顿";case "telemetry_only"->"收到遥测（旧模组无心跳）";default->"等待游戏连接";};}
}
