package cn.vabpnn.telemetry;
import com.google.gson.*;
import net.fabricmc.loader.api.FabricLoader;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;

final class TelemetryTransport {
 static final Gson GSON=new Gson();
 static volatile JsonObject settings=new JsonObject();
 static volatile String status="waiting",reward="{}";
 static volatile long dropped=0,sent=0;
 private static final AtomicReference<String> pending=new AtomicReference<>();
 private static final HttpClient HTTP=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(1)).build();
 private static String endpoint="http://127.0.0.1:9876";
 private static long nextConfig=0;
 private static final String INSTANCE=java.util.UUID.randomUUID().toString();
 private static volatile JsonObject game=new JsonObject();
 private static GameLogTailer logTailer;
 private static long nextHeartbeat=0,logAcknowledged=0;
 private static String scorerStream="";
 static void gameState(JsonObject state){game=state;}
 static void start(){
  try{Path path=FabricLoader.getInstance().getConfigDir().resolve("vabpnn-telemetry.json");if(Files.exists(path)){JsonObject c=JsonParser.parseString(Files.readString(path)).getAsJsonObject();if(c.has("scorer_url"))endpoint=c.get("scorer_url").getAsString();}else Files.writeString(path,"{\"scorer_url\":\"http://127.0.0.1:9876\"}");
   endpoint=System.getProperty("vabpnn.scorer.url",endpoint);
   URI uri=URI.create(endpoint);if(!java.util.Set.of("127.0.0.1","localhost","::1","[::1]").contains(uri.getHost())||!uri.getScheme().equals("http"))throw new IllegalArgumentException("Scorer must be local HTTP");
  }catch(Exception e){TelemetryMod.LOG.error("Telemetry configuration invalid; collector transport disabled",e);status=e.toString();return;}
  logTailer=new GameLogTailer(FabricLoader.getInstance().getGameDir().resolve("logs/latest.log"));
  ScheduledExecutorService executor=Executors.newSingleThreadScheduledExecutor(r->{Thread t=new Thread(r,"vabpnn-telemetry-http");t.setDaemon(true);return t;});executor.scheduleWithFixedDelay(TelemetryTransport::flush,0,100,TimeUnit.MILLISECONDS);
 }
 static int setting(String key,int fallback,int low,int high){try{return Math.max(low,Math.min(high,settings.get(key).getAsInt()));}catch(Exception e){return fallback;}}
 static void submit(JsonObject packet){String body=GSON.toJson(packet);if(body.getBytes(java.nio.charset.StandardCharsets.UTF_8).length>1_500_000){dropped++;status="packet too large";return;}if(pending.getAndSet(body)!=null)dropped++;}
 private static void flush(){
  heartbeat();
  try{
   if(System.currentTimeMillis()>nextConfig){nextConfig=System.currentTimeMillis()+10000;var r=HTTP.send(HttpRequest.newBuilder(URI.create(endpoint+"/v1/collector-config")).timeout(Duration.ofSeconds(2)).GET().build(),HttpResponse.BodyHandlers.ofString());if(r.statusCode()==200)settings=JsonParser.parseString(r.body()).getAsJsonObject();}
   String body=pending.getAndSet(null);if(body==null)return;
   var response=HTTP.send(HttpRequest.newBuilder(URI.create(endpoint+"/v1/telemetry")).timeout(Duration.ofSeconds(2)).header("Content-Type","application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build(),HttpResponse.BodyHandlers.ofString());
   if(response.statusCode()==200){sent++;status="connected";reward=response.body();}else{dropped++;status="HTTP "+response.statusCode()+": "+response.body().substring(0,Math.min(300,response.body().length()));}
  }catch(Exception e){status=e.getClass().getSimpleName()+": "+e.getMessage();}
 }
 private static void heartbeat(){
  if(System.currentTimeMillis()<nextHeartbeat)return;nextHeartbeat=System.currentTimeMillis()+1000;
  try{
   logTailer.poll();JsonObject state=game.deepCopy();state.addProperty("instance_id",INSTANCE);state.addProperty("transport_status",status);state.addProperty("sent",sent);state.addProperty("dropped",dropped);state.add("log",logTailer.snapshot(logAcknowledged));
   var r=HTTP.send(HttpRequest.newBuilder(URI.create(endpoint+"/v1/game-status")).timeout(Duration.ofSeconds(2)).header("Content-Type","application/json").POST(HttpRequest.BodyPublishers.ofString(GSON.toJson(state))).build(),HttpResponse.BodyHandlers.ofString());
   if(r.statusCode()==200){JsonObject reply=JsonParser.parseString(r.body()).getAsJsonObject();String stream=reply.get("stream_id").getAsString();if(!scorerStream.isEmpty()&&!scorerStream.equals(stream))logAcknowledged=0;else logAcknowledged=reply.get("log_ack").getAsLong();scorerStream=stream;}
  }catch(Exception ignored){/* A lost heartbeat must not discard a pending training observation. */}
 }
}

