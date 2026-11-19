package cn.vabpnn.mod.client.capture;

import com.google.gson.*;
import com.google.gson.stream.*;
import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.*;
import java.nio.charset.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.Flow;

/** HTTP only. Completion callbacks are not on Minecraft's thread; callers own action/session pairing. */
public final class ScorerRewardTransport implements AutoCloseable {
    private static final int MAX_BODY=2_000_000;
    private final URI endpoint;
    private final ExecutorService executor=Executors.newFixedThreadPool(2,r->{
        Thread thread=new Thread(r,"vabpnn-scorer-reward-http");thread.setDaemon(true);return thread;
    });
    private final HttpClient client;
    private final AtomicBoolean closed=new AtomicBoolean();
    private final Semaphore capacity=new Semaphore(16);
    private final Set<CompletableFuture<JsonObject>> pending=ConcurrentHashMap.newKeySet();

    public ScorerRewardTransport(URI endpoint) {
        Objects.requireNonNull(endpoint,"endpoint");
        String host=endpoint.getHost();
        if(!"http".equalsIgnoreCase(endpoint.getScheme())||host==null
                ||!Set.of("127.0.0.1","localhost","::1","[::1]").contains(host.toLowerCase(Locale.ROOT))
                ||endpoint.getUserInfo()!=null||endpoint.getRawQuery()!=null||endpoint.getRawFragment()!=null
                ||!(endpoint.getPath()==null||endpoint.getPath().isEmpty()||endpoint.getPath().equals("/"))
                ||endpoint.getPort()==0||endpoint.getPort()>65535)
            throw new IllegalArgumentException("Scorer endpoint must be a loopback HTTP origin without credentials, path, query or fragment");
        this.endpoint=URI.create("http://"+endpoint.getRawAuthority());
        client=HttpClient.newBuilder().connectTimeout(Duration.ofMillis(500))
                .followRedirects(HttpClient.Redirect.NEVER).executor(executor)
                .proxy(new ProxySelector(){
                    public List<Proxy> select(URI uri){return List.of(Proxy.NO_PROXY);}
                    public void connectFailed(URI uri,SocketAddress address,IOException error){}
                }).build();
    }

    public CompletableFuture<JsonObject> handshake(){return request("/v1/training",null);}
    public CompletableFuture<JsonObject> rewards(long after,String streamId){
        if(after<0||streamId==null||streamId.length()>256)
            return CompletableFuture.failedFuture(new IllegalArgumentException("Invalid reward cursor or stream"));
        return request("/v1/rewards?after="+after+"&stream_id="+URLEncoder.encode(streamId,StandardCharsets.UTF_8),null);
    }
    public CompletableFuture<JsonObject> report(JsonObject status){
        if(status==null)return CompletableFuture.failedFuture(new IllegalArgumentException("Training status required"));
        String body=status.toString();
        if(body.getBytes(StandardCharsets.UTF_8).length>16000)
            return CompletableFuture.failedFuture(new IllegalArgumentException("Training status exceeds 16000 bytes"));
        return request("/v1/training-status",body);
    }

    private CompletableFuture<JsonObject> request(String path,String body){
        if(closed.get())return CompletableFuture.failedFuture(new IOException("Scorer transport closed"));
        if(!capacity.tryAcquire())return CompletableFuture.failedFuture(new IOException("Too many pending Scorer requests"));
        try{
            HttpRequest.Builder request=HttpRequest.newBuilder(endpoint.resolve(path))
                    .timeout(Duration.ofMillis(1400)).header("Accept","application/json");
            if(body==null)request.GET();else request.header("Content-Type","application/json; charset=utf-8")
                    .POST(HttpRequest.BodyPublishers.ofString(body,StandardCharsets.UTF_8));
            CompletableFuture<HttpResponse<byte[]>> wire=client.sendAsync(request.build(),info->{
                BoundedBody subscriber=new BoundedBody();
                if(info.headers().firstValueAsLong("Content-Length").orElse(0)>MAX_BODY)
                    subscriber.body.completeExceptionally(new IOException("Scorer response exceeds 2 MB"));
                return subscriber;
            });
            CompletableFuture<JsonObject> result=wire.thenApplyAsync(response->{
                try{return decode(response);}catch(IOException error){throw new CompletionException(error);}
            },executor);
            pending.add(result);
            result.orTimeout(1500,TimeUnit.MILLISECONDS).whenComplete((value,error)->{
                pending.remove(result);capacity.release();if(error!=null)wire.cancel(true);
            });
            if(closed.get())result.completeExceptionally(new IOException("Scorer transport closed"));
            return result;
        }catch(RuntimeException error){capacity.release();return CompletableFuture.failedFuture(error);}
    }

    private static JsonObject decode(HttpResponse<byte[]> response)throws IOException{
        String body=StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(response.body())).toString();
        JsonObject object=null;Exception malformed=null;
        try(JsonReader reader=new JsonReader(new StringReader(body))){
            reader.setStrictness(Strictness.STRICT);
            JsonElement value=JsonParser.parseReader(reader);
            if(!value.isJsonObject()||reader.peek()!=JsonToken.END_DOCUMENT)throw new IOException("Expected one JSON object");
            object=value.getAsJsonObject();
        }catch(Exception error){malformed=error;}
        if(response.statusCode()<200||response.statusCode()>=300)
            throw new ResponseException(response.statusCode(),object);
        if(object==null)throw new IOException("Malformed Scorer JSON response",malformed);
        return object;
    }

    /** Preserves bounded structured server errors while rejecting every non-success HTTP response. */
    public static final class ResponseException extends IOException {
        private final int statusCode;
        private final JsonObject error;
        ResponseException(int status,JsonObject error){
            super("Scorer HTTP "+status+(error!=null&&error.has("error")?": "+brief(error.get("error").toString()):""));
            statusCode=status;this.error=error==null?new JsonObject():error.deepCopy();
        }
        private static String brief(String text){return text.substring(0,Math.min(300,text.length()));}
        public int statusCode(){return statusCode;}
        public JsonObject error(){return error.deepCopy();}
    }

    private static final class BoundedBody implements HttpResponse.BodySubscriber<byte[]> {
        final CompletableFuture<byte[]> body=new CompletableFuture<>();
        private final ByteArrayOutputStream bytes=new ByteArrayOutputStream(8192);
        private Flow.Subscription subscription;
        public CompletionStage<byte[]> getBody(){return body;}
        public void onSubscribe(Flow.Subscription value){
            subscription=value;if(body.isDone())value.cancel();else value.request(1);
        }
        public void onNext(List<ByteBuffer> chunks){
            if(body.isDone()){subscription.cancel();return;}
            for(ByteBuffer chunk:chunks){
                if(chunk.remaining()>MAX_BODY-bytes.size()){
                    body.completeExceptionally(new IOException("Scorer response exceeds 2 MB"));subscription.cancel();return;
                }
                byte[] part=new byte[chunk.remaining()];chunk.get(part);bytes.writeBytes(part);
            }
            subscription.request(1);
        }
        public void onError(Throwable error){body.completeExceptionally(error);}
        public void onComplete(){body.complete(bytes.toByteArray());}
    }

    @Override public void close(){
        if(!closed.compareAndSet(false,true))return;
        for(CompletableFuture<JsonObject> future:pending)future.completeExceptionally(new IOException("Scorer transport closed"));
        client.shutdownNow();executor.shutdownNow();
    }
}
