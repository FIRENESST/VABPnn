package cn.vabpnn.mod.client.capture;

import com.google.gson.*;
import com.sun.net.httpserver.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/** Local HTTP contract regression, callable from the existing GL smoke runner. */
public final class ScorerRewardTransportTest {
    private record Reply(int code,byte[] body,boolean chunked,long delay){}
    private static Reply reply(int code,String body){return new Reply(code,body.getBytes(StandardCharsets.UTF_8),false,0);}
    private static void check(boolean ok,String message){if(!ok)throw new AssertionError(message);}
    private static Throwable failure(CompletableFuture<?> future)throws Exception{
        try{future.get(3,TimeUnit.SECONDS);throw new AssertionError("Request unexpectedly succeeded");}
        catch(ExecutionException error){Throwable cause=error.getCause();while(cause instanceof CompletionException&&cause.getCause()!=null)cause=cause.getCause();return cause;}
    }
    public static void main(String[] args)throws Exception{run();}
    public static void run()throws Exception{
        for(String bad:List.of("http://example.invalid","https://127.0.0.1","http://user@127.0.0.1","http://127.0.0.1:0","http://127.0.0.1/v1","http://127.0.0.1?q=x","http://127.0.0.1#fragment")){
            boolean rejected=false;try(var ignored=new ScorerRewardTransport(URI.create(bad))){}catch(IllegalArgumentException expected){rejected=true;}
            check(rejected,"Reject unsafe or ambiguous endpoint: "+bad);
        }
        var executor=Executors.newFixedThreadPool(3,r->{Thread thread=new Thread(r,"scorer-transport-test-server");thread.setDaemon(true);return thread;});
        HttpServer server=HttpServer.create(new InetSocketAddress("127.0.0.1",0),8);server.setExecutor(executor);
        AtomicReference<Reply> answer=new AtomicReference<>(reply(200,"{\"schema\":1,\"protocol\":\"vabpnn-reward-v1\",\"ready\":true}"));
        AtomicReference<String> query=new AtomicReference<>(),report=new AtomicReference<>();
        server.createContext("/",exchange->{
            Reply current=answer.get();String path=exchange.getRequestURI().getPath();
            if(path.equals("/v1/rewards"))query.set(exchange.getRequestURI().getRawQuery());
            if(path.equals("/v1/training-status"))report.set(new String(exchange.getRequestBody().readNBytes(16001),StandardCharsets.UTF_8));
            if(current.code==302)exchange.getResponseHeaders().set("Location","http://127.0.0.1:1/");
            exchange.getResponseHeaders().set("Content-Type","application/json; charset=utf-8");
            try{
                exchange.sendResponseHeaders(current.code,current.chunked?0:current.body.length);
                OutputStream output=exchange.getResponseBody();
                if(current.delay>0){output.write(current.body,0,1);output.flush();Thread.sleep(current.delay);output.write(current.body,1,current.body.length-1);}
                else output.write(current.body);
            }catch(IOException ignored){/* Response limit / timeout deliberately cancels the socket. */}
            catch(InterruptedException error){Thread.currentThread().interrupt();}
            finally{exchange.close();}
        });
        server.start();URI endpoint=URI.create("http://127.0.0.1:"+server.getAddress().getPort()+"/");
        try(var transport=new ScorerRewardTransport(endpoint)){
            check(transport.handshake().get(2,TimeUnit.SECONDS).get("ready").getAsBoolean(),"Handshake round trip");
            answer.set(new Reply(200,"{\"ready\":true}".getBytes(StandardCharsets.UTF_8),false,450));
            long started=System.nanoTime();var asynchronous=transport.handshake();long returnMillis=(System.nanoTime()-started)/1_000_000;
            check(returnMillis<250&&!asynchronous.isDone(),"Slow body never blocks the calling game thread");
            check(asynchronous.get(2,TimeUnit.SECONDS).get("ready").getAsBoolean(),"Asynchronous complete body parsing");
            answer.set(reply(200,"{\"stream_id\":\"new\",\"gap\":true,\"rewards\":[],\"error\":\"changed\"}"));
            JsonObject batch=transport.rewards(7,"a+b/c space").get(2,TimeUnit.SECONDS);
            check(batch.get("gap").getAsBoolean()&&batch.get("error").getAsString().equals("changed"),"Preserve protocol reset/error fields");
            check(query.get().equals("after=7&stream_id=a%2Bb%2Fc+space"),"Encode complete stream identity without query injection");
            answer.set(reply(200,"{\"accepted\":true}"));JsonObject status=new JsonObject();status.addProperty("schema",1);status.addProperty("trainer_id","中文测试");
            check(transport.report(status).get(2,TimeUnit.SECONDS).get("accepted").getAsBoolean(),"Training report round trip");
            check(JsonParser.parseString(report.get()).getAsJsonObject().get("trainer_id").getAsString().equals("中文测试"),"UTF-8 report preserved");
            answer.set(reply(503,"{\"error\":\"temporarily_unavailable\",\"retry\":true}"));
            Throwable error=failure(transport.handshake());check(error instanceof ScorerRewardTransport.ResponseException,"HTTP failures have structured exception");
            var http=(ScorerRewardTransport.ResponseException)error;check(http.statusCode()==503&&http.error().get("retry").getAsBoolean(),"Preserve HTTP code and structured error JSON");
            answer.set(reply(302,"{}"));check(failure(transport.handshake()) instanceof ScorerRewardTransport.ResponseException,"Redirects are rejected, never followed");
            for(String malformed:List.of("{", "[]", "{} garbage", "{/* comment */\"x\":1}")){
                answer.set(reply(200,malformed));check(failure(transport.handshake()) instanceof IOException,"Reject malformed/non-object JSON");
            }
            byte[] huge=new byte[2_000_001];Arrays.fill(huge,(byte)' ');
            for(boolean chunked:new boolean[]{false,true}){
                answer.set(new Reply(200,huge,chunked,0));check(failure(transport.handshake()).getMessage().contains("2 MB"),"Bound announced and streaming response sizes");
            }
            answer.set(new Reply(200,"{\"ready\":true}".getBytes(StandardCharsets.UTF_8),false,2400));
            started=System.nanoTime();error=failure(transport.handshake());long elapsed=(System.nanoTime()-started)/1_000_000;
            check((error instanceof TimeoutException||error instanceof java.net.http.HttpTimeoutException)&&elapsed<2100,"Timeout covers slow body, not only response headers");
            check(failure(transport.rewards(-1,"s")) instanceof IllegalArgumentException,"Reject negative cursor");
            var pending=transport.handshake();transport.close();check(failure(pending) instanceof IOException,"Close cancels in-flight transport");
            check(failure(transport.handshake()) instanceof IOException,"No requests after close");transport.close();
        }finally{server.stop(0);executor.shutdownNow();}
        System.out.println("PASS: Scorer transport async handshake/rewards/status, loopback/no redirects, UTF-8/error JSON, strict parsing, 2 MB limits, whole-body timeout and close cancellation.");
    }
}
