package cn.vabpnn.telemetry;

import com.google.gson.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;

/** Called only on the HTTP worker. Bounded tailing, UTF-8 line assembly and rotation detection. */
public final class GameLogTailer {
 private final Path path;
 private final Deque<JsonObject> lines=new ArrayDeque<>();
 private final ByteArrayOutputStream partial=new ByteArrayOutputStream();
 private long offset=-1,sequence=0,skippedBytes=0;
 private Object fileKey;
 private boolean skipPartial;
 private String status="等待日志文件";
 public GameLogTailer(Path path){this.path=path;}
 public void poll(){
  try{
   if(!Files.isRegularFile(path)){status="等待 latest.log";return;}
   BasicFileAttributes attrs=Files.readAttributes(path,BasicFileAttributes.class);
   if(offset<0||attrs.size()<offset||!Objects.equals(fileKey,attrs.fileKey())){
    offset=Math.max(0,attrs.size()-32768);fileKey=attrs.fileKey();partial.reset();skipPartial=offset>0;skippedBytes+=offset;
   }
   if(attrs.size()-offset>131072){long next=attrs.size()-32768;skippedBytes+=next-offset;offset=next;partial.reset();skipPartial=true;}
   try(RandomAccessFile file=new RandomAccessFile(path.toFile(),"r")){
    file.seek(offset);byte[] buffer=new byte[32768];int n=file.read(buffer);
    if(n>0){offset+=n;for(int i=0;i<n;i++){
     int b=buffer[i]&255;
     if(b==10){if(!skipPartial)line(partial.toByteArray());partial.reset();skipPartial=false;}
     else if(!skipPartial){if(partial.size()<8192)partial.write(b);else{line(partial.toByteArray());partial.reset();skipPartial=true;}}
    }}
   }
   status="正在读取 latest.log";
  }catch(IOException e){status="日志读取失败："+e.getClass().getSimpleName();}
 }
 private void line(byte[] bytes){
  String text=new String(bytes,StandardCharsets.UTF_8).stripTrailing();if(text.isEmpty())return;
  JsonObject row=new JsonObject();row.addProperty("id",++sequence);row.addProperty("text",text);row.addProperty("level",text.contains("/ERROR]")||text.contains("/FATAL]")?"ERROR":text.contains("/WARN]")?"WARN":"INFO");
  lines.addLast(row);while(lines.size()>300)lines.removeFirst();
 }
 public JsonObject snapshot(long after){
  JsonObject result=new JsonObject();JsonArray batch=new JsonArray();for(JsonObject row:lines)if(row.get("id").getAsLong()>after&&batch.size()<100)batch.add(row.deepCopy());
  result.add("lines",batch);result.addProperty("latest_id",sequence);result.addProperty("oldest_id",lines.isEmpty()?0:lines.getFirst().get("id").getAsLong());result.addProperty("skipped_bytes",skippedBytes);result.addProperty("status",status);return result;
 }
}
