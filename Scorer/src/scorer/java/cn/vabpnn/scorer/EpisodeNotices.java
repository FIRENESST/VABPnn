package cn.vabpnn.scorer;
import com.google.gson.*;import java.io.*;import java.nio.file.*;import java.util.*;import java.util.stream.*;
/** Durable, disk-backed lifecycle outbox; summaries are bounded and no game NBT is stored here. */
final class EpisodeNotices {
 private final Path directory,cursorFile;
 private final LinkedHashMap<String,Long> cursors=new LinkedHashMap<>();
 private long pending;
 private JsonObject last=new JsonObject();
 EpisodeNotices(Path root)throws IOException{
  directory=root.resolve("episode-notices");cursorFile=directory.resolve("cursors.json");Files.createDirectories(directory);
  if(Files.exists(cursorFile))for(var e:Json.read(cursorFile).entrySet())cursors.put(e.getKey(),e.getValue().getAsLong());
  try(var files=Files.list(directory)){pending=files.filter(EpisodeNotices::noticeFile).count();}
 }
 private static boolean noticeFile(Path p){return p.getFileName().toString().matches("[0-9]{13}-[0-9]{20}-[0-9a-f-]{36}\\.json");}
 private static String cursorKey(JsonObject e){return Json.str(e,"world_id","")+"|"+Json.str(e,"player_id","")+"|"+Json.str(e,"server_session_id","");}
 boolean enqueue(JsonObject event)throws IOException{
  String key=cursorKey(event);long index=(long)Json.num(event,"episode_index",0);
  if(index<=cursors.getOrDefault(key,-1L))return false;
  String id=UUID.fromString(Json.str(event,"event_id","")).toString();
  long time=(long)Json.num(event,"observed_at_ms",System.currentTimeMillis());
  String name=String.format(Locale.ROOT,"%013d-%020d-%s.json",time,index,id);
  Path file=directory.resolve(name);if(!Files.exists(file)){Json.write(file,event);pending++;}
  cursors.remove(key);cursors.put(key,index);while(cursors.size()>128)cursors.remove(cursors.keySet().iterator().next());
  JsonObject saved=new JsonObject();cursors.forEach(saved::addProperty);Json.write(cursorFile,saved);
  last=event.deepCopy();return true;
 }
 List<JsonObject> batch()throws IOException {
  List<Path> files;try(var stream=Files.list(directory)){files=stream.filter(EpisodeNotices::noticeFile).sorted().limit(16).toList();}
  var result=new ArrayList<JsonObject>();for(Path p:files){var e=Json.read(p);e.addProperty("_notice_file",p.getFileName().toString());result.add(e);}return result;
 }
 void acknowledge(JsonObject event)throws IOException{
  Path file=directory.resolve(Json.str(event,"_notice_file","")).normalize();
  if(!file.getParent().equals(directory.normalize())||!noticeFile(file))throw new IOException("Invalid notice acknowledgement");
  if(Files.deleteIfExists(file))pending--;
 }
 long pending(){return pending;}
 JsonObject last(){return last.deepCopy();}
}
