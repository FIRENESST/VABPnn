package cn.vabpnn.scorer;
import com.google.gson.*;
import java.nio.file.*;
import java.io.*;
final class Json {
 static final Gson G=new GsonBuilder().setPrettyPrinting().disableHtmlEscaping().create(), COMPACT=new Gson();
 static JsonObject obj(JsonObject o,String k){return o.has(k)&&o.get(k).isJsonObject()?o.getAsJsonObject(k):new JsonObject();}
 static JsonArray arr(JsonObject o,String k){return o.has(k)&&o.get(k).isJsonArray()?o.getAsJsonArray(k):new JsonArray();}
 static String str(JsonObject o,String k,String f){return o.has(k)&&o.get(k).isJsonPrimitive()?o.get(k).getAsString():f;}
 static double num(JsonObject o,String k,double f){try{double v=o.get(k).getAsDouble();return Double.isFinite(v)?v:f;}catch(Exception e){return f;}}
 static boolean bool(JsonObject o,String k,boolean f){try{return o.get(k).getAsBoolean();}catch(Exception e){return f;}}
 static JsonObject read(Path p)throws IOException{return JsonParser.parseString(Files.readString(p)).getAsJsonObject();}
 static void write(Path p,JsonObject o)throws IOException{Files.createDirectories(p.toAbsolutePath().getParent());Path tmp=p.resolveSibling(p.getFileName()+".tmp");Files.writeString(tmp,G.toJson(o));try{Files.move(tmp,p,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);}catch(AtomicMoveNotSupportedException e){Files.move(tmp,p,StandardCopyOption.REPLACE_EXISTING);}}
 static double clamp(double x,double lo,double hi){return Math.max(lo,Math.min(hi,x));}
}

