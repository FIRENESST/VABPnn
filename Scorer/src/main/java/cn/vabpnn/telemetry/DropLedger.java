package cn.vabpnn.telemetry;

import com.google.gson.*;
import java.util.*;

/** Provenance belongs to ground entities, never to inventory item components. */
public final class DropLedger {
 private final LinkedHashMap<String,Integer> amounts=new LinkedHashMap<>();
 public void add(String source,int count){if(count>0&&(amounts.containsKey(source)||amounts.size()<128))amounts.merge(source,count,(a,b)->Math.min(1000000,a+b));}
 public JsonObject take(int count){JsonObject out=new JsonObject();var iterator=amounts.entrySet().iterator();while(count>0&&iterator.hasNext()){var entry=iterator.next();int n=Math.min(count,entry.getValue());out.addProperty(entry.getKey(),n);count-=n;if(n==entry.getValue())iterator.remove();else entry.setValue(entry.getValue()-n);}return out;}
 public void add(JsonObject entries){for(var entry:entries.entrySet())add(entry.getKey(),entry.getValue().getAsInt());}
 public String save(){JsonObject out=new JsonObject();amounts.forEach(out::addProperty);return out.toString();}
 public void load(String text){amounts.clear();if(text.length()>20000)return;try{JsonObject o=JsonParser.parseString(text).getAsJsonObject();for(var e:o.entrySet())if(e.getKey().length()<=100)add(e.getKey(),Math.max(0,Math.min(1000000,e.getValue().getAsInt())));}catch(RuntimeException ignored){amounts.clear();}}
}
