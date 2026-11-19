package cn.vabpnn.telemetry;

import com.google.gson.*;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.entity.item.ItemEntity;
import java.util.UUID;

public final class HarvestDrops {
 public static final ThreadLocal<Context> CURRENT=new ThreadLocal<>();
 public static final class Context {
  public final String id=UUID.randomUUID().toString();
  public final JsonObject drops=new JsonObject();
  public JsonObject evidence(){JsonObject out=new JsonObject();out.addProperty("source_id",id);out.add("items",drops.deepCopy());return out;}
 }
 public static void created(ItemEntity entity){
  Context c=CURRENT.get();if(c==null||entity.level().isClientSide())return;
  int count=entity.getItem().getCount();String item=BuiltInRegistries.ITEM.getKey(entity.getItem().getItem()).toString();
  ((TrackedDrop)entity).scorer$origins().add(c.id,count);c.drops.addProperty(item,count+(c.drops.has(item)?c.drops.get(item).getAsInt():0));
 }
}
