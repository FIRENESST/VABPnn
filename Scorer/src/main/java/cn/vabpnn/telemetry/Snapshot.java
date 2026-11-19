package cn.vabpnn.telemetry;
import com.google.gson.*;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.*;
import net.minecraft.world.item.*;
import net.minecraft.core.registries.*;
import net.minecraft.core.component.DataComponents;
import net.minecraft.stats.*;
import net.minecraft.util.ProblemReporter;
import net.minecraft.world.level.storage.TagValueOutput;
import java.util.*;

final class Snapshot {
 private static String previousPlayer="";
 private static long previousTick;
 private static double previousX,previousY,previousZ;
 static void reset(){previousPlayer="";previousTick=0;previousX=previousY=previousZ=0;}
 static JsonObject capture(MinecraftServer server,ServerPlayer p){
  JsonObject out=new JsonObject(),player=entity(p,true);var food=p.getFoodData();
  String context=p.getUUID()+"|"+p.level().dimension().identifier();
  long nowTick=p.level().getGameTime();double elapsed=(nowTick-previousTick)/20.0;
  boolean velocityValid=context.equals(previousPlayer)&&elapsed>0&&elapsed<=10;
  player.addProperty("server_vx",p.getDeltaMovement().x*20);player.addProperty("server_vy",p.getDeltaMovement().y*20);player.addProperty("server_vz",p.getDeltaMovement().z*20);
  player.addProperty("vx",velocityValid?(p.getX()-previousX)/elapsed:0);player.addProperty("vy",velocityValid?(p.getY()-previousY)/elapsed:0);player.addProperty("vz",velocityValid?(p.getZ()-previousZ)/elapsed:0);player.addProperty("velocity_valid",velocityValid);player.addProperty("velocity_source","position_delta_per_game_second");
  previousPlayer=context;previousTick=nowTick;previousX=p.getX();previousY=p.getY();previousZ=p.getZ();
  player.addProperty("food",food.getFoodLevel());player.addProperty("saturation",food.getSaturationLevel());player.addProperty("air",p.getAirSupply());player.addProperty("max_air",p.getMaxAirSupply());player.addProperty("underwater",p.isUnderWater());player.addProperty("in_water",p.isInWater());player.addProperty("in_lava",p.isInLava());player.addProperty("on_fire",p.isOnFire());player.addProperty("on_ground",p.onGround());player.addProperty("sprinting",p.isSprinting());player.addProperty("sneaking",p.isShiftKeyDown());player.addProperty("swimming",p.isSwimming());player.addProperty("blocking",p.isBlocking());player.addProperty("using_item",p.isUsingItem());player.addProperty("sleeping",p.isSleeping());player.addProperty("xp_level",p.experienceLevel);player.addProperty("xp_total",p.totalExperience);player.addProperty("game_mode",p.gameMode.getGameModeForPlayer().getName());player.addProperty("yaw",p.getYRot());player.addProperty("pitch",p.getXRot());player.addProperty("last_damage_source",p.getLastDamageSource()==null?"":p.getLastDamageSource().getMsgId());
  JsonArray effects=new JsonArray();for(var effect:p.getActiveEffects()){JsonObject e=new JsonObject();e.addProperty("id",BuiltInRegistries.MOB_EFFECT.getKey(effect.getEffect().value()).toString());e.addProperty("amplifier",effect.getAmplifier());e.addProperty("duration_ticks",effect.getDuration());effects.add(e);}player.add("effects",effects);
  JsonObject inventory=new JsonObject();JsonArray slots=new JsonArray();int damage=0;boolean tool=false;
  for(int i=0;i<p.getInventory().getContainerSize();i++){ItemStack stack=p.getInventory().getItem(i);if(stack.isEmpty())continue;String id=BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();inventory.addProperty(id,(inventory.has(id)?inventory.get(id).getAsInt():0)+stack.getCount());JsonObject slot=new JsonObject();slot.addProperty("slot",i);slot.addProperty("stack_identity",Integer.toUnsignedLong(System.identityHashCode(stack)));slot.addProperty("item",id);slot.addProperty("count",stack.getCount());slot.addProperty("damage",stack.getDamageValue());slot.addProperty("max_damage",stack.getMaxDamage());slot.addProperty("components",truncate(stack.getComponentsPatch().toString(),1024));slots.add(slot);damage+=stack.getDamageValue();tool|=id.endsWith("_pickaxe")||id.endsWith("_axe");}
  player.addProperty("equipment_damage",damage);player.addProperty("has_tool",tool);out.add("player",player);out.add("inventory",inventory);out.add("inventory_slots",slots);
  JsonObject world=new JsonObject();world.addProperty("dimension",p.level().dimension().identifier().toString());world.addProperty("game_time",p.level().getGameTime());world.addProperty("day_time",p.level().getOverworldClockTime());world.addProperty("difficulty",p.level().getDifficulty().getSerializedName());world.addProperty("raining",p.level().isRaining());world.addProperty("thundering",p.level().isThundering());world.addProperty("biome",p.level().getBiome(p.blockPosition()).unwrapKey().map(k->k.identifier().toString()).orElse("unknown"));out.add("world",world);
  int radius=TelemetryMod.setting("entity_radius",64,8,160),cap=TelemetryMod.setting("max_entities",64,1,256),nbts=TelemetryMod.setting("nbt_entities",8,0,32);
  var entities=new ArrayList<>(p.level().getEntities(p,p.getBoundingBox().inflate(radius),e->e.isAlive()));entities.sort(Comparator.comparingDouble(p::distanceToSqr));JsonArray ea=new JsonArray();for(int i=0;i<Math.min(cap,entities.size());i++)ea.add(entity(entities.get(i),i<nbts));out.add("entities",ea);out.addProperty("entity_count_in_query",entities.size());out.addProperty("entity_radius",radius);out.addProperty("entities_truncated",entities.size()>cap);
  JsonArray completed=new JsonArray();JsonObject progress=new JsonObject();for(var holder:server.getAdvancements().getAllAdvancements()){String id=holder.id().toString();if(id.contains(":recipes/"))continue;var ap=p.getAdvancements().getOrStartProgress(holder);if(ap.isDone())completed.add(id);if(ap.hasProgress()){JsonObject pr=new JsonObject();pr.addProperty("complete",ap.isDone());pr.addProperty("fraction",ap.getPercent());JsonArray done=new JsonArray();ap.getCompletedCriteria().forEach(done::add);pr.add("completed_criteria",done);JsonArray remaining=new JsonArray();ap.getRemainingCriteria().forEach(remaining::add);pr.add("remaining_criteria",remaining);progress.add(id,pr);}}out.add("advancements",completed);out.add("advancement_progress",progress);
  JsonObject stats=new JsonObject();for(var id:BuiltInRegistries.CUSTOM_STAT){int value=p.getStats().getValue(Stats.CUSTOM.get(id));if(value!=0)stats.addProperty(id.getPath(),value);}
  long mined=0,crafted=0,eaten=0;JsonObject minedBy=new JsonObject(),craftedBy=new JsonObject(),brokenBy=new JsonObject();
  for(var block:BuiltInRegistries.BLOCK){int n=p.getStats().getValue(Stats.BLOCK_MINED,block);if(n>0){mined+=n;minedBy.addProperty(BuiltInRegistries.BLOCK.getKey(block).toString(),n);}}
  for(var item:BuiltInRegistries.ITEM){String key=BuiltInRegistries.ITEM.getKey(item).toString();int n=p.getStats().getValue(Stats.ITEM_CRAFTED,item);if(n>0){crafted+=n;craftedBy.addProperty(key,n);}int broken=p.getStats().getValue(Stats.ITEM_BROKEN,item);if(broken>0)brokenBy.addProperty(key,broken);if(item.components().has(DataComponents.FOOD))eaten+=p.getStats().getValue(Stats.ITEM_USED,item);}
  out.add("broken_by_type",brokenBy);JsonObject equipment=new JsonObject();for(var slot:EquipmentSlot.values()){ItemStack item=p.getItemBySlot(slot);if(!item.isEmpty())equipment.addProperty(slot.getName(),BuiltInRegistries.ITEM.getKey(item.getItem()).toString());}out.add("equipment",equipment);
  stats.addProperty("blocks_mined",mined);stats.addProperty("items_crafted",crafted);stats.addProperty("items_eaten",eaten);out.add("stats",stats);out.add("mined_by_type",minedBy);out.add("crafted_by_type",craftedBy);return out;
 }
 private static JsonObject entity(Entity e,boolean includeNbt){
  JsonObject o=new JsonObject();o.addProperty("uuid",e.getUUID().toString());o.addProperty("type",BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString());o.addProperty("x",e.getX());o.addProperty("y",e.getY());o.addProperty("z",e.getZ());var v=e.getDeltaMovement();o.addProperty("vx",v.x*20);o.addProperty("vy",v.y*20);o.addProperty("vz",v.z*20);o.addProperty("velocity_units","blocks_per_second_at_20_tps");JsonArray tags=new JsonArray();e.entityTags().forEach(tags::add);o.add("tags",tags);
  if(e instanceof LivingEntity living){o.addProperty("health",living.getHealth());o.addProperty("max_health",living.getMaxHealth());o.addProperty("armor",living.getArmorValue());}
  int limit=TelemetryMod.setting("nbt_chars",4096,0,16384);if(includeNbt&&limit>0){TagValueOutput data=TagValueOutput.createWithContext(ProblemReporter.DISCARDING,e.registryAccess());e.saveWithoutId(data);String nbt=data.buildResult().toString();o.addProperty("nbt_snbt",truncate(nbt,limit));o.addProperty("nbt_truncated",nbt.length()>limit);o.addProperty("nbt_original_chars",nbt.length());}return o;
 }
 private static String truncate(String s,int n){return s.length()>n?s.substring(0,n):s;}
}

