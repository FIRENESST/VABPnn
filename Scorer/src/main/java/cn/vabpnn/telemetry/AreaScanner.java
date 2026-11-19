package cn.vabpnn.telemetry;
import com.google.gson.*;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.core.registries.*;
import net.minecraft.world.level.chunk.*;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import java.util.*;

final class AreaScanner {
 private int cx=Integer.MIN_VALUE,cz,radius,index,loaded;private String dimension="";private long started;
 private final Set<String> registry=new TreeSet<>();
 private final Map<String,JsonObject> structures=new LinkedHashMap<>();
 private final Map<String,JsonObject> blockEntities=new LinkedHashMap<>();
 private final List<int[]> offsets=new ArrayList<>();
 private JsonObject complete=new JsonObject();
 void reset(){cx=Integer.MIN_VALUE;complete=new JsonObject();}
 void tick(ServerPlayer p){
  int r=TelemetryMod.setting("chunk_radius",10,1,10);int x=p.chunkPosition().x(),z=p.chunkPosition().z();String dim=p.level().dimension().identifier().toString();
  if(cx!=x||cz!=z||radius!=r||!dimension.equals(dim)||index>=(2*r+1)*(2*r+1)){
   cx=x;cz=z;radius=r;dimension=dim;index=loaded=0;started=p.level().getGameTime();registry.clear();structures.clear();blockEntities.clear();offsets.clear();
   for(int dx=-r;dx<=r;dx++)for(int dz=-r;dz<=r;dz++)offsets.add(new int[]{dx,dz});
   offsets.sort(Comparator.comparingInt(a->a[0]*a[0]+a[1]*a[1]));
   if(!complete.has("dimension")||!complete.get("dimension").getAsString().equals(dim)||!complete.has("center_x")||complete.get("center_x").getAsInt()!=x||complete.get("center_z").getAsInt()!=z)complete=new JsonObject();
  }
  int side=2*r+1,total=side*side,budget=TelemetryMod.setting("chunks_per_tick",2,1,8);
  for(int n=0;n<budget&&index<total;n++,index++){
   int sx=cx+offsets.get(index)[0],sz=cz+offsets.get(index)[1];ChunkAccess chunk=p.level().getChunkSource().getChunkNow(sx,sz);if(chunk==null)continue;loaded++;
   for(LevelChunkSection section:chunk.getSections())if(!section.hasOnlyAir())section.getStates().getAll(state->{if(!state.isAir())registry.add(BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString());});
   if(chunk instanceof LevelChunk full)for(var be:full.getBlockEntities().values()){
    if(blockEntities.size()>=TelemetryMod.setting("nbt_block_entities",16,0,64))break;String key=be.getBlockPos().toShortString();JsonObject data=new JsonObject();data.addProperty("pos",key);data.addProperty("type",BuiltInRegistries.BLOCK_ENTITY_TYPE.getKey(be.getType()).toString());int limit=TelemetryMod.setting("nbt_chars",4096,0,16384);
    if(limit>0){String nbt=be.saveWithFullMetadata(p.registryAccess()).toString();data.addProperty("nbt_snbt",nbt.substring(0,Math.min(limit,nbt.length())));data.addProperty("nbt_truncated",nbt.length()>limit);}blockEntities.put(key,data);
   }
   chunk.getAllStarts().forEach((type,start)->{
    if(!start.isValid()||structures.size()>=512)return;
    String id=p.registryAccess().lookupOrThrow(Registries.STRUCTURE).getKey(type).toString(),key=dim+"|"+id+"|"+start.getChunkPos().x()+","+start.getChunkPos().z();
    JsonObject s=new JsonObject();s.addProperty("key",key);s.addProperty("type",id);s.addProperty("chunk_x",start.getChunkPos().x());s.addProperty("chunk_z",start.getChunkPos().z());var b=start.getBoundingBox();JsonArray box=new JsonArray();for(int v:new int[]{b.minX(),b.minY(),b.minZ(),b.maxX(),b.maxY(),b.maxZ()})box.add(v);s.add("bounds",box);s.addProperty("player_inside_bounds",b.isInside(p.blockPosition()));structures.put(key,s);
   });
  }
  if(index>=total){complete=snapshot();complete.addProperty("cycle_complete",true);}
 }
 private JsonObject snapshot(){
  JsonObject o=new JsonObject();o.addProperty("dimension",dimension);o.addProperty("center_x",cx);o.addProperty("center_z",cz);o.addProperty("chunk_radius",radius);o.addProperty("scan_start_tick",started);o.addProperty("chunks_examined",index);o.addProperty("chunks_loaded",loaded);o.addProperty("chunks_requested",(2*radius+1)*(2*radius+1));o.addProperty("cycle_complete",false);o.addProperty("registry_kind","loaded_chunk_palette_not_visibility");JsonArray ids=new JsonArray();registry.forEach(ids::add);o.add("block_registry",ids);JsonArray s=new JsonArray();structures.values().forEach(s::add);o.add("structures",s);o.addProperty("structure_count",s.size());o.addProperty("structures_truncated",s.size()>=512);return o;
 }
 JsonObject view(){JsonObject o=complete.isEmpty()?snapshot():complete.deepCopy();o.addProperty("next_cycle_progress",index);JsonArray entities=new JsonArray();blockEntities.values().forEach(entities::add);o.add("block_entities",entities);o.addProperty("block_entity_limit",TelemetryMod.setting("nbt_block_entities",16,0,64));return o;}
}

