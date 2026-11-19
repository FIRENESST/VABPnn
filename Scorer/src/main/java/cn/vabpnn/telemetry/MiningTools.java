package cn.vabpnn.telemetry;

import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.tags.BlockTags;
import net.minecraft.tags.ItemTags;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.state.BlockState;

/** Facts from the live block state and item components, including modded tool rules. */
public final class MiningTools {
 public static JsonObject describe(ServerPlayer player,BlockPos pos){
  BlockState state=player.level().getBlockState(pos);ItemStack held=player.getMainHandItem();
  String family=state.is(BlockTags.MINEABLE_WITH_PICKAXE)?"pickaxe":state.is(BlockTags.MINEABLE_WITH_AXE)?"axe":state.is(BlockTags.MINEABLE_WITH_SHOVEL)?"shovel":state.is(BlockTags.MINEABLE_WITH_HOE)?"hoe":"special_or_hand";
  boolean matches=switch(family){case "pickaxe"->held.is(ItemTags.PICKAXES);case "axe"->held.is(ItemTags.AXES);case "shovel"->held.is(ItemTags.SHOVELS);case "hoe"->held.is(ItemTags.HOES);default->false;};
  boolean needs=state.requiresCorrectToolForDrops(),harvest=!needs||held.isCorrectToolForDrops(state);
  float speed=held.getDestroySpeed(state),best=harvest?speed:0;String bestItem=harvest?id(held):"";
  for(int i=0;i<player.getInventory().getContainerSize();i++){
   ItemStack item=player.getInventory().getItem(i);if(item.isEmpty()||(needs&&!item.isCorrectToolForDrops(state)))continue;
   float candidate=item.getDestroySpeed(state);if(candidate>best){best=candidate;bestItem=id(item);}
  }
  JsonObject result=new JsonObject();result.addProperty("held_item",id(held));result.addProperty("recommended_family",family);
  result.addProperty("requires_correct_tool",needs);result.addProperty("can_harvest",harvest);result.addProperty("speed",speed);
  result.addProperty("family_matches",matches||speed>1);result.addProperty("held_is_tool",held.is(ItemTags.PICKAXES)||held.is(ItemTags.AXES)||held.is(ItemTags.SHOVELS)||held.is(ItemTags.HOES)||held.is(ItemTags.SWORDS)||speed>1);
  result.addProperty("best_item",bestItem);result.addProperty("best_speed",best);result.addProperty("hardness",state.getDestroySpeed(player.level(),pos));
  result.addProperty("block",BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString());return result;
 }
 private static String id(ItemStack item){return item.isEmpty()?"minecraft:air":BuiltInRegistries.ITEM.getKey(item.getItem()).toString();}
}
