package cn.vabpnn.telemetry.mixin;
import cn.vabpnn.telemetry.TelemetryMod;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.context.BlockPlaceContext;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.core.registries.BuiltInRegistries;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.*;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;
@Mixin(BlockItem.class)
abstract class BlockItemMixin {
 @Inject(method="placeBlock",at=@At("RETURN"))
 private void scorerPlaced(BlockPlaceContext context,BlockState state,CallbackInfoReturnable<Boolean> result){
  if(result.getReturnValueZ()&&context.getPlayer() instanceof ServerPlayer player)TelemetryMod.record(player,"block_place",context.getClickedPos(),BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString());
 }
}

