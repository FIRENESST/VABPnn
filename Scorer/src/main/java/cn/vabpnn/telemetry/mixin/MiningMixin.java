package cn.vabpnn.telemetry.mixin;

import cn.vabpnn.telemetry.MiningTools;
import cn.vabpnn.telemetry.TelemetryMod;
import cn.vabpnn.telemetry.HarvestDrops;
import com.llamalad7.mixinextras.injector.wrapmethod.WrapMethod;
import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.level.ServerPlayerGameMode;
import org.spongepowered.asm.mixin.*;
import org.spongepowered.asm.mixin.injection.*;
import org.spongepowered.asm.mixin.injection.callback.*;

@Mixin(ServerPlayerGameMode.class)
public abstract class MiningMixin {
 @Shadow @Final protected ServerPlayer player;
 @Shadow private boolean isDestroyingBlock;
 @Shadow private BlockPos destroyPos;
 @Shadow private boolean hasDelayedDestroy;
 @Shadow private BlockPos delayedDestroyPos;

 @Inject(method="tick",at=@At("HEAD"))
 private void scorer$hold(CallbackInfo ci){
  if(isDestroyingBlock)TelemetryMod.miningAttempt(player,destroyPos);
  else if(hasDelayedDestroy)TelemetryMod.miningAttempt(player,delayedDestroyPos);
 }
 @WrapMethod(method="destroyBlock")
 private boolean scorer$break(BlockPos pos,Operation<Boolean> original){
  if(!TelemetryMod.controls(player))return original.call(pos);
  JsonObject tool=MiningTools.describe(player,pos);HarvestDrops.Context prior=HarvestDrops.CURRENT.get(),harvest=new HarvestDrops.Context();HarvestDrops.CURRENT.set(harvest);
  try{boolean success=original.call(pos);if(success){tool.add("harvest",harvest.evidence());TelemetryMod.record(player,"block_break",pos,tool.get("block").getAsString(),tool);}return success;}
  finally{if(prior==null)HarvestDrops.CURRENT.remove();else HarvestDrops.CURRENT.set(prior);}
 }
}
