package cn.vabpnn.telemetry.mixin;

import cn.vabpnn.telemetry.*;
import com.google.gson.*;
import com.llamalad7.mixinextras.injector.wrapmethod.WrapMethod;
import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.storage.*;
import org.spongepowered.asm.mixin.*;
import org.spongepowered.asm.mixin.injection.*;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(ItemEntity.class)
public abstract class ItemEntityMixin implements TrackedDrop {
 @Unique private final DropLedger scorer$ledger=new DropLedger();
 @Override public DropLedger scorer$origins(){return scorer$ledger;}
 @Inject(method="<init>(Lnet/minecraft/world/level/Level;DDDLnet/minecraft/world/item/ItemStack;)V",at=@At("RETURN"))
 private void scorer$born(Level level,double x,double y,double z,ItemStack stack,CallbackInfo ci){HarvestDrops.created((ItemEntity)(Object)this);}
 @Inject(method="<init>(Lnet/minecraft/world/level/Level;DDDLnet/minecraft/world/item/ItemStack;DDD)V",at=@At("RETURN"))
 private void scorer$born(Level level,double x,double y,double z,ItemStack stack,double vx,double vy,double vz,CallbackInfo ci){HarvestDrops.created((ItemEntity)(Object)this);}
 @WrapMethod(method="merge(Lnet/minecraft/world/entity/item/ItemEntity;Lnet/minecraft/world/item/ItemStack;Lnet/minecraft/world/entity/item/ItemEntity;Lnet/minecraft/world/item/ItemStack;)V")
 private static void scorer$merge(ItemEntity target,ItemStack targetStack,ItemEntity source,ItemStack sourceStack,Operation<Void> original){
  int before=sourceStack.getCount();original.call(target,targetStack,source,sourceStack);int moved=Math.max(0,before-sourceStack.getCount());
  ((TrackedDrop)target).scorer$origins().add(((TrackedDrop)source).scorer$origins().take(moved));
 }
 @WrapOperation(method="playerTouch",at=@At(value="INVOKE",target="Lnet/minecraft/world/entity/player/Inventory;add(Lnet/minecraft/world/item/ItemStack;)Z"))
 private boolean scorer$pickup(Inventory inventory,ItemStack stack,Operation<Boolean> original){
  ItemEntity entity=(ItemEntity)(Object)this;int before=stack.getCount();String item=BuiltInRegistries.ITEM.getKey(stack.getItem()).toString();
  boolean result=original.call(inventory,stack);int picked=Math.max(0,before-stack.getCount());if(picked==0)return result;
  JsonObject origins=scorer$ledger.take(picked);
  if(inventory.player instanceof ServerPlayer p&&TelemetryMod.controls(p)){JsonObject detail=new JsonObject();detail.addProperty("item",item);detail.addProperty("count",picked);detail.addProperty("entity_id",entity.getUUID().toString());detail.add("origins",origins);TelemetryMod.recordDetail(p,"item_pickup",entity.blockPosition(),detail);}return result;
 }
 @Inject(method="addAdditionalSaveData",at=@At("RETURN"))
 private void scorer$save(ValueOutput output,CallbackInfo ci){output.putString("vabpnn_scorer_origins",scorer$ledger.save());}
 @Inject(method="readAdditionalSaveData",at=@At("RETURN"))
 private void scorer$load(ValueInput input,CallbackInfo ci){scorer$ledger.load(input.getStringOr("vabpnn_scorer_origins","{}"));}
 @Inject(method="restoreFrom",at=@At("RETURN"))
 private void scorer$copy(Entity source,CallbackInfo ci){if(source instanceof TrackedDrop tracked)scorer$ledger.load(tracked.scorer$origins().save());}
}
