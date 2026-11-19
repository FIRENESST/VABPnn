package cn.vabpnn.mod.mixin;

import cn.vabpnn.mod.VabpnnMod;
import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Minimal example shared by integrated servers and dedicated servers. */
@Mixin(MinecraftServer.class)
public abstract class MinecraftServerMixin {
    @Inject(method = "runServer", at = @At("HEAD"))
    private void vabpnn$onServerStart(CallbackInfo ci) {
        VabpnnMod.LOGGER.info("VABPnn server mixin active.");
    }
}