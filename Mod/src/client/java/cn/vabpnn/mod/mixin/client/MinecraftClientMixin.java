package cn.vabpnn.mod.mixin.client;

import cn.vabpnn.mod.VabpnnMod;
import cn.vabpnn.mod.client.capture.CaptureController;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Minecraft.class)
public abstract class MinecraftClientMixin {
    @Inject(method = "run", at = @At("HEAD"))
    private void vabpnn$onClientStart(CallbackInfo ci) {
        VabpnnMod.LOGGER.info("VABPnn client mixin active.");
    }

    @Inject(method = "renderFrame", at = @At(value = "INVOKE",
            target = "Lcom/mojang/blaze3d/pipeline/RenderTarget;blitToScreen()V", shift = At.Shift.AFTER))
    private void vabpnn$capturePresentedFrame(boolean advanceGameTime, CallbackInfo ci) {
        CaptureController.onPresentedFrame();
    }

    @Inject(method = "close", at = @At("HEAD"))
    private void vabpnn$releaseCapture(CallbackInfo ci) {
        CaptureController.close();
    }
}