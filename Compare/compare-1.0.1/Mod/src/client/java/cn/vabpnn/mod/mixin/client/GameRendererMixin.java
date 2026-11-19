package cn.vabpnn.mod.mixin.client;

import cn.vabpnn.mod.client.capture.CaptureController;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(GameRenderer.class)
public abstract class GameRendererMixin {
    @Inject(method = "render", at = @At("TAIL"))
    private void vabpnn$verifyRenderTail(DeltaTracker deltaTracker, boolean advanceGameTime, CallbackInfo ci) {
        CaptureController.onGameRenderTail();
    }
}