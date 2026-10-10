package cn.vabpnn.mod.mixin.client;

import cn.vabpnn.mod.client.capture.CaptureController;
import net.minecraft.client.KeyboardHandler;
import net.minecraft.client.input.KeyEvent;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(KeyboardHandler.class)
public abstract class KeyboardHandlerMixin {
    @Inject(method = "keyPress", at = @At("HEAD"))
    private void vabpnn$onCaptureKey(long handle, int action, KeyEvent event, CallbackInfo ci) {
        CaptureController.onKey(handle, action, event);
    }
}