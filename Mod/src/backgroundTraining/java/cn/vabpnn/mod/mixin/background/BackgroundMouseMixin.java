package cn.vabpnn.mod.mixin.background;

import cn.vabpnn.mod.client.capture.BackgroundTraining;
import com.mojang.blaze3d.platform.InputConstants;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Keep vanilla logical mouse state, without capturing, hiding or warping the desktop cursor. */
@Mixin(InputConstants.class)
public abstract class BackgroundMouseMixin {
    @Inject(method="grabOrReleaseMouse", at=@At("HEAD"), cancellable=true)
    private static void vabpnn$noDesktopGrab(CallbackInfo callback) {
        if (BackgroundTraining.enabled()) callback.cancel();
    }
}
