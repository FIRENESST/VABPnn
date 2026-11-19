package cn.vabpnn.mod.mixin.background;

import cn.vabpnn.mod.client.capture.BackgroundTraining;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** Only the explicitly isolated background instance accepts its own synthetic input while hidden. */
@Mixin(Minecraft.class)
public abstract class BackgroundClientMixin {
    @Inject(method="isWindowActive", at=@At("HEAD"), cancellable=true)
    private void vabpnn$logicalInputContext(CallbackInfoReturnable<Boolean> callback) {
        if (BackgroundTraining.enabled()) callback.setReturnValue(true);
    }
}
