package cn.vabpnn.mod.mixin.background;

import cn.vabpnn.mod.client.capture.BackgroundTraining;
import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;

/** GUI coordinates are already updated synchronously in the production sink. Skip its OS warp. */
@Mixin(targets="cn.vabpnn.mod.client.capture.MinecraftInputSink", remap=false)
public abstract class BackgroundCursorMixin {
    @WrapOperation(method="move", at=@At(value="INVOKE",
            target="Lorg/lwjgl/glfw/GLFW;glfwSetCursorPos(JDD)V"))
    private void vabpnn$noDesktopWarp(long window, double x, double y, Operation<Void> original) {
        if (!BackgroundTraining.enabled()) original.call(window, x, y);
    }
}
