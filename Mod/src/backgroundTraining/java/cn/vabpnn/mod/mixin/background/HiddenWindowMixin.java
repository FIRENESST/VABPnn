package cn.vabpnn.mod.mixin.background;

import cn.vabpnn.mod.client.capture.BackgroundTraining;
import com.mojang.blaze3d.platform.Window;
import org.lwjgl.glfw.GLFW;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** Make the actual GPU window invisible at creation, before Windows can show or focus it. */
@Mixin(Window.class)
public abstract class HiddenWindowMixin {
    @Inject(method="createGlfwWindow", at=@At(value="INVOKE",
            target="Lorg/lwjgl/glfw/GLFW;glfwCreateWindow(IILjava/lang/CharSequence;JJ)J"))
    private static void vabpnn$hidden(CallbackInfoReturnable<Long> callback) {
        if (BackgroundTraining.enabled()) {
            GLFW.glfwWindowHint(GLFW.GLFW_VISIBLE, GLFW.GLFW_FALSE);
            GLFW.glfwWindowHint(GLFW.GLFW_FOCUSED, GLFW.GLFW_FALSE);
            GLFW.glfwWindowHint(GLFW.GLFW_FOCUS_ON_SHOW, GLFW.GLFW_FALSE);
        }
    }
}
