package cn.vabpnn.mod.mixin.client;
import cn.vabpnn.mod.client.capture.CaptureController;
import net.minecraft.client.Minecraft;
import net.minecraft.client.MouseHandler;
import net.minecraft.client.input.MouseButtonInfo;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
@Mixin(MouseHandler.class)
public abstract class HumanMouseMixin {
    @Inject(method="onMove", at=@At("HEAD"))
    private void vabpnn$realMove(long handle, double x, double y, CallbackInfo ci) {
        if (handle != Minecraft.getInstance().getWindow().handle()) return;
        MouseHandlerAccess mouse = (MouseHandlerAccess)(Object)this;
        if (!mouse.vabpnn$ignoreFirstMove())
            CaptureController.onHumanMove(x-mouse.vabpnn$getXpos(), y-mouse.vabpnn$getYpos());
    }
    @Inject(method="onButton", at=@At("HEAD"))
    private void vabpnn$realButton(long handle, MouseButtonInfo button, int action, CallbackInfo ci) {
        if (handle == Minecraft.getInstance().getWindow().handle()) CaptureController.onHumanButton(button.button(),action);
    }
    @Inject(method="onScroll", at=@At("HEAD"))
    private void vabpnn$realScroll(long handle, double x, double y, CallbackInfo ci) {
        if (handle == Minecraft.getInstance().getWindow().handle()) CaptureController.onHumanScroll(y);
    }
}
