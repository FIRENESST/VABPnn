package cn.vabpnn.mod.mixin.client;

import net.minecraft.client.MouseHandler;
import net.minecraft.client.input.MouseButtonInfo;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(MouseHandler.class)
public interface MouseHandlerAccess {
    @Invoker("onButton") void vabpnn$button(long window, MouseButtonInfo button, int action);
    @Invoker("onMove") void vabpnn$move(long window, double x, double y);
    @Accessor("isLeftPressed") void vabpnn$setLeftPressed(boolean value);
    @Accessor("isRightPressed") void vabpnn$setRightPressed(boolean value);
    @Invoker("onScroll") void vabpnn$scroll(long window, double horizontal, double vertical);
    @Accessor("accumulatedDX") double vabpnn$getAccumulatedDX();
    @Accessor("accumulatedDX") void vabpnn$setAccumulatedDX(double value);
    @Accessor("accumulatedDY") double vabpnn$getAccumulatedDY();
    @Accessor("accumulatedDY") void vabpnn$setAccumulatedDY(double value);
    @Accessor("ignoreFirstMove") boolean vabpnn$ignoreFirstMove();
    @Accessor("xpos") double vabpnn$getXpos();
    @Accessor("ypos") double vabpnn$getYpos();
}

