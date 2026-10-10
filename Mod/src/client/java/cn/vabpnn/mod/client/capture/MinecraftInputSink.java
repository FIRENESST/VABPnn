package cn.vabpnn.mod.client.capture;

import cn.vabpnn.mod.mixin.client.KeyMappingAccess;
import cn.vabpnn.mod.mixin.client.KeyboardHandlerInvoker;
import cn.vabpnn.mod.mixin.client.ToggleKeyMappingAccess;
import cn.vabpnn.mod.mixin.client.MouseHandlerAccess;
import com.mojang.blaze3d.platform.InputConstants;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.input.KeyEvent;
import net.minecraft.client.input.MouseButtonInfo;
import org.lwjgl.glfw.GLFW;

/** Minecraft 26.1 callbacks, logical key state and cursor position share the client thread. */
final class MinecraftInputSink implements ActionExecutor.Sink {
    private static final int[] KEY_CODES = {87,65,83,68,69,32};
    private static final int[] KEY_ACTIONS = {0,1,2,3,4,12};
    private final Minecraft client;
    private Object commandLevel, commandScreen;
    private boolean commandGrabbed;
    private int touchedKeys, touchedButtons;
    MinecraftInputSink(Minecraft client) { this.client = client; }

    @Override public void beginCommand() {
        commandLevel = client.level; commandScreen = client.screen;
        commandGrabbed = client.mouseHandler.isMouseGrabbed();
    }
    @Override public boolean contextValid() {
        return client.level == commandLevel && client.screen == commandScreen
                && client.mouseHandler.isMouseGrabbed() == commandGrabbed;
    }
    private boolean physicalKey(int code) {
        return InputConstants.isKeyDown(client.getWindow(),code);
    }
    private boolean physicalButton(int button) {
        return GLFW.glfwGetMouseButton(client.getWindow().handle(),button) == GLFW.GLFW_PRESS;
    }
    @Override public void key(int code, boolean down) {
        if (!down && physicalKey(code)) return;
        if (down) for (int i=0;i<KEY_CODES.length;i++) if(KEY_CODES[i]==code) touchedKeys |= 1<<i;
        var event = new KeyEvent(code,GLFW.glfwGetKeyScancode(code),0);
        CaptureController.beginSynthetic();
        try {
            ((KeyboardHandlerInvoker) client.keyboardHandler).vabpnn$pressKey(client.getWindow().handle(),
                    down ? GLFW.GLFW_PRESS : GLFW.GLFW_RELEASE,event);
            // A GUI may consume keyReleased before vanilla updates its gameplay mappings.
            if (!down || client.screen==null && client.getOverlay()==null)
                forceMapping(InputConstants.Type.KEYSYM,code,down,false);
        } finally { CaptureController.endSynthetic(); }
    }
    @Override public void button(int button, boolean down) {
        if (!down && physicalButton(button)) return;
        if (down) touchedButtons |= 1<<button;
        CaptureController.beginSynthetic();
        try {
            MouseHandlerAccess mouse = (MouseHandlerAccess) client.mouseHandler;
            mouse.vabpnn$button(client.getWindow().handle(),new MouseButtonInfo(button,0),
                    down ? GLFW.GLFW_PRESS : GLFW.GLFW_RELEASE);
            // Container callbacks can return early, so release the world state as well.
            if (!down || client.screen==null && client.getOverlay()==null)
                forceMapping(InputConstants.Type.MOUSE,button,down,false);
            if (!down) {
                if (button==0) mouse.vabpnn$setLeftPressed(false);
                if (button==1) mouse.vabpnn$setRightPressed(false);
            }
        } finally { CaptureController.endSynthetic(); }
    }
    @Override public void move(double dx, double dy) {
        MouseHandlerAccess mouse = (MouseHandlerAccess) client.mouseHandler;
        if (client.mouseHandler.isMouseGrabbed()) {
            mouse.vabpnn$setAccumulatedDX(mouse.vabpnn$getAccumulatedDX()+dx);
            mouse.vabpnn$setAccumulatedDY(mouse.vabpnn$getAccumulatedDY()+dy);
            CaptureController.onModelMove(dx,dy);
        } else {
            double beforeX=mouse.vabpnn$getXpos(),beforeY=mouse.vabpnn$getYpos();
            double x = Math.clamp(beforeX+dx,0,Math.max(0,client.getWindow().getScreenWidth()-1));
            double y = Math.clamp(beforeY+dy,0,Math.max(0,client.getWindow().getScreenHeight()-1));
            CaptureController.beginSynthetic();
            try {
                // Update Minecraft synchronously; GLFW's deferred callback must not decide click position.
                if (mouse.vabpnn$ignoreFirstMove())
                    mouse.vabpnn$move(client.getWindow().handle(),mouse.vabpnn$getXpos(),mouse.vabpnn$getYpos());
                mouse.vabpnn$move(client.getWindow().handle(),x,y);
                GLFW.glfwSetCursorPos(client.getWindow().handle(),x,y);
                client.mouseHandler.handleAccumulatedMovement();
            } finally { CaptureController.endSynthetic(); }
            CaptureController.onModelMove(x-beforeX,y-beforeY);
        }
    }
    @Override public void scroll(double steps) {
        CaptureController.beginSynthetic();
        try { ((MouseHandlerAccess) client.mouseHandler).vabpnn$scroll(client.getWindow().handle(),0,steps); }
        finally { CaptureController.endSynthetic(); }
    }
    /** Recover holds reset by real input or vanilla focus restoration, without another click edge. */
    void maintainHeldInputs(int keys,int buttons) {
        if (client.screen!=null || client.getOverlay()!=null) return;
        for(int i=0;i<KEY_CODES.length;i++)
            if ((keys & (1<<KEY_ACTIONS[i]))!=0)
                forceMapping(InputConstants.Type.KEYSYM,KEY_CODES[i],true,false);
        MouseHandlerAccess mouse=(MouseHandlerAccess)client.mouseHandler;
        if ((buttons&1)!=0) { forceMapping(InputConstants.Type.MOUSE,0,true,false); mouse.vabpnn$setLeftPressed(true); }
        if ((buttons&2)!=0) { forceMapping(InputConstants.Type.MOUSE,1,true,false); mouse.vabpnn$setRightPressed(true); }
    }
    @Override public void cancelPending() {
        if (touchedKeys==0 && touchedButtons==0) return;
        for(int i=0;i<KEY_CODES.length;i++)
            if ((touchedKeys & (1<<i))!=0 && !physicalKey(KEY_CODES[i]))
                forceMapping(InputConstants.Type.KEYSYM,KEY_CODES[i],false,true);
        for(int button=0;button<2;button++)
            if ((touchedButtons & (1<<button))!=0 && !physicalButton(button)) {
                forceMapping(InputConstants.Type.MOUSE,button,false,true);
                MouseHandlerAccess mouse=(MouseHandlerAccess)client.mouseHandler;
                if(button==0) mouse.vabpnn$setLeftPressed(false); else mouse.vabpnn$setRightPressed(false);
            }
        touchedKeys=touchedButtons=0;
    }
    private void forceMapping(InputConstants.Type type,int code,boolean down,boolean clearClicks) {
        for(KeyMapping mapping:client.options.keyMappings) {
            KeyMappingAccess access=(KeyMappingAccess)mapping;
            InputConstants.Key key=access.vabpnn$boundKey();
            if(key.getType()==type && key.getValue()==code) {
                // Bypass ToggleKeyMapping.setDown/release: model objects have explicit holds/releases.
                access.vabpnn$forceDown(down);
                if(clearClicks) access.vabpnn$setClickCount(0);
                if(!down && mapping instanceof ToggleKeyMappingAccess toggle)
                    toggle.vabpnn$setScreenRestore(false);
            }
        }
    }
}
