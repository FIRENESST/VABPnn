package cn.vabpnn.mod.client.capture;

import java.lang.reflect.Field;
import java.util.List;
import java.util.function.Consumer;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.chat.GuiMessage;
import net.minecraft.client.gui.screens.ChatScreen;
import net.minecraft.client.input.KeyEvent;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.contents.TranslatableContents;
import org.lwjgl.glfw.GLFW;

/** Checks real local chat routing, action-bar throttling and repeated asynchronous saves. */
final class MinecraftFeedbackChecks {
    private static void require(boolean condition,String message) {
        if(!condition) throw new AssertionError(message);
    }
    private static Object field(Object target,String name) {
        try {
            Field field=target.getClass().getDeclaredField(name);
            field.setAccessible(true); return field.get(target);
        } catch(ReflectiveOperationException error) { throw new AssertionError(error); }
    }
    @SuppressWarnings("unchecked")
    private static List<GuiMessage> messages(Minecraft client) {
        return (List<GuiMessage>)field(client.gui.getChat(),"allMessages");
    }
    private static boolean has(Component component,String key) {
        if(component==null) return false;
        if(component.getContents() instanceof TranslatableContents translated) {
            if(translated.getKey().equals(key)) return true;
            for(Object argument:translated.getArgs())
                if(argument instanceof Component nested && has(nested,key)) return true;
        }
        for(Component sibling:component.getSiblings()) if(has(sibling,key)) return true;
        return false;
    }
    private static long chatCount(Minecraft client,String key) {
        return messages(client).stream().filter(message -> has(message.content(),key)).count();
    }
    private static boolean overlay(Minecraft client,String key) {
        return (int)field(client.gui,"overlayMessageTime")>0
                && has((Component)field(client.gui,"overlayMessageString"),key);
    }
    private static void command(ClientGameTestContext context,String command) {
        command(context,command,client -> {});
    }
    private static void command(ClientGameTestContext context,String command,Consumer<Minecraft> afterCommand) {
        context.runOnClient(client -> {
            ChatScreen chat=new ChatScreen("",false);
            client.setScreen(chat);
            chat.handleChatInput("/vabpnn "+command,true);
            afterCommand.accept(client);
        });
    }
    private static void key(ClientGameTestContext context,int code) {
        context.runOnClient(client -> CaptureController.onKey(client.getWindow().handle(),
                GLFW.GLFW_PRESS,new KeyEvent(code,0,0)));
    }
    static void verifyHotkey(Minecraft client,int code) {
        String key=switch(code) {
            case GLFW.GLFW_KEY_F8 -> CaptureController.inputState().modelControl() ? "model_on":"demo_on";
            case GLFW.GLFW_KEY_F9 -> CaptureController.inputState().capturing() ? "continuous_on":"continuous_off";
            default -> null;
        };
        if(key!=null) require(chatCount(client,"message.vabpnn."+key)>0,"hotkey feedback reached actual local chat");
    }
    static void verifyEmptyModel(ClientGameTestContext context) {
        context.waitFor(client -> {
            var state=CaptureController.feedbackState(true);
            return state.decisionFrame()>0 && state.input().frame()>0 && state.input().selectedMask()==0;
        },120);
        context.runOnClient(client -> {
            var state=CaptureController.feedbackState(true);
            require(has(state.gate(),"message.vabpnn.state.idle_k")
                    || has(state.gate(),"message.vabpnn.state.idle_threshold")
                    || has(state.gate(),"message.vabpnn.state.idle"),
                    "zero-action packets report model selection instead of falsely claiming execution");
        });
        context.waitFor(client -> chatCount(client,"message.vabpnn.model_idle")>0,120);
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: F8 zero-action diagnostics distinguish GPU selection from input delivery in real chat.");
    }
    static void verifySilentInventoryFeedback(ClientGameTestContext context) {
        context.runOnClient(client -> {
            require(chatCount(client,"message.vabpnn.inventory_recorded")==0,
                    "inventory recording does not post chat feedback");
            require(!overlay(client,"message.vabpnn.inventory_recorded"),
                    "inventory recording does not post action-bar feedback");
        });
    }
    private static Object staticField(Class<?> type,String name) {
        try {
            Field field=type.getDeclaredField(name); field.setAccessible(true); return field.get(null);
        } catch(ReflectiveOperationException error) { throw new AssertionError(error); }
    }
    private static void requireCold() {
        require(!CaptureController.isStarted() && !CaptureController.inputState().capturing(),
                "default startup is standby");
        require(!CaptureController.pipelineAllocated() && staticField(CaptureController.class,"surfaces")==null,
                "no CUDA bridge or FBO allocation before start");
        require(!(boolean)staticField(NativeCudaBridge.class,"loaded"),"CUDA DLL is not loaded before start");
        var monitor=(cn.vabpnn.mod.client.scope.NetworkMonitor)staticField(CaptureController.class,"monitor");
        require(!monitor.visible() && field(monitor,"window")==null,"no scope window before start");
    }
    static void verify(ClientGameTestContext context) {
        context.runOnClient(client -> requireCold());
        context.waitFor(client -> chatCount(client,"message.vabpnn.welcome")>0,120);
        command(context,"status");
        context.runOnClient(client -> {
            require(chatCount(client,"message.vabpnn.state.not_started")>0,"status explains manual start");
            require(chatCount(client,"message.vabpnn.mode.standby")>0,"standby is distinct from demonstration mode");
            client.setScreen(null);
        });
        for(int code:new int[]{GLFW.GLFW_KEY_F7,GLFW.GLFW_KEY_F8,GLFW.GLFW_KEY_F9,GLFW.GLFW_KEY_F10})
            key(context,code);
        context.waitTicks(20);
        context.runOnClient(client -> {
            requireCold();
            require(chatCount(client,"message.vabpnn.not_started")>=4,"all pipeline hotkeys require start first");
        });
        command(context,"start",client ->
                require(!CaptureController.pipelineAllocated(),"command defers GL/CUDA allocation to a presented frame"));
        context.runOnClient(client -> {
            require(CaptureController.isStarted() && CaptureController.inputState().capturing(),
                    "real client start command opens the capture gate");
            require(!CaptureController.inputState().modelControl(),"first start uses demonstration mode");
            require(chatCount(client,"message.vabpnn.started")>0,"start is acknowledged in local chat");
            client.setScreen(null);
        });
        context.waitFor(client -> CaptureController.pipelineAllocated()
                && CaptureController.feedbackState(true).phase()==NativeCudaBridge.READY
                && CaptureController.feedbackState(false).processed()>0,600);
        Object pipeline=context.computeOnClient(client -> staticField(CaptureController.class,"bridge"));
        command(context,"start");
        context.runOnClient(client -> {
            require(staticField(CaptureController.class,"bridge")==pipeline,"repeated start preserves the pipeline");
            require(chatCount(client,"message.vabpnn.already_started")>0,"repeated start is reported");
            client.setScreen(null);
        });
        command(context,"status");
        context.runOnClient(client -> {
            for(String section:List.of("mode","actions","gpu","learning"))
                require(chatCount(client,"message.vabpnn.status."+section)>0,
                        "client status command reports "+section+" in actual chat");
            require(chatCount(client,"message.vabpnn.state.chat")>0,"chat query explains suspended input");
            require(!CaptureController.inputState().modelControl(),"status query keeps demonstration mode");
        });
        command(context,"help");
        context.runOnClient(client -> {
            require(chatCount(client,"message.vabpnn.help")>0 && chatCount(client,"message.vabpnn.commands")>0,
                    "client help command works without server permissions");
            client.setScreen(null);
        });
        key(context,GLFW.GLFW_KEY_F9);
        command(context,"status");
        context.runOnClient(client -> {
            require(chatCount(client,"message.vabpnn.state.paused")>0,"paused status is distinct from chat suspension");
            client.setScreen(null);
        });
        key(context,GLFW.GLFW_KEY_F9);
        command(context,"hud off");
        context.runOnClient(client -> client.setScreen(null));
        context.waitTicks(70);
        context.runOnClient(client -> require(!overlay(client,"message.vabpnn.hud"),"HUD can be disabled"));
        command(context,"hud on");
        context.runOnClient(client -> client.setScreen(null));
        context.waitFor(client -> overlay(client,"message.vabpnn.hud"),160);
        context.runOnClient(client -> client.options.keyUp.setDown(true));
        context.waitFor(client -> overlay(client,"message.vabpnn.hud")
                && ((Component)field(client.gui,"overlayMessageString")).getString().contains("W"),30);
        context.runOnClient(client -> client.options.keyUp.setDown(false));
        int count=context.computeOnClient(client -> messages(client).size());
        context.waitTicks(30);
        context.runOnClient(client -> require(messages(client).size()==count,"live HUD never floods chat history"));
        for(int attempt=0;attempt<2;attempt++) {
            long previous=context.computeOnClient(client -> chatCount(client,"message.vabpnn.save_finished"));
            key(context,GLFW.GLFW_KEY_F10);
            long request=context.computeOnClient(client -> {
                require(chatCount(client,"message.vabpnn.save_requested")>0,"save request is acknowledged");
                try {
                    Field pending=InGameFeedback.class.getDeclaredField("pendingSave");
                    pending.setAccessible(true); return pending.getLong(null);
                } catch(ReflectiveOperationException error) { throw new AssertionError(error); }
            });
            require(request>0,"save request has a worker completion serial");
            key(context,GLFW.GLFW_KEY_F10);
            context.runOnClient(client -> require(chatCount(client,"message.vabpnn.save_pending")>0,
                    "pending save is coalesced instead of falsely reported as complete"));
            context.waitFor(client -> CaptureController.feedbackState(true).savedRequest()>=request
                    && CaptureController.feedbackState(false).checkpoint()==3,1200);
            context.waitFor(client -> chatCount(client,"message.vabpnn.save_finished")>previous,120);
        }
        for(String mode:new String[]{"off","independent","cooperate","compete","on"}) {
            command(context,"blocks "+mode); context.runOnClient(client -> client.setScreen(null));
            context.waitFor(client -> {
                var blocks=CaptureController.hierarchyState();
                return blocks.frame()>0 && (mode.equals("off") ? blocks.enabled()==0:
                        mode.equals("on") ? blocks.enabled()==1:blocks.mode()==switch(mode) {
                            case "independent" -> 0; case "compete" -> 2; default -> 1; });
            },300);
        }
        command(context,"blocks cooperate"); context.runOnClient(client -> client.setScreen(null));
        context.waitFor(client -> CaptureController.hierarchyState().enabled()==1
                && CaptureController.hierarchyState().mode()==1,300);
        command(context,"blocks");
        context.runOnClient(client -> {
            require(CaptureController.hierarchyState().samples()>0,"real game human demonstration scores high-level controllers");
            client.setScreen(null);
        });
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: /vabpnn blocks off/on/independent/cooperate/compete applies to real completed GPU frames.");
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: cold standby without CUDA/FBO/scope; only /vabpnn start opens the pipeline; repeated start, local status/help/hud, held-key HUD and two F10 save completions.");
    }
    static void verifyStopRestart(ClientGameTestContext context) {
        key(context,GLFW.GLFW_KEY_F8);
        context.waitFor(client -> CaptureController.inputState().allowed()
                && CaptureController.inputState().frame()>0,600);
        command(context,"start");
        context.runOnClient(client -> {
            require(CaptureController.inputState().modelControl(),"repeated start preserves model-control mode");
            client.setScreen(null);
        });
        context.waitFor(client -> CaptureController.inputState().allowed()
                && CaptureController.inputState().frame()>0,600);
        Object pipeline=context.computeOnClient(client -> staticField(CaptureController.class,"bridge"));
        command(context,"stop");
        context.runOnClient(client -> {
            var input=CaptureController.inputState();
            require(!CaptureController.isStarted() && !input.capturing() && !input.allowed()
                    && input.frame()==0 && input.heldKeys()==0 && input.heldButtons()==0,
                    "stop immediately restores standby and releases owned model input");
            require(!(boolean)staticField(CaptureController.class,"scopeSampling"),"stop disables scope sampling");
            require(chatCount(client,"message.vabpnn.stopped")>0,"stop reports its result");
            client.setScreen(null);
        });
        long submitted=context.computeOnClient(client -> ((NativeCudaBridge)pipeline).stats().submitted());
        key(context,GLFW.GLFW_KEY_F9);
        context.waitTicks(30);
        context.runOnClient(client -> {
            require(!CaptureController.isStarted() && !CaptureController.inputState().capturing(),
                    "F9 cannot restart after stop");
            require(((NativeCudaBridge)pipeline).stats().submitted()==submitted,"stop prevents all new frame submissions");
        });
        command(context,"start");
        context.runOnClient(client -> {
            require(CaptureController.isStarted() && !CaptureController.inputState().modelControl(),
                    "restart defaults to human demonstration");
            require(staticField(CaptureController.class,"bridge")==pipeline,"restart preserves learned GPU parameters");
            client.setScreen(null);
        });
        context.waitFor(client -> ((NativeCudaBridge)pipeline).stats().submitted()>submitted,200);
        // Also check command-based resume retains the configured mode after an F9 pause.
        key(context,GLFW.GLFW_KEY_F8);
        key(context,GLFW.GLFW_KEY_F9);
        command(context,"start");
        context.runOnClient(client -> {
            require(CaptureController.inputState().capturing() && CaptureController.inputState().modelControl(),
                    "start resumes an F9 pause without changing model-control mode");
            require(chatCount(client,"message.vabpnn.resumed")>0,"resume is reported");
            client.setScreen(null);
        });
        context.waitFor(client -> CaptureController.inputState().allowed()
                && CaptureController.inputState().frame()>0,600);
        // The test framework disconnects next; observe the production disconnect callback.
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: /vabpnn stop releases model control, F9 cannot bypass it, no stopped-frame submissions, warm restart preserves parameters, and command resume preserves mode.");
    }
    static void verifyDisconnect(ClientGameTestContext context) {
        context.waitFor(client -> client.level==null && !CaptureController.isStarted(),120);
        context.runOnClient(client -> {
            var input=CaptureController.inputState();
            require(!CaptureController.isStarted() && !input.capturing() && !input.allowed()
                    && input.heldKeys()==0 && input.heldButtons()==0,"disconnect restores standby");
        });
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: leaving a world restores standby and requires /vabpnn start on the next connection.");
    }
}
