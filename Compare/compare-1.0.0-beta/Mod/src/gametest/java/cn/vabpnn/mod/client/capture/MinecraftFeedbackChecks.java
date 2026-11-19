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
            case GLFW.GLFW_KEY_F6 -> CaptureController.memeWindowVisible() ? "meme_scope_on":"meme_scope_off";
            case GLFW.GLFW_KEY_F8 -> CaptureController.inputState().modelControl() ? "model_on":"demo_on";
            case GLFW.GLFW_KEY_F9 -> CaptureController.inputState().capturing() ? "continuous_on":"continuous_off";
            default -> null;
        };
        if(key!=null) require(chatCount(client,"message.vabpnn."+key)>0,"hotkey feedback reached actual local chat");
    }
    static void verifyContinuousModel(ClientGameTestContext context) {
        context.waitFor(client -> {
            var state=CaptureController.feedbackState(true);
            return state.decisionFrame()>0 && state.input().frame()>0 && state.input().allowed()
                    && state.eligibleMask()==ActionLayout.KNOWN_MASK;
        },200);
        var before=context.computeOnClient(client -> CaptureController.feedbackState(true));
        var timeBefore=context.computeOnClient(client -> CaptureController.temporalState());
        context.waitTicks(20);
        context.runOnClient(client -> {
            var after=CaptureController.feedbackState(true);
            require(after.input().modelControl() && after.input().appliedCommands()>before.input().appliedCommands()
                    && after.decisionFrame()>before.decisionFrame() && after.eligibleMask()==ActionLayout.KNOWN_MASK,
                    "F8 continuously consumes real GPU top-k packets with every valid action eligible");
            require(after.examples()==before.examples() && after.decisionSteps()==before.decisionSteps()
                    && CaptureController.temporalState().minuteSteps()==timeBefore.minuteSteps()
                    && CaptureController.temporalState().bayesianMotionSteps()==timeBefore.bayesianMotionSteps(),
                    "Continuous action packets do not unfreeze neural learning");
            if(after.requestedK()==0) require(after.selectedCount()==0,"Learned rest remains a valid zero-action decision");
            else require(after.selectedCount()>0,"Positive learned k cannot be suppressed by a confidence threshold");
        });
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: F8 continuously predicts and applies real GPU packets without a confidence gate; neural weights stay frozen.");
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
        var memes=(cn.vabpnn.mod.client.scope.MemePropagationMonitor)staticField(CaptureController.class,"memeMonitor");
        require(!memes.visible() && field(memes,"window")==null,"no meme window before start");
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
        for(int code:new int[]{GLFW.GLFW_KEY_F6,GLFW.GLFW_KEY_F7,GLFW.GLFW_KEY_F8,GLFW.GLFW_KEY_F9,GLFW.GLFW_KEY_F10})
            key(context,code);
        context.waitTicks(20);
        context.runOnClient(client -> {
            requireCold();
            require(chatCount(client,"message.vabpnn.not_started")>=5,"all pipeline hotkeys require start first");
        });
        command(context,"predict");context.runOnClient(client -> { requireCold();client.setScreen(null); });
        command(context,"start predict",client ->
                require(!CaptureController.pipelineAllocated(),"command defers GL/CUDA allocation to a presented frame"));
        context.runOnClient(client -> {
            require(CaptureController.isStarted() && CaptureController.inputState().capturing(),
                    "real client start command opens the capture gate");
            require(CaptureController.inputState().predictionOnly() && CaptureController.inputState().learningFrozen()
                    && !CaptureController.inputState().modelControl(),"explicit cold start freezes prediction from the first frame");
            require(chatCount(client,"message.vabpnn.predict_started")>0,"start is acknowledged in local chat");
            client.setScreen(null);
        });
        context.waitFor(client -> CaptureController.pipelineAllocated()
                && CaptureController.feedbackState(true).phase()==NativeCudaBridge.READY
                && CaptureController.feedbackState(false).processed()>0,600);
        context.waitFor(client -> CaptureController.predictionState(System.nanoTime()).valid(),200);
        context.runOnClient(client -> {
            var state=CaptureController.feedbackState(true);
            require(state.input().predictionOnly() && !state.input().allowed() && state.input().appliedCommands()==0,
                    "cold prediction-only never injects actions");
            require(state.examples()==0 && state.pixelSteps()==0 && state.decisionSteps()==0
                    && state.jepaSteps()==0 && state.motionSteps()==0 && CaptureController.positionState().positionSteps()==0
                    && CaptureController.temporalState().minuteSteps()==0 && CaptureController.temporalState().bayesianMotionSteps()==0,
                    "no learner has updated in a prediction-only cold start");
        });
        command(context,"predict off");context.runOnClient(client -> {
            require(!CaptureController.inputState().learningFrozen(),"explicit predict off resumes human demonstration");
            client.setScreen(null);
        });
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: /vabpnn start predict freezes every learner from the first GPU frame, no model input; predict before start stays cold.");
        command(context,"memes window on");context.runOnClient(client -> client.setScreen(null));
        context.waitFor(client -> CaptureController.memeWindowVisible() && CaptureController.memeWindowFrame()>0,200);
        context.runOnClient(client -> require(!CaptureController.feedbackState(false).scopeVisible(),"F6 window is independent of F7"));
        command(context,"memes window off");context.runOnClient(client -> {
            require(!CaptureController.memeWindowVisible() && !(boolean)staticField(CaptureController.class,"memeScopeSampling"),
                    "window command hides viewer and immediately disables telemetry");client.setScreen(null);
        });
        key(context,GLFW.GLFW_KEY_F6);
        context.waitFor(client -> CaptureController.memeWindowFrame()>0,200);
        key(context,GLFW.GLFW_KEY_F6);
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: separate 5 Hz meme window via real F6 and /vabpnn memes window on/off; F7 remains independent.");
        Object pipeline=context.computeOnClient(client -> staticField(CaptureController.class,"bridge"));
        command(context,"start");
        context.runOnClient(client -> {
            require(staticField(CaptureController.class,"bridge")==pipeline,"repeated start preserves the pipeline");
            require(chatCount(client,"message.vabpnn.already_started")>0,"repeated start is reported");
            client.setScreen(null);
        });
        command(context,"status");
        context.runOnClient(client -> {
            for(String section:List.of("mode","actions","gpu","learning","time"))
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
                if(blocks.frame()==0 || (mode.equals("off") && blocks.enabled()!=0)) {
                    var state=CaptureController.feedbackState(true);
                    cn.vabpnn.mod.VabpnnMod.LOGGER.debug("Block wait: mode={}, frame={}, enabled={}, phase={}, processed={}, capturing={}",mode,blocks.frame(),blocks.enabled(),state.phase(),state.processed(),state.input().capturing());
                }
                return blocks.frame()>0 && (mode.equals("off") ? blocks.enabled()==0:
                        mode.equals("on") ? blocks.enabled()==1:blocks.mode()==switch(mode) {
                            case "independent" -> 0; case "compete" -> 2; default -> 1; });
            },300);
        }
        for(String enabled:new String[]{"off","on"}) {
            command(context,"memes "+enabled);context.runOnClient(client -> client.setScreen(null));
            context.waitFor(client -> {
                var memes=CaptureController.memeState();
                return memes.frame()>0 && memes.enabled()==enabled.equals("on")
                    && (!enabled.equals("off") || memes.localActive()==0 && memes.importActive()==0);
            },300);
        }
        command(context,"memes");context.runOnClient(client -> client.setScreen(null));
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: /vabpnn memes on/off applies at completed GPU frames and off clears active packets.");
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
    
    static void verifyPredictionOnly(ClientGameTestContext context) {
        // Start from frozen model control with genuine owned holds to check the transition release.
        context.runOnClient(client -> {
            long now=System.nanoTime(),id=CaptureController.inputState().frame()+1;
            CaptureController.applyModelCommand(new long[]{id,now,1|(1<<7)|(1<<13)|(1<<14),1,0,0,0},now);
            require(CaptureController.inputState().heldKeys()!=0 && client.mouseHandler.isLeftPressed(),
                    "fixture starts with genuine model-owned key/button holds");
        });
        command(context,"predict");context.runOnClient(client -> {
            var state=CaptureController.inputState();
            require(state.predictionOnly() && state.learningFrozen() && !state.modelControl() && !state.allowed()
                    && state.frame()==0 && state.heldKeys()==0 && state.heldButtons()==0,
                    "prediction mode immediately releases model ownership and blocks injection");
            require(!client.options.keyUp.isDown() && !client.options.keySprint.isDown()
                    && !client.options.keyShift.isDown() && !client.mouseHandler.isLeftPressed(),
                    "old W/Ctrl/Shift/left press released on transition");
            require(chatCount(client,"message.vabpnn.predict_on")>0,"read-only mode has real chat feedback");
            client.setScreen(null);
        });
        context.waitFor(client -> CaptureController.predictionState(System.nanoTime()).valid(),200);
        command(context,"memes window on");context.runOnClient(client -> client.setScreen(null));
        context.waitFor(client -> CaptureController.memeWindowFrame()>0 && CaptureController.evaluationSampleReady(),200);
        var before=context.computeOnClient(client -> CaptureController.feedbackState(true));
        var blocks=context.computeOnClient(client -> CaptureController.hierarchyState());
        var memes=context.computeOnClient(client -> CaptureController.memeState());
        long coordinate=context.computeOnClient(client -> CaptureController.positionState().positionSteps());
        var time=context.computeOnClient(client -> CaptureController.temporalState());
        long[] connections=context.computeOnClient(client -> CaptureController.connectionState());
        context.waitTicks(30);
        context.runOnClient(client -> {
            var after=CaptureController.feedbackState(true);var b=CaptureController.hierarchyState();var m=CaptureController.memeState();
            long[] c=CaptureController.connectionState();
            require(after.processed()>before.processed() && after.prediction().valid() && after.input().predictionOnly(),
                    "GPU prediction keeps producing fresh proposals");
            require(after.examples()==before.examples() && after.pixelSteps()==before.pixelSteps()
                    && after.decisionSteps()==before.decisionSteps() && after.jepaSteps()==before.jepaSteps()
                    && after.motionSteps()==before.motionSteps() && CaptureController.positionState().positionSteps()==coordinate
                    && CaptureController.temporalState().minuteSteps()==time.minuteSteps()
                    && CaptureController.temporalState().bayesianMotionSteps()==time.bayesianMotionSteps(),
                    "pixel/decision/JEPA/mouse/position learners remain frozen");
            require(b.samples()==blocks.samples() && b.updates()==blocks.updates() && b.rankings()==blocks.rankings()
                    && b.mutations()==blocks.mutations() && m.samples()==memes.samples() && m.updates()==memes.updates()
                    && c[3]>connections[3] && c[7]>connections[7],"controller/meme neural weights frozen while contribution and CSR remain dynamic");
            require(after.input().appliedCommands()==before.input().appliedCommands() && !after.input().allowed(),
                    "none of the real GPU packets were injected");
            long now=System.nanoTime(),id=after.prediction().frame()+1;
            int mask=1|(1<<5)|(1<<6)|(1<<7)|(1<<12)|(1<<13)|(1<<14);
            double x=client.player.getX(),y=client.player.getY(),z=client.player.getZ();
            float yaw=client.player.getYRot(),pitch=client.player.getXRot();int selected=client.player.getInventory().getSelectedSlot();
            CaptureController.applyModelCommand(new long[]{id++,now,mask,1,Float.floatToRawIntBits(25f),Float.floatToRawIntBits(-15f),0},now);
            require(CaptureController.predictionState(now).selectedMask()==mask,"nonempty proposal stays visible to observers");
            require(CaptureController.inputState().frame()==0 && CaptureController.inputState().heldKeys()==0
                    && CaptureController.inputState().heldButtons()==0 && !client.options.keyUp.isDown()
                    && !client.options.keyJump.isDown() && !client.options.keySprint.isDown() && !client.options.keyShift.isDown()
                    && !client.mouseHandler.isLeftPressed(),"nonempty proposal cannot inject movement, jump, modifiers or mouse press");
            CaptureController.applyModelCommand(new long[]{id,now,(1<<5)|(1<<6)|(1<<11),1,Float.floatToRawIntBits(5f),
                    Float.floatToRawIntBits(4f),Float.floatToRawIntBits(1f)},now);
            require(client.player.getX()==x && client.player.getY()==y && client.player.getZ()==z
                    && client.player.getYRot()==yaw && client.player.getXRot()==pitch
                    && client.player.getInventory().getSelectedSlot()==selected,"proposals do not alter player/camera/scroll");
            client.options.keyUp.setDown(true);
            CaptureController.applyModelCommand(new long[]{id+1,now,0,1,0,0,0},now);
            require(client.options.keyUp.isDown(),"prediction-only never cancels the human's W input");
            client.options.keyUp.setDown(false);
        });
        command(context,"predict on");context.runOnClient(client -> {
            require(chatCount(client,"message.vabpnn.predict_already")>0,"repeated predict on is idempotent");
            client.setScreen(null);
        });
        command(context,"status");context.runOnClient(client -> {
            require(chatCount(client,"message.vabpnn.mode.predict")>0 && chatCount(client,"message.vabpnn.status.prediction")>0,
                    "status distinguishes proposals from actual applied input");
            client.setScreen(null);
        });
        key(context,GLFW.GLFW_KEY_F9);
        context.runOnClient(client -> require(CaptureController.inputState().predictionOnly()
                && !CaptureController.inputState().capturing(),"pause keeps prediction mode"));
        command(context,"start");context.runOnClient(client -> {
            require(CaptureController.inputState().predictionOnly(),"plain start resumes the paused prediction mode");
            client.setScreen(null);
        });
        context.waitFor(client -> CaptureController.predictionState(System.nanoTime()).valid(),200);
        command(context,"predict off");context.runOnClient(client -> {
            require(!CaptureController.inputState().learningFrozen(),"prediction off returns to demonstrations");
            client.setScreen(null);
        });
        context.waitFor(client -> CaptureController.feedbackState(true).examples()>before.examples(),200);
        command(context,"predict");context.runOnClient(client -> client.setScreen(null));
        context.waitFor(client -> CaptureController.predictionState(System.nanoTime()).valid(),200);
        key(context,GLFW.GLFW_KEY_F8);
        context.waitFor(client -> CaptureController.inputState().modelControl() && !CaptureController.inputState().predictionOnly()
                && CaptureController.inputState().allowed() && CaptureController.inputState().frame()>0,200);
        command(context,"memes window off");context.runOnClient(client -> client.setScreen(null));
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: actual read-only GPU prediction, neural weights frozen and salience/CSR dynamic, owned input released, nonempty key/mouse/camera/scroll proposals never injected, human input preserved, pause/resume/demo/F8 transitions.");
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
            require(!CaptureController.memeWindowVisible() && !(boolean)staticField(CaptureController.class,"memeScopeSampling"),
                    "stop hides the separate viewer and stops its sampling");
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
            require(!CaptureController.memeWindowVisible(),"disconnect hides meme window");
        });
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: leaving a world restores standby and requires /vabpnn start on the next connection.");
    }
}

