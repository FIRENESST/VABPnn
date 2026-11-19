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

/** Checks local chat-only feedback, absence of mod overlays and repeated asynchronous saves. */
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
    private static void requireNoModOverlay(Minecraft client) {
        require((int)field(client.gui,"overlayMessageTime")<=0
                || !((Component)field(client.gui,"overlayMessageString")).getString().startsWith("[VABPnn] "),
                "mod feedback stays in chat without action-bar overlays");
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
            case GLFW.GLFW_KEY_F7 -> CaptureController.feedbackState(false).scopeVisible() ? "scope_on":"scope_off";
            case GLFW.GLFW_KEY_F8 -> CaptureController.inputState().modelControl() ? "model_on":"demo_on";
            case GLFW.GLFW_KEY_F9 -> CaptureController.inputState().capturing() ? "continuous_on":"continuous_off";
            default -> null;
        };
        if(key!=null) {
            require(chatCount(client,"message.vabpnn."+key)>0,"hotkey feedback reached actual local chat");
            requireNoModOverlay(client);
        }
    }
    static void verifyContinuousModel(ClientGameTestContext context) {
        context.waitFor(client -> {
            var state=CaptureController.feedbackState(true);
            return state.decisionFrame()>0 && state.input().frame()>0 && state.input().allowed()
                    && (state.eligibleMask() & ~ActionLayout.KNOWN_MASK)==0;
        },200);
        var before=context.computeOnClient(client -> CaptureController.feedbackState(true));
        var timeBefore=context.computeOnClient(client -> CaptureController.temporalState());
        context.waitTicks(20);
        context.runOnClient(client -> {
            var after=CaptureController.feedbackState(true);
            require(after.input().modelControl() && after.input().appliedCommands()>before.input().appliedCommands()
                    && after.decisionFrame()>before.decisionFrame() && (after.eligibleMask() & ~ActionLayout.KNOWN_MASK)==0,
                    "F8 continuously consumes fresh GPU packets while the learned confidence gate remains active");
            require(after.examples()==before.examples() && after.decisionSteps()==before.decisionSteps()
                    && CaptureController.temporalState().minuteSteps()==timeBefore.minuteSteps()
                    && CaptureController.temporalState().bayesianMotionSteps()==timeBefore.bayesianMotionSteps(),
                    "Continuous action packets do not unfreeze neural learning");
            if(after.requestedK()==0) require(after.selectedCount()==0,"Learned rest remains a valid zero-action decision");
            else if(after.eligibleMask()==0) require(after.selectedCount()==0,"An empty threshold-filtered set suppresses every action even when k is positive");
            require(after.selectedCount()<=Integer.bitCount(after.eligibleMask()),"Only threshold-eligible objects can be selected");
        });
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: F8 continuously predicts and applies real GPU packets with a learned confidence gate; neural weights stay frozen.");
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
        context.runOnClient(client -> {
            requireCold();
            var tree=InGameFeedback.commandTree().build();
            var names=tree.getChildren().stream().map(node->node.getName()).collect(java.util.stream.Collectors.toSet());
            require(names.equals(java.util.Set.of("start","stop","selftrain","explore")),"exact public command whitelist");
            require(tree.getChild("start").getChildren().isEmpty(),"start has no removed prediction alias");
            for(String branch:List.of("selftrain","explore"))require(tree.getChild(branch).getChildren().stream().map(node->node.getName()).collect(java.util.stream.Collectors.toSet()).equals(java.util.Set.of("on","off")),"only on/off arguments remain");
            long mappings=java.util.Arrays.stream(CaptureController.class.getDeclaredFields()).filter(f->f.getType()==net.minecraft.client.KeyMapping.class).count();
            require(mappings==3,"only three VABPnn key bindings remain");
        });
        context.waitFor(client -> chatCount(client,"message.vabpnn.welcome")>0,120);
        for(int code:new int[]{GLFW.GLFW_KEY_F7,GLFW.GLFW_KEY_F8,GLFW.GLFW_KEY_F9})key(context,code);
        context.runOnClient(client -> {
            requireCold();long messages=chatCount(client,"message.vabpnn.not_started");
            require(messages>=3,"retained keys require explicit start");
            CaptureController.onKey(client.getWindow().handle(),GLFW.GLFW_PRESS,new KeyEvent(GLFW.GLFW_KEY_F6,0,0));
            CaptureController.onKey(client.getWindow().handle(),GLFW.GLFW_PRESS,new KeyEvent(GLFW.GLFW_KEY_F10,0,0));
            require(chatCount(client,"message.vabpnn.not_started")==messages,"F6/F10 are no longer intercepted");
        });
        command(context,"start",client -> require(!CaptureController.pipelineAllocated(),"start defers CUDA allocation"));
        context.runOnClient(client->{require(CaptureController.isStarted()&&CaptureController.inputState().capturing(),"public start runs the pipeline");client.setScreen(null);});
        context.waitFor(client->CaptureController.pipelineAllocated()&&CaptureController.feedbackState(true).processed()>8&&CaptureController.feedbackState(true).examples()>0,600);
        context.waitFor(client->CaptureController.hierarchyState().enabled()==1&&CaptureController.memeState().enabled()&&memoryStatus()[10]==1,200);
        context.runOnClient(client->{
            var state=CaptureController.inputState();var trainer=(SelfTrainingController)staticField(CaptureController.class,"selfTraining");
            require(!state.modelControl()&&!state.predictionOnly()&&!state.learningFrozen()&&!state.allowed(),"start defaults to human learning without model action injection");
            require(!trainer.armed()&&trainer.exploration()==0&&!trainer.bootstrap(),"reward learning and threshold-exempt exploration default off");
            require(!CaptureController.feedbackState(false).scopeVisible()&&!CaptureController.memeWindowVisible(),"both monitor windows default hidden after start");
        });
        key(context,GLFW.GLFW_KEY_F7);
        context.waitFor(client->CaptureController.feedbackState(false).scopeVisible()&&CaptureController.memeWindowVisible()&&CaptureController.memeWindowFrame()>0,200);
        key(context,GLFW.GLFW_KEY_F7);
        context.runOnClient(client->require(!CaptureController.feedbackState(false).scopeVisible()&&!CaptureController.memeWindowVisible()&&!(boolean)staticField(CaptureController.class,"scopeSampling")&&!(boolean)staticField(CaptureController.class,"memeScopeSampling"),"F7 closes both viewers and GPU sampling"));
        key(context,GLFW.GLFW_KEY_F9);context.runOnClient(client->require(!CaptureController.inputState().capturing(),"F9 pauses"));
        key(context,GLFW.GLFW_KEY_F9);context.waitFor(client->CaptureController.inputState().capturing(),80);
        Object pipeline=context.computeOnClient(client->staticField(CaptureController.class,"bridge"));
        command(context,"start");context.runOnClient(client->{require(staticField(CaptureController.class,"bridge")==pipeline,"repeated start retains model allocation");client.setScreen(null);});
        command(context,"");context.runOnClient(client->{require(chatCount(client,"message.vabpnn.commands")>0,"bare root shows the short control list");client.setScreen(null);});
        context.runOnClient(client->{requireNoModOverlay(client);client.options.keyUp.setDown(true);});
        int count=context.computeOnClient(client->messages(client).size());context.waitTicks(30);
        context.runOnClient(client->{
            client.options.keyUp.setDown(false);
            requireNoModOverlay(client);
            require(messages(client).size()==count,"continuous state never floods chat");
        });
        // Save worker remains callable for autosave/shutdown; F10 and /save are absent.
        for(int attempt=0;attempt<2;attempt++){
            long before=context.computeOnClient(client->chatCount(client,"message.vabpnn.save_finished"));
            long request=context.computeOnClient(client->{InGameFeedback.requestSave();long serial=(long)staticField(InGameFeedback.class,"pendingSave");InGameFeedback.requestSave();require(serial>0&&chatCount(client,"message.vabpnn.save_pending")>0,"worker save coalesces duplicate requests");return serial;});
            context.waitFor(client->CaptureController.feedbackState(true).savedRequest()>=request&&CaptureController.feedbackState(false).checkpoint()==3,1200);
            context.waitFor(client->chatCount(client,"message.vabpnn.save_finished")>before,120);
        }
        // Simulate disabled legacy module settings, then verify a fresh public start restores defaults.
        context.runOnClient(client->{CaptureController.configureBlocks(0,-1);CaptureController.configureMemes(0);CaptureController.configureLongMemory(false);});
        context.waitFor(client->CaptureController.hierarchyState().enabled()==0&&!CaptureController.memeState().enabled()&&memoryStatus()[10]==0,200);
        command(context,"stop");command(context,"start");context.runOnClient(client->client.setScreen(null));
        context.waitFor(client->CaptureController.hierarchyState().enabled()==1&&CaptureController.memeState().enabled()&&memoryStatus()[10]==1,200);
        context.runOnClient(client->{var trainer=(SelfTrainingController)staticField(CaptureController.class,"selfTraining");require(!trainer.armed()&&trainer.exploration()==0&&!CaptureController.feedbackState(false).scopeVisible()&&!CaptureController.memeWindowVisible(),"fresh start preserves all opt-in defaults");});
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: exact start/stop/selftrain/explore command whitelist, only F7/F8/F9, cold standby, all modules enabled despite old disabled flags, human learning on, control/reward/exploration/windows off, F7 paired viewers, F9, chat-only feedback without HUD overlays and automatic-save worker preserved.");
    }
    private static long[] memoryStatus(){var bridge=(NativeCudaBridge)staticField(CaptureController.class,"bridge");long[] s=new long[24];bridge.readLongMemory(s);return s;}
    // Exercise the retained controller state directly; no removed public command is reintroduced.
    private static void prediction(ClientGameTestContext c,boolean on){c.runOnClient(mc->{String key=CaptureController.setPredictionOnly(on);InGameFeedback.hotkey(Component.translatable("message.vabpnn."+key));mc.setScreen(null);});}
    private static void monitors(ClientGameTestContext c,boolean visible){if(c.computeOnClient(mc->CaptureController.feedbackState(false).scopeVisible())!=visible)key(c,GLFW.GLFW_KEY_F7);}

    static void verifyPredictionOnly(ClientGameTestContext context) {
        // Start from frozen model control with genuine owned holds to check the transition release.
        context.runOnClient(client -> {
            long now=System.nanoTime(),id=CaptureController.inputState().frame()+1;
            CaptureController.applyModelCommand(new long[]{id,now,1|(1<<7)|(1<<13)|(1<<14),1,0,0,0},now);
            require(CaptureController.inputState().heldKeys()!=0 && client.mouseHandler.isLeftPressed(),
                    "fixture starts with genuine model-owned key/button holds");
        });
        prediction(context,true);context.runOnClient(client -> {
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
        monitors(context,true);context.runOnClient(client -> client.setScreen(null));
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
        prediction(context,true);context.runOnClient(client -> {
            require(chatCount(client,"message.vabpnn.predict_already")>0,"repeated predict on is idempotent");
            client.setScreen(null);
        });
        command(context,"");context.runOnClient(client -> {
            require(CaptureController.inputState().predictionOnly()&&CaptureController.predictionState(System.nanoTime()).valid(),
                    "root overview preserves prediction mode and live proposals");
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
        prediction(context,false);context.runOnClient(client -> {
            require(!CaptureController.inputState().learningFrozen(),"prediction off returns to demonstrations");
            client.setScreen(null);
        });
        context.waitFor(client -> CaptureController.feedbackState(true).examples()>before.examples(),200);
        prediction(context,true);context.runOnClient(client -> client.setScreen(null));
        context.waitFor(client -> CaptureController.predictionState(System.nanoTime()).valid(),200);
        key(context,GLFW.GLFW_KEY_F8);
        context.waitFor(client -> CaptureController.inputState().modelControl() && !CaptureController.inputState().predictionOnly()
                && CaptureController.inputState().allowed() && CaptureController.inputState().frame()>0,200);
        monitors(context,false);context.runOnClient(client -> client.setScreen(null));
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

