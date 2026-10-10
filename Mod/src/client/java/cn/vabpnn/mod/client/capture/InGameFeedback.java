package cn.vabpnn.mod.client.capture;

import cn.vabpnn.mod.VabpnnMod;
import java.util.Locale;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.minecraft.ChatFormatting;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

/** Local chat feedback and a 2 Hz action bar. Reads only the worker's published CPU counters. */
final class InGameFeedback {
    private static boolean welcomePending,hudEnabled=true;
    private static long nextHud,flashUntil,pendingSave;
    private static int lastCheckpoint;
    private static boolean wasModel,idleNotice;
    private static long idleNoticeAfter;
    private static final String[] ACTIONS={"W","A","S","D","E","x","y","left_down","right_down",
            "left_up","right_up","wheel","space"};
    private InGameFeedback() {}

    static void initialize() {
        ClientPlayConnectionEvents.JOIN.register((handler,sender,client) -> {
            welcomePending=true; nextHud=0; lastCheckpoint=0;
        });
        ClientPlayConnectionEvents.DISCONNECT.register((handler,client) -> {
            CaptureController.stopPipeline();
            welcomePending=false; nextHud=0; flashUntil=0;
            wasModel=false; idleNotice=false;
        });
        ClientCommandRegistrationCallback.EVENT.register((dispatcher,buildContext) ->
                dispatcher.register(ClientCommands.literal("vabpnn")
                        .executes(context -> status(context.getSource()))
                        .then(ClientCommands.literal("status").executes(context -> status(context.getSource())))
                        .then(ClientCommands.literal("help").executes(context -> help(context.getSource())))
                        .then(ClientCommands.literal("start").executes(context -> {
                            post(text(CaptureController.startPipeline()),false); return 1;
                        }))
                        .then(ClientCommands.literal("stop").executes(context -> {
                            post(text(CaptureController.stopPipeline() ? "stopped":"already_stopped"),false);
                            return 1;
                        }))
                        .then(ClientCommands.literal("save").executes(context -> {
                            requestSave(); return 1;
                        }))
                        .then(ClientCommands.literal("blocks")
                                .executes(context -> blockStatus(context.getSource()))
                                .then(ClientCommands.literal("on").executes(context -> setBlocks(1,-1)))
                                .then(ClientCommands.literal("off").executes(context -> setBlocks(0,-1)))
                                .then(ClientCommands.literal("independent").executes(context -> setBlocks(-1,0)))
                                .then(ClientCommands.literal("cooperate").executes(context -> setBlocks(-1,1)))
                                .then(ClientCommands.literal("compete").executes(context -> setBlocks(-1,2))))
                        .then(ClientCommands.literal("hud")
                                .then(ClientCommands.literal("on").executes(context -> setHud(true)))
                                .then(ClientCommands.literal("off").executes(context -> setHud(false))))));
    }
    private static Component text(String key,Object... args) {
        return Component.translatable("message.vabpnn."+key,args);
    }
    private static Component labelled(Component message,boolean error) {
        return Component.literal("[VABPnn] ").withStyle(error ? ChatFormatting.RED:ChatFormatting.AQUA)
                .append(message.copy().withStyle(error ? ChatFormatting.RED:ChatFormatting.WHITE));
    }
    private static void post(Component message,boolean error) {
        Minecraft client=Minecraft.getInstance();
        Component output=labelled(message,error);
        VabpnnMod.LOGGER.info("{}",output.getString());
        if(client.player!=null) {
            client.gui.getChat().addClientSystemMessage(output);
            client.gui.setOverlayMessage(output,false);
            flashUntil=System.nanoTime()+3_000_000_000L;
        }
    }
    static void hotkey(Component message) {
        post(text("hotkey",message,CaptureController.feedbackState(false).gate()),false);
    }
    static void requestSave() {
        if(pendingSave!=0) { post(text("save_pending"),false); return; }
        long request=CaptureController.requestCheckpoint();
        if(request==0) { post(text("save_unavailable"),true); return; }
        pendingSave=request;
        post(text("save_requested"),false);
    }
    static void failure(Throwable error) {
        pendingSave=0;
        String detail=error.getMessage();
        if(detail==null || detail.isBlank()) detail=error.getClass().getSimpleName();
        if(detail.length()>160) detail=detail.substring(0,160)+"…";
        post(text("pipeline_failed",detail),true);
    }
    private static int setBlocks(int enabled,int mode) {
        CaptureController.configureBlocks(enabled,mode);
        String message=enabled==0 ? "高层控制关闭请求已排队":enabled==1 ? "高层控制开启请求已排队":
                "高层模式请求已排队："+switch(mode) { case 0 -> "独立"; case 2 -> "竞争"; default -> "合作"; };
        post(Component.literal(message+"；/vabpnn blocks 查看完成帧状态"),false);
        return 1;
    }
    private static int blockStatus(FabricClientCommandSource source) {
        var s=CaptureController.hierarchyState();
        source.sendFeedback(labelled(Component.literal(String.format(Locale.ROOT,
                "高层 %s / %s，半径 %d 块，完成帧 %d；真实示范 %d，策略更新 %d，排名 %d，累计变异 %d，最近处罚 %d",
                s.enabled()==1 ? "开启":"关闭",s.modeName(),s.radius(),s.frame(),s.samples(),s.updates(),
                s.rankings(),s.mutations(),s.punished())),false));
        return 1;
    }
    private static int setHud(boolean enabled) {
        hudEnabled=enabled; nextHud=0;
        post(text(enabled ? "hud_on":"hud_off"),false);
        return 1;
    }
    private static int help(FabricClientCommandSource source) {
        source.sendFeedback(labelled(text("help",hotkeys()),false));
        source.sendFeedback(labelled(text("commands"),false));
        source.sendFeedback(labelled(Component.literal("/vabpnn blocks [on|off|independent|cooperate|compete]"),false));
        return 1;
    }
    private static Component hotkeys() {
        return CaptureController.feedbackHotkeys();
    }
    private static int status(FabricClientCommandSource source) {
        var state=CaptureController.feedbackState(true);
        var input=state.input();
        source.sendFeedback(labelled(text("status.mode",mode(input.modelControl()),state.gate(),
                text(state.scopeVisible() ? "shown":"hidden")),false));
        source.sendFeedback(labelled(text("status.actions",input.frame(),
                names(input.selectedMask(),true),held(source.getClient(),input)),false));
        source.sendFeedback(labelled(text("status.decision",state.decisionFrame(),state.requestedK(),
                names(state.eligibleMask(),true),state.selectedCount(),names(state.lastHumanLabel(),true)),false));
        var position=CaptureController.positionState();
        source.sendFeedback(labelled(text("status.position",
                String.format(Locale.ROOT,"%.3f",position.x()),String.format(Locale.ROOT,"%.3f",position.y()),
                String.format(Locale.ROOT,"%.3f",position.z()),
                text(position.valid() ? "position.connected":"position.unavailable"),position.positionSteps()),false));
        var human=state.human();
        source.sendFeedback(labelled(text("status.mouse",human.leftPress(),human.rightPress(),
                human.leftRelease(),human.rightRelease()),false));
        source.sendFeedback(labelled(text("status.gpu",state.processed(),state.dropped(),
                millis(state.gpuUs()),millis(state.maximumGpuUs()),state.overBudget()),false));
        source.sendFeedback(labelled(text("status.learning",state.examples(),state.pixelSteps(),
                state.decisionSteps(),state.jepaSteps(),state.motionSteps(),checkpoint(state)),false));
        blockStatus(source);
        return 1;
    }
    private static Component checkpoint(CaptureController.FeedbackState state) {
        if(!CaptureController.pipelineAllocated()) return text("checkpoint.not_started");
        if(pendingSave!=0 && state.savedRequest()<pendingSave) return text("checkpoint.pending");
        return text("checkpoint."+switch(state.checkpoint()) {
            case 1 -> "new";
            case 2 -> "restored";
            case 3 -> "saved";
            case -1 -> "failed";
            default -> "starting";
        });
    }
    private static Component mode(boolean model) {
        return text(!CaptureController.isStarted() ? "mode.standby":model ? "mode.model":"mode.demo");
    }
    private static String millis(long micros) { return micros<=0 ? "--":String.format(Locale.ROOT,"%.2f",micros/1000.0); }
    private static Component names(int mask,boolean allActions) {
        var output=Component.empty();
        boolean first=true;
        for(int k=0;k<13;k++) {
            if((mask&(1<<k))==0 || (!allActions && k>4 && k!=12)) continue;
            if(!first) output.append(" + ");
            output.append(k<5 ? Component.literal(ACTIONS[k]):text("action."+ACTIONS[k]));
            first=false;
        }
        return first ? text("none"):output;
    }
    private static Component held(Minecraft client,CaptureController.InputState input) {
        if(!CaptureController.isStarted()) return text("none");
        int keys=input.heldKeys(),buttons=input.heldButtons();
        if(!input.modelControl()) {
            var human=CaptureController.humanInputState();
            keys=0;
            if(client.options.keyUp.isDown()) keys|=1;
            if(client.options.keyLeft.isDown()) keys|=2;
            if(client.options.keyDown.isDown()) keys|=4;
            if(client.options.keyRight.isDown()) keys|=8;
            if(client.options.keyInventory.isDown()) keys|=16;
            if(client.options.keyJump.isDown()) keys|=1<<12;
            if(client.screen instanceof net.minecraft.client.gui.screens.inventory.AbstractContainerScreen<?>)
                keys=human.heldKeys();
            buttons=human.heldButtons();
        }
        if(keys==0 && buttons==0) return text("none");
        var output=Component.empty();
        if(keys!=0) output.append(names(keys,false));
        if((buttons&1)!=0) {
            if(keys!=0) output.append(" + ");
            output.append(text("action.left"));
        }
        if((buttons&2)!=0) {
            if(keys!=0 || (buttons&1)!=0) output.append(" + ");
            output.append(text("action.right"));
        }
        return output;
    }
    static void tick(Minecraft client,long now) {
        if(client.player==null || client.level==null || now<nextHud) return;
        nextHud=now+500_000_000L;
        if(welcomePending) {
            welcomePending=false;
            post(text("welcome",hotkeys()),false);
        }
        var state=CaptureController.feedbackState(false);
        if(pendingSave!=0 && state.savedRequest()>=pendingSave) {
            pendingSave=0;
            boolean error=state.checkpoint()<0;
            post(error ? text("save_failed",state.checkpointMessage()):text("save_finished"),error);
        } else if(state.checkpoint()!=lastCheckpoint) {
            if(state.checkpoint()<0) post(text("checkpoint_error",state.checkpointMessage()),true);
            else if(state.checkpoint()==1) post(text("model_new"),false);
            else if(state.checkpoint()==2) post(text("model_restored"),false);
        }
        lastCheckpoint=state.checkpoint();
        boolean model=state.input().modelControl();
        if(model!=wasModel) { wasModel=model; idleNotice=false; idleNoticeAfter=now+1_000_000_000L; }
        if(model && state.input().allowed() && state.input().frame()>0 && !idleNotice && now>=idleNoticeAfter
                && state.input().selectedMask()==0 && state.input().heldKeys()==0 && state.input().heldButtons()==0
                && state.decisionFrame()>0) {
            idleNotice=true;
            post(text("model_idle",state.gate(),state.requestedK(),names(state.eligibleMask(),true)),false);
        }
        if(!hudEnabled || now<flashUntil || (client.screen!=null
                && !(client.screen instanceof net.minecraft.client.gui.screens.inventory.AbstractContainerScreen<?>))
                || client.getOverlay()!=null) return;
        client.gui.setOverlayMessage(labelled(text("hud",mode(state.input().modelControl()),state.gate(),
                held(client,state.input()),millis(state.gpuUs())),false),false);
    }
}
