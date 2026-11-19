package cn.vabpnn.mod.client.capture;

import cn.vabpnn.mod.VabpnnMod;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.command.v2.FabricClientCommandSource;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.minecraft.ChatFormatting;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

/** Local chat feedback polled at 2 Hz. Reads only the worker's published CPU counters. */
final class InGameFeedback {
    private static boolean welcomePending;
    private static long nextFeedback,pendingSave;
    private static int lastCheckpoint;
    private static boolean wasModel,idleNotice;
    private static long idleNoticeAfter;
    private static final String[] ACTIONS={"W","A","S","D","E","x","y","left_down","right_down",
            "left_up","right_up","wheel","space","ctrl","shift"};
    private InGameFeedback() {}

    static void initialize() {
        ClientPlayConnectionEvents.JOIN.register((handler,sender,client) -> {
            welcomePending=true; nextFeedback=0; lastCheckpoint=0;
        });
        ClientPlayConnectionEvents.DISCONNECT.register((handler,client) -> {
            CaptureController.stopPipeline();
            welcomePending=false; nextFeedback=0;
            wasModel=false; idleNotice=false;
        });
        ClientCommandRegistrationCallback.EVENT.register((dispatcher,buildContext) -> dispatcher.register(commandTree()));
    }
    // This is the entire public surface. Internal diagnostic and training APIs stay available.
    static LiteralArgumentBuilder<FabricClientCommandSource> commandTree() {
        return ClientCommands.literal("vabpnn")
                .executes(context -> controls(context.getSource()))
                .then(ClientCommands.literal("start").executes(context -> {
                    post(text(CaptureController.startPipeline()),false);return 1;
                }))
                .then(ClientCommands.literal("stop").executes(context -> {
                    post(text(CaptureController.stopPipeline() ? "stopped":"already_stopped"),false);return 1;
                }))
                .then(ClientCommands.literal("selftrain")
                        .executes(context -> {context.getSource().sendFeedback(labelled(Component.literal(CaptureController.selfTrainingStatus()),false));return 1;})
                        .then(ClientCommands.literal("on").executes(context -> {post(text(CaptureController.setSelfTraining(true)),false);return 1;}))
                        .then(ClientCommands.literal("off").executes(context -> {post(text(CaptureController.setSelfTraining(false)),false);return 1;})))
                .then(ClientCommands.literal("explore")
                        .executes(context -> {context.getSource().sendFeedback(labelled(Component.literal(CaptureController.explorationStatus()),false));return 1;})
                        .then(ClientCommands.literal("on").executes(context -> explore(.2f,true)))
                        .then(ClientCommands.literal("off").executes(context -> explore(0,false))));
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
    private static int explore(float probability,boolean bootstrap) {
        post(Component.literal(CaptureController.setExploration(probability,bootstrap)),false);return 1;
    }
    private static int controls(FabricClientCommandSource source) {
        source.sendFeedback(labelled(text("commands"),false));
        source.sendFeedback(labelled(hotkeys(),false));
        source.sendFeedback(labelled(Component.literal(CaptureController.selfTrainingStatus()),false));
        return 1;
    }
    private static Component hotkeys() {return CaptureController.feedbackHotkeys();}
    private static Component names(int mask,boolean allActions) {
        var output=Component.empty();
        boolean first=true;
        for(int k=0;k<ActionLayout.ACTIONS;k++) {
            if((mask&(1<<k))==0 || (!allActions && (ActionLayout.KEY_MASK & (1<<k))==0)) continue;
            if(!first) output.append(" + ");
            output.append(k<5 ? Component.literal(ACTIONS[k]):text("action."+ACTIONS[k]));
            first=false;
        }
        return first ? text("none"):output;
    }
    static void tick(Minecraft client,long now) {
        if(client.player==null || client.level==null || now<nextFeedback) return;
        nextFeedback=now+500_000_000L;
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

    }
}


