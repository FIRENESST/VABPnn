package cn.vabpnn.mod.client.capture;

import cn.vabpnn.mod.VabpnnMod;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.file.*;
import java.time.Instant;
import java.util.concurrent.*;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.server.MinecraftServer;
import net.minecraft.sounds.SoundSource;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.gamerules.GameRules;
import net.minecraft.world.level.storage.LevelResource;
import org.lwjgl.glfw.GLFW;

/** Opt-in local runner; prediction, input receipts and reward updates use the unchanged production path. */
public final class BackgroundTraining implements ClientModInitializer {
    private static Boolean enabled;
    private Path directory;
    private final ExecutorService writer=Executors.newSingleThreadExecutor(r -> {
        Thread t=new Thread(r,"vabpnn-background-status"); t.setDaemon(true); return t;
    });
    private MinecraftServer server;
    private LocalPlayer armedPlayer;
    private long nextStatus, nextCheckpoint, initialNanos=System.nanoTime();
    private int stableTicks;
    private boolean stopping;

    public static boolean enabled() {
        if (enabled!=null) return enabled;
        if (!Boolean.getBoolean("vabpnn.background.training")) return enabled=false;
        Path game=FabricLoader.getInstance().getGameDir().toAbsolutePath().normalize();
        Path expected=Path.of(System.getProperty("vabpnn.background.directory", "")).toAbsolutePath().normalize();
        if (!game.equals(expected) || !Files.isRegularFile(game.resolve(".vabpnn-background")))
            throw new IllegalStateException("Background trainer requires its own marked game directory");
        return enabled=true;
    }
    @Override public void onInitializeClient() {
        if (!enabled()) return;
        directory=FabricLoader.getInstance().getGameDir();
        ClientTickEvents.END_CLIENT_TICK.register(this::tick);
        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
            publish(client,"stopped"); writer.shutdown();
            try { writer.awaitTermination(3,TimeUnit.SECONDS); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        });
        VabpnnMod.LOGGER.info("Background trainer enabled: invisible GPU window; in-process input; exploration OFF; directory {}",directory);
    }
    private void tick(Minecraft client) {
        long now=System.nanoTime();
        if (stopping) return;
        try {
            if (GLFW.glfwGetWindowAttrib(client.getWindow().handle(),GLFW.GLFW_VISIBLE)!=GLFW.GLFW_FALSE)
                throw new IllegalStateException("Background GPU window unexpectedly visible");
            client.options.pauseOnLostFocus=false;
            client.options.framerateLimit().set(30);
            client.options.renderDistance().set(8);
            client.options.simulationDistance().set(6);
            client.options.getSoundSourceOptionInstance(SoundSource.MASTER).set(0.0);
            if (now>=nextStatus) {
                nextStatus=now+5_000_000_000L;
                if (Files.exists(directory.resolve("stop.request"))) {
                    stopping=true; CaptureController.setSelfTraining(false); publish(client,"stopping"); client.stop(); return;
                }
                publish(client,CaptureController.selfTrainingArmed() ? "self_training":"loading");
            }
            if (client.player==null || client.level==null || client.getSingleplayerServer()==null) {
                stableTicks=0;
                if (server!=null || now-initialNanos>180_000_000_000L)
                    throw new IllegalStateException("Background training world is unavailable");
                return;
            }
            if (server!=client.getSingleplayerServer()) {
                server=client.getSingleplayerServer();
                Path actual=server.getWorldPath(LevelResource.ROOT).toAbsolutePath().normalize();
                Path expected=directory.resolve("saves/training-world").toAbsolutePath().normalize();
                if (!actual.equals(expected)) throw new IllegalStateException("Refusing to train outside isolated save");
                server.execute(() -> {
                    server.getGameRules().set(GameRules.IMMEDIATE_RESPAWN,true,server);
                    for (var player:server.getPlayerList().getPlayers())
                        if (player.gameMode.getGameModeForPlayer()!=GameType.SURVIVAL) player.setGameMode(GameType.SURVIVAL);
                });
            }
            if (!client.player.isAlive() || client.getOverlay()!=null || client.isPaused()) { stableTicks=0; return; }
            if (++stableTicks<40) return;
            if (!CaptureController.isStarted()) CaptureController.startPipeline(true);
            if (CaptureController.pipelineAllocated()) {
                var health=CaptureController.feedbackState(false);
                if (health.checkpoint()<0) throw new IllegalStateException(health.checkpointMessage());
                if (health.phase()<0) throw new IllegalStateException("GPU capture failed");
            }
            if (client.player!=armedPlayer && !CaptureController.selfTrainingArmed()) {
                String result=CaptureController.setSelfTraining(true);
                if (!result.equals("selftrain_on")) throw new IllegalStateException(result);
                armedPlayer=client.player;
                VabpnnMod.LOGGER.info("Background self-training armed for current life; all neural branches train; normal confidence thresholds retained; sub-threshold exploration OFF");
            }
            if (!CaptureController.selfTrainingArmed() && client.player==armedPlayer)
                throw new IllegalStateException("Self-training disarmed: "+CaptureController.selfTrainingStatus());
            if (now>=nextCheckpoint && CaptureController.feedbackState(false).checkpoint()>0) {
                nextCheckpoint=now+120_000_000_000L;
                CaptureController.requestCheckpoint();
            }
        } catch (RuntimeException | LinkageError error) {
            stopping=true;
            if (CaptureController.isStarted()) CaptureController.setSelfTraining(false);
            VabpnnMod.LOGGER.error("Background trainer stopped",error);
            publish(client,"error: "+error.getMessage());
            client.stop();
        }
    }
    private void publish(Minecraft client,String phase) {
        JsonObject state=new JsonObject();
        state.addProperty("pid",ProcessHandle.current().pid());
        state.addProperty("observed_at",Instant.now().toString());
        state.addProperty("phase",phase);
        state.addProperty("window_visible",GLFW.glfwGetWindowAttrib(client.getWindow().handle(),GLFW.GLFW_VISIBLE)!=0);
        state.addProperty("width",client.getWindow().getWidth()); state.addProperty("height",client.getWindow().getHeight());
        state.addProperty("exploration",false);
        state.addProperty("armed",CaptureController.selfTrainingArmed());
        state.addProperty("reward_updates",CaptureController.rewardUpdates());
        state.addProperty("training_status",CaptureController.selfTrainingStatus());
        var input=CaptureController.inputState();
        state.addProperty("input_allowed",input.allowed()); state.addProperty("actions_delivered",input.appliedCommands());
        state.addProperty("selected_mask",input.selectedMask());
        if (CaptureController.pipelineAllocated()) {
            var stats=CaptureController.feedbackState(true);
            state.addProperty("processed_frames",stats.processed());
            state.addProperty("checkpoint_status",stats.checkpoint());
            state.addProperty("checkpoint_message",stats.checkpointMessage());
            state.addProperty("neural_weights_frozen",input.learningFrozen());
            state.addProperty("learning_scope",input.fullLearning()?"all":"frozen_or_policy_heads");
            state.addProperty("jepa_selfsupervised_steps",stats.jepaSteps());
            state.addProperty("demonstration_examples",stats.examples());
            JsonObject branches=new JsonObject();String[] names={"updates","seconds","pixels","minutes","position","bayes_motion","jepa_motion","memory","blocks","memes","wheel","cached","dropped","bytes"};
            long[] values=CaptureController.fullLearningCounters();for(int i=0;i<names.length;i++)branches.addProperty(names[i],values[i]);state.add("branch_updates",branches);
        }
        if (client.player!=null) {
            state.addProperty("alive",client.player.isAlive());
            state.addProperty("x",client.player.getX()); state.addProperty("y",client.player.getY()); state.addProperty("z",client.player.getZ());
        }
        String text=state.toString();
        writer.execute(() -> {
            Path temporary=directory.resolve("status.json.tmp"), target=directory.resolve("status.json");
            try {
                Files.writeString(temporary,text);
                try { Files.move(temporary,target,StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE); }
                catch (AtomicMoveNotSupportedException ignored) { Files.move(temporary,target,StandardCopyOption.REPLACE_EXISTING); }
            } catch (IOException error) { VabpnnMod.LOGGER.warn("Cannot write background training status",error); }
        });
    }
}
