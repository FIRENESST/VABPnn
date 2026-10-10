package cn.vabpnn.mod.client.capture;

import cn.vabpnn.mod.VabpnnMod;
import cn.vabpnn.mod.client.scope.NetworkMonitor;
import cn.vabpnn.mod.client.scope.MouseMotionTracker;
import cn.vabpnn.mod.client.scope.ScopeSnapshot;
import com.mojang.blaze3d.platform.InputConstants;
import com.mojang.blaze3d.systems.RenderSystem;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.gui.screens.inventory.AbstractContainerScreen;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.options.controls.KeyBindsScreen;
import net.minecraft.client.input.KeyEvent;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import org.lwjgl.glfw.GLFW;

import static org.lwjgl.opengl.GL33C.*;

/** Continuous GL -> CUDA device queue -> GPU operator. Does not read or save frames. */
public final class CaptureController {
    private static KeyMapping continuousKey, controlKey, saveKey, scopeKey;
    private static NetworkMonitor monitor;
    private static final MouseMotionTracker mouseMotion=new MouseMotionTracker();
    private static final long[] scopeMeta=new long[ScopeSnapshot.META];
    private static final float[] scopeValues=new float[ScopeSnapshot.FLOATS];
    private static final int[] scopeDiscrete=new int[ScopeSnapshot.INTS];
    private static boolean scopeConfigured,scopeSampling;
    private static int scopeTile=-1;
    private static long nextScopeRead;
    private static final DemonstrationInput demonstration = new DemonstrationInput();
    private static final PlayerPositionSampler playerPosition=new PlayerPositionSampler();
    private static final long[] positionStatus=new long[7];
    private static final long[] blockStatus=new long[14];
    private static int requestedBlocks=-1,requestedBlockMode=-1;
    public static NetworkMonitor.Hierarchy hierarchyState() {
        if(bridge!=null) bridge.readBlocks(blockStatus);
        return NetworkMonitor.Hierarchy.from(blockStatus);
    }
    static void configureBlocks(int enabled,int mode) {
        if(enabled>=0) requestedBlocks=enabled;
        if(mode>=0) requestedBlockMode=mode;
        if(bridge!=null) bridge.configureBlocks(enabled,mode);
    }
    private static final long[] demoData = new long[15], trainingStatus = new long[9];
    private static final double[] demoMotion = new double[3];
    private static final boolean[] physicalKeys = new boolean[6];
    private static final int[] demoKeyCodes = {87,65,83,68,69,32};
    private static boolean modelControl, inputActive;
    private static int syntheticDepth;
    private static int lastCheckpointStatus;
    private static final long[] saveStatus=new long[3], decisionStatus=new long[7];
    private static long nextFeedbackRead;
    private static NativeCudaBridge.Stats feedbackStats;
    private static String checkpointText="";
    private static final long[] gpuTiming = new long[8];
    private static NativeCudaBridge bridge;
    private static final ActionExecutor actions = new ActionExecutor();
    private static final long[] controlCommand = new long[7];
    private static MinecraftInputSink inputSink;
    private static Object actionScene, actionScreen;
    private static boolean actionMouseGrabbed, controlAllowed;
    public record InputState(boolean modelControl,boolean capturing,boolean allowed,long frame,
                             int selectedMask,int heldKeys,int heldButtons,long appliedCommands) {}
    /** Read on the Minecraft thread; consumers may publish this immutable snapshot. */
    public static InputState inputState() {
        return new InputState(modelControl,started && enabled && !failed && !closed,controlAllowed,actions.appliedFrame(),
                actions.appliedMask(),actions.heldKeys(),actions.heldButtons(),actions.appliedCommands());
    }
    record HumanInputState(int heldKeys,int heldButtons,long leftPress,long rightPress,long leftRelease,long rightRelease) {}
    static HumanInputState humanInputState() {
        return new HumanInputState(demonstration.heldKeys(),demonstration.heldButtons(),
                demonstration.buttonEdges(0),demonstration.buttonEdges(1),
                demonstration.buttonEdges(2),demonstration.buttonEdges(3));
    }
    public record PositionState(boolean valid,long sampleNanos,long epoch,double x,double y,double z,
                                long samples,long gpuSampleNanos,long positionSteps) {}
    public static PositionState positionState() {
        if(bridge!=null) bridge.readPosition(positionStatus);
        return new PositionState(started && enabled && playerPosition.nanos()!=0 && positionStatus[3]!=0
                && positionStatus[1]==playerPosition.epoch()
                && System.nanoTime()-playerPosition.nanos()<=200_000_000L,
            playerPosition.nanos(),playerPosition.epoch(),playerPosition.x(),playerPosition.y(),playerPosition.z(),
            playerPosition.samples(),positionStatus[0],positionStatus[2]);
    }
    private static OpenGlCaptureSurface[] surfaces;
    private static int width, height;
    private static int capacity;
    private static boolean started, enabled, closed, failed, resizing;
    private static long renderedFrame, nextLog;
    private static boolean loggedRenderTail;
    private static final int smokeFrames = Integer.getInteger("vabpnn.cuda.smokeFrames", 0);

    private CaptureController() {}

    public static void initialize() {
        inputSink = new MinecraftInputSink(Minecraft.getInstance());
        ClientTickEvents.START_CLIENT_TICK.register(CaptureController::onClientTick);
        InGameFeedback.initialize();
        KeyMapping.Category category = KeyMapping.Category.register(
                Identifier.fromNamespaceAndPath(VabpnnMod.MOD_ID, "capture"));
        continuousKey = KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.vabpnn.continuous", InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_F9, category));
        controlKey = KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.vabpnn.model_control", InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_F8, category));
        saveKey = KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.vabpnn.save_model", InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_F10, category));
        scopeKey=KeyMappingHelper.registerKeyMapping(new KeyMapping(
                "key.vabpnn.network_scope",InputConstants.Type.KEYSYM,GLFW.GLFW_KEY_F7,category));
        monitor=new NetworkMonitor(message -> VabpnnMod.LOGGER.warn(message));
        monitor.status(false,false,false,"待命：/vabpnn start");
        VabpnnMod.LOGGER.info("Network scope is deferred until /vabpnn start; F7 then toggles visibility.");
        capacity = Math.max(3, Math.min(5, Integer.getInteger("vabpnn.cuda.capacity", 4)));
        VabpnnMod.LOGGER.info("Human demonstration mode; F8 toggles model control, F10 saves model/optimizer.");
        VabpnnMod.LOGGER.info("Waiting for /vabpnn start: {} slots configured; no GPU pipeline is allocated before starting.", capacity);
    }

    public static boolean isStarted() { return started; }
    static boolean pipelineAllocated() { return bridge!=null; }

    /** Command callbacks only change gates; GL/CUDA allocation starts on the next presented frame. */
    static String startPipeline() {
        if(closed) return "start_unavailable";
        if(started && enabled && !failed) return "already_started";
        boolean fresh=!started;
        started=true; enabled=true; failed=false;
        if(fresh) modelControl=false;
        resetInputEpoch();
        nextLog=0; nextFeedbackRead=0;
        if(fresh && monitor!=null)
            monitor.show(Boolean.parseBoolean(System.getProperty("vabpnn.scope.enabled","true")));
        refreshMonitor("启动中");
        return fresh ? "started":"resumed";
    }
    /** Keep learned parameters for a quick restart; stop all new frame submissions and owned input. */
    static boolean stopPipeline() {
        if(!started || closed) return false;
        started=false; enabled=false; modelControl=false;
        resetInputEpoch();
        disableScopeSampling();
        if(monitor!=null) monitor.show(false);
        refreshMonitor("待命：/vabpnn start");
        return true;
    }
    private static void disableScopeSampling() {
        if(bridge!=null && scopeSampling) bridge.scopeEnabled(false,scopeTile);
        scopeSampling=false;
    }

    public static void beginSynthetic() { syntheticDepth++; }
    public static void endSynthetic() { syntheticDepth--; }
    public static void onHumanMove(double dx, double dy) {
        if (syntheticDepth == 0 && !modelControl && inputActive) {
            demonstration.move(dx, dy);
            mouseMotion.human(dx, dy);
        }
    }
    static void onModelMove(double dx,double dy) {
        if(modelControl && inputActive && controlAllowed) mouseMotion.executed(dx,dy);
    }
    public static void onHumanButton(int button, int action) {
        if(syntheticDepth!=0 || closed) return;
        if(inputSink!=null) syncInputContext(Minecraft.getInstance());
        boolean record=!modelControl && inputActive;
        demonstration.button(button,action,record);
    }
    public static void onHumanScroll(double steps) {
        if (syntheticDepth == 0 && !modelControl && inputActive) demonstration.wheel(steps);
    }
    public static void onKey(long window, int action, KeyEvent event) {
        Minecraft minecraft = Minecraft.getInstance();
        if (closed || continuousKey == null || window != minecraft.getWindow().handle()) return;
        if (syntheticDepth == 0 && !modelControl) demonstration.key(event.key(), action, System.nanoTime());
        if (action != GLFW.GLFW_PRESS || syntheticDepth != 0 || minecraft.screen instanceof KeyBindsScreen) return;
        Component message;
        if(!started && (scopeKey.matches(event) || controlKey.matches(event)
                || saveKey.matches(event) || continuousKey.matches(event))) {
            InGameFeedback.hotkey(Component.translatable("message.vabpnn.not_started"));
            return;
        }
        if(scopeKey.matches(event)) {
            monitor.show(!monitor.visible());
            message=Component.translatable(monitor.visible() ? "message.vabpnn.scope_on":"message.vabpnn.scope_off");
        } else if (controlKey.matches(event)) {
            modelControl = !modelControl;
            resetInputEpoch();
            message = Component.translatable(modelControl ? "message.vabpnn.model_on" : "message.vabpnn.demo_on");
        } else if (saveKey.matches(event)) {
            InGameFeedback.requestSave();
            return;
        } else if (continuousKey.matches(event)) {
            if (failed) { failed = false; enabled = true; } else enabled = !enabled;
            resetInputEpoch();
            if(!enabled) disableScopeSampling();
            message = Component.translatable(enabled ? "message.vabpnn.continuous_on" : "message.vabpnn.continuous_off");
        } else return;
        InGameFeedback.hotkey(message);
    }

    public static void onGameRenderTail() {
        renderedFrame++;
        if (!loggedRenderTail) {
            loggedRenderTail = true;
            VabpnnMod.LOGGER.info("GameRenderer TAIL verified; continuous capture runs after blitToScreen, before swap.");
        }
    }

    public static void onPresentedFrame() {
        if (closed || failed) return;
        Minecraft minecraft = Minecraft.getInstance();
        // Opt-in development integration run: exercise real key matching and shutdown lifecycle.
        if (smokeFrames > 0) {
            int key = renderedFrame == smokeFrames / 6 || renderedFrame == smokeFrames / 6+20 ? GLFW.GLFW_KEY_F7
                    : renderedFrame == smokeFrames / 3 || renderedFrame == smokeFrames / 3 + 20 ? GLFW.GLFW_KEY_F8
                    : renderedFrame == smokeFrames * 2 / 3 || renderedFrame == smokeFrames * 2 / 3 + 40 ? GLFW.GLFW_KEY_F9
                    : renderedFrame == smokeFrames - 100 ? GLFW.GLFW_KEY_F10 : 0;
            if (key != 0) onKey(minecraft.getWindow().handle(), GLFW.GLFW_PRESS, new KeyEvent(key, 0, 0));
            if (renderedFrame >= smokeFrames) { minecraft.stop(); return; }
        }
        // Waiting/paused frames must never initialize or resize the GL/CUDA pool.
        if(!started || !enabled) return;
        int newWidth = minecraft.getWindow().getWidth();
        int newHeight = minecraft.getWindow().getHeight();
        if (newWidth <= 0 || newHeight <= 0) { controlAllowed=false; actions.stop(inputSink); refreshMonitor("窗口不可用"); return; }
        try {
            syncInputContext(minecraft);
            if (bridge == null) {
                if (!enabled) return;
                bridge = new NativeCudaBridge(capacity);
                bridge.checkpoint(FabricLoader.getInstance().getGameDir().resolve("vabpnn/models/preview-1.0.0-pre-v1.bin"));
                if(requestedBlocks>=0 || requestedBlockMode>=0) bridge.configureBlocks(requestedBlocks,requestedBlockMode);
                allocateSurfaces(newWidth, newHeight);
                VabpnnMod.LOGGER.info("Shared-context CUDA worker started; OpenGL renderer: {}", glGetString(GL_RENDERER));
            }
            samplePlayerPosition(minecraft,System.nanoTime());
            int phase = bridge.phase();
            if (phase < 0) throw new IllegalStateException(bridge.error());
            if (!resizing && (width != newWidth || height != newHeight)) {
                bridge.retire();
                resizing = true;
            }
            if (resizing) {
                controlAllowed=false; actions.stop(inputSink); refreshMonitor("调整尺寸");
                if (bridge.phase() == NativeCudaBridge.RETIRED) {
                    closeSurfaces();
                    allocateSurfaces(newWidth, newHeight);
                    resizing = false;
                    VabpnnMod.LOGGER.info("GPU pool resized to {}x{} after previous CUDA ownership retired.", width, height);
                }
                return;
            }
            updateNetworkScope(System.nanoTime(),phase);
            if (enabled && phase == NativeCudaBridge.READY) captureFrame();
            long now = System.nanoTime();
            bridge.readControl(controlCommand);
            applyModelCommand(controlCommand,now);
            refreshMonitor("");
            if (now >= nextLog) {
                nextLog = now + 5_000_000_000L;
                var stats = bridge.stats();
                bridge.readTraining(trainingStatus);
                bridge.readTiming(gpuTiming);
                bridge.readDecision(decisionStatus);
                var human=humanInputState();
                var position=positionState();
                VabpnnMod.LOGGER.info("Player position: source=local-player, target=30 Hz, valid={}, xyz={}/{}/{}, samples={}, epoch={}, GPU sample={}, adapterSteps={}",
                        position.valid(),position.x(),position.y(),position.z(),position.samples(),position.epoch(),
                        position.gpuSampleNanos(),position.positionSteps());
                VabpnnMod.LOGGER.info("Decision gate: frame={}, valid={}, eligibleMask={}, learnedK={}, selectedCount={}, lastHumanLabel={}; real mouse press L/R={}/{}, release L/R={}/{}",
                        decisionStatus[0],decisionStatus[1],Long.toHexString(decisionStatus[2]),
                        decisionStatus[3],decisionStatus[4],Long.toHexString(decisionStatus[5]),
                        human.leftPress(),human.rightPress(),human.leftRelease(),human.rightRelease());
                VabpnnMod.LOGGER.info("Minecraft input: allowed={}, appliedFrame={}, mask={}, heldKeys={}, heldButtons={}, appliedCommands={}",
                        controlAllowed,actions.appliedFrame(),Integer.toHexString(actions.appliedMask()),
                        Integer.toHexString(actions.heldKeys()),actions.heldButtons(),actions.appliedCommands());
                VabpnnMod.LOGGER.info("Network scope: visible={}, sampledFrame={}, tile={}, valid={}",monitor.visible(),scopeMeta[0],scopeDiscrete[1],scopeDiscrete[0]);
                VabpnnMod.LOGGER.info("SNN connections: localEdges={}, anchors={}, longEdges={}, topologyUpdates={}, contributionUpdates={}; GPU operator last={} ms, max={} ms, over10ms={}",
                        gpuTiming[4], gpuTiming[5], gpuTiming[6], gpuTiming[3], gpuTiming[7], gpuTiming[0] / 1000.0, gpuTiming[1] / 1000.0, gpuTiming[2]);
                VabpnnMod.LOGGER.info("Online learning: mode={}, examples={}, rejected={}, pixelSteps={}, decisionSteps={}, JEPA={}, mouse={}, decisionLoss={}, wheelLoss={}, checkpoint={}",
                        modelControl ? "MODEL" : "HUMAN", trainingStatus[0], trainingStatus[1], trainingStatus[2],
                        trainingStatus[3], trainingStatus[4], trainingStatus[5],
                        Float.intBitsToFloat((int)trainingStatus[6]), Float.intBitsToFloat((int)trainingStatus[7]),
                        bridge.checkpointMessage());
                if (trainingStatus[8] != lastCheckpointStatus) {
                    lastCheckpointStatus = (int)trainingStatus[8];
                    if (lastCheckpointStatus < 0) VabpnnMod.LOGGER.warn("Model checkpoint: {}", bridge.checkpointMessage());
                }
                VabpnnMod.LOGGER.info(
                        "GPU pipeline: device={}, submitted={}, copied={}, processed={}, skipped={}, dropped={}, lastFrame={}, pool={} slots, allocation={} MiB, models={} MiB, registrations={}, phase={}, actionMask={}",
                        bridge.deviceName(), stats.submitted(), stats.copied(), stats.processed(),
                        stats.skipped(), stats.dropped(), stats.lastProcessed(), stats.capacity(),
                        stats.memoryBytes() / 1048576L, stats.modelBytes() / 1048576L,
                        stats.registrations(), stats.phase(), Long.toHexString(controlCommand[2]));
            }
        } catch (RuntimeException | LinkageError | OutOfMemoryError error) {
            failed = true; enabled = false;
            if(monitor!=null) monitor.status(false,modelControl,false,"GPU 停止");
            VabpnnMod.LOGGER.error("Continuous GPU capture stopped; game rendering can continue.", error);
            InGameFeedback.failure(error);
            releasePipeline();
        }
    }

    private static void samplePlayerPosition(Minecraft minecraft,long now) {
        boolean active=minecraft.level!=null && minecraft.player!=null && minecraft.player.isAlive() && !minecraft.isPaused();
        if(!playerPosition.poll(minecraft.level,minecraft.player,active,now)) return;
        if(active) {
            // AtomSpike sidecar uses the same LocalPlayer.position() accessor.
            // Minecraft state is ticked at 20 Hz; a 30 Hz sample may repeat a tick's coordinates.
            var point=minecraft.player.position();
            playerPosition.coordinates(point.x,point.y,point.z);
        }
        bridge.position(playerPosition.nanos(),playerPosition.epoch(),
                playerPosition.x(),playerPosition.y(),playerPosition.z());
    }
    private static boolean canInteract(Minecraft minecraft) {
        return started && enabled && !failed && !closed && minecraft.level!=null && minecraft.player!=null
                && minecraft.player.isAlive() && minecraft.getOverlay()==null
                && minecraft.isWindowActive() && !minecraft.isPaused()
                && (minecraft.screen==null || minecraft.screen instanceof AbstractContainerScreen<?>);
    }
    /** Single production delivery path, shared by presented frames and tick lease recovery. */
    static void applyModelCommand(long[] command,long now) {
        syncInputContext(Minecraft.getInstance());
        controlAllowed=modelControl && inputActive && !resizing && bridge!=null
                && bridge.phase()==NativeCudaBridge.READY;
        if(inputActive && !resizing) mouseMotion.prediction(command,now);
        actions.apply(command,now,controlAllowed,inputSink);
    }
    private static void resetInputEpoch() {
        controlAllowed=false;
        mouseMotion.reset();
        if(monitor!=null) monitor.motion(mouseMotion.current());
        if(inputSink!=null) actions.stop(inputSink);
        java.util.Arrays.fill(controlCommand,0);
        java.util.Arrays.fill(decisionStatus,0);
        nextFeedbackRead=0;
        if(!started || !enabled) playerPosition.reset();
        demonstration.newScope(System.nanoTime());
        if(bridge!=null) bridge.resetDecision(renderedFrame+1);
    }
    private static void syncInputContext(Minecraft minecraft) {
        boolean active=canInteract(minecraft), grabbed=minecraft.mouseHandler.isMouseGrabbed();
        if(active!=inputActive || actionScene!=minecraft.level || actionScreen!=minecraft.screen || actionMouseGrabbed!=grabbed) {
            resetInputEpoch();
            inputActive=active; actionScene=minecraft.level; actionScreen=minecraft.screen; actionMouseGrabbed=grabbed;
        }
    }
    private static void onClientTick(Minecraft minecraft) {
        if(inputSink==null || closed) return;
        if(!failed) try {
            applyModelCommand(controlCommand,System.nanoTime());
            if(controlAllowed) inputSink.maintainHeldInputs(actions.heldKeys(),actions.heldButtons());
            refreshMonitor(resizing ? "调整尺寸":"");
        } catch(RuntimeException | LinkageError | OutOfMemoryError error) {
            failed=true; enabled=false;
            VabpnnMod.LOGGER.error("Minecraft model input stopped.",error);
            InGameFeedback.failure(error);
            releasePipeline();
            refreshMonitor("输入停止");
        }
        InGameFeedback.tick(minecraft,System.nanoTime());
    }
    /** Immutable, low-frequency host counters for chat/HUD; no image or tensor readback. */
    record FeedbackState(InputState input,Component gate,int phase,boolean scopeVisible,
                         long processed,long dropped,long examples,long pixelSteps,long decisionSteps,
                         long jepaSteps,long motionSteps,long gpuUs,long maximumGpuUs,long overBudget,
                         int checkpoint,String checkpointMessage,long savedRequest,
                         long decisionFrame,int eligibleMask,int requestedK,int selectedCount,int lastHumanLabel,
                         HumanInputState human) {}

    static Component feedbackHotkeys() {
        return Component.translatable("message.vabpnn.hotkeys",scopeKey.getTranslatedKeyMessage(),
                controlKey.getTranslatedKeyMessage(),continuousKey.getTranslatedKeyMessage(),saveKey.getTranslatedKeyMessage());
    }
    static long requestCheckpoint() {
        if(!started || bridge==null || closed || failed || bridge.phase()!=NativeCudaBridge.READY) return 0;
        long request=bridge.saveCheckpoint();
        nextFeedbackRead=0;
        return request;
    }
    static FeedbackState feedbackState(boolean refresh) {
        long now=System.nanoTime();
        int phase=bridge==null ? NativeCudaBridge.INITIALIZING:bridge.phase();
        if(bridge!=null && (refresh || now>=nextFeedbackRead)) {
            nextFeedbackRead=now+1_000_000_000L;
            bridge.readTraining(trainingStatus);
            bridge.readTiming(gpuTiming);
            bridge.readSaveStatus(saveStatus);
            bridge.readDecision(decisionStatus);
            feedbackStats=bridge.stats();
            checkpointText=bridge.checkpointMessage();
        }
        Minecraft client=Minecraft.getInstance();
        String reason;
        if(!started) reason="not_started";
        else if(failed || phase<0) reason="error";
        else if(!enabled || closed) reason="paused";
        else if(bridge==null || phase!=NativeCudaBridge.READY || resizing) reason=resizing ? "resizing":"starting";
        else if(client.level==null || client.player==null) reason="no_world";
        else if(!client.player.isAlive()) reason="dead";
        else if(client.getOverlay()!=null) reason="loading";
        else if(!client.isWindowActive()) reason="unfocused";
        else if(client.screen instanceof net.minecraft.client.gui.screens.ChatScreen) reason="chat";
        else if(client.isPaused()) reason="game_paused";
        else if(client.screen!=null && !(client.screen instanceof AbstractContainerScreen<?>)) reason="menu";
        else if(modelControl) {
            if(actions.appliedFrame()==0) reason="waiting";
            else if(actions.appliedMask()!=0 || actions.heldKeys()!=0 || actions.heldButtons()!=0) reason="executing";
            else if(decisionStatus[0]==0) reason="waiting";
            else reason=decisionStatus[3]==0 ? "idle_k":decisionStatus[2]==0 ? "idle_threshold":"idle";
        }
        else reason="demonstrating";
        return new FeedbackState(inputState(),Component.translatable("message.vabpnn.state."+reason),phase,
                monitor!=null && monitor.visible(),feedbackStats==null ? 0:feedbackStats.processed(),
                feedbackStats==null ? 0:feedbackStats.dropped(),trainingStatus[0],trainingStatus[2],
                trainingStatus[3],trainingStatus[4],trainingStatus[5],gpuTiming[0],gpuTiming[1],gpuTiming[2],
                (int)saveStatus[2],checkpointText,saveStatus[1],
                decisionStatus[0],(int)decisionStatus[2],(int)decisionStatus[3],(int)decisionStatus[4],
                (int)decisionStatus[5],humanInputState());
    }
    private static void refreshMonitor(String note) {
        if(!started) note="待命：/vabpnn start";
        if(monitor!=null) {
            monitor.motion(mouseMotion.sample(System.nanoTime(),started && enabled && !failed && !closed
                    && inputActive && !resizing));
            monitor.status(started && enabled,modelControl,inputActive,note,actions.appliedFrame(),
                    actions.appliedMask(),actions.heldKeys(),actions.heldButtons());
        }
    }
    private static void updateNetworkScope(long now,int phase) {
        if(monitor==null) return;
        boolean sample=monitor.visible() && enabled && phase==NativeCudaBridge.READY;
        int tile=monitor.inspectedTile();
        if(!scopeConfigured || sample!=scopeSampling || tile!=scopeTile) {
            bridge.scopeEnabled(sample,tile); scopeConfigured=true; scopeSampling=sample; scopeTile=tile;
        }
        if(sample && now>=nextScopeRead) {
            nextScopeRead=now+100_000_000L;
            bridge.readScope(scopeMeta,scopeValues,scopeDiscrete);
            monitor.publish(scopeMeta,scopeValues,scopeDiscrete,now);
            monitor.hierarchy(hierarchyState());
        }
    }

    private static void allocateSurfaces(int newWidth, int newHeight) {
        surfaces = new OpenGlCaptureSurface[capacity];
        int[] textures = new int[capacity];
        for (int i = 0; i < capacity; i++) {
            surfaces[i] = new OpenGlCaptureSurface();
            surfaces[i].prepare(newWidth, newHeight);
            textures[i] = surfaces[i].texture();
        }
        glFlush();
        bridge.configure(textures, newWidth, newHeight);
        width = newWidth;
        height = newHeight;
    }

    private static void captureFrame() {
        int slot = bridge.acquire();
        if (slot < 0) return;
        long fence = 0;
        boolean submitted = false;
        try {
            surfaces[slot].copyPresentedFrame();
            fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            if (fence == 0) throw new IllegalStateException("Could not create GL completion fence");
            glFlush();
            long captured = System.nanoTime();
            long handle = Minecraft.getInstance().getWindow().handle();
            for (int k = 0; k < physicalKeys.length; k++)
                physicalKeys[k] = GLFW.glfwGetKey(handle, demoKeyCodes[k]) == GLFW.GLFW_PRESS;
            demonstration.physicalKeys(physicalKeys, captured);
            demonstration.snapshot(captured, inputActive, modelControl, demoData, demoMotion);
            demoData[14] = Minecraft.getInstance().mouseHandler.isMouseGrabbed() ? 1 : 0;
            bridge.demonstration(demoData, demoMotion);
            submitted = bridge.submit(slot, fence, renderedFrame, captured);
            if (submitted) fence = 0; // Ownership transferred to native worker.
        } finally {
            if (fence != 0) glDeleteSync(fence);
            if (!submitted) bridge.cancel(slot);
        }
    }

    private static void closeSurfaces() {
        if (surfaces != null) {
            for (OpenGlCaptureSurface surface : surfaces) if (surface != null) surface.close();
            surfaces = null;
        }
    }

    private static void releasePipeline() {
        // Shutdown alone may wait for native work. Steady-state capture/resize never wait on CUDA.
        controlAllowed=false;
        if (inputSink != null) actions.stop(inputSink);
        java.util.Arrays.fill(controlCommand,0);
        if (bridge != null) { bridge.close(); bridge = null; }
        if (RenderSystem.isOnRenderThread() && GLFW.glfwGetCurrentContext() != 0) closeSurfaces();
        scopeConfigured=false; scopeSampling=false; nextScopeRead=0;
        nextFeedbackRead=0; feedbackStats=null; checkpointText="";
        java.util.Arrays.fill(saveStatus,0);
        java.util.Arrays.fill(decisionStatus,0);
        java.util.Arrays.fill(scopeMeta,0);
        playerPosition.reset();
        java.util.Arrays.fill(positionStatus,0);
        java.util.Arrays.fill(blockStatus,0);
        width = height = 0;
        resizing = false;
    }

    public static void close() {
        if (closed) return;
        closed = true;
        releasePipeline();
        if(monitor!=null) monitor.close();
    }
}

