package cn.vabpnn.mod.client.capture;

import com.mojang.blaze3d.systems.RenderSystem;
import cn.vabpnn.mod.client.scope.ScopeSnapshot;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;
import org.lwjgl.system.MemoryStack;

import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL33C.*;

/** Hidden-window integration test. No game pixels are read back or written to disk. */
public final class OpenGlCaptureSmokeTest {
    private static final int CAPACITY = 4;
    private static long frame;
    public static void main(String[] args) throws Exception {
        ActionExecutorTest.run();
        DemonstrationInputTest.run();
        PlayerPositionSamplerTest.run();
        GLFWErrorCallback callback = GLFWErrorCallback.createPrint(System.err);
        glfwSetErrorCallback(callback);
        try {
            require(glfwInit(), "GLFW initialization");
            RenderSystem.initRenderThread();
            testWindow(0);
            testWindow(4);
            System.out.println("PASS: CUDA texture registration/map/D2D copy/operator completion, backpressure, resize and GL state; no frame readback.");
        } finally {
            glfwTerminate();
            glfwSetErrorCallback(null);
            callback.free();
        }
    }

    private static void testWindow(int samples) throws Exception {
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, samples);
        long window = glfwCreateWindow(96, 64, "VABPnn CUDA test", 0, 0);
        require(window != 0, "hidden window");
        glfwMakeContextCurrent(window);
        GL.createCapabilities();
        NativeCudaBridge bridge = null;
        OpenGlCaptureSurface[] surfaces = null;
        try {
            int actualSamples = glGetInteger(GL_SAMPLE_BUFFERS) == 0 ? 0 : glGetInteger(GL_SAMPLES);
            require(samples == 0 ? actualSamples == 0 : actualSamples > 0, "source sample count");
            bridge = new NativeCudaBridge(CAPACITY);
            int[] size = size(window);
            bridge.scopeEnabled(true,32*120+60);
            surfaces = allocate(bridge, size[0], size[1]);
            waitPhase(bridge, NativeCudaBridge.READY);
            stress(bridge, surfaces, size[0], size[1]);
            waitProcessed(bridge);
            var before = bridge.stats();
            require(before.capacity() == CAPACITY && before.registrations() == CAPACITY, "fixed pool/register once");
            long pixelModelBytes = 76_677_120L + 13L * 7680 * Integer.BYTES + 13L * Integer.BYTES;
            require(before.modelBytes() > pixelModelBytes + 4_000_000L && before.modelBytes() < pixelModelBytes + 480_000_000L,
                    "persistent pixel/decision/SNN weights and bounded JEPA/AdamW workspace");
            long[] timing = new long[8];
            bridge.readTiming(timing);
            require(timing[0] > 0 && timing[1] >= timing[0] && timing[4] <= 1966080L * 4,
                    "CUDA event timing and bounded connection counters");
            long[] command = new long[7];
            bridge.readControl(command);
            require(command[0] > 0 && command[3] == 1 && command[2] == 0,
                    "neutral untrained heads publish a valid zero-action command");
            require(command[4] == 0 && command[5] == 0 && command[6] == 0, "zero initial motion proposal");
            long[] decision=new long[7]; bridge.readDecision(decision);
            require(decision[0]>0 && decision[1]==1 && decision[2]==0 && decision[3]==0 && decision[4]==0,
                    "neutral GPU diagnostics explain k=0 and no eligible action");
            long[] scopeMeta=new long[ScopeSnapshot.META];
            float[] scopeValues=new float[ScopeSnapshot.FLOATS]; int[] scopeInts=new int[ScopeSnapshot.INTS];
            bridge.readScope(scopeMeta,scopeValues,scopeInts);
            require(scopeMeta[0]>0 && scopeMeta[0]<=bridge.stats().lastProcessed() && scopeInts[0]==1
                && scopeInts[1]==32*120+60,"completed-frame scope snapshot through JNI");
            for(int k=0;k<13;++k) require(Math.abs(scopeValues[ScopeSnapshot.POSTERIOR+k]-.5f)<1e-6
                && scopeInts[ScopeSnapshot.VOTES+k]==0,"real neutral GPU probabilities/raw votes in the scope");
            for(int i=ScopeSnapshot.MAP_POS;i<ScopeSnapshot.LOCAL_MEM;++i)
                require(Float.isFinite(scopeValues[i]) && scopeValues[i]>=0 && scopeValues[i]<=1,"bounded signed block rates");
            require(before.skipped() + before.dropped() > 0, "backpressure without blocking producer");

            bridge.retire();
            waitPhase(bridge, NativeCudaBridge.RETIRED);
            closeSurfaces(surfaces);
            surfaces = null;
            require(bridge.stats().memoryBytes() == 0, "CUDA frame pool released after retirement");
            require(bridge.stats().modelBytes() == before.modelBytes(), "weights retained during frame-pool retirement");
            glfwSetWindowSize(window, 176, 110);
            glfwPollEvents();
            glfwSwapBuffers(window);
            size = size(window);
            surfaces = allocate(bridge, size[0], size[1]);
            waitPhase(bridge, NativeCudaBridge.READY);
            stress(bridge, surfaces, size[0], size[1]);
            long oldProcessed = before.processed();
            long deadline = System.nanoTime() + 20_000_000_000L;
            while (bridge.stats().processed() <= oldProcessed && System.nanoTime() < deadline) {
                require(bridge.phase() >= 0, bridge.error());
                Thread.sleep(10);
            }
            var after = bridge.stats();
            require(after.processed() > oldProcessed, "processing after resize");
            require(after.registrations() == CAPACITY * 2, "register once per resized texture");
            require(after.modelBytes() == before.modelBytes(), "fixed model allocation after resize");
            blockModesThroughJni(bridge,surfaces);
            teachThroughJni(bridge, surfaces);
            System.out.printf("PASS: MSAA=%d, device=%s, stats=%s%n", actualSamples, bridge.deviceName(), after);
        } finally {
            if (bridge != null) bridge.close();
            closeSurfaces(surfaces);
            GL.setCapabilities(null);
            glfwMakeContextCurrent(0);
            glfwDestroyWindow(window);
        }
    }

    private static void blockModesThroughJni(NativeCudaBridge bridge,OpenGlCaptureSurface[] surfaces) throws Exception {
        // No checkpoint path is configured in this bridge, so commands must still reach the GPU.
        long[] data=new long[15],status=new long[14]; double[] movement=new double[3];
        data[0]=222; data[2]=1; data[3]=1; data[14]=1;
        int[][] modes={{0,-1,0,1},{1,0,1,0},{-1,1,1,1},{-1,2,1,2},{-1,1,1,1}};
        long nanos=System.nanoTime();
        for(var mode:modes) {
            bridge.configureBlocks(mode[0],mode[1]); nanos+=20_000_000;
            submitDemo(bridge,surfaces,data,movement,nanos); bridge.readBlocks(status);
            require(status[6]==mode[2] && status[7]==mode[3] && status[13]==frame,
                    "off/on/independent/cooperate/compete reaches completed GPU state without a checkpoint");
        }
        System.out.println("PASS: actual JNI block mode application with no checkpoint path.");
    }
    private static void teachThroughJni(NativeCudaBridge bridge, OpenGlCaptureSurface[] surfaces) throws Exception {
        long[] data = new long[15], training = new long[9];
        double[] movement = new double[3];
        data[0] = 101; data[1] = 1; data[2] = 1; data[14] = 1;
        long nanos = System.nanoTime();
        for (int step = 0; step < 44; step++) {
            data[8] += 20_000_000L; movement[0] += 0.4;
            nanos += 20_000_000L;
            submitDemo(bridge, surfaces, data, movement, nanos);
        }
        bridge.readTraining(training);
        require(training[0] >= 18 && training[2] >= 4 && training[5] >= 16,
                "real GL -> JNI sample attachment -> GPU policy / pixels / mouse training");
        long[] decision=new long[7]; bridge.readDecision(decision);
        require(decision[0]>0 && decision[1]==1 && (decision[5]&1)!=0,
                "GPU publishes the last genuine demonstration label for diagnostics");
        data[0]++; data[14]=0; data[1]=0;
        nanos+=20_000_000L; submitDemo(bridge,surfaces,data,movement,nanos);
        for(int button=0;button<2;button++) {
            data[1]=1L<<(7+button); data[4+button]++;
            nanos+=20_000_000L; submitDemo(bridge,surfaces,data,movement,nanos);
            bridge.readDecision(decision);
            require((decision[5]&(1L<<(7+button)))!=0,"ungrabbed container button enters GPU labels");
            nanos+=20_000_000L; submitDemo(bridge,surfaces,data,movement,nanos);
            bridge.readDecision(decision);
            require((decision[5]&(1L<<(7+button)))!=0,"container hold remains a GPU label without a new edge");
            data[1]=0; data[6+button]++;
            nanos+=20_000_000L; submitDemo(bridge,surfaces,data,movement,nanos);
            bridge.readDecision(decision);
            require((decision[5]&(1L<<(9+button)))!=0,"container release enters GPU labels");
        }
        bridge.readTraining(training);
        long[] timing = new long[8]; bridge.readTiming(timing);
        require(timing[3] > 0 && timing[7] >= 18, "JNI full pipeline computes supervised SNN contribution and rewires every five predictions");
        long[] blocks=new long[14]; bridge.readBlocks(blocks);
        require(blocks[0]>=18 && blocks[1]>0,"real human GL/JNI feedback reaches hierarchical scores and AdamW after warmup");
        long blockExamples=blocks[0],blockUpdates=blocks[1];
        long contributionSteps = timing[7];
        long examples = training[0], mouseSteps = training[5];
        long[] scopeMeta=new long[ScopeSnapshot.META]; float[] scopeValues=new float[ScopeSnapshot.FLOATS];
        int[] scopeInts=new int[ScopeSnapshot.INTS];
        bridge.readScope(scopeMeta,scopeValues,scopeInts); long sampledFrame=scopeMeta[0];
        bridge.scopeEnabled(false,32*120+60);
        data[0]++; data[3] = 1;
        for (int step = 0; step < 2; step++) {
            nanos += 20_000_000L; movement[0] += 1;
            submitDemo(bridge, surfaces, data, movement, nanos);
        }
        bridge.readTraining(training);
        require(training[0] == examples && training[5] == mouseSteps, "JNI model-mode metadata prevents imitation");
        bridge.readTiming(timing);
        bridge.readBlocks(blocks);
        require(blocks[0]==blockExamples && blocks[1]==blockUpdates,"model mode cannot train hierarchical actors or scores");
        require(timing[7] == contributionSteps, "model control is not a supervised connection target");
        bridge.readScope(scopeMeta,scopeValues,scopeInts);
        require(scopeMeta[0]==sampledFrame,"hiding scope stops GPU observation publication, not the model");
        bridge.scopeEnabled(true,137);
        for(int i=0;i<12;++i) {
            nanos+=20_000_000L; submitDemo(bridge,surfaces,data,movement,nanos); Thread.sleep(12);
        }
        bridge.readScope(scopeMeta,scopeValues,scopeInts);
        require(scopeMeta[0]>sampledFrame && scopeInts[1]==137,"scope reopens and inspects another real 16x16 tile");
        System.out.println("PASS: real GL/JNI network telemetry, visibility gates and tile inspection.");
        System.out.println("PASS: real GL frames plus JNI cumulative human input train GPU pixel/decision/mouse models; model mode excludes BC.");
    }
    private static void submitDemo(NativeCudaBridge bridge, OpenGlCaptureSurface[] surfaces,
                                   long[] data, double[] movement, long nanos) throws Exception {
        int slot = bridge.acquire();
        require(slot >= 0, "free texture for ordered demonstration");
        surfaces[slot].copyPresentedFrame();
        long fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush(); long id = ++frame;
        bridge.demonstration(data, movement);
        bridge.position(nanos-5_000_000L,data[0],42.25,64,-7.125);
        require(bridge.submit(slot, fence, id, nanos), "submit human metadata and frame together");
        long deadline = System.nanoTime() + 20_000_000_000L;
        while (bridge.stats().lastProcessed() < id && System.nanoTime() < deadline) {
            require(bridge.phase() >= 0, bridge.error()); Thread.sleep(5);
        }
        require(bridge.stats().lastProcessed() == id, "complete matching demonstration frame");
        long[] position=new long[7]; bridge.readPosition(position);
        require(position[0]==nanos-5_000_000L && position[1]==data[0] && position[3]==1,
                "coordinate timestamp and world epoch stay paired with the submitted GL frame");
        require(Double.longBitsToDouble(position[4])==42.25 && Double.longBitsToDouble(position[5])==64
                && Double.longBitsToDouble(position[6])==-7.125,"JNI coordinate precision reaches actual GPU decision");
    }

    private static OpenGlCaptureSurface[] allocate(NativeCudaBridge bridge, int width, int height) {
        OpenGlCaptureSurface[] surfaces = new OpenGlCaptureSurface[CAPACITY];
        int[] textures = new int[CAPACITY];
        int previousTexture = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, previousTexture);
        int unpack = glGenBuffers();
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpack);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, 16L, GL_STREAM_DRAW);
        try {
            for (int i = 0; i < CAPACITY; i++) {
                surfaces[i] = new OpenGlCaptureSurface();
                surfaces[i].prepare(width, height);
                textures[i] = surfaces[i].texture();
                require(glGetInteger(GL_TEXTURE_BINDING_2D) == previousTexture, "allocation texture state");
                require(glGetInteger(GL_PIXEL_UNPACK_BUFFER_BINDING) == unpack, "allocation unpack state");
            }
            glFlush();
            bridge.configure(textures, width, height);
            return surfaces;
        } catch (RuntimeException | Error error) {
            closeSurfaces(surfaces);
            throw error;
        } finally {
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glDeleteTextures(previousTexture);
            glDeleteBuffers(unpack);
        }
    }

    private static void stress(NativeCudaBridge bridge, OpenGlCaptureSurface[] surfaces, int width, int height) {
        int priorRead = glGenFramebuffers();
        int priorDraw = glGenFramebuffers();
        try {
            for (int i = 0; i < 2000; i++) {
                int slot = bridge.acquire();
                if (slot < 0) continue;
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
                glDrawBuffer(GL_BACK);
                glDisable(GL_SCISSOR_TEST);
                glClearColor(i % 2, 0.5f, 1, 1);
                glClear(GL_COLOR_BUFFER_BIT);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
                glReadBuffer(GL_FRONT);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, priorRead);
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, priorDraw);
                glEnable(GL_SCISSOR_TEST);
                glScissor(1, 1, 2, 2);
                glEnable(GL_FRAMEBUFFER_SRGB);
                surfaces[slot].copyPresentedFrame();
                require(glGetInteger(GL_READ_FRAMEBUFFER_BINDING) == priorRead, "read FBO restoration");
                require(glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING) == priorDraw, "draw FBO restoration");
                require(glIsEnabled(GL_SCISSOR_TEST) && glIsEnabled(GL_FRAMEBUFFER_SRGB), "enable restoration");
                int[] scissor = new int[4];
                glGetIntegerv(GL_SCISSOR_BOX, scissor);
                require(scissor[0] == 1 && scissor[1] == 1 && scissor[2] == 2 && scissor[3] == 2, "scissor rectangle");
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
                require(glGetInteger(GL_READ_BUFFER) == GL_FRONT, "default FBO read buffer");
                glBindFramebuffer(GL_READ_FRAMEBUFFER, priorRead);
                long fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                glFlush();
                require(fence != 0, "producer GL fence");
                if (!bridge.submit(slot, fence, ++frame, System.nanoTime())) {
                    glDeleteSync(fence);
                    bridge.cancel(slot);
                    throw new AssertionError("submission failed: " + bridge.error());
                }
                require(glGetError() == GL_NO_ERROR, "producer GL error");
            }
        } finally {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteFramebuffers(priorRead);
            glDeleteFramebuffers(priorDraw);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_FRAMEBUFFER_SRGB);
        }
    }

    private static void waitPhase(NativeCudaBridge bridge, int phase) throws Exception {
        long deadline = System.nanoTime() + 30_000_000_000L;
        while (bridge.phase() != phase && System.nanoTime() < deadline) {
            require(bridge.phase() >= 0, bridge.error());
            Thread.sleep(10);
        }
        require(bridge.phase() == phase, "pool phase " + phase + ": " + bridge.error());
    }
    private static void waitProcessed(NativeCudaBridge bridge) throws Exception {
        long deadline = System.nanoTime() + 20_000_000_000L;
        while (bridge.stats().processed() == 0 && System.nanoTime() < deadline) {
            require(bridge.phase() >= 0, bridge.error());
            Thread.sleep(10);
        }
        require(bridge.stats().copied() > 0 && bridge.stats().processed() > 0, "GPU copy and operator completion");
    }
    private static void closeSurfaces(OpenGlCaptureSurface[] surfaces) {
        if (surfaces != null) for (var surface : surfaces) if (surface != null) surface.close();
    }
    private static int[] size(long window) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            var w = stack.mallocInt(1);
            var h = stack.mallocInt(1);
            glfwGetFramebufferSize(window, w, h);
            return new int[]{w.get(0), h.get(0)};
        }
    }
    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}

