package cn.vabpnn.mod.client.capture;

import com.mojang.blaze3d.systems.RenderSystem;
import cn.vabpnn.mod.client.scope.ScopeSnapshot;
import cn.vabpnn.mod.client.scope.MemeSnapshot;
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
        ScorerRewardTransportTest.run();
        RewardLedgerTest.run();
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
            comparisonCheckpointSequential();
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
            long pixelParameters = 1920L * 1024 * 3 * ActionLayout.ACTIONS;
            long pixelModelBytes = pixelParameters * (Byte.BYTES + Float.BYTES)
                    + (long) ActionLayout.ACTIONS * 7680 * Integer.BYTES + (long) ActionLayout.ACTIONS * Integer.BYTES;
            require(before.modelBytes() > pixelModelBytes + 4_700_000_000L && before.modelBytes() < 7_500_000_000L,
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
            for(int k=0;k<ScopeSnapshot.ACTIONS;++k) require(Math.abs(scopeValues[ScopeSnapshot.POSTERIOR+k]-.5f)<1e-6
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
            memeScopeThroughJni(bridge,surfaces);
            blockModesThroughJni(bridge,surfaces);
            teachThroughJni(bridge, surfaces);
            // The dense checkpoint test runs sequentially after these windows.
            System.out.printf("PASS: MSAA=%d, device=%s, stats=%s%n", actualSamples, bridge.deviceName(), after);
        } finally {
            if (bridge != null) bridge.close();
            closeSurfaces(surfaces);
            GL.setCapabilities(null);
            glfwMakeContextCurrent(0);
            glfwDestroyWindow(window);
        }
    }

    
    private static void memeScopeThroughJni(NativeCudaBridge bridge,OpenGlCaptureSurface[] surfaces) throws Exception {
        long[] meta=new long[MemeSnapshot.META],ids=new long[MemeSnapshot.LONGS],data=new long[ActionLayout.DEMO_LONGS];
        float[] values=new float[MemeSnapshot.FLOATS];int[] ints=new int[MemeSnapshot.INTS];double[] motion=new double[3];
        data[0]=204;data[2]=1;data[3]=1;data[ActionLayout.POINTER_MODE_INDEX]=1;
        bridge.scopeEnabled(false,0);bridge.memeScopeEnabled(true,0);
        long start=System.nanoTime(),previous=0,previousCapture=0;int distinct=0;
        while(System.nanoTime()-start<1_250_000_000L) {
            submitDemo(bridge,surfaces,data,motion,System.nanoTime());
            bridge.readMemeScope(meta,values,ints,ids);
            if(meta[0]!=0 && meta[0]!=previous) {
                require(meta[0]<=bridge.stats().lastProcessed() && meta[3]==0 && meta[39]==1,"coherent completed 5 Hz JNI frame");
                if(previousCapture!=0) require(meta[1]-previousCapture>=180_000_000L,"GPU telemetry is throttled to 5 Hz");
                previous=meta[0];previousCapture=meta[1];distinct++;
            }
            Thread.sleep(8);
        }
        require(distinct>=3 && distinct<=7,"independent native 5 Hz telemetry keeps updating");
        long hiddenFrame=meta[0],processed=bridge.stats().processed();
        bridge.memeScopeEnabled(false,0);
        for(int i=0;i<20;i++)submitDemo(bridge,surfaces,data,motion,System.nanoTime());
        bridge.readMemeScope(meta,values,ints,ids);
        require(meta[0]==hiddenFrame && bridge.stats().processed()>processed,"hiding meme window stops readback while inference continues");
        bridge.memeScopeEnabled(true,137);
        long deadline=System.nanoTime()+2_000_000_000L;
        do {
            submitDemo(bridge,surfaces,data,motion,System.nanoTime());bridge.readMemeScope(meta,values,ints,ids);
        } while(meta[3]!=137 && System.nanoTime()<deadline);
        require(meta[0]>hiddenFrame && meta[3]==137,"reopening selects the actual GPU tile");
        boolean rejected=false;
        try {bridge.readMemeScope(new long[1],values,ints,ids);}catch(IllegalStateException e){rejected=true;}
        require(rejected,"JNI rejects mismatched buffers");
        bridge.memeScopeEnabled(false,137);bridge.scopeEnabled(true,32*120+60);
        System.out.println("PASS: independent real 5 Hz GPU/JNI meme samples, coherent maps, hide/reopen/selection, no hidden readback.");
    }
    private static void blockModesThroughJni(NativeCudaBridge bridge,OpenGlCaptureSurface[] surfaces) throws Exception {
        // No checkpoint path is configured in this bridge, so commands must still reach the GPU.
        long[] data=new long[ActionLayout.DEMO_LONGS],status=new long[14]; double[] movement=new double[3];
        data[0]=222; data[2]=1; data[3]=1; data[ActionLayout.POINTER_MODE_INDEX]=1;
        int[][] modes={{0,-1,0,1},{1,0,1,0},{-1,1,1,1},{-1,2,1,2},{-1,1,1,1}};
        long nanos=System.nanoTime();
        for(var mode:modes) {
            bridge.configureBlocks(mode[0],mode[1]); nanos+=20_000_000;
            submitDemo(bridge,surfaces,data,movement,nanos); bridge.readBlocks(status);
            require(status[6]==mode[2] && status[7]==mode[3] && status[13]==frame,
                    "off/on/independent/cooperate/compete reaches completed GPU state without a checkpoint");
        }
        long[] memes=new long[12];
        for(int enabled:new int[]{0,1,0,1}) {
            bridge.configureMemes(enabled);nanos+=20_000_000;
            submitDemo(bridge,surfaces,data,movement,nanos);bridge.readMemes(memes);
            require(memes[7]==enabled && memes[11]==frame,"meme toggle reaches completed worker state");
            if(enabled==0)require(memes[8]==0 && memes[9]==0,"off clears local/import active status");
        }
        System.out.println("PASS: actual JNI block modes and meme on/off application without a checkpoint path.");
    }
    private static void teachThroughJni(NativeCudaBridge bridge, OpenGlCaptureSurface[] surfaces) throws Exception {
        long[] data = new long[ActionLayout.DEMO_LONGS], training = new long[9];
        double[] movement = new double[3];
        data[0] = 101; data[1] = 1; data[2] = 1; data[ActionLayout.POINTER_MODE_INDEX] = 1;
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
        data[0]++; data[ActionLayout.POINTER_MODE_INDEX]=0; data[1]=0;
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
        long[] memes=new long[12];bridge.readMemes(memes);
        require(memes[0]>0 && memes[1]>0 && memes[7]==1,"all-frame causal observations and human supervision reach meme branch");
        long memeSamples=memes[1],memeUpdates=memes[2];
        long contributionSteps = timing[7];
        long examples = training[0], mouseSteps = training[5];
        long pixelSteps=training[2],decisionSteps=training[3],jepaSteps=training[4],topologySteps=timing[3];
        long rankings=blocks[2],mutations=blocks[3];
        long[] scopeMeta=new long[ScopeSnapshot.META]; float[] scopeValues=new float[ScopeSnapshot.FLOATS];
        int[] scopeInts=new int[ScopeSnapshot.INTS];
        bridge.readScope(scopeMeta,scopeValues,scopeInts); long sampledFrame=scopeMeta[0];
        bridge.scopeEnabled(false,32*120+60);
        data[0]++; data[3] = 1;
        for (int step = 0; step < 12; step++) {
            nanos += 20_000_000L; movement[0] += 1;
            submitDemo(bridge, surfaces, data, movement, nanos);
        }
        bridge.readTraining(training);
        require(training[0]==examples && training[2]==pixelSteps && training[3]==decisionSteps
                && training[4]==jepaSteps && training[5]==mouseSteps,"JNI evaluation freezes every training counter including JEPA");
        bridge.readTiming(timing);
        bridge.readBlocks(blocks);
        require(blocks[0]==blockExamples && blocks[1]==blockUpdates && blocks[2]==rankings && blocks[3]==mutations,
                "evaluation freezes hierarchical learning, ranking and mutation");
        bridge.readMemes(memes);require(memes[1]==memeSamples && memes[2]==memeUpdates,
            "model control does not supervise meme reliability or projection optimizer");
        require(timing[7]>contributionSteps && timing[3]>topologySteps,"evaluation keeps contribution statistics and rewiring dynamic");
        bridge.readScope(scopeMeta,scopeValues,scopeInts);
        require(scopeMeta[0]==sampledFrame,"hiding scope stops GPU observation publication, not the model");
        bridge.scopeEnabled(true,137);
        for(int i=0;i<12;++i) {
            nanos+=20_000_000L; submitDemo(bridge,surfaces,data,movement,nanos); Thread.sleep(12);
        }
        bridge.readScope(scopeMeta,scopeValues,scopeInts);
        require(scopeMeta[0]>sampledFrame && scopeInts[1]==137,"scope reopens and inspects another real 16x16 tile");
        System.out.println("PASS: real GL/JNI network telemetry, visibility gates and tile inspection.");
        bridge.readTraining(training); bridge.readTiming(timing);
        require(training[4]==jepaSteps && timing[3]>topologySteps,"reopened scope retains neural freeze and dynamic connectivity");
        data[0]++;data[3]=0;
        for(int step=0;step<12;++step) {
            nanos+=20_000_000L;data[8]+=20_000_000L;movement[0]+=.4;
            submitDemo(bridge,surfaces,data,movement,nanos);
        }
        bridge.readTraining(training);bridge.readTiming(timing);
        require(training[0]>examples && training[2]>pixelSteps && training[3]>decisionSteps
                && training[4]>jepaSteps && training[5]>mouseSteps && timing[3]>topologySteps,
                "returning to human mode restores learning and plasticity");
        System.out.println("PASS: actual GL/JNI evaluation freezes neural/JEPA learning while salience/placement/plasticity and inference/scope continue and demonstration resumes training.");
    }

    private static void comparisonCheckpointSequential() throws Exception {
        glfwDefaultWindowHints();glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
        long window=glfwCreateWindow(96,64,"VABPnn gated dense checkpoint",0,0);
        require(window!=0,"comparison checkpoint GL window");glfwMakeContextCurrent(window);GL.createCapabilities();
        var directory=java.nio.file.Files.createTempDirectory(java.nio.file.Path.of("build/native"),"memory-jni-");
        var current=directory.resolve("compare-1.0.2.bin");var reference=directory.resolve("frozen-reference.bin");
        var beta=directory.resolve("compare-1.0.0-beta.bin");
        String fixture=System.getProperty("vabpnn.test.checkpoint","");
        long[] inherited=null;
        if(!fixture.isBlank()) {
            java.nio.file.Files.copy(java.nio.file.Path.of(fixture),beta);
            inherited=readCheckpointTraining(beta);
        }
        long[] sample=new long[ActionLayout.DEMO_LONGS],expected=new long[9],memory=new long[7];
        double[] movement=new double[3];sample[0]=501;sample[1]=1;sample[2]=1;sample[ActionLayout.POINTER_MODE_INDEX]=1;
        long nanos=System.nanoTime(),gateSteps=0;
        try {
            try(var bridge=new NativeCudaBridge(CAPACITY)) {
                var surfaces=allocate(bridge,96,64);
                try {
                    waitPhase(bridge,NativeCudaBridge.READY);bridge.checkpoint(current);
                    if(inherited!=null) {
                        sample[3]=1;
                        for(int i=0;i<4;++i){nanos+=20_000_000L;submitDemo(bridge,surfaces,sample,movement,nanos);}
                        bridge.readTraining(expected);bridge.readMemory(memory);
                        for(int i=0;i<6;++i)require(expected[i]==inherited[i],"read-only beta migration keeps trained counters");
                        require(bridge.checkpointMessage().contains("restored beta"),"actual worker selected beta migration");
                        require(memory[2]==0 && memory[1]>=4 && memory[3]==1,"fresh untrained gates with dynamic migrated-model memory");
                        waitSave(bridge);assertBetaInheritance(beta,current);
                        System.out.println("PASS: real trained 1.94 GB beta checkpoint migrates read-only; all neural weights/moments and old counters exact, new gate optimizer starts at zero.");
                        sample[0]++;sample[3]=0;
                    }
                    final long oldExamples=expected[0],oldDenseSteps=expected[3];
                    for(int i=0;i<16;++i) {
                        nanos+=20_000_000L;sample[8]+=20_000_000L;movement[0]+=.2;
                        submitDemo(bridge,surfaces,sample,movement,nanos);
                    }
                    bridge.readTraining(expected);bridge.readMemory(memory);gateSteps=memory[2];
                    require(expected[0]>oldExamples && expected[3]>oldDenseSteps && gateSteps>0,"real demonstration learns both inherited full graph and new gates");
                    sample[0]++;sample[3]=1;
                    bridge.readMemory(memory);long priorMemoryFrames=memory[1];
                    for(int i=0;i<4;++i){nanos+=20_000_000L;submitDemo(bridge,surfaces,sample,movement,nanos);}
                    bridge.readMemory(memory);
                    require(memory[1]>priorMemoryFrames && memory[2]==gateSteps,"frozen actual JNI predictions still advance matrix memory");
                    waitSave(bridge);
                    require(java.nio.file.Files.size(current)==1_939_227_548L,"independent v11 24-part gated/reward checkpoint size");
                    var dense=readComparisonSteps(current);
                    require(dense[0]==expected[3] && dense[1]==gateSteps,"causal dense and fresh gate optimizer counters persist");
                    java.nio.file.Files.copy(current,reference);
                } finally {bridge.close();closeSurfaces(surfaces);}
            }
            // Only one complete gated model exists at a time on this 8 GiB GPU.
            try(var restored=new NativeCudaBridge(CAPACITY)) {
                var surfaces=allocate(restored,96,64);
                try {
                    waitPhase(restored,NativeCudaBridge.READY);restored.checkpoint(current);
                    for(int i=0;i<8;++i){nanos+=20_000_000L;submitDemo(restored,surfaces,sample,movement,nanos);}
                    long[] actual=new long[9];restored.readTraining(actual);restored.readMemory(memory);
                    for(int i=0;i<6;++i)require(actual[i]==expected[i],"v11 restart preserves every learner counter");
                    require(restored.checkpointMessage().contains("restored"),"actual v11 restored in worker");
                    require(memory[1]>=8 && memory[2]==gateSteps && memory[3]==1 && Float.intBitsToFloat((int)memory[5])>0,
                            "fresh recurrent state evolves after restart with frozen trained gates");
                    waitSave(restored);
                } finally {restored.close();closeSurfaces(surfaces);}
            }
            assertComparisonNeuralParts(reference,current);
            require(java.util.Arrays.equals(readComparisonSteps(reference),readComparisonSteps(current)),
                    "both dense and gate AdamW counters freeze after restart");
            if(inherited!=null)require(rangeHash(java.nio.file.Path.of(fixture),0,java.nio.file.Files.size(beta))
                    .equals(rangeHash(beta,0,java.nio.file.Files.size(beta))),"original beta fixture is unchanged");
            System.out.println("PASS: GL/JNI full gated model trains, saves/restores v11, freezes all neural/gate weights and moments while matrix memory continues.");
            rewardCheckpointSequential(directory,current);
            System.out.println("COMPARE_CHECKPOINT_DIR="+directory.toAbsolutePath());
        } finally {GL.setCapabilities(null);glfwMakeContextCurrent(0);glfwDestroyWindow(window);}
    }
    private static void rewardCheckpointSequential(java.nio.file.Path directory,java.nio.file.Path trained)throws Exception {
        var model=directory.resolve("reward-model.bin");var initial=directory.resolve("reward-initial.bin");var updated=directory.resolve("reward-updated.bin");
        CheckpointTestSupport.seedPolicy(trained,model);java.nio.file.Files.copy(model,initial);
        long[] sample=new long[ActionLayout.DEMO_LONGS],status=new long[12],proposal=new long[10],command=new long[7],training=new long[9];
        sample[0]=777;sample[2]=1;sample[3]=1;sample[ActionLayout.POINTER_MODE_INDEX]=1;
        double[] motion=new double[3];long updates=0;
        try(var bridge=new NativeCudaBridge(CAPACITY)){
            var surfaces=allocate(bridge,96,64);
            try {
                waitPhase(bridge,NativeCudaBridge.READY);bridge.checkpoint(model);
                submitDemo(bridge,surfaces,sample,motion,System.nanoTime());bridge.readTraining(training);
                bridge.rewardMode(10,true);Thread.sleep(30);
                for(int step=0;step<3;step++){
                    submitDemo(bridge,surfaces,sample,motion,System.nanoTime());bridge.readControl(command);bridge.readRewardProposal(command[0],proposal);
                    require(proposal[8]==1 && proposal[2]==10 && command[2]==proposal[4],"matching stochastic proposal and final control");
                    require((proposal[4]&~proposal[5])==0 && (proposal[5]&~3)==0,"real JNI respects learned fixture thresholds");
                    require(!bridge.reward(10,step+1,100,120,1,new long[]{command[0]}),"no successful execution proof rejected");
                    require(!bridge.rewardExecution(10,command[0],command[1],(int)command[2]^4,101,101,System.nanoTime(),proposal[3]),"wrong actual mask rejected");
                    var sink=new ActionExecutor.Sink(){public void key(int c,boolean d){}public void button(int b,boolean d){}public void move(double x,double y){}public void scroll(double w){}};
                    var executor=new ActionExecutor();executor.apply(command,System.nanoTime(),true,sink);
                    require(executor.appliedCommands()==1 && executor.appliedMask()==command[2],"real ActionExecutor accepted packet");
                    require(bridge.rewardExecution(10,command[0],command[1],executor.appliedMask(),101,101,System.nanoTime(),proposal[3]),"ack actual successful application");
                    require(!bridge.rewardExecution(10,command[0],command[1],executor.appliedMask(),101,101,System.nanoTime(),proposal[3]),"no duplicate execution receipt");
                    require(!bridge.reward(10,step+1,101,120,1,new long[]{command[0]}),"open interval excludes start-tick action");
                    float credit=step==0?1:step==1?-1:0;
                    require(bridge.reward(10,step+1,100,120,credit,new long[]{command[0]}),"matched receipt queued");
                    require(!bridge.reward(10,step+1,100,120,credit,new long[]{command[0]}),"duplicate reward rejected");
                    long deadline=System.nanoTime()+10_000_000_000L;do{bridge.readRewardStatus(status);if(status[5]>=step+1)break;Thread.sleep(5);}while(System.nanoTime()<deadline);
                    require(status[5]==step+1 && status[2]==Math.min(step+1,2),"positive/negative update, zero receipt no update");
                    executor.stop(sink);
                }
                updates=status[2];require(updates==2,"exactly two reward updates");
                long[] after=new long[9];bridge.readTraining(after);for(int i=0;i<6;i++)require(training[i]==after[i],"all demo/PNN/JEPA learners frozen during reward");
                bridge.rewardMode(11,false);
                require(!bridge.reward(10,4,100,120,1,new long[]{command[0]}),"F8/predict epoch invalidates queued reward");
                for(int i=0;i<3;i++)submitDemo(bridge,surfaces,sample,motion,System.nanoTime());
                bridge.readRewardStatus(status);require(status[2]==updates&&status[7]==0,"reward weights frozen");
                waitSave(bridge);CheckpointTestSupport.frozen(initial,model,true);java.nio.file.Files.copy(model,updated);
            }finally{bridge.close();closeSurfaces(surfaces);}
        }
        require(java.util.Arrays.equals(CheckpointTestSupport.rewardSteps(model),new long[]{2,2}),"separate reward optimizer saved");
        try(var bridge=new NativeCudaBridge(CAPACITY)){
            var surfaces=allocate(bridge,96,64);try{
                waitPhase(bridge,NativeCudaBridge.READY);bridge.checkpoint(model);
                for(int i=0;i<4;i++)submitDemo(bridge,surfaces,sample,motion,System.nanoTime());
                bridge.readRewardStatus(status);require(status[2]==updates && status[1]==2 && status[7]==0,"v11 restart publishes independent optimizer without enabling training");
                waitSave(bridge);CheckpointTestSupport.frozen(updated,model,false);
            }finally{bridge.close();closeSurfaces(surfaces);}
        }
        System.out.println("PASS: real GL/JNI/CUDA reward proposals -> successful ActionExecutor receipts -> positive/negative/zero, dedupe/open interval/F8 protections; only 1022 policy weights update; all other neural parts exact; reward Adam checkpoint/restart frozen.");
    }
    private static String rangeHash(java.nio.file.Path path,long offset,long length) throws Exception {
        try(var file=java.nio.channels.FileChannel.open(path,java.nio.file.StandardOpenOption.READ)) {
            var hash=java.security.MessageDigest.getInstance("SHA-256");
            for(long i=0;i<length;) {
                int count=(int)Math.min(1<<20,length-i);hash.update(readCheckpointBytes(file,offset+i,count));i+=count;
            }
            return java.util.HexFormat.of().formatHex(hash.digest());
        }
    }
    private static void assertBetaInheritance(java.nio.file.Path beta,java.nio.file.Path current) throws Exception {
        try(var b=java.nio.channels.FileChannel.open(beta,java.nio.file.StandardOpenOption.READ);
            var c=java.nio.channels.FileChannel.open(current,java.nio.file.StandardOpenOption.READ)) {
            var hb=readCheckpointBytes(b,0,256);var hc=readCheckpointBytes(c,0,256);
            require(hb.getLong(8)==9 && hc.getLong(8)==11 && hb.getLong(16)==23 && hc.getLong(16)==24,"real beta-to-official layout");
            long before=256,after=256;int[] frozen={0,1,2,3,4,5,7,8,11,14,18,20,21,22};
            for(int part=0;part<23;++part) {
                long bs=hb.getLong(32+part*8),cs=hc.getLong(32+part*8);
                if(part==19) {
                    require(bs==1_530_656 && cs==1_532_056,"new gate checkpoint expansion");
                    for(int bank=0;bank<3;++bank) {
                        require(rangeHash(beta,before+bank*510_216L,510_216L)
                                .equals(rangeHash(current,after+bank*510_680L,510_216L)),"inherited dense parameter/moment prefix "+bank);
                        var gate=readCheckpointBytes(c,after+bank*510_680L+510_216L,464);
                        for(int h=0;h<2;++h)for(int g=0;g<3;++g)for(int f=0;f<19;++f) {
                            float expected=bank==0 && g==0 && f==18 ? -2.252168461f:0;
                            require(gate.getFloat(((h*3+g)*19+f)*4)==expected,"fresh gate weights and moments");
                        }
                        for(int h=0;h<2;++h)require(gate.getFloat((114+h)*4)==(bank==0 ? 1:0),"fresh gate read gain/moments");
                    }
                    require(readCheckpointBytes(b,before+1_530_648L,8).getLong(0)==readCheckpointBytes(c,after+1_532_040L,8).getLong(0)
                            && readCheckpointBytes(c,after+1_532_048L,8).getLong(0)==0,"inherited beta optimizer counter, independent gate counter");
                } else {
                    require(bs==cs,"unchanged old part layout");
                    if(java.util.Arrays.binarySearch(frozen,part)>=0)
                        require(rangeHash(beta,before,bs).equals(rangeHash(current,after,cs)),"inherited exact neural tensor "+part);
                }
                before+=bs;after+=cs;
            }
        }
    }
    private static long[] readComparisonSteps(java.nio.file.Path path) throws Exception {
        try(var file=java.nio.channels.FileChannel.open(path,java.nio.file.StandardOpenOption.READ)) {
            var header=readCheckpointBytes(file,0,256);
            require(header.getLong(8)==11 && header.getLong(16)==24,"comparison v11 has twenty-four parts");
            long denseOffset=256;
            for(int i=0;i<19;++i)denseOffset+=header.getLong(32+i*8);
            require(header.getLong(32+19*8)==1_532_056L,"comparison small Transformer/AdamW/gate state");
            for(int i=20;i<23;++i)require(header.getLong(32+i*8)==471_859_200L,"full two-head pair tensor and both AdamW moments");
            var steps=readCheckpointBytes(file,denseOffset+1_532_040L,16);return new long[]{steps.getLong(0),steps.getLong(8)};
        }
    }
    private static void assertComparisonNeuralParts(java.nio.file.Path a,java.nio.file.Path b) throws Exception {
        try(var first=java.nio.channels.FileChannel.open(a,java.nio.file.StandardOpenOption.READ);
            var second=java.nio.channels.FileChannel.open(b,java.nio.file.StandardOpenOption.READ)) {
            var h1=readCheckpointBytes(first,0,256);var h2=readCheckpointBytes(second,0,256);
            int[] frozen={0,1,2,3,4,5,7,8,11,14,18,19,20,21,22,23};
            long offset=256;
            for(int part=0;part<24;++part) {
                long length=h1.getLong(32+part*8);
                require(length==h2.getLong(32+part*8),"checkpoint part layout stable");
                if(java.util.Arrays.binarySearch(frozen,part)>=0) {
                    var d1=java.security.MessageDigest.getInstance("SHA-256");
                    var d2=java.security.MessageDigest.getInstance("SHA-256");
                    for(long pos=0;pos<length;) {
                        int count=(int)Math.min(1<<20,length-pos);
                        d1.update(readCheckpointBytes(first,offset+pos,count));
                        d2.update(readCheckpointBytes(second,offset+pos,count));pos+=count;
                    }
                    require(java.util.Arrays.equals(d1.digest(),d2.digest()),"frozen comparison neural/optimizer part "+part);
                }
                offset+=length;
            }
        }
    }
    private static long[] readCheckpointTraining(java.nio.file.Path path) throws Exception {
        try(var input=java.nio.channels.FileChannel.open(path,java.nio.file.StandardOpenOption.READ)) {
            var header=readCheckpointBytes(input,0,256);
            require((header.getLong(8)==6 && header.getLong(16)==17) || (header.getLong(8)==7 && header.getLong(16)==18)
                    || (header.getLong(8)==8 && header.getLong(16)==19) || (header.getLong(8)==9 && header.getLong(16)==23) || (header.getLong(8)==10 && header.getLong(16)==23) || (header.getLong(8)==11 && header.getLong(16)==24),"real checkpoint layout");
            int parts=(int)header.getLong(16);long[] starts=new long[parts],sizes=new long[parts];long offset=256;
            for(int i=0;i<parts;++i){sizes[i]=header.getLong(32+i*8);starts[i]=offset;offset+=sizes[i];}
            require(offset==input.size() && sizes[6]==64 && sizes[8]==34_488,"real checkpoint part sizes");
            var online=readCheckpointBytes(input,starts[8]+sizes[8]-48,48);
            var optimizers=readCheckpointBytes(input,starts[6],64);
            return new long[]{online.getLong(0),online.getLong(8),online.getLong(16),online.getLong(24),
                    optimizers.getLong(0),optimizers.getLong(32)};
        }
    }
    private static long[] readCheckpointClockSteps(java.nio.file.Path path) throws Exception {
        try(var input=java.nio.channels.FileChannel.open(path,java.nio.file.StandardOpenOption.READ)) {
            var header=readCheckpointBytes(input,0,256);
            if(header.getLong(8)<8)return new long[2];
            require(header.getLong(8)==8 && header.getLong(16)==19,"compatible v8 temporal layout");
            long offset=256;for(int i=0;i<18;++i)offset+=header.getLong(32+i*8);
            require(header.getLong(32+18*8)==14_216,"compatible minute/Bayesian mouse state");
            var part=readCheckpointBytes(input,offset,14_216);
            return new long[]{part.getLong(13_408),part.getLong(14_208)};
        }
    }
    private static java.nio.ByteBuffer readCheckpointBytes(java.nio.channels.FileChannel input,long offset,int count) throws Exception {
        var bytes=java.nio.ByteBuffer.allocate(count).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        while(bytes.hasRemaining())require(input.read(bytes,offset+bytes.position())>0,"complete checkpoint metadata");
        bytes.flip();return bytes;
    }

    private static void waitSave(NativeCudaBridge bridge) throws Exception {
        long request=bridge.saveCheckpoint(),deadline=System.nanoTime()+20_000_000_000L;
        long[] status=new long[3];
        do {
            bridge.readSaveStatus(status);
            if(status[1]>=request && status[2]==3)return;
            require(status[2]>=0,bridge.checkpointMessage());Thread.sleep(5);
        } while(System.nanoTime()<deadline);
        throw new AssertionError("checkpoint save completion");
    }
    private static byte[] fileDigest(java.nio.file.Path path) throws Exception {
        var digest=java.security.MessageDigest.getInstance("SHA-256");byte[] bytes=new byte[1<<20];
        try(var input=java.nio.file.Files.newInputStream(path)) {
            for(int length;(length=input.read(bytes))>=0;)if(length>0)digest.update(bytes,0,length);
        }
        return digest.digest();
    }

    private static void submitDemo(NativeCudaBridge bridge, OpenGlCaptureSurface[] surfaces,
                                   long[] data, double[] movement, long nanos) throws Exception {
        int slot;
        long acquireDeadline = System.nanoTime() + 20_000_000_000L;
        while ((slot = bridge.acquire()) < 0 && System.nanoTime() < acquireDeadline) {
            require(bridge.phase() >= 0, bridge.error()); Thread.sleep(2);
        }
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


