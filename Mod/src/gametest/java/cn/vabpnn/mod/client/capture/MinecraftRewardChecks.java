package cn.vabpnn.mod.client.capture;
import com.google.gson.*;
import java.nio.file.*;import java.net.*;import java.net.http.*;import java.time.*;
import java.lang.reflect.*;import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.client.Minecraft;import net.minecraft.client.gui.screens.ChatScreen;import net.minecraft.client.input.KeyEvent;
import org.lwjgl.glfw.GLFW;
/** Real production controller/telemetry/Scorer reward integration, isolated world and checkpoint. */
public final class MinecraftRewardChecks {
    private static void require(boolean b,String s){if(!b)throw new AssertionError(s);}
    private static Object field(String key){
        try{var f=CaptureController.class.getDeclaredField(key);f.setAccessible(true);return f.get(null);}
        catch(ReflectiveOperationException e){throw new AssertionError(e);}
    }
    private static SelfTrainingController trainer(){return (SelfTrainingController)field("selfTraining");}
    private static NativeCudaBridge bridge(){return (NativeCudaBridge)field("bridge");}
    private static long[] rewardState(ClientGameTestContext c){return c.computeOnClient(mc->{var v=new long[12];bridge().readRewardStatus(v);return v;});}
    private static void command(ClientGameTestContext c,String text){
        c.runOnClient(mc->{var chat=new ChatScreen("",false);mc.setScreen(chat);chat.handleChatInput("/vabpnn "+text,true);mc.setScreen(null);});
        c.waitTick();
    }
    private static void key(ClientGameTestContext c,int key){
        c.runOnClient(mc->CaptureController.onKey(mc.getWindow().handle(),GLFW.GLFW_PRESS,new KeyEvent(key,0,0)));
        c.waitTick();
    }
    private static void save(ClientGameTestContext c)throws Exception {
        long request=c.computeOnClient(mc->bridge().saveCheckpoint());
        c.waitFor(mc->{long[] s=new long[3];bridge().readSaveStatus(s);return s[1]>=request && s[2]==3;},600);
    }
    private static String status(ClientGameTestContext c){return c.computeOnClient(mc->CaptureController.selfTrainingStatus()+" | "+CaptureController.inputState());}
    private static void running(ClientGameTestContext c){
        c.waitFor(mc->trainer().ready(),600);
        c.waitFor(mc->trainer().pending()>0 && CaptureController.inputState().appliedCommands()>0,300);
    }
    private static byte[][] frozenHashes(Path model)throws Exception {
        var offsets=CheckpointTestSupport.layout(model);int[] parts=offsets.length>22 ? new int[]{0,1,2,3,4,5,7,8,11,14,18,19,20,21}:new int[]{0,1,2,3,4,5,7,8,11,14,18,19,20};
        byte[][] hashes=new byte[parts.length][];
        for(int i=0;i<parts.length;i++){int part=parts[i];hashes[i]=CheckpointTestSupport.hash(model,offsets[part],offsets[part+1]-offsets[part]);}
        return hashes;
    }
    private static void freezeCheck(ClientGameTestContext c,String mode)throws Exception {
        c.waitTicks(6);save(c);Path model=Path.of(System.getProperty("vabpnn.model.path"));
        byte[][] frozen=frozenHashes(model);byte[] frozenMemory=java.security.MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(Path.of(model+".vdb")));
        var memoryBefore=MinecraftMemoryChecks.freezeBaseline(c);
        var before=rewardState(c);long frames=c.computeOnClient(mc->CaptureController.feedbackState(true).processed());
        c.waitTicks(35);var after=rewardState(c);
        require(before[2]==after[2]&&after[7]==0,mode+" freezes reward optimizer");
        require(c.computeOnClient(mc->CaptureController.feedbackState(true).processed())>frames,mode+" continues GPU inference");
        require(!c.computeOnClient(mc->CaptureController.selfTrainingArmed()),mode+" keeps explicit self-training disabled");
        require(c.computeOnClient(mc->trainer().exploration())==0,mode+" also disables random exploration");
        save(c);require(java.util.Arrays.deepEquals(frozen,frozenHashes(model)),mode+" freezes every neural/optimizer checkpoint part");
        require(java.util.Arrays.equals(frozenMemory,java.security.MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(Path.of(model+".vdb")))),mode+" freezes every external memory database byte");
        MinecraftMemoryChecks.verifyFrozen(c,memoryBefore);
    }
    public static void run(ClientGameTestContext c)throws Exception {
        try(var world=c.worldBuilder().create()){
            world.getClientLevel().waitForChunksDownload();world.getServer().runCommand("gamemode survival @p");
            world.getServer().runCommand("tp @p 0.5 -60 0.5 180 0");
            c.waitFor(mc->mc.player!=null&&mc.player.onGround(),200);
            c.runOnClient(mc->{mc.options.framerateLimit().set(60);mc.options.pauseOnLostFocus=false;});
            c.runOnClient(mc->CaptureController.startPipeline(true));c.waitTick();
            c.waitFor(mc->bridge()!=null&&bridge().phase()==NativeCudaBridge.READY&&CaptureController.feedbackState(true).processed()>0,600);
            Path model=Path.of(System.getProperty("vabpnn.model.path"));Path directory=model.toAbsolutePath().getParent();
            save(c);Path baseline=directory.resolve("game-baseline.bin");Files.copy(model,baseline);byte[] initialMemory=java.security.MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(Path.of(model+".vdb")));
            long initialExamples=c.computeOnClient(mc->CaptureController.feedbackState(true).examples());
            command(c,"explore on");running(c);
            boolean all=c.computeOnClient(mc->trainer().fullLearning());
            long initialJepa=c.computeOnClient(mc->CaptureController.feedbackState(true).jepaSteps());
            require(c.computeOnClient(mc->trainer().bootstrap() && trainer().exploration()==.2f),"real command enables configured bootstrap exploration");
            c.waitFor(mc->{long[] state=new long[12];bridge().readRewardStatus(state);return state[11]>0;},300);
            long explored=rewardState(c)[11];
            c.runOnClient(mc->CaptureController.setExploration(.3f,false));c.waitTick();running(c);
            require(c.computeOnClient(mc->!trainer().bootstrap() && trainer().exploration()==.3f),"retained internal strict policy preserves threshold candidates");
            command(c,"explore off");running(c);
            require(c.computeOnClient(mc->trainer().exploration())==0 && c.computeOnClient(mc->CaptureController.selfTrainingArmed()),"explore off retains normal scorer policy mode");
            command(c,"explore on");running(c);
            System.out.println("EXPLORATION_GAME commands/config/handshake/execution PASS; proposals="+explored);
            long start=rewardState(c)[2];boolean positive=false;
            for(int attempt=0;attempt<6&&!positive;attempt++){
                c.waitTicks(22);world.getServer().runCommand("give @p minecraft:oak_log 8");c.waitTicks(28);
                var s=rewardState(c);positive=s[2]>start&&Float.intBitsToFloat((int)s[10])>0;
                System.out.println("REWARD_GAME positive attempt="+attempt+" status="+status(c)+" native="+java.util.Arrays.toString(s));
            }
            require(positive,"actual positive Scorer telemetry reward reaches CUDA: "+status(c));
            long negativeStart=rewardState(c)[2];boolean negative=false;
            for(int attempt=0;attempt<4&&!negative;attempt++){
                c.waitTicks(24);world.getServer().runCommand("damage @p 2 minecraft:generic");c.waitTicks(28);
                var s=rewardState(c);negative=s[2]>negativeStart&&Float.intBitsToFloat((int)s[10])<0;
                System.out.println("REWARD_GAME negative attempt="+attempt+" status="+status(c)+" native="+java.util.Arrays.toString(s));
            }
            require(negative,"actual damage Scorer reward reaches CUDA: "+status(c));
            command(c,"selftrain off");c.waitTicks(4);
            save(c);
            if(all){
                long[] full=c.computeOnClient(mc->trainer().fullCounters());
                require(full[0]>0&&full[1]>0&&full[2]>0&&full[3]>0&&full[7]>0,"Scorer updates full seconds/PNN/minute/memory branches: "+java.util.Arrays.toString(full));
                require(c.computeOnClient(mc->CaptureController.feedbackState(true).jepaSteps())>initialJepa,"full learning resumes real JEPA self-supervision");
                require(!java.util.Arrays.equals(initialMemory,java.security.MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(Path.of(model+".vdb")))),"full learning writes external memory");
                var oldLayout=CheckpointTestSupport.layout(baseline);var newLayout=CheckpointTestSupport.layout(model);
                require(!java.util.Arrays.equals(CheckpointTestSupport.hash(baseline,oldLayout[1]+4,11732),CheckpointTestSupport.hash(model,newLayout[1]+4,11732)),"seconds multihead and feed-forward core really learns");
            }else{CheckpointTestSupport.frozen(baseline,model,true);require(java.util.Arrays.equals(initialMemory,java.security.MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(Path.of(model+".vdb")))),"explicit heads-only diagnosis freezes external memory");}
            require(c.computeOnClient(mc->CaptureController.feedbackState(true).examples())==initialExamples,"real reward loop leaves demo learner frozen");
            command(c,"explore on");running(c);
            key(c,GLFW.GLFW_KEY_F8);freezeCheck(c,"F8");
            command(c,"explore on");running(c);
            command(c,"selftrain off");freezeCheck(c,"prediction-only");
            command(c,"explore on");running(c);c.waitTicks(6);
            c.runOnClient(mc->CaptureController.onKey(mc.getWindow().handle(),GLFW.GLFW_PRESS,new KeyEvent(GLFW.GLFW_KEY_W,0,0)));
            c.waitFor(mc->!CaptureController.selfTrainingArmed() && CaptureController.inputState().predictionOnly(),80);
            freezeCheck(c,"manual takeover");
            command(c,"selftrain on");running(c);
            world.getServer().runCommand("effect give @p minecraft:instant_health 1 2 true");
            // Stop exploration rewards in this isolated fixture, keeping real sampled actions/receipts.
            // Otherwise a legitimate late ordinary-policy update may invalidate the death window.
            world.getServer().runCommand("effect give @p minecraft:slowness infinite 255 true");c.waitTicks(70);
            long deathBefore=rewardState(c)[2];System.out.println("REWARD_GAME death baseline updates="+deathBefore+" "+status(c));world.getServer().runCommand("kill @p");
            c.waitFor(mc->!mc.player.isAlive(),120);
            c.waitFor(mc->!CaptureController.selfTrainingArmed()&&CaptureController.inputState().predictionOnly(),160);
            var dead=rewardState(c);String reason=c.computeOnClient(mc->trainer().reason());
            System.out.println("REWARD_GAME death reason="+reason+" native="+java.util.Arrays.toString(dead));
            require(reason.equals("terminal_reward_settled") && dead[2]>deathBefore && Float.intBitsToFloat((int)dead[10])<0,"death window matched and GPU receipt settled before freeze");
            c.runOnClient(mc->mc.player.respawn());c.waitFor(mc->mc.player.isAlive(),200);c.runOnClient(mc->mc.setScreen(null));c.waitTicks(8);
            require(!c.computeOnClient(mc->CaptureController.selfTrainingArmed()),"respawn never implicitly re-enables reward learning");
            freezeCheck(c,"terminal respawn");
            command(c,"selftrain on");running(c);
            key(c,GLFW.GLFW_KEY_F9);c.waitTicks(22);var paused=rewardState(c);c.waitTicks(22);
            require(rewardState(c)[2]==paused[2]&&rewardState(c)[7]==0,"capture pause clears training receipts");
            key(c,GLFW.GLFW_KEY_F9);running(c);
            world.getServer().runCommand("kill @p");
            c.waitFor(mc->!mc.player.isAlive(),120);key(c,GLFW.GLFW_KEY_F9);
            var deathPaused=rewardState(c);c.waitTicks(70);
            require(rewardState(c)[2]==deathPaused[2] && rewardState(c)[7]==0,"F9 during death grace prevents terminal reward update");
            command(c,"selftrain off");
            c.runOnClient(mc->mc.player.respawn());c.waitFor(mc->mc.player.isAlive(),200);c.runOnClient(mc->mc.setScreen(null));
            key(c,GLFW.GLFW_KEY_F9);freezeCheck(c,"explicit off");
            save(c);
            command(c,"stop");
            System.out.println("PASS: actual Minecraft actions -> authoritative telemetry -> real Scorer positive/damage/death rewards -> policy CUDA updates; F8/predict/manual/respawn/pause/off protection; all requested network branches learn; evaluation remains byte-frozen.");
        }
    }
}
