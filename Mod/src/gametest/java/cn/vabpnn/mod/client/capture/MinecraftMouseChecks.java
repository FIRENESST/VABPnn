package cn.vabpnn.mod.client.capture;
import java.lang.reflect.*;import java.nio.file.*;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.client.Minecraft;import net.minecraft.client.gui.screens.ChatScreen;import net.minecraft.client.input.KeyEvent;
import org.lwjgl.glfw.GLFW;
public final class MinecraftMouseChecks {
    static void require(boolean v,String s){if(!v)throw new AssertionError(s);}
    static void command(ClientGameTestContext c,String v){c.runOnClient(mc->{mc.setScreen(new ChatScreen("",false));((ChatScreen)mc.screen).handleChatInput("/vabpnn "+v,true);mc.setScreen(null);});c.waitTick();}
    static void focus(Minecraft mc,boolean v)throws Exception{var f=mc.getWindow().getClass().getDeclaredField("focused");f.setAccessible(true);f.setBoolean(mc.getWindow(),v);}
    static boolean advised(){var v=CaptureController.mouseTelemetry().suggestion();return v.fresh(System.nanoTime())&&v.paired()&&v.enabled()&&Math.abs(v.dx())>.01&&Math.abs(v.dy())>.01;}
    static NativeCudaBridge bridge()throws Exception{var f=CaptureController.class.getDeclaredField("bridge");f.setAccessible(true);return (NativeCudaBridge)f.get(null);}
    static void save(ClientGameTestContext c)throws Exception{
        long request=c.computeOnClient(mc->{try{return bridge().saveCheckpoint();}catch(Exception e){throw new AssertionError(e);}});
        c.waitFor(mc->{try{long[] s=new long[3];bridge().readSaveStatus(s);return s[1]>=request&&s[2]==3;}catch(Exception e){throw new AssertionError(e);}},600);
    }
    public static void run(ClientGameTestContext c)throws Exception {
        Path model=Path.of(System.getProperty("vabpnn.model.path")),reference=model.getParent().resolve("mouse-frozen-baseline.bin");
        try(var world=c.worldBuilder().create()){
            world.getClientLevel().waitForChunksDownload();world.getServer().runCommand("gamemode survival @p");world.getServer().runCommand("tp @p 0.5 -60 0.5 0 0");
            c.waitFor(mc->mc.player!=null&&mc.player.onGround(),200);
            c.runOnClient(mc->{mc.options.framerateLimit().set(60);mc.options.pauseOnLostFocus=false;mc.options.smoothCamera=false;mc.options.sensitivity().set(.5);});
            c.runOnClient(mc->CaptureController.startPipeline(true));c.waitTick();c.waitFor(mc->advised(),600);save(c);Files.copy(model,reference);
            var before=c.computeOnClient(mc->CaptureController.feedbackState(true));
            final float yaw=c.computeOnClient(mc->mc.player.getYRot()),pitch=c.computeOnClient(mc->mc.player.getXRot());
            try{
                c.runOnClient(mc->{try{focus(mc,false);}catch(Exception e){throw new AssertionError(e);}});
                c.waitTicks(6);c.waitFor(mc->!CaptureController.inputState().allowed()&&advised(),300);c.waitTicks(20);
                c.runOnClient(mc->{var s=CaptureController.mouseTelemetry().suggestion();
                    require(s.fresh(System.nanoTime())&&s.paired()&&s.enabled()&&Math.abs(s.dx())>.01&&Math.abs(s.dy())>.01,"inactive prediction-only retains real GPU advice");
                    require(CaptureController.inputState().appliedCommands()==before.input().appliedCommands()&&mc.player.getYRot()==yaw&&mc.player.getXRot()==pitch,"inactive prediction does not move camera");
                    System.out.println("MOUSE_GAME inactive advice="+s+" camera unchanged");
                });
            }finally{c.runOnClient(mc->{try{focus(mc,true);}catch(Exception e){throw new AssertionError(e);}});}
            c.waitFor(mc->advised(),300);save(c);CheckpointTestSupport.frozen(reference,model,false);
            c.runOnClient(mc->CaptureController.onKey(mc.getWindow().handle(),GLFW.GLFW_PRESS,new KeyEvent(GLFW.GLFW_KEY_F8,0,0)));
            c.waitFor(mc->CaptureController.inputState().allowed()&&CaptureController.inputState().frame()>0,300);
            c.waitFor(mc->Math.abs(mc.player.getYRot()-yaw)>.1&&Math.abs(mc.player.getXRot()-pitch)>.1,300);
            c.runOnClient(mc->{require(CaptureController.inputState().modelControl()&&(CaptureController.inputState().selectedMask()&96)==96,"real GPU axes reached production F8 controller");
                System.out.println("MOUSE_GAME F8 camera yaw="+mc.player.getYRot()+" pitch="+mc.player.getXRot()+" state="+CaptureController.inputState());
            });
            try {
                c.runOnClient(mc->{try{focus(mc,false);}catch(Exception e){throw new AssertionError(e);}});
                c.waitTicks(6);c.waitFor(mc->advised()&&!CaptureController.inputState().allowed(),300);
                final float backgroundYaw=c.computeOnClient(mc->mc.player.getYRot()),backgroundPitch=c.computeOnClient(mc->mc.player.getXRot());
                long applied=c.computeOnClient(mc->CaptureController.inputState().appliedCommands());c.waitTicks(20);
                c.runOnClient(mc->require(advised()&&mc.player.getYRot()==backgroundYaw&&mc.player.getXRot()==backgroundPitch
                    &&CaptureController.inputState().appliedCommands()==applied,"inactive F8 keeps advice and blocks real camera input"));
                c.runOnClient(mc->CaptureController.setPredictionOnly(true));c.waitTick();
            } finally {c.runOnClient(mc->{try{focus(mc,true);}catch(Exception e){throw new AssertionError(e);}});}
            c.waitFor(mc->advised(),300);
            final float frozenYaw=c.computeOnClient(mc->mc.player.getYRot()),frozenPitch=c.computeOnClient(mc->mc.player.getXRot());c.waitTicks(20);
            c.runOnClient(mc->require(mc.player.getYRot()==frozenYaw&&mc.player.getXRot()==frozenPitch&&!CaptureController.inputState().allowed(),"prediction stops actual camera while advice stays visible"));
            save(c);CheckpointTestSupport.frozen(reference,model,false);command(c,"stop");
            System.out.println("PASS: inactive/foreground prediction advice, actual GPU F8 camera movement, mode switch stop, neural/checkpoint optimizer freeze.");
        }
    }
}
