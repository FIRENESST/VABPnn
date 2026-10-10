package cn.vabpnn.mod.client.capture;

import cn.vabpnn.mod.VabpnnMod;
import cn.vabpnn.mod.mixin.client.MouseHandlerAccess;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.inventory.InventoryScreen;
import net.minecraft.client.input.KeyEvent;
import net.minecraft.client.input.MouseButtonEvent;
import net.minecraft.client.input.MouseButtonInfo;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.Blocks;
import org.lwjgl.glfw.GLFW;

/** Real callbacks and game state in a separate temporary singleplayer save. No screenshots. */
public final class MinecraftActionsGameTest implements FabricClientGameTest {
    private final ActionExecutor executor=new ActionExecutor();
    private MinecraftInputSink sink;
    private long frame=1;
    private static void require(boolean ok,String message) { if(!ok) throw new AssertionError(message); }
    private void send(ClientGameTestContext context,int mask,float dx,float dy,float wheel) {
        context.runOnClient(client -> {
            long now=System.nanoTime();
            executor.apply(new long[]{frame++,now,mask,1,Float.floatToRawIntBits(dx),
                    Float.floatToRawIntBits(dy),Float.floatToRawIntBits(wheel)},now,true,sink);
        });
    }
    private void hotkey(ClientGameTestContext context,int key) {
        context.runOnClient(client -> {
            CaptureController.onKey(client.getWindow().handle(),
                    GLFW.GLFW_PRESS,new KeyEvent(key,GLFW.glfwGetKeyScancode(key),0));
            MinecraftFeedbackChecks.verifyHotkey(client,key);
        });
    }
    @Override public void runTest(ClientGameTestContext context) {
        try(var world=context.worldBuilder().create()) {
            world.getClientLevel().waitForChunksDownload();
            world.getServer().runCommand("gamemode survival @p");
            world.getServer().runCommand("tp @p 0.5 -60 0.5 0 0");
            context.waitFor(client -> client.player!=null && client.player.onGround(),200);
            context.runOnClient(client -> {
                sink=new MinecraftInputSink(client);
                client.options.sensitivity().set(.5);
                client.options.smoothCamera=false;
                client.options.invertMouseX().set(false);
                client.options.invertMouseY().set(false);
                require(!CaptureController.inputState().modelControl(),"startup remains demonstration mode");
            });
            MinecraftFeedbackChecks.verify(context);
            verifyPlayerPosition(context);
            // A live GPU packet must reach this world's production controller, not just a fake sink.
            long positionSteps=context.computeOnClient(client -> CaptureController.positionState().positionSteps());
            hotkey(context,GLFW.GLFW_KEY_F8);
            context.runOnClient(client -> require(CaptureController.positionState().positionSteps()>=positionSteps,
                    "F8 history reset retains the published coordinate training count"));
            context.waitFor(client -> CaptureController.inputState().allowed()
                    && CaptureController.inputState().frame()>0
                    && CaptureController.inputState().appliedCommands()>0,600);
            MinecraftFeedbackChecks.verifyEmptyModel(context);
            // Feed deterministic nonempty packets through the SAME production controller as JNI.
            context.runOnClient(client -> {
                long id=CaptureController.inputState().frame()+1,now=System.nanoTime();
                CaptureController.applyModelCommand(new long[]{id++,now,1,1,0,0,0},now);
                require(CaptureController.inputState().allowed() && client.options.keyUp.isDown()
                        && CaptureController.inputState().selectedMask()==1,"F8 production path injects nonempty W");
                CaptureController.applyModelCommand(new long[]{id++,now,1<<7,1,0,0,0},now);
                require(client.mouseHandler.isLeftPressed() && client.options.keyAttack.isDown(),
                        "F8 production path injects left press");
                CaptureController.applyModelCommand(new long[]{id++,now,1<<9,1,0,0,0},now);
                require(!client.mouseHandler.isLeftPressed() && !client.options.keyAttack.isDown(),
                        "F8 production path injects left release");
                CaptureController.applyModelCommand(new long[]{id++,now,1<<8,1,0,0,0},now);
                require(client.mouseHandler.isRightPressed() && client.options.keyUse.isDown(),
                        "F8 production path injects right press");
                CaptureController.applyModelCommand(new long[]{id++,now,1<<10,1,0,0,0},now);
                require(!client.mouseHandler.isRightPressed() && !client.options.keyUse.isDown(),
                        "F8 production path injects right release");
            });
            hotkey(context,GLFW.GLFW_KEY_F9);
            context.runOnClient(client -> {
                var state=CaptureController.inputState();
                require(!state.allowed() && state.frame()==0 && state.heldKeys()==0 && state.heldButtons()==0,
                        "F9 releases actual production control immediately");
            });
            hotkey(context,GLFW.GLFW_KEY_F8); // back to demonstration before deterministic fixtures
            hotkey(context,GLFW.GLFW_KEY_F9);
            verifyInventoryDemonstrations(context);
            for(int k=0;k<4;k++) {
                double x=context.computeOnClient(client -> client.player.getX());
                double z=context.computeOnClient(client -> client.player.getZ());
                final int action=k;
                send(context,1<<k,0,0,0);
                context.runOnClient(client -> require(switch(action) {
                    case 0 -> client.options.keyUp.isDown();
                    case 1 -> client.options.keyLeft.isDown();
                    case 2 -> client.options.keyDown.isDown();
                    default -> client.options.keyRight.isDown();
                },"W/A/S/D is the actual game key mapping"));
                context.waitTicks(8);
                context.runOnClient(client -> require(Math.hypot(client.player.getX()-x,client.player.getZ()-z)>.05,
                        "movement object moves the actual player"));
                send(context,0,0,0,0);
            }
            send(context,15,0,0,0);
            context.runOnClient(client -> require(client.options.keyUp.isDown() && client.options.keyDown.isDown()
                    && client.options.keyLeft.isDown() && client.options.keyRight.isDown(),"opposed movement remains permitted"));
            send(context,0,0,0,0);
            context.waitFor(client -> client.player.onGround(),60);
            double ground=context.computeOnClient(client -> client.player.getY());
            send(context,1<<12,0,0,0);
            context.waitFor(client -> client.player.getY()>ground+.05,30);
            send(context,0,0,0,0);

            send(context,1<<4,0,0,0);
            context.waitForScreen(InventoryScreen.class);
            send(context,0,0,0,0);
            send(context,1<<4,0,0,0);
            context.waitForScreen(null);
            send(context,0,0,0,0);
            context.waitTicks(3);
            float yaw=context.computeOnClient(client -> client.player.getYRot());
            float pitch=context.computeOnClient(client -> client.player.getXRot());
            send(context,(1<<5)|(1<<6),10,-8,0);
            context.waitTick();
            context.runOnClient(client -> require(Math.abs(client.player.getYRot()-yaw)>.1
                    && Math.abs(client.player.getXRot()-pitch)>.1,"X/Y permissions turn the actual camera"));

            send(context,1<<7,0,0,0);
            context.runOnClient(client -> require(client.mouseHandler.isLeftPressed() && client.options.keyAttack.isDown(),
                    "left press reaches vanilla attack input"));
            send(context,1<<9,0,0,0);
            context.runOnClient(client -> require(!client.mouseHandler.isLeftPressed() && !client.options.keyAttack.isDown(),
                    "left release clears vanilla attack input"));
            send(context,1<<8,0,0,0);
            context.runOnClient(client -> require(client.mouseHandler.isRightPressed() && client.options.keyUse.isDown(),
                    "right press reaches vanilla use input"));
            send(context,1<<10,0,0,0);
            context.runOnClient(client -> require(!client.mouseHandler.isRightPressed() && !client.options.keyUse.isDown(),
                    "right release clears vanilla use input"));
            int slot=context.computeOnClient(client -> client.player.getInventory().getSelectedSlot());
            send(context,1<<11,0,0,1);
            context.runOnClient(client -> require(client.player.getInventory().getSelectedSlot()!=slot,
                    "wheel permission changes the actual hotbar slot"));
            send(context,0,0,0,0);

            // A single legal press+release must still cause a real creative block attack.
            world.getServer().runCommand("gamemode creative @p");
            world.getServer().runCommand("tp @p 0.5 -60 0.5 0 0");
            world.getServer().runCommand("setblock 0 -59 3 minecraft:stone");
            context.waitTicks(4);
            send(context,(1<<7)|(1<<9),0,0,0);
            context.waitFor(client -> client.level.getBlockState(new BlockPos(0,-59,3)).isAir(),80);

            world.getServer().runCommand("gamemode survival @p");
            context.waitFor(client -> !client.player.getAbilities().instabuild,40);

            // Use a vanilla inventory subclass to observe the click coordinate without reading pixels.
            context.runOnClient(client -> client.setScreen(new ProbeInventory(client.player)));
            context.waitTick();
            double oldX=context.computeOnClient(client -> ((MouseHandlerAccess)client.mouseHandler).vabpnn$getXpos());
            double oldY=context.computeOnClient(client -> ((MouseHandlerAccess)client.mouseHandler).vabpnn$getYpos());
            send(context,(1<<5)|(1<<6)|(1<<7)|(1<<9),18,-12,0);
            context.runOnClient(client -> {
                ProbeInventory screen=(ProbeInventory)client.screen;
                double expectedX=Math.clamp(oldX+18,0,client.getWindow().getScreenWidth()-1)
                        *client.getWindow().getGuiScaledWidth()/client.getWindow().getScreenWidth();
                double expectedY=Math.clamp(oldY-12,0,client.getWindow().getScreenHeight()-1)
                        *client.getWindow().getGuiScaledHeight()/client.getWindow().getScreenHeight();
                require(Math.abs(screen.clickX-expectedX)<.01 && Math.abs(screen.clickY-expectedY)<.01,
                        "same packet cursor motion precedes the real inventory click");
                require(!client.options.keyAttack.isDown() && !client.mouseHandler.isLeftPressed(),
                        "inventory click/release leaves no world attack hold");
                executor.stop(sink);
                require(!client.options.keyAttack.consumeClick(),"stop cancels pending owned click counters");
                client.setScreen(null);
            });
            context.waitTick();

            // Vanilla can reset a hold during focus restoration; recover state without another edge.
            send(context,1|(1<<7),0,0,0);
            context.runOnClient(client -> {
                client.options.keyUp.setDown(false); client.options.keyAttack.setDown(false);
                sink.maintainHeldInputs(executor.heldKeys(),executor.heldButtons());
                require(client.options.keyUp.isDown() && client.options.keyAttack.isDown(),"held logical inputs recover");
                VabpnnMod.LOGGER.info("Input cleanup before screen: {}",cleanupState(client,sink));
                client.setScreen(new ProbeInventory(client.player));
                executor.stop(sink);
                VabpnnMod.LOGGER.info("Input cleanup after stop: {}",cleanupState(client,sink));
                require(!client.options.keyUp.isDown() && !client.options.keyAttack.isDown()
                        && !client.mouseHandler.isLeftPressed(),"stop inside inventory clears prior world holds");
                require(!client.options.keyAttack.consumeClick(),"world attack edge cannot leak after inventory closes");
                client.setScreen(null);
            });
            context.runOnClient(client -> {
                client.options.toggleAttack().set(true);
                client.options.toggleUse().set(true);
            });
            send(context,1<<7,0,0,0);
            context.runOnClient(client -> {
                sink.maintainHeldInputs(executor.heldKeys(),executor.heldButtons());
                sink.maintainHeldInputs(executor.heldKeys(),executor.heldButtons());
                require(client.options.keyAttack.isDown(),"toggle attack option does not flip a model hold");
            });
            send(context,1<<9,0,0,0);
            context.runOnClient(client -> require(!client.options.keyAttack.isDown(),"explicit release works with toggle attack enabled"));
            send(context,1<<8,0,0,0);
            context.runOnClient(client -> {
                sink.maintainHeldInputs(executor.heldKeys(),executor.heldButtons());
                sink.maintainHeldInputs(executor.heldKeys(),executor.heldButtons());
                require(client.options.keyUse.isDown(),"toggle use option does not flip a model hold");
            });
            send(context,1<<10,0,0,0);
            context.runOnClient(client -> require(!client.options.keyUse.isDown(),"explicit release works with toggle use enabled"));
            context.runOnClient(client -> {
                executor.stop(sink);
                client.options.toggleAttack().set(false); client.options.toggleUse().set(false);
            });
            // Verify the whole live pipeline resumes, then control is released before leaving the world.
            hotkey(context,GLFW.GLFW_KEY_F8);
            context.waitFor(client -> CaptureController.inputState().allowed() && CaptureController.inputState().frame()>0,600);
            hotkey(context,GLFW.GLFW_KEY_F8);
            context.runOnClient(client -> require(!CaptureController.inputState().modelControl()
                    && CaptureController.inputState().frame()==0,"F8 returns control to the player"));
            VabpnnMod.LOGGER.info("PASS: live GPU commands applied in a real world; all 13 objects, player movement/jump, inventory, camera, attack/release, hotbar, GUI click position, held-input recovery and F8/F9 releases.");
            MinecraftFeedbackChecks.verifyStopRestart(context);
        } finally {
            if(sink!=null) context.runOnClient(client -> executor.stop(sink));
        }
        MinecraftFeedbackChecks.verifyDisconnect(context);
    }
    private static void verifyPlayerPosition(ClientGameTestContext context) {
        context.waitFor(client -> {
            var p=CaptureController.positionState();
            return p.valid() && p.gpuSampleNanos()>0 && p.positionSteps()>0;
        },240);
        final long[] initial=context.computeOnClient(client -> {
            var p=CaptureController.positionState();
            require(Math.abs(p.x()-client.player.getX())<1 && Math.abs(p.y()-client.player.getY())<1
                    && Math.abs(p.z()-client.player.getZ())<1,"sampled XYZ comes from the actual LocalPlayer");
            return new long[]{p.samples(),System.nanoTime(),p.epoch()};
        });
        context.waitTicks(20);
        context.runOnClient(client -> {
            var p=CaptureController.positionState();
            long count=p.samples()-initial[0],elapsed=System.nanoTime()-initial[1];
            require(count>0 && count*PlayerPositionSampler.PERIOD_NANOS<=elapsed+2*PlayerPositionSampler.PERIOD_NANOS,
                    "live render callbacks sample coordinates without exceeding 30 Hz");
            require(p.valid() && p.epoch()==initial[2] && p.sampleNanos()>=p.gpuSampleNanos(),
                    "CUDA consumes causal coordinates from this world");
            VabpnnMod.LOGGER.info("PASS: real Minecraft XYZ at target=30 Hz, measured={} Hz; coordinate sample={} reaches GPU prediction, adapterSteps={}.",
                    count*1e9/elapsed,p.gpuSampleNanos(),p.positionSteps());
        });
    }
    private static void verifyInventoryDemonstrations(ClientGameTestContext context) {
        context.runOnClient(client -> client.setScreen(new ProbeInventory(client.player)));
        context.waitTicks(3);
        context.waitFor(client -> CaptureController.feedbackState(true).processed()>0,120);
        for(int button=0;button<2;button++) {
            final int index=button;
            var before=context.computeOnClient(client -> CaptureController.humanInputState());
            long examples=context.computeOnClient(client -> CaptureController.feedbackState(true).examples());
            context.runOnClient(client -> {
                ((MouseHandlerAccess)client.mouseHandler).vabpnn$button(client.getWindow().handle(),
                        new MouseButtonInfo(index,0),GLFW.GLFW_PRESS);
                var human=CaptureController.humanInputState();
                require((human.heldButtons()&(1<<index))!=0,"inventory click records physical hold even when vanilla consumes it");
                require((index==0 ? human.leftPress()-before.leftPress():human.rightPress()-before.rightPress())==1,
                        "inventory real callback increments its human press counter");
                require(!client.mouseHandler.isLeftPressed() && !client.mouseHandler.isRightPressed(),
                        "vanilla world pressed flags do not report consumed inventory clicks");
            });
            context.waitFor(client -> {
                var state=CaptureController.feedbackState(true);
                return state.examples()>examples && (state.lastHumanLabel()&(1<<(7+index)))!=0;
            },160);
            context.runOnClient(client -> {
                ((MouseHandlerAccess)client.mouseHandler).vabpnn$button(client.getWindow().handle(),
                        new MouseButtonInfo(index,0),GLFW.GLFW_RELEASE);
                var human=CaptureController.humanInputState();
                require((human.heldButtons()&(1<<index))==0,"inventory release clears the recorded hold");
                require((index==0 ? human.leftRelease()-before.leftRelease():human.rightRelease()-before.rightRelease())==1,
                        "inventory release is separately recorded");
            });
            context.waitTicks(3);
        }
        MinecraftFeedbackChecks.verifySilentInventoryFeedback(context);
        context.runOnClient(client -> client.setScreen(null));
        context.waitTick();
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: real inventory-consumed left/right callbacks, holds and releases are recorded; both press labels reach online GPU learning without inventory chat notifications.");
    }
    private static String cleanupState(Minecraft client,MinecraftInputSink sink) {
        try {
        var counter=net.minecraft.client.KeyMapping.class.getDeclaredField("clickCount"); counter.setAccessible(true);
        var touched=MinecraftInputSink.class.getDeclaredField("touchedButtons"); touched.setAccessible(true);
        var key=((cn.vabpnn.mod.mixin.client.KeyMappingAccess)client.options.keyAttack).vabpnn$boundKey();
        return "clicks="+counter.getInt(client.options.keyAttack)+", touched="+touched.getInt(sink)
                +", nativeButton="+GLFW.glfwGetMouseButton(client.getWindow().handle(),0)
                +", inMappings="+java.util.Arrays.asList(client.options.keyMappings).contains(client.options.keyAttack)
                +", key="+key.getType()+":"+key.getValue();
        } catch(ReflectiveOperationException e) { throw new AssertionError(e); }
    }
    private static final class ProbeInventory extends InventoryScreen {
        double clickX=Double.NaN,clickY=Double.NaN;
        ProbeInventory(LocalPlayer player) { super(player); }
        @Override public boolean mouseClicked(MouseButtonEvent event,boolean doubleClick) {
            clickX=event.x(); clickY=event.y();
            return super.mouseClicked(event,doubleClick);
        }
    }
}
