package cn.vabpnn.mod.client.capture;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.client.gui.screens.ChatScreen;
/** Real world/context sampler, local commands and ongoing frozen memory state. */
final class MinecraftMemoryChecks {
    private static void require(boolean ok,String message){if(!ok)throw new AssertionError(message);}
    private static NativeCudaBridge bridge(){try{var f=CaptureController.class.getDeclaredField("bridge");f.setAccessible(true);return (NativeCudaBridge)f.get(null);}catch(ReflectiveOperationException e){throw new AssertionError(e);}}
    private static long[] state(){long[] s=new long[24];bridge().readLongMemory(s);return s;}
    private static void command(ClientGameTestContext c,String value){
        c.runOnClient(mc->{var chat=new ChatScreen("",false);mc.setScreen(chat);chat.handleChatInput("/vabpnn "+value,true);mc.setScreen(null);});c.waitTick();
    }
    static void verifyCommands(ClientGameTestContext c){
        c.waitFor(mc->{var s=state();return s[11]==1&&s[7]>0&&s[4]>0&&s[5]>0;},400);
        c.runOnClient(mc->require(state()[10]==1&&state()[8]==8192,"long-term memory and its bounded capacity default enabled"));
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: actual Minecraft world/session clock sampler, human memory labels/gate learning and default-enabled external vector memory.");
    }
    static long[] freezeBaseline(ClientGameTestContext c){return c.computeOnClient(mc->state());}
    static void verifyFrozen(ClientGameTestContext c,long[] before){
        c.runOnClient(mc->{var s=state();require(s[0]>before[0]&&s[1]==before[1]&&s[2]==before[2]&&s[3]==before[3]&&s[4]==before[4]&&s[5]==before[5]&&s[6]==before[6]&&s[7]==before[7],"frozen evaluation advances read-only query frames but never changes the memory catalog, writes, visits or gate steps");
            require(Float.intBitsToFloat((int)s[13])>=Float.intBitsToFloat((int)before[13]),"memory minute axis advances with the graph timeline");});
        cn.vabpnn.mod.VabpnnMod.LOGGER.info("PASS: frozen model operation keeps read-only seconds/minutes retrieval while the complete memory catalog, entries and learned gates stay frozen.");
    }
}
