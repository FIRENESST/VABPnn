package cn.vabpnn.mod.client.capture;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.UUID;
import net.minecraft.client.Minecraft;
import net.minecraft.world.level.storage.LevelResource;

/** Only world identity and two clocks cross JNI; no images or learned vectors. */
final class MemoryContextSampler {
    private Object level,player;
    private UUID world,session;
    private final long[] values=new long[6];
    void update(Minecraft client,NativeCudaBridge bridge) {
        if(bridge==null)return;
        if(client.level==null||client.player==null||!client.player.isAlive()) {
            java.util.Arrays.fill(values,0);level=player=null;bridge.memoryContext(values);return;
        }
        if(level!=client.level||player!=client.player){
            level=client.level;player=client.player;session=UUID.randomUUID();
            String identity;
            if(client.getSingleplayerServer()!=null){
                Path path=client.getSingleplayerServer().getWorldPath(LevelResource.ROOT).toAbsolutePath().normalize();
                identity="singleplayer:"+path;
            }else identity="server:"+(client.getCurrentServer()==null ? "unidentified":client.getCurrentServer().ip);
            identity+="|"+client.level.dimension().identifier()+"|"+client.player.getUUID();
            world=UUID.nameUUIDFromBytes(identity.getBytes(StandardCharsets.UTF_8));
        }
        values[0]=world.getMostSignificantBits();values[1]=world.getLeastSignificantBits();
        values[2]=session.getMostSignificantBits();values[3]=session.getLeastSignificantBits();
        values[4]=System.nanoTime();values[5]=System.currentTimeMillis();
        bridge.memoryContext(values);
    }
    void reset(){level=player=null;world=session=null;java.util.Arrays.fill(values,0);}
}
