package cn.vabpnn.mod.client.capture;
import java.nio.file.*;
public final class MouseFixtureBuilder {
    public static void main(String[] args)throws Exception{
        Path source=Path.of(args[0]),target=Path.of(args[1]);
        byte[] before=CheckpointTestSupport.hash(source,0,Files.size(source));
        CheckpointTestSupport.seedMousePolicy(source,target);
        if(!java.util.Arrays.equals(before,CheckpointTestSupport.hash(source,0,Files.size(source))))throw new AssertionError("fixture source changed");
        System.out.println("PASS: isolated constant mouse head/allowed-axis fixture built; original checkpoint exact. "+target);
    }
}
