package cn.vabpnn.mod.client.capture;
import java.nio.*;import java.nio.channels.*;import java.nio.file.*;import java.security.*;import java.util.*;
/** Isolated weight fixtures only. Never accepts a path outside this verification build tree. */
public final class CheckpointTestSupport {
    public static ByteBuffer read(Path path,long offset,int count)throws Exception {
        try(var in=FileChannel.open(path,StandardOpenOption.READ)){var b=ByteBuffer.allocate(count).order(ByteOrder.LITTLE_ENDIAN);
            while(b.hasRemaining())if(in.read(b,offset+b.position())<=0)throw new AssertionError("Incomplete fixture");return b.flip();}
    }
    public static long[] layout(Path path)throws Exception {
        var h=read(path,0,256);if(h.getLong(8)!=11||h.getLong(16)!=24||h.getLong(32+23*8)!=8200)throw new AssertionError("v11 reward layout");
        var out=new long[25];out[0]=256;for(int i=0;i<24;i++)out[i+1]=out[i]+h.getLong(32+i*8);
        if(out[24]!=Files.size(path))throw new AssertionError("Fixture size");return out;
    }
    public static void seedPolicy(Path source,Path target)throws Exception {
        Path build=Path.of("build").toAbsolutePath().normalize();target=target.toAbsolutePath().normalize();
        if(!target.startsWith(build)||source.toAbsolutePath().normalize().equals(target))throw new AssertionError("Isolated fixture path");
        Files.createDirectories(target.getParent());Files.copy(source,target);
        var offsets=layout(target);if(offsets[2]-offsets[1]!=17028)throw new AssertionError("Verified CUDA decision ABI");
        var heads=read(target,offsets[1],17028);heads.putFloat(0,3);
        for(int a=0;a<15;a++){
            for(int d=0;d<16;d++){heads.putFloat(11736+(a*16+d)*4,0);heads.putFloat(12816+(a*16+d)*4,0);}
            heads.putFloat(12696+a*4,a<2?4:-10);heads.putFloat(12756+a*4,0);
            heads.putFloat(13776+a*4,a<2?-10:10);heads.putFloat(16844+a*4,0);
        }
        for(int k=0;k<16;k++){for(int f=0;f<46;f++)heads.putFloat(13836+(k*46+f)*4,0);heads.putFloat(16780+k*4,k==1?.5f:k<3?0:-20);}
        try(var out=FileChannel.open(target,StandardOpenOption.WRITE)){
            while(heads.hasRemaining())out.write(heads,offsets[1]+heads.position());
            var hash=hash(target,256,offsets[24]-256);out.write(ByteBuffer.wrap(hash),224);
        }
    }
    public static void seedMousePolicy(Path source,Path target)throws Exception {
        seedPolicy(source,target);var o=layout(target);
        if(o[3]-o[2]!=510722L*4 || o[19]-o[18]!=14216)throw new AssertionError("Verified mouse/clock ABI");
        var heads=read(target,o[1],17028);heads.putFloat(0,3);
        for(int a=0;a<15;a++){heads.putFloat(12696+a*4,a==5||a==6?10:-10);heads.putFloat(13776+a*4,a==5||a==6?-10:30);}
        for(int k=0;k<16;k++)heads.putFloat(16780+k*4,k==2?10:-10);
        var motion=ByteBuffer.allocate(514*4).order(ByteOrder.LITTLE_ENDIAN);
        motion.putFloat(512*4,.2f);motion.putFloat(513*4,-.1f);
        var residual=ByteBuffer.allocate(66*4);
        try(var out=FileChannel.open(target,StandardOpenOption.WRITE)){
            while(heads.hasRemaining())out.write(heads,o[1]+heads.position());
            while(motion.hasRemaining())out.write(motion,o[3]-514*4+motion.position());
            while(residual.hasRemaining())out.write(residual,o[19]-800+residual.position());
            out.write(ByteBuffer.wrap(hash(target,256,o[24]-256)),224);
        }
    }
    public static byte[] hash(Path p,long offset,long length)throws Exception {
        var h=MessageDigest.getInstance("SHA-256");
        try(var in=FileChannel.open(p,StandardOpenOption.READ)){
            var b=ByteBuffer.allocate(1<<20);for(long n=0;n<length;){b.clear();b.limit((int)Math.min(b.capacity(),length-n));
                while(b.hasRemaining())if(in.read(b,offset+n+b.position())<=0)throw new AssertionError("Fixture hash read");
                n+=b.position();b.flip();h.update(b);}
        }return h.digest();
    }
    public static void frozen(Path a,Path b,boolean policyMayChange)throws Exception {
        var x=layout(a);var y=layout(b);if(!Arrays.equals(x,y))throw new AssertionError("Fixture layout changed");
        int[] frozen={0,1,2,3,4,5,7,8,11,14,18,19,20,21,22,23};
        for(int part:frozen)if(!(policyMayChange&&(part==1||part==23)))
            if(!Arrays.equals(hash(a,x[part],x[part+1]-x[part]),hash(b,y[part],y[part+1]-y[part])))
                throw new AssertionError("Neural/optimizer freeze part "+part);
        if(policyMayChange){
            byte[] before=read(a,x[1],17028).array(),after=read(b,y[1],17028).array();
            for(int i=0;i<17028;i++){boolean allowed=i>=11736&&i<12816||i>=13836&&i<16844;
                if(!allowed&&before[i]!=after[i])throw new AssertionError("Reward changed unapproved decision byte "+i);}
            if(Arrays.equals(before,after))throw new AssertionError("Reward never changed actual policy weights");
        }
    }
    public static long[] rewardSteps(Path p)throws Exception {long offset=layout(p)[23];var bytes=read(p,offset+8176,16);return new long[]{bytes.getLong(0),bytes.getLong(8)};}
}
