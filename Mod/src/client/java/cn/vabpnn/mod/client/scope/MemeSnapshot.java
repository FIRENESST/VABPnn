package cn.vabpnn.mod.client.scope;

/** One completed CUDA frame, immutable across Minecraft and Swing threads. No image data. */
public final class MemeSnapshot {
    public static final int TILES=120*64,CHANNELS=15,META=40;
    public static final int OWN=0,IMPORTED=TILES,AGE=2*TILES,CONFIDENCE=3*TILES;
    public static final int KNOWLEDGE=4*TILES,COMPATIBILITY=5*TILES,DETAIL=6*TILES,PACKET=28;
    public static final int VISUAL=DETAIL+3*PACKET,FEATURES=VISUAL+8,ACTION=FEATURES+CHANNELS;
    public static final int CONFIG=ACTION+CHANNELS,FLOATS=CONFIG+10;
    public static final int ORIGIN=0,HOPS=TILES,SENDER=2*TILES,CHANNEL=3*TILES,INTS=4*TILES,LONGS=2*TILES;
    private final long[] meta,identities;
    private final float[] values;
    private final int[] discrete;
    private final long receivedNanos;
    private final int activeLocal,activeImported;

    public MemeSnapshot(long[] meta,float[] values,int[] discrete,long[] identities,long receivedNanos) {
        if(meta.length!=META || values.length!=FLOATS || discrete.length!=INTS || identities.length!=LONGS)
            throw new IllegalArgumentException("Meme snapshot shape");
        if(meta[0]<=0 || meta[3]<0 || meta[3]>=TILES || meta[39]!=1)
            throw new IllegalArgumentException("Meme snapshot frame/tile/schema");
        this.meta=meta.clone();this.values=values.clone();this.discrete=discrete.clone();
        this.identities=identities.clone();this.receivedNanos=receivedNanos;
        int local=0,imported=0;
        for(int tile=0;tile<TILES;tile++) { if(value(OWN+tile)>0) local++;if(value(IMPORTED+tile)>0) imported++; }
        activeLocal=local;activeImported=imported;
    }
    public long frame() { return meta[0]; }
    public int tile() { return (int)meta[3]; }
    public long meta(int index) { return meta[index]; }
    public float value(int index) { float v=values[index];return Float.isFinite(v) ? v:0; }
    public int integer(int index) { return discrete[index]; }
    public long identity(int index) { return identities[index]; }
    public int activeLocal() { return activeLocal; }
    public int activeImported() { return activeImported; }
    public long receivedNanos() { return receivedNanos; }
    public boolean stale(long now) { return now-receivedNanos>750_000_000L; }
    public boolean selected(int tile) { return tile()==tile; }
    public boolean enabled() { return meta[4]!=0 && meta[5]!=0; }
    public static String block(long tile) {
        return tile<0 || tile>=TILES ? "未知":("#"+tile+" ("+(tile%120)+", "+(tile/120)+")");
    }
}
