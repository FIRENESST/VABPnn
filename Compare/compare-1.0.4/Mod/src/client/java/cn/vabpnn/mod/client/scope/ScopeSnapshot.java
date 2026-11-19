package cn.vabpnn.mod.client.scope;

/** Immutable, bounded GPU observation. Buffers contain neural state only, never game RGB. */
public final class ScopeSnapshot {
    public static final int TILES=120*64, CELLS=256, EDGES=512, ACTIONS=15;
    public static final int MAP_MEM=8, MAP_POS=MAP_MEM+TILES, MAP_NEG=MAP_POS+TILES;
    public static final int LOCAL_MEM=MAP_NEG+TILES, POSTERIOR=LOCAL_MEM+CELLS;
    public static final int THRESHOLD=POSTERIOR+ACTIONS, PRIOR=THRESHOLD+ACTIONS;
    public static final int BLOCK_SCORE=PRIOR+ACTIONS, BLOCK_SELECTED=BLOCK_SCORE+TILES, MEME_MAP=BLOCK_SELECTED+8;
    public static final int MEME_EVIDENCE=MEME_MAP+TILES, MEME_SELECTED=MEME_EVIDENCE+ACTIONS, FLOATS=MEME_SELECTED+8;
    public static final int LOCAL_SPIKE=8, OUTGOING=LOCAL_SPIKE+CELLS, ANCHORS=OUTGOING+1024;
    public static final int EDGE_PAIRS=ANCHORS+256, VOTES=EDGE_PAIRS+EDGES*2, BLOCK_DATA=VOTES+ACTIONS, MEME_DATA=BLOCK_DATA+8, INTS=MEME_DATA+8, META=24;
    public static final String[] ACTION_NAMES={"W","A","S","D","E","鼠标 X","鼠标 Y","左键按下","右键按下","左键松开","右键松开","滚轮","空格","Ctrl 疾跑","Shift 潜行"};
    final long[] meta;
    final float[] values;
    final int[] discrete;
    final long receivedNanos;

    public ScopeSnapshot(long[] meta,float[] values,int[] discrete,long now) {
        if(meta.length!=META || values.length!=FLOATS || discrete.length!=INTS)
            throw new IllegalArgumentException("Network scope buffer shape mismatch");
        this.meta=meta.clone(); this.values=values.clone(); this.discrete=discrete.clone(); receivedNanos=now;
    }
    public long frame() { return meta[0]; }
    public int tile() { return Math.clamp(discrete[1],0,TILES-1); }
    public float value(int i) { return Float.isFinite(values[i]) ? values[i]:0; }
    public float probability(int k) { return clamp(value(POSTERIOR+k)); }
    public float threshold(int k) { return clamp(value(THRESHOLD+k)); }
    public float prior(int k) { return clamp(value(PRIOR+k)); }
    public boolean selected(int k) { return (discrete[5]&(1<<k))!=0; }
    public boolean eligible(int k) { return (discrete[4]&(1<<k))!=0; }
    public boolean valid() { return frame()>0 && discrete[0]==1 && discrete[7]==0; }
    public boolean stale(long now) { return now-receivedNanos>750_000_000L; }
    public static float clamp(float x) { return Float.isFinite(x) ? Math.clamp(x,0,1):0; }
}

