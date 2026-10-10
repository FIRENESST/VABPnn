package cn.vabpnn.mod.client.capture;

/** Phase-preserving 30 Hz sampling on the Minecraft thread; no timers, UDP, allocation or catch-up bursts. */
final class PlayerPositionSampler {
    static final long PERIOD_NANOS=1_000_000_000L/30;
    private Object scene,actor;
    private boolean available;
    private long epoch,nextNanos,lastPoll,sampleNanos,samples;
    private double x,y,z;

    /** Returns true when a new observation or an invalidation must be published. */
    boolean poll(Object world,Object player,boolean active,long now) {
        boolean changed=scene!=world || actor!=player || available!=active || now<lastPoll;
        lastPoll=now;
        if(changed) {
            scene=world; actor=player; available=active;
            ++epoch; sampleNanos=0; x=y=z=0; nextNanos=now;
        }
        if(!available) return changed;
        if(now<nextNanos) return false;
        nextNanos+=((now-nextNanos)/PERIOD_NANOS+1)*PERIOD_NANOS;
        sampleNanos=now;
        return true;
    }
    void coordinates(double px,double py,double pz) {
        if(!Double.isFinite(px) || !Double.isFinite(py) || !Double.isFinite(pz)
                || Math.abs(px)>30_000_000 || Math.abs(py)>30_000_000 || Math.abs(pz)>30_000_000) {
            sampleNanos=0; x=y=z=0; return;
        }
        x=px; y=py; z=pz; ++samples;
    }
    void reset() {
        scene=actor=null; available=false; ++epoch; sampleNanos=nextNanos=lastPoll=0; x=y=z=0;
    }
    long nanos() { return sampleNanos; }
    long epoch() { return epoch; }
    long samples() { return samples; }
    double x() { return x; }
    double y() { return y; }
    double z() { return z; }
}
