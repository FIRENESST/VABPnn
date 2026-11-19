package cn.vabpnn.mod.client.scope;

import cn.vabpnn.mod.client.capture.ActionExecutor;

/** Client-thread accumulators; Swing receives only immutable, bounded scalar samples. */
public final class MouseMotionTracker {
    public static final long SAMPLE_NANOS=1_000_000_000L/30;
    public static final long STALE_NANOS=250_000_000L;
    public record Sample(long nanos,long intervalNanos,boolean active,
                         double humanDx,double humanDy,double predictedDx,double predictedDy,
                         double executedDx,double executedDy) {
        public boolean fresh(long now) {
            long age=now-nanos;
            return active && age>=0 && age<=STALE_NANOS;
        }
    }
    public static final Sample EMPTY=new Sample(0,0,false,0,0,0,0,0,0);
    private Sample published=EMPTY;
    private long previousNanos,lastPredictionFrame;
    private boolean sampling;
    private double humanDx,humanDy,predictedDx,predictedDy,executedDx,executedDy;

    private static double sum(double previous,double delta) {
        double result=previous+delta;
        return Double.isFinite(result) ? result:previous;
    }
    public void human(double dx,double dy) {
        if(!Double.isFinite(dx) || !Double.isFinite(dy)) return;
        humanDx=sum(humanDx,dx); humanDy=sum(humanDy,dy);
    }
    /** The final GPU packet already includes axis permissions and per-frame dt. */
    public void prediction(long[] command,long now) {
        if(command.length!=7 || command[3]!=1 || command[0]<=lastPredictionFrame) return;
        long age=now-command[1];
        if(age<0 || age>ActionExecutor.COMMAND_LEASE_NANOS) return;
        int mask=(int)command[2];
        if((mask & ~0x7fff)!=0) return;
        float dx=Float.intBitsToFloat((int)command[4]),dy=Float.intBitsToFloat((int)command[5]);
        if(!Float.isFinite(dx) || !Float.isFinite(dy)) return;
        lastPredictionFrame=command[0];
        predictedDx=sum(predictedDx,(mask & (1<<5))!=0 ? dx:0);
        predictedDy=sum(predictedDy,(mask & (1<<6))!=0 ? dy:0);
    }
    /** Called after delivery; GUI samples use the displacement after edge clamping. */
    public void executed(double dx,double dy) {
        if(!Double.isFinite(dx) || !Double.isFinite(dy)) return;
        executedDx=sum(executedDx,dx); executedDy=sum(executedDy,dy);
    }
    private void clearInterval() {
        humanDx=humanDy=predictedDx=predictedDy=executedDx=executedDy=0;
    }
    public void reset() {
        clearInterval(); sampling=false; published=EMPTY;
        // Keep the frame cursor: a cached packet must never become a new prediction.
    }
    public Sample current() { return published; }
    public Sample sample(long now,boolean active) {
        if(!active) { reset(); return published; }
        if(!sampling) { sampling=true; previousNanos=now; return published; }
        long interval=now-previousNanos;
        if(interval<SAMPLE_NANOS) return published;
        published=new Sample(now,interval,true,humanDx,humanDy,predictedDx,predictedDy,executedDx,executedDy);
        clearInterval(); previousNanos=now;
        return published;
    }
}

