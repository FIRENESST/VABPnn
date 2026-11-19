package cn.vabpnn.mod.client.scope;

import cn.vabpnn.mod.client.capture.ActionExecutor;

/** Client-thread accumulators; Swing receives only immutable, bounded scalar samples. */
public final class MouseMotionTracker {
    public static final long SAMPLE_NANOS=1_000_000_000L/30;
    public static final long STALE_NANOS=250_000_000L;
    public record Suggestion(long frame,long captured,boolean paired,boolean enabled,
            double dx,double dy,double probabilityX,double probabilityY,double thresholdX,double thresholdY,
            int eligible,int selected) {
        public boolean fresh(long now) {long age=now-captured;return frame>0 && age>=0 && age<=ActionExecutor.COMMAND_LEASE_NANOS;}
        public String axis(int bit) {return (selected&bit)!=0 ? "选中":(eligible&bit)!=0 ? "候选":"未过阈值";}
    }
    public static final Suggestion NO_SUGGESTION=new Suggestion(0,0,false,false,0,0,0,0,0,0,0,0);
    public record Sample(long nanos,long intervalNanos,boolean active,
                         double humanDx,double humanDy,double predictedDx,double predictedDy,
                         double executedDx,double executedDy,Suggestion suggestion) {
        public Sample(long nanos,long intervalNanos,boolean active,double humanDx,double humanDy,
                double predictedDx,double predictedDy,double executedDx,double executedDy) {
            this(nanos,intervalNanos,active,humanDx,humanDy,predictedDx,predictedDy,executedDx,executedDy,NO_SUGGESTION);
        }
        public boolean fresh(long now) {
            long age=now-nanos;
            return active && age>=0 && age<=STALE_NANOS;
        }
    }
    public static final Sample EMPTY=new Sample(0,0,false,0,0,0,0,0,0);
    private Sample published=EMPTY;
    private long previousNanos,lastPredictionFrame,lastSuggestionFrame;
    private Suggestion suggested=NO_SUGGESTION;
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
    public boolean needsSuggestion(long frame) {return frame>lastSuggestionFrame;}
    /** Pair advice with the exact completed control packet; never combine two GPU frames. */
    public void suggestion(long[] values,long[] command,long now) {
        if(values.length!=17 || command.length!=7 || command[3]!=1 || values[2]!=1
                || values[0]!=command[0] || values[1]!=command[1] || values[0]<=lastSuggestionFrame)return;
        long age=now-values[1];if(age<0 || age>ActionExecutor.COMMAND_LEASE_NANOS)return;
        for(int i=5;i<=14;i++)if(!Float.isFinite(Float.intBitsToFloat((int)values[i])))return;
        int eligible=(int)values[15],selected=(int)values[16];
        if((eligible&~96)!=0 || (selected&~96)!=0 || selected!=((int)command[2]&96))return;
        double px=Float.intBitsToFloat((int)values[11]),py=Float.intBitsToFloat((int)values[12]),
                tx=Float.intBitsToFloat((int)values[13]),ty=Float.intBitsToFloat((int)values[14]);
        if(px<0 || px>1 || py<0 || py>1 || tx<0 || tx>1 || ty<0 || ty>1)return;
        lastSuggestionFrame=values[0];
        suggested=new Suggestion(values[0],values[1],values[3]==1,values[4]==1,
                Float.intBitsToFloat((int)values[5]),Float.intBitsToFloat((int)values[6]),
                px,py,tx,ty,eligible,selected);
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
        clearInterval(); sampling=false; published=EMPTY;suggested=NO_SUGGESTION;
        // Keep the frame cursor: a cached packet must never become a new prediction.
    }
    public Sample current() { return published; }
    public Sample sample(long now,boolean active) {
        if(!active) { reset(); return published; }
        if(!sampling) { sampling=true; previousNanos=now; return published; }
        long interval=now-previousNanos;
        if(interval<SAMPLE_NANOS) return published;
        published=new Sample(now,interval,true,humanDx,humanDy,predictedDx,predictedDy,executedDx,executedDy,
                suggested.fresh(now) ? suggested:NO_SUGGESTION);
        clearInterval(); previousNanos=now;
        return published;
    }
}

