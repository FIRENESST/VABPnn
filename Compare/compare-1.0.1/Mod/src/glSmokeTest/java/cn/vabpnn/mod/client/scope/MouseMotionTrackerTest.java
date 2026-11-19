package cn.vabpnn.mod.client.scope;

/** Deterministic timestamps and scalar packets; no GPU or OS input is used. */
final class MouseMotionTrackerTest {
    private static void require(boolean ok,String message) {
        if(!ok) throw new AssertionError(message);
    }
    private static long[] command(long frame,long now,int mask,float dx,float dy) {
        return new long[]{frame,now,mask,1,Float.floatToRawIntBits(dx),Float.floatToRawIntBits(dy),0};
    }
    static void run() {
        var tracker=new MouseMotionTracker();
        long start=1_000_000_000L;
        require(!tracker.sample(start,true).active(),"wait for the first complete interval");
        tracker.human(1.25,-2); tracker.human(.75,-.5);
        var first=command(1,start,(1<<5)|(1<<6),4,-3);
        tracker.prediction(first,start); tracker.prediction(first,start+1);
        tracker.executed(3,-3); // Cursor clamping can reduce the delivered X displacement.
        require(tracker.sample(start+MouseMotionTracker.SAMPLE_NANOS-1,true)==MouseMotionTracker.EMPTY,
                "publication does not run once per input callback");
        var sample=tracker.sample(start+MouseMotionTracker.SAMPLE_NANOS,true);
        require(sample.humanDx()==2 && sample.humanDy()==-2.5,"accumulate signed manual events");
        require(sample.predictedDx()==4 && sample.predictedDy()==-3,"cached GPU packets counted once");
        require(sample.executedDx()==3 && sample.executedDy()==-3,"execution remains separate from prediction");
        require(sample.fresh(sample.nanos()+MouseMotionTracker.STALE_NANOS)
                && !sample.fresh(sample.nanos()+MouseMotionTracker.STALE_NANOS+1),"stale motion is never kept on screen");
        require(tracker.sample(start+MouseMotionTracker.SAMPLE_NANOS+1,true)==sample,"reuse immutable sample between intervals");
        long next=start+2*MouseMotionTracker.SAMPLE_NANOS;
        tracker.prediction(first,next);
        sample=tracker.sample(next,true);
        require(sample.humanDx()==0 && sample.predictedDx()==0 && sample.executedDx()==0,"idle intervals clear old movement");
        tracker.prediction(command(2,next,1<<5,2,9),next);
        tracker.prediction(command(3,next,1<<6,9,-5),next);
        tracker.human(Double.NaN,1); tracker.human(1,Double.POSITIVE_INFINITY);
        tracker.executed(Double.NEGATIVE_INFINITY,2);
        tracker.prediction(command(4,next,1<<5,Float.NaN,0),next);
        tracker.prediction(command(5,next-500_000_001L,1<<5,100,0),next);
        tracker.prediction(command(6,next+1,1<<5,100,0),next);
        var invalid=command(7,next,1<<5,100,0); invalid[3]=0; tracker.prediction(invalid,next);
        sample=tracker.sample(next+MouseMotionTracker.SAMPLE_NANOS,true);
        require(sample.predictedDx()==2 && sample.predictedDy()==-5,"axis permission gates and invalid/expired packets");
        require(sample.humanDx()==0 && sample.humanDy()==0 && sample.executedDx()==0,"nonfinite events ignored");
        tracker.human(8,9); tracker.executed(5,6);
        require(!tracker.sample(next+MouseMotionTracker.SAMPLE_NANOS+1,false).active(),"pause publishes zeros immediately");
        long resumed=next+10*MouseMotionTracker.SAMPLE_NANOS;
        tracker.sample(resumed,true); tracker.prediction(first,resumed);
        sample=tracker.sample(resumed+MouseMotionTracker.SAMPLE_NANOS,true);
        require(sample.humanDx()==0 && sample.executedDx()==0 && sample.predictedDx()==0,"resume discards old events and cached packets");
        tracker.human(7,-8);
        sample=tracker.sample(resumed+20*MouseMotionTracker.SAMPLE_NANOS,true);
        require(sample.humanDx()==7 && sample.humanDy()==-8,"slow rendering publishes current totals without replaying intervals");
        require(tracker.sample(sample.nanos()+1,true)==sample,"no catch-up publication burst");
        System.out.println("PASS: mouse delta accumulation, 30 Hz limit, command deduplication, axis gates, GUI execution values, idle/pause/resume/stale and invalid input.");
    }
}
