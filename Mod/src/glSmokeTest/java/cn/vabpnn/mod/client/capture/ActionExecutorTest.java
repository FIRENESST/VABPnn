package cn.vabpnn.mod.client.capture;

import java.util.ArrayList;
import java.util.List;

/** Synthetic commands and a fake sink; never sends input to a real window. */
final class ActionExecutorTest {
    private static class RecordingSink implements ActionExecutor.Sink {
        final List<String> events = new ArrayList<>();
        @Override public void key(int code, boolean down) { events.add("key:" + code + ":" + down); }
        @Override public void button(int button, boolean down) { events.add("button:" + button + ":" + down); }
        @Override public void move(double dx, double dy) { events.add("move:" + dx + ":" + dy); }
        @Override public void scroll(double steps) { events.add("scroll:" + steps); }
    }
    private static long[] command(long frame, long nanos, int mask, float dx, float dy, float wheel) {
        return new long[]{frame, nanos, mask, 1, Float.floatToRawIntBits(dx),
                Float.floatToRawIntBits(dy), Float.floatToRawIntBits(wheel)};
    }
    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
    static void run() {
        ActionExecutor executor = new ActionExecutor();
        RecordingSink sink = new RecordingSink();
        long now = 1_000_000_000L;
        executor.apply(command(1, now, 0x101f, 0, 0, 0), now, true, sink);
        require(sink.events.equals(List.of("key:87:true","key:65:true","key:83:true",
                "key:68:true","key:69:true","key:32:true")), "literal W A S D E Space; W/S and A/D may coexist");
        sink.events.clear();
        long[] motion = command(2, now, (1 << 5) | (1 << 6) | (1 << 11), 2, -3, 1);
        executor.apply(motion, now, true, sink);
        require(sink.events.contains("move:2.0:-3.0") && sink.events.contains("scroll:1.0"),
                "three independent motion permission gates");
        int count = sink.events.size();
        executor.apply(motion, now + 1, true, sink);
        require(sink.events.size() == count, "cached command is not replayed");
        sink.events.clear();
        executor.apply(command(3, now, (1 << 7) | (1 << 9), 9, 8, 7), now, true, sink);
        require(sink.events.equals(List.of("button:0:true","button:0:false")), "left press then release is a legal click; disabled motion is ignored");
        sink.events.clear();
        executor.apply(command(4, now, 1 << 8, 0, 0, 0), now, true, sink);
        executor.apply(command(5, now, 0, 0, 0, 0), now, true, sink);
        require(sink.events.equals(List.of("button:1:true")), "mouse press remains until its release operation");
        executor.apply(command(6, now, 1 << 10, 0, 0, 0), now, true, sink);
        require(sink.events.getLast().equals("button:1:false"), "right release object");
        for (int[] pair : new int[][]{{7,8},{9,10},{7,11},{8,11}}) {
            sink.events.clear();
            executor.apply(command(7, now, (1 << pair[0]) | (1 << pair[1]), 1, 1, 1), now, true, sink);
            require(sink.events.isEmpty(), "reject exactly the four declared conflicting pairs");
        }
        sink.events.clear();
        executor.apply(command(8, now, 1 << 7, 0, 0, 0), now, true, sink);
        executor.apply(command(8, now, 1 << 7, 0, 0, 0), now, false, sink);
        require(sink.events.equals(List.of("button:0:true","button:0:false")), "focus loss/pause releases owned button");
        sink.events.clear();
        executor.apply(command(8, now, 1 << 7, 0, 0, 0), now, true, sink);
        require(sink.events.isEmpty(), "resuming does not replay old press");
        executor.apply(command(9, now, 1, 0, 0, 0), now, true, sink);
        executor.apply(command(9, now, 1, 0, 0, 0), now + ActionExecutor.COMMAND_LEASE_NANOS + 1, true, sink);
        require(sink.events.equals(List.of("key:87:true","key:87:false")), "expired command releases W");
        sink.events.clear();
        executor.apply(command(10, now, 1 << 5, Float.NaN, 0, 0), now, true, sink);
        require(sink.events.isEmpty(), "nonfinite displacement does not reach input handlers");
        sink.events.clear();
        executor.apply(command(11,now,(1<<5)|(1<<6)|(1<<7)|(1<<9),4,-2,0),now,true,sink);
        require(sink.events.equals(List.of("move:4.0:-2.0","button:0:true","button:0:false")),
                "same packet moves before click and preserves click order");
        require(executor.appliedFrame()==11 && executor.appliedMask()==((1<<5)|(1<<6)|(1<<7)|(1<<9)),
                "published execution status matches accepted packet");
        class ContextSink extends RecordingSink {
            boolean same=true;
            @Override public void key(int code,boolean down) {
                super.key(code,down);
                if(code==69 && down) same=false;
            }
            @Override public boolean contextValid() { return same; }
        }
        ContextSink changing=new ContextSink();
        executor.apply(command(12,now,(1<<4)|(1<<5)|(1<<7),8,0,0),now,true,changing);
        require(changing.events.equals(List.of("key:69:true","key:69:false")),
                "screen changed by E: later cursor/click operations are discarded");
        require(executor.appliedFrame()==0 && executor.heldKeys()==0 && executor.heldButtons()==0,
                "context loss clears execution and holds");
        System.out.println("PASS: 13 action mappings, declared conflicts, motion gates, mouse press/release, command deduplication and pause/expiry releases.");
    }
}
