package cn.vabpnn.mod.client.capture;

/** Applies final GPU control packets on the Minecraft thread. */
public final class ActionExecutor {
    public interface Sink {
        void key(int code, boolean down);
        void button(int button, boolean down);
        void move(double dx, double dy);
        void scroll(double steps);
        default void beginCommand() {}
        default boolean contextValid() { return true; }
        default void cancelPending() {}
    }
    public static final long COMMAND_LEASE_NANOS = 500_000_000L;
    private static final int[] KEY_ACTIONS = ActionLayout.KEY_ACTIONS;
    private static final int[] KEY_CODES = ActionLayout.KEY_CODES;
    private static final int KEY_MASK = ActionLayout.KEY_MASK;
    private long lastFrame, appliedFrame, appliedCommands;
    private int heldKeys, appliedMask;
    private boolean leftDown, rightDown;

    public long appliedFrame() { return appliedFrame; }
    public long appliedCommands() { return appliedCommands; }
    public int appliedMask() { return appliedMask; }
    public int heldKeys() { return heldKeys; }
    public int heldButtons() { return (leftDown ? 1 : 0) | (rightDown ? 2 : 0); }
    private static boolean bit(int mask, int index) { return (mask & (1 << index)) != 0; }
    private static boolean conflicts(int mask) {
        return bit(mask, 7) && bit(mask, 8) || bit(mask, 9) && bit(mask, 10)
                || bit(mask, 7) && bit(mask, 11) || bit(mask, 8) && bit(mask, 11);
    }
    private boolean keepContext(Sink sink) {
        if (sink.contextValid()) return true;
        stop(sink);
        return false;
    }
    public void apply(long[] command, long now, boolean active, Sink sink) {
        if (command.length != 7) throw new IllegalArgumentException("Seven-long control command required");
        int mask = (int) command[2];
        long age = now - command[1];
        if (!active || command[3] != 1 || command[0] <= 0 || age < 0
                || age > COMMAND_LEASE_NANOS || (mask & ~ActionLayout.KNOWN_MASK) != 0 || conflicts(mask)) {
            stop(sink);
            return;
        }
        if (command[0] <= lastFrame) return; // Never replay motion, scrolling or click edges.
        lastFrame = command[0];
        float dx = Float.intBitsToFloat((int) command[4]);
        float dy = Float.intBitsToFloat((int) command[5]);
        float wheel = Float.intBitsToFloat((int) command[6]);
        if (!Float.isFinite(dx) || !Float.isFinite(dy) || !Float.isFinite(wheel)) {
            stop(sink);
            return;
        }
        sink.beginCommand();
        int desiredKeys = mask & KEY_MASK;
        // Modifier releases/presses precede ordinary keys and mouse events in the same packet.
        for (int pass=0;pass<2;pass++) for (int i=0;i<KEY_ACTIONS.length;i++) {
            if ((KEY_ACTIONS[i]>=13)!=(pass==0)) continue;
            int keyBit=1<<KEY_ACTIONS[i];
            if ((heldKeys & keyBit)!=(desiredKeys & keyBit)) {
                boolean down=(desiredKeys & keyBit)!=0;
                sink.key(KEY_CODES[i],down);
                heldKeys=down ? heldKeys | keyBit:heldKeys & ~keyBit;
                if (!keepContext(sink)) return;
            }
        }
        // GUI clicks/drag releases must use the new cursor position from this packet.
        double x = bit(mask, 5) ? dx : 0, y = bit(mask, 6) ? dy : 0;
        if (x != 0 || y != 0) {
            sink.move(x, y);
            if (!keepContext(sink)) return;
        }
        // Preserve the specified ordering for a legal press+release of the same button.
        if (bit(mask, 7) && !leftDown) {
            sink.button(0, true); leftDown = true;
            if (!keepContext(sink)) return;
        }
        if (bit(mask, 8) && !rightDown) {
            sink.button(1, true); rightDown = true;
            if (!keepContext(sink)) return;
        }
        if (bit(mask, 9) && leftDown) {
            sink.button(0, false); leftDown = false;
            if (!keepContext(sink)) return;
        }
        if (bit(mask, 10) && rightDown) {
            sink.button(1, false); rightDown = false;
            if (!keepContext(sink)) return;
        }
        if (bit(mask, 11) && wheel != 0) {
            sink.scroll(wheel);
            if (!keepContext(sink)) return;
        }
        appliedFrame = command[0]; appliedMask = mask; appliedCommands++;
    }
    /** Release the model's holds and discard its pending vanilla click counters. */
    public void stop(Sink sink) {
        for (int i = 0; i < KEY_ACTIONS.length; i++)
            if (bit(heldKeys, KEY_ACTIONS[i])) sink.key(KEY_CODES[i], false);
        heldKeys = 0;
        if (leftDown) sink.button(0, false);
        if (rightDown) sink.button(1, false);
        leftDown = rightDown = false;
        sink.cancelPending();
        appliedFrame = 0; appliedMask = 0;
        // Keep lastFrame: resuming cannot replay a previously consumed command.
    }
}

