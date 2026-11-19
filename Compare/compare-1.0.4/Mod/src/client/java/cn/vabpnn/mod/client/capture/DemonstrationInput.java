package cn.vabpnn.mod.client.capture;

/** Only cumulative human control metadata; no pixels, labels invented by a model, or per-frame allocation. */
public final class DemonstrationInput {
    private static final int[] KEYS = ActionLayout.PHYSICAL_KEYS;
    private final boolean[] physical = new boolean[KEYS.length];
    private static final int[] BITS = ActionLayout.KEY_ACTIONS;
    private final long[] heldTime = new long[8], edges = new long[4];
    private final boolean[] held = new boolean[8], buttons = new boolean[2];
    private long lastNanos, epoch = 1;
    private double x, y, wheel;

    private void advance(long now) {
        if (lastNanos != 0 && now > lastNanos) {
            long dt = now - lastNanos;
            for (int k = 0; k < held.length; k++) if (held[k]) heldTime[k] += dt;
        }
        lastNanos = now;
    }
    public void key(int key, int action, long now) {
        advance(now);
        for (int k = 0; k < KEYS.length; k++) if (key == KEYS[k]) physical[k] = action != 0;
        updateHolds();
    }
    public void physicalKeys(boolean[] down, long now) {
        advance(now);
        if (down.length != physical.length) throw new IllegalArgumentException("Ten physical key states required");
        System.arraycopy(down, 0, physical, 0, physical.length);
        updateHolds();
    }
    private void updateHolds() {
        System.arraycopy(physical,0,held,0,held.length);
        // Both sides share one logical action and one duration; releasing one side keeps the other held.
        held[6] |= physical[8];
        held[7] |= physical[9];
    }
    public boolean button(int button, int action) { return button(button,action,true); }
    /** Track real state in every mode; only human, interactive edges are teaching targets. */
    public boolean button(int button, int action, boolean record) {
        if (button < 0 || button > 1) return false;
        boolean down = action != 0;
        if (buttons[button] == down) return false;
        buttons[button] = down;
        if (record) edges[button + (down ? 0 : 2)]++;
        return record;
    }
    public int heldButtons() { return (buttons[0] ? 1:0)|(buttons[1] ? 2:0); }
    public long buttonEdges(int edge) { return edges[edge]; }
    public int heldKeys() {
        int mask=0;
        for(int k=0;k<held.length;k++) if(held[k]) mask|=1<<BITS[k];
        return mask;
    }
    public void move(double dx, double dy) {
        if (Double.isFinite(dx) && Double.isFinite(dy)) { x += dx; y += dy; }
    }
    public void wheel(double steps) { if (Double.isFinite(steps)) wheel += steps; }
    public void newScope(long now) { advance(now); epoch++; }
    public void snapshot(long now, boolean active, boolean model, long[] data, double[] motion) {
        if (data.length != ActionLayout.DEMO_LONGS || motion.length != 3) throw new IllegalArgumentException("Invalid demonstration buffers");
        advance(now);
        // A held mouse button is a continuing press target, including inside containers.
        int mask = heldKeys() | (buttons[0] ? 1<<7:0) | (buttons[1] ? 1<<8:0);
        data[0] = epoch; data[1] = mask; data[2] = active ? 1 : 0; data[3] = model ? 1 : 0;
        System.arraycopy(edges, 0, data, 4, 4); System.arraycopy(heldTime, 0, data, 8, heldTime.length);
        motion[0] = x; motion[1] = y; motion[2] = wheel;
    }
}

