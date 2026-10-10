package cn.vabpnn.mod.client.capture;
public final class DemonstrationInputTest {
    private static void require(boolean ok, String message) { if (!ok) throw new AssertionError(message); }
    public static void run() {
        DemonstrationInput input = new DemonstrationInput();
        long[] data = new long[15]; double[] movement = new double[3];
        input.key(87, 1, 1_000_000_000);
        input.move(2.5, -1); input.button(0, 1); input.wheel(-1);
        input.snapshot(1_020_000_000, true, false, data, movement);
        require(data[1] == (1|(1<<7)) && data[8] == 20_000_000 && data[4] == 1, "W duration and left press");
        require(movement[0] == 2.5 && movement[1] == -1 && movement[2] == -1, "raw signed mouse/wheel totals");
        input.snapshot(1_030_000_000,true,false,data,movement);
        require((data[1]&(1<<7))!=0 && data[4]==1 && input.heldButtons()==1,
                "held left button remains a press target without repeating its edge");
        long epoch = data[0], held = data[8]; double x = movement[0];
        input.move(1, 0); input.move(3, 0); input.button(0, 0);
        input.snapshot(1_120_000_000, true, false, data, movement);
        require(data[8] - held == 90_000_000 && movement[0] - x == 4, "cumulative targets survive skipped frames");
        require(data[6] == 1 && (data[1]&(1<<7))==0, "left release clears held state and remains an event");
        input.button(1,1); input.snapshot(1_125_000_000,true,false,data,movement);
        require(data[5]==1 && (data[1]&(1<<8))!=0 && input.heldButtons()==2,
                "right button is tracked independently of vanilla world-only pressed flags");
        input.button(1,0,false); input.snapshot(1_126_000_000,true,false,data,movement);
        require(data[7]==0 && input.heldButtons()==0,"real release in model mode synchronizes state without a human target");
        input.newScope(1_130_000_000); input.key(87, 0, 1_140_000_000);
        input.snapshot(1_150_000_000, true, true, data, movement);
        require(data[0] > epoch && data[1] == 0 && data[3] == 1, "mode scope and physical key release");
        input.move(Double.NaN, 2); input.wheel(Double.POSITIVE_INFINITY);
        input.snapshot(1_160_000_000, false, false, data, movement);
        require(Double.isFinite(movement[0]) && data[2] == 0, "invalid metadata ignored and inactive flagged");
        System.out.println("PASS: cumulative human input timing, dropped frames, button edges, mode epochs and validity.");
    }
}
