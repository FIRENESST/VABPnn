package cn.vabpnn.mod.client.capture;

/** Stable action numbering shared by human input, GPU packets and Minecraft execution. */
public final class ActionLayout {
    private ActionLayout() {}
    public static final int ACTIONS=15, KNOWN_MASK=(1<<ACTIONS)-1;
    public static final int[] KEY_ACTIONS={0,1,2,3,4,12,13,14};
    public static final int[] KEY_CODES={87,65,83,68,69,32,341,340};
    public static final int[] PHYSICAL_KEYS={87,65,83,68,69,32,341,340,345,344};
    public static final int KEY_MASK=0x701f;
    public static final int DEMO_LONGS=17, POINTER_MODE_INDEX=16;
}
