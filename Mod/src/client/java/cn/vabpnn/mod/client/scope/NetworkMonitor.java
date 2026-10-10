package cn.vabpnn.mod.client.scope;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.GraphicsEnvironment;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/** Swing owns painting; Minecraft owns all JNI calls and bridge lifetime. */
public final class NetworkMonitor implements AutoCloseable {
    public record Status(boolean capturing,boolean modelControl,boolean inputActive,String note,
                         long appliedFrame,int appliedMask,int heldKeys,int heldButtons) {
        public Status(boolean capturing,boolean modelControl,boolean inputActive,String note) {
            this(capturing,modelControl,inputActive,note,0,0,0,0);
        }
    }
    public record Hierarchy(long samples,long updates,long rankings,long mutations,long controls,long broadcasts,
                            int enabled,int mode,int radius,int eligible,int candidates,int punished,int invalid,long frame) {
        public static Hierarchy from(long[] v) {
            if(v.length!=14) throw new IllegalArgumentException("Block status shape");
            return new Hierarchy(v[0],v[1],v[2],v[3],v[4],v[5],(int)v[6],(int)v[7],(int)v[8],
                    (int)v[9],(int)v[10],(int)v[11],(int)v[12],v[13]);
        }
        public String modeName() { return switch(mode) { case 0 -> "独立"; case 2 -> "竞争"; default -> "合作"; }; }
    }
    private final AtomicReference<Hierarchy> hierarchy=new AtomicReference<>(Hierarchy.from(new long[14]));
    private final AtomicBoolean visible=new AtomicBoolean(),closed=new AtomicBoolean();
    private final AtomicReference<ScopeSnapshot> latest=new AtomicReference<>();
    private final AtomicReference<MouseMotionTracker.Sample> motion=new AtomicReference<>(MouseMotionTracker.EMPTY);
    private final AtomicReference<Status> status=new AtomicReference<>(new Status(true,false,false,"等待 CUDA"));
    private final AtomicInteger tile=new AtomicInteger(32*120+60);
    private final Consumer<String> errors;
    private JFrame window;
    private ScopePanel panel;
    private Timer timer;

    public NetworkMonitor(Consumer<String> errors) { this.errors=errors; }
    public boolean visible() { return visible.get() && !closed.get(); }
    public int inspectedTile() { return tile.get(); }
    public void show(boolean show) {
        if(closed.get()) return;
        visible.set(show);
        SwingUtilities.invokeLater(() -> {
            if(closed.get()) return;
            try {
                if(show && window==null) createWindow();
                if(window!=null) {
                    window.setVisible(visible());
                    if(visible()) timer.start(); else timer.stop();
                }
            } catch(RuntimeException | LinkageError e) {
                visible.set(false);
                if(timer!=null) timer.stop();
                errors.accept("Network monitor unavailable: "+e);
            }
        });
    }
    private void createWindow() {
        if(GraphicsEnvironment.isHeadless()) throw new IllegalStateException("desktop display unavailable");
        panel=new ScopePanel(latest,status,tile,motion,hierarchy);
        window=new JFrame("VABPnn · 实时脉冲网络  |  F7 显示 / 隐藏");
        window.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        // Set both before the first show: monitoring must not steal Minecraft's focus.
        window.setFocusableWindowState(false);
        window.setAutoRequestFocus(false);
        if(window.isAlwaysOnTopSupported()) window.setAlwaysOnTop(true);
        window.setContentPane(panel);
        Rectangle area=GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice()
                .getDefaultConfiguration().getBounds();
        var insets=Toolkit.getDefaultToolkit().getScreenInsets(window.getGraphicsConfiguration());
        int usableH=area.height-insets.top-insets.bottom-64;
        double scale=Math.min(1.0,Math.min((area.width-40)/(double)ScopePanel.LOGICAL_WIDTH,
                usableH/(double)ScopePanel.LOGICAL_HEIGHT));
        window.setSize((int)(ScopePanel.LOGICAL_WIDTH*scale)+16,(int)(ScopePanel.LOGICAL_HEIGHT*scale)+38);
        window.setMinimumSize(new java.awt.Dimension(460,580));
        window.setLocation(area.x+area.width-window.getWidth()-insets.right-16,area.y+insets.top+12);
        timer=new Timer(16,e -> { panel.advance(System.nanoTime()); panel.repaint(); });
        timer.setCoalesce(true);
        window.addWindowListener(new WindowAdapter() {
            @Override public void windowClosing(WindowEvent event) {
                visible.set(false); timer.stop();
            }
        });
    }
    public void status(boolean capturing,boolean control,boolean active,String note) {
        status(capturing,control,active,note,0,0,0,0);
    }
    public void status(boolean capturing,boolean control,boolean active,String note,long frame,int mask,int keys,int buttons) {
        String normalized=note==null ? "" : note;
        Status previous=status.get();
        if(previous.capturing()==capturing && previous.modelControl()==control
                && previous.inputActive()==active && previous.appliedFrame()==frame
                && previous.appliedMask()==mask && previous.heldKeys()==keys
                && previous.heldButtons()==buttons && previous.note().equals(normalized)) return;
        status.set(new Status(capturing,control,active,normalized,frame,mask,keys,buttons));
    }
    public void hierarchy(Hierarchy sample) { if(!closed.get()) hierarchy.set(sample); }
    public void motion(MouseMotionTracker.Sample sample) {
        if(!closed.get() && motion.get()!=sample) motion.set(sample);
    }
    public void publish(long[] meta,float[] values,int[] discrete,long now) {
        if(closed.get() || meta[0]==0) return;
        ScopeSnapshot old=latest.get();
        if(old==null || old.frame()!=meta[0]) latest.set(new ScopeSnapshot(meta,values,discrete,now));
    }
    @Override public void close() {
        closed.set(true); visible.set(false);
        SwingUtilities.invokeLater(() -> {
            if(timer!=null) timer.stop();
            if(window!=null) { window.dispose(); window=null; }
            latest.set(null);
        });
    }
}
