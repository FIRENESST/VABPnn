package cn.vabpnn.mod.client.scope;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.Dimension;
import java.awt.GraphicsEnvironment;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/** Ordinary desktop window, opened without requesting focus. JNI stays on the Minecraft thread. */
public final class MemePropagationMonitor implements AutoCloseable {
    public static final int REFRESH_MILLIS=200;
    public static final long SAMPLE_NANOS=200_000_000L;
    public record Status(boolean capturing,boolean evaluation,String note,boolean predictionOnly) {
        public Status(boolean capturing,boolean evaluation,String note) { this(capturing,evaluation,note,false); }
    }
    private final AtomicBoolean visible=new AtomicBoolean(),closed=new AtomicBoolean();
    private final AtomicInteger tile=new AtomicInteger(32*120+60);
    private final AtomicReference<MemeSnapshot> latest=new AtomicReference<>();
    private final AtomicReference<Status> status=new AtomicReference<>(new Status(false,false,"待命：/vabpnn start"));
    private final Consumer<String> errors;
    private volatile long nextSample;
    private JFrame window;
    private Timer timer;
    private MemePropagationPanel panel;

    public MemePropagationMonitor(Consumer<String> errors) { this.errors=errors; }
    public boolean visible() { return visible.get() && !closed.get(); }
    public int inspectedTile() { return tile.get(); }
    public long sampledFrame() { MemeSnapshot sample=latest.get();return sample==null ? 0:sample.frame(); }
    public boolean shouldSample(long now) {
        if(!visible() || !status.get().capturing() || now<nextSample) return false;
        nextSample=now+SAMPLE_NANOS;return true;
    }
    public void status(boolean capturing,boolean evaluation,String note) {
        status(capturing,evaluation,note,false);
    }
    public void status(boolean capturing,boolean evaluation,String note,boolean predictionOnly) {
        String text=note==null ? "":note;
        Status old=status.get();
        if(old.capturing()==capturing && old.evaluation()==evaluation && old.predictionOnly()==predictionOnly && old.note().equals(text)) return;
        status.set(new Status(capturing,evaluation,text,predictionOnly));
    }
    public void publish(long[] meta,float[] values,int[] discrete,long[] identities,long now) {
        if(!visible() || meta[0]==0) return;
        MemeSnapshot old=latest.get();
        if(old==null || old.frame()!=meta[0]) latest.set(new MemeSnapshot(meta,values,discrete,identities,now));
    }
    public void clear() { latest.set(null);nextSample=0; }
    public void show(boolean show) {
        if(closed.get()) return;
        if(visible.getAndSet(show)!=show) clear();
        SwingUtilities.invokeLater(() -> {
            if(closed.get()) return;
            try {
                if(visible() && window==null) createWindow();
                if(window!=null) {
                    window.setVisible(visible());
                    if(visible()) timer.start();else timer.stop();
                }
            } catch(RuntimeException | LinkageError error) {
                visible.set(false);if(timer!=null) timer.stop();
                errors.accept("Meme propagation viewer unavailable: "+error);
            }
        });
    }
    private void createWindow() {
        if(GraphicsEnvironment.isHeadless()) throw new IllegalStateException("desktop display unavailable");
        panel=new MemePropagationPanel(latest,status,tile);
        window=new JFrame("VABPnn 1.4.1 · 模因传播  |  5 Hz  |  F6 显示 / 隐藏");
        window.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        window.setAutoRequestFocus(false);
        window.setContentPane(panel);
        Rectangle area=GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice()
                .getDefaultConfiguration().getBounds();
        var insets=Toolkit.getDefaultToolkit().getScreenInsets(window.getGraphicsConfiguration());
        double scale=Math.min(1.0,Math.min((area.width-insets.left-insets.right-48)/(double)MemePropagationPanel.WIDTH,
                (area.height-insets.top-insets.bottom-80)/(double)MemePropagationPanel.HEIGHT));
        window.setSize((int)(MemePropagationPanel.WIDTH*scale)+16,(int)(MemePropagationPanel.HEIGHT*scale)+38);
        window.setMinimumSize(new Dimension(660,450));
        window.setLocation(area.x+insets.left+16,area.y+insets.top+12);
        timer=new Timer(REFRESH_MILLIS,event -> { panel.advance(System.nanoTime());panel.repaint(); });
        timer.setCoalesce(true);
        window.addWindowListener(new WindowAdapter() {
            @Override public void windowClosing(WindowEvent event) { visible.set(false);timer.stop();clear(); }
        });
    }
    @Override public void close() {
        closed.set(true);visible.set(false);clear();
        SwingUtilities.invokeLater(() -> {
            if(timer!=null) timer.stop();
            if(window!=null) { window.dispose();window=null; }
        });
    }
}
