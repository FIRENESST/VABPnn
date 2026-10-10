package cn.vabpnn.mod.client.scope;

import javax.imageio.ImageIO;
import javax.swing.SwingUtilities;
import javax.swing.JFrame;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.concurrent.atomic.*;
import static cn.vabpnn.mod.client.scope.ScopeSnapshot.*;

/** UI-only fixtures. Exported preview uses synthetic neural telemetry, not Minecraft pixels. */
public final class ScopeRenderTest {
    private static void require(boolean ok,String msg) { if(!ok) throw new AssertionError(msg); }
    private static BufferedImage render(ScopePanel panel) {
        var image=new BufferedImage(ScopePanel.LOGICAL_WIDTH,ScopePanel.LOGICAL_HEIGHT,BufferedImage.TYPE_INT_RGB);
        var graphics=image.createGraphics(); panel.paint(graphics); graphics.dispose(); return image;
    }
    private static int changedMotionPixels(BufferedImage left,BufferedImage right) {
        int count=0;
        for(int y=882;y<954;y++) for(int x=300;x<746;x++)
            if(left.getRGB(x,y)!=right.getRGB(x,y)) count++;
        return count;
    }
    public static void main(String[] args) throws Exception {
        MouseMotionTrackerTest.run();
        long[] meta=new long[META]; float[] values=new float[FLOATS]; int[] ints=new int[INTS];
        meta[0]=420; meta[2]=3760; meta[3]=6150; meta[10]=3_934_743; meta[11]=24; meta[12]=25_620;
        meta[13]=84; meta[14]=420; meta[15]=412; meta[16]=103; meta[17]=412; meta[18]=105; meta[19]=401;
        meta[20]=535_989_571; meta[8]=14; meta[22]=3;
        ints[0]=1; ints[1]=32*120+60; ints[2]=3; ints[3]=3; ints[4]=0x1021; ints[5]=0x1021; ints[6]=64;
        values[0]=.45f; values[2]=.0034f;
        for(int t=0;t<TILES;t++) {
            values[MAP_MEM+t]=(float)(.08+.33*(.5+.5*Math.sin(t%120*.13+t/120*.19)));
            values[MAP_POS+t]=(float)(.09+.08*Math.sin(t*.08));
            values[MAP_NEG+t]=(float)(.04+.03*Math.cos(t*.07));
        }
        for(int k=0;k<ACTIONS;k++) {
            values[POSTERIOR+k]=(ints[5]&(1<<k))!=0 ? .83f:.17f+.02f*k;
            values[THRESHOLD+k]=.54f; values[PRIOR+k]=.5f;
            ints[VOTES+k]=(k%2==0 ? 1:-1)*(k+1)*18_374;
        }
        for(int i=0;i<CELLS;i++) {
            values[LOCAL_MEM+i]=(i%17)*.025f;
            ints[LOCAL_SPIKE+i]=i%13==0 ? 1:i%17==0 ? -1:0;
            for(int slot=0;slot<4;slot++) ints[OUTGOING+i*4+slot]=((i+slot*41+17)&255)|(slot<<8)|1024;
        }
        for(int i=0;i<24;i++) ints[ANCHORS+i]=(12+i/6*9)*1920+(15+i%6*290);
        for(int i=0;i<64;i++) {
            ints[EDGE_PAIRS+i*2]=ints[ANCHORS+i%24];
            ints[EDGE_PAIRS+i*2+1]=ints[ANCHORS+(i*7+1)%24]|((i%4)<<21);
        }
        ints[BLOCK_DATA]=1; ints[BLOCK_DATA+1]=1; ints[BLOCK_DATA+2]=412; ints[BLOCK_DATA+3]=76;
        values[BLOCK_SELECTED]=.0521f; values[BLOCK_SELECTED+1]=.73f; values[BLOCK_SELECTED+2]=.81f;
        values[BLOCK_SELECTED+3]=.963f; values[BLOCK_SELECTED+4]=.0032f;
        values[BLOCK_SELECTED+5]=1.082f; values[BLOCK_SELECTED+6]=1.041f;
        for(int t=0;t<TILES;t++) values[BLOCK_SCORE+t]=(float)(.13*Math.sin(t%120*.13+t/120*.19));
        var snapshot=new ScopeSnapshot(meta,values,ints,System.nanoTime());
        meta[0]=999; values[POSTERIOR]=0; ints[5]=0;
        require(snapshot.frame()==420 && snapshot.probability(0)>.8 && snapshot.selected(0),"immutable publication");
        require(!snapshot.stale(snapshot.receivedNanos+500_000_000) && snapshot.stale(snapshot.receivedNanos+800_000_000),"stale data detection");
        require(ScopePanel.tileAt(new Point(24,138))==0 && ScopePanel.tileAt(new Point(383,329))==7679,"overview coordinates");
        require(ScopePanel.tileAt(new Point(384,330))==-1,"overview boundaries");
        require(ScopePanel.localAt(new Point(24,395))==0 && ScopePanel.localAt(new Point(167,538))==255,"16x16 coordinates");
        var latest=new AtomicReference<>(snapshot);
        var state=new AtomicReference<>(new NetworkMonitor.Status(true,false,true,"测试数据预览"));
        var tile=new AtomicInteger(snapshot.tile());
        var motion=new AtomicReference<>(MouseMotionTracker.EMPTY);
        var hierarchy=new AtomicReference<>(new NetworkMonitor.Hierarchy(412,7680*76,3,21,84,1512,1,1,1,7652,104,7,0,420));
        require(FLOATS==31031 && INTS==2589,"Java/CUDA block scope layout agreement");
        Path preview=Path.of("build/native/network-scope-preview.png"); java.nio.file.Files.createDirectories(preview.getParent());
        SwingUtilities.invokeAndWait(() -> {
            try {
                var panel=new ScopePanel(latest,state,tile,motion,hierarchy);
                panel.setSize(ScopePanel.LOGICAL_WIDTH,ScopePanel.LOGICAL_HEIGHT);
                for(int i=0;i<180;i++) {
                    long[] m=snapshot.meta.clone(); m[0]=i+1;
                    latest.set(new ScopeSnapshot(m,snapshot.values,snapshot.discrete,System.nanoTime()));
                    panel.advance(System.nanoTime());
                }
                panel.dispatchEvent(new MouseEvent(panel,MouseEvent.MOUSE_CLICKED,System.currentTimeMillis(),0,45,147,1,false));
                require(tile.get()==3*120+7,"map click chooses intended GPU tile");
                tile.set(snapshot.tile());
                panel.dispatchEvent(new MouseEvent(panel,MouseEvent.MOUSE_CLICKED,System.currentTimeMillis(),0,600,116,1,false));
                panel.dispatchEvent(new MouseEvent(panel,MouseEvent.MOUSE_CLICKED,System.currentTimeMillis(),0,600,116,1,false));
                require(snapshot.value(BLOCK_SCORE+snapshot.tile())!=0 && snapshot.value(BLOCK_SELECTED+3)>.9f,
                        "high-level score and selected controller are real immutable snapshot fields");
                render(panel); // Warm up desktop font discovery before the 250 ms freshness assertion.
                var example=new MouseMotionTracker.Sample(System.nanoTime(),33_333_333,true,12.5,-7.25,4.25,-2.5,3.75,-2.5);
                motion.set(example);
                BufferedImage image=render(panel);
                ImageIO.write(image,"PNG",preview.toFile());
                motion.set(MouseMotionTracker.EMPTY);
                var zero=render(panel);
                require(changedMotionPixels(image,zero)>40,"signed mouse deltas are visible in the new panel");
                motion.set(new MouseMotionTracker.Sample(System.nanoTime()-MouseMotionTracker.STALE_NANOS-1,
                        33_333_333,true,12.5,-7.25,4.25,-2.5,3.75,-2.5));
                require(changedMotionPixels(render(panel),zero)==0,"stale mouse deltas render as zero");
                motion.set(new MouseMotionTracker.Sample(System.nanoTime(),33_333_333,true,12.5,-7.25,4.25,-2.5,3.75,-2.5));
                state.set(new NetworkMonitor.Status(false,false,false,"暂停"));
                require(changedMotionPixels(render(panel),zero)==0,"paused mouse deltas render as zero");
                panel.setSize(ScopePanel.LOGICAL_WIDTH/2,ScopePanel.LOGICAL_HEIGHT/2);
                panel.dispatchEvent(new MouseEvent(panel,MouseEvent.MOUSE_CLICKED,System.currentTimeMillis(),0,28,79,1,false));
                require(tile.get()==6*120+10,"resized panel retains overview hit coordinates");
            } catch(Exception e) { throw new RuntimeException(e); }
        });
        AtomicReference<String> error=new AtomicReference<>();
        var monitor=new NetworkMonitor(error::set); monitor.show(true);
        SwingUtilities.invokeAndWait(() -> {});
        require(error.get()==null && monitor.visible(),"desktop scope opens");
        var field=NetworkMonitor.class.getDeclaredField("window"); field.setAccessible(true);
        var window=(JFrame)field.get(monitor);
        require(!window.getFocusableWindowState() && !window.isAutoRequestFocus(),"window cannot take game focus");
        monitor.show(false); SwingUtilities.invokeAndWait(() -> {});
        require(!window.isVisible(),"F7 hide leaves model independent");
        monitor.show(true); SwingUtilities.invokeAndWait(() -> {});
        require(window.isVisible(),"hidden scope can reopen");
        monitor.close(); SwingUtilities.invokeAndWait(() -> {});
        require(!window.isDisplayable(),"timer/window shutdown releases desktop resources");
        System.out.println("PASS: immutable telemetry, stale data, map/tile interactions, mouse delta rendering/clearing, scaled hit coordinates, no-focus window, hide/reopen/close.");
        System.out.println("UI preview (synthetic neural values): "+preview.toAbsolutePath());
    }
}
