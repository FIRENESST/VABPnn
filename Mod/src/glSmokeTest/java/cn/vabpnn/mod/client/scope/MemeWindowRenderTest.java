package cn.vabpnn.mod.client.scope;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.concurrent.atomic.*;
import static cn.vabpnn.mod.client.scope.MemeSnapshot.*;

public final class MemeWindowRenderTest {
    private static void require(boolean ok,String message) { if(!ok) throw new AssertionError(message); }
    public static void main(String[] args) throws Exception {
        long[] meta=new long[META],ids=new long[LONGS];float[] values=new float[FLOATS];int[] ints=new int[INTS];
        java.util.Arrays.fill(ints,ORIGIN,ORIGIN+TILES,-1);java.util.Arrays.fill(ints,SENDER,SENDER+TILES,-1);
        int inspected=32*120+60,origin=21*120+29,sender=32*120+59;
        meta[0]=10425;meta[1]=1000000000;meta[3]=inspected;meta[4]=1;meta[5]=1;meta[6]=1;
        meta[7]=10342;meta[8]=9399;meta[9]=92310;meta[10]=2085;meta[11]=743321;meta[12]=380840;
        meta[13]=643009;meta[17]=10420;meta[18]=sender;meta[19]=1;meta[21]=origin;meta[22]=36*120+85;
        meta[23]=5341;meta[24]=203;meta[25]=inspected;
        meta[26]=5341;meta[27]=203;meta[28]=10425;meta[29]=inspected;
        meta[31]=4271;meta[32]=180;meta[33]=10408;meta[34]=origin;meta[35]=4;meta[36]=1392;meta[38]=4782;meta[39]=1;
        for(int t=0;t<TILES;t++) {
            values[OWN+t]=Math.max(0,(float)(.5+.4*Math.sin(t%120*.18+t/120*.21)));
            values[IMPORTED+t]=t%7==0 ? 0:Math.max(0,(float)(.3+.2*Math.cos(t%120*.11-t/120*.27)));
            values[AGE+t]=(t%30)/20f;values[CONFIDENCE+t]=.81f;values[KNOWLEDGE+t]=.91f;
            values[COMPATIBILITY+t]=.93f;ints[ORIGIN+t]=origin;ints[HOPS+t]=4;ids[t]=10408;
        }
        ints[SENDER+inspected]=sender;ints[CHANNEL+inspected]=1;ids[TILES+inspected]=10420;
        for(int slot=0;slot<3;slot++) {
            int start=DETAIL+slot*PACKET;
            for(int k=0;k<CHANNELS;k++) values[start+k]=(float)Math.sin(k*.58+slot*.17)*.75f;
            for(int d=0;d<8;d++) values[start+CHANNELS+d]=d<3 ? .36f+d*.16f:.12f;
            values[start+23]=.67f;values[start+24]=.86f;values[start+25]=.79f;
            values[start+26]=slot==2 ? .81f:.91f;values[start+27]=slot==2 ? .32f:0;
        }
        for(int d=0;d<8;d++)values[VISUAL+d]=d<3 ? .35f+d*.16f:.1f;
        for(int k=0;k<CHANNELS;k++) { values[FEATURES+k]=.46f*(float)Math.sin(k*.58);values[ACTION+k]=.4f*(float)Math.sin(k*.58); }
        float[] config={16,8,.15f,.35f,.1f,.2f,2,.75f,.48f,1};
        System.arraycopy(config,0,values,CONFIG,10);
        long now=System.nanoTime();
        MemeSnapshot snapshot=new MemeSnapshot(meta,values,ints,ids,now);
        require(FLOATS==46212 && INTS==30720 && LONGS==15360,"CUDA/JNI/Java telemetry layout");
        meta[0]=999;values[IMPORTED+inspected]=0;ints[SENDER+inspected]=-1;ids[inspected]=0;
        require(snapshot.frame()==10425 && snapshot.value(IMPORTED+inspected)>0
                && snapshot.integer(SENDER+inspected)==sender && snapshot.identity(inspected)==10408,"all snapshot buffers immutable");
        require(!snapshot.stale(now+500000000) && snapshot.stale(now+800000000),"stopped frames are marked stale");
        var latest=new AtomicReference<>(snapshot);
        var state=new AtomicReference<>(new MemePropagationMonitor.Status(true,false,"合成遥测预览"));
        var tile=new AtomicInteger(inspected);
        Path preview=Path.of("build/native/meme-propagation-preview.png");java.nio.file.Files.createDirectories(preview.getParent());
        SwingUtilities.invokeAndWait(() -> {
            try {
                var panel=new MemePropagationPanel(latest,state,tile);panel.setSize(MemePropagationPanel.WIDTH,MemePropagationPanel.HEIGHT);
                panel.advance(now);
                var image=new BufferedImage(MemePropagationPanel.WIDTH,MemePropagationPanel.HEIGHT,BufferedImage.TYPE_INT_RGB);
                var graphics=image.createGraphics();panel.paint(graphics);graphics.dispose();ImageIO.write(image,"PNG",preview.toFile());
                panel.inspectAt(24+7*6+3,196+3*6+3);require(tile.get()==3*120+7,"full block-map click");
                panel.setSize(MemePropagationPanel.WIDTH/2,MemePropagationPanel.HEIGHT/2);
                panel.inspectAt((24+10*6+3)/2,(196+6*6+3)/2);require(tile.get()==6*120+10,"scaled map hit coordinates");
                tile.set(inspected);panel.setSize(MemePropagationPanel.WIDTH,MemePropagationPanel.HEIGHT);
                panel.inspectAt(40,620);require(tile.get()==origin,"source chip jumps to exact originating block");
                tile.set(inspected);panel.inspectAt(290,620);require(tile.get()==sender,"last-hop chip jumps to actual sending block");
                panel.inspectAt(900,850);require(tile.get()==sender,"outside map does not change selection");
            } catch(Exception e) { throw new RuntimeException(e); }
        });
        var error=new AtomicReference<String>();var monitor=new MemePropagationMonitor(error::set);
        require(!monitor.visible() && !monitor.shouldSample(1000000000),"no cold-start sampling/window");
        monitor.status(true,false,"测试");monitor.show(true);SwingUtilities.invokeAndWait(() -> {});
        require(error.get()==null && monitor.visible(),"independent desktop window opens");
        require(monitor.shouldSample(1000000000) && !monitor.shouldSample(1199999999)
                && monitor.shouldSample(1200000000) && !monitor.shouldSample(1200000001),"5 Hz sampling, no duplicate or catch-up burst");
        var windowField=MemePropagationMonitor.class.getDeclaredField("window");windowField.setAccessible(true);
        var timerField=MemePropagationMonitor.class.getDeclaredField("timer");timerField.setAccessible(true);
        JFrame window=(JFrame)windowField.get(monitor);Timer timer=(Timer)timerField.get(monitor);
        require(timer.getDelay()==200 && timer.isCoalesce(),"paint timer is exactly 5 Hz");
        require(!window.isAlwaysOnTop() && window.getFocusableWindowState() && !window.isAutoRequestFocus(),"ordinary non-topmost window without automatic focus requests");
        monitor.status(false,true,"暂停");require(!monitor.shouldSample(3000000000L),"F9 pause suppresses sampling");
        monitor.show(false);SwingUtilities.invokeAndWait(() -> {});
        require(!window.isVisible() && !timer.isRunning() && !monitor.shouldSample(4000000000L),"hide stops painting and sampling");
        monitor.status(true,true,"评估");monitor.show(true);SwingUtilities.invokeAndWait(() -> {});
        require(window.isVisible() && timer.isRunning() && monitor.shouldSample(4000000000L),"reopen restarts immediately");
        SwingUtilities.invokeAndWait(() -> window.dispatchEvent(new WindowEvent(window,WindowEvent.WINDOW_CLOSING)));
        require(!monitor.visible() && !timer.isRunning(),"title-bar close disables telemetry");
        monitor.show(true);SwingUtilities.invokeAndWait(() -> {});monitor.close();SwingUtilities.invokeAndWait(() -> {});
        require(!window.isDisplayable() && !timer.isRunning(),"shutdown disposes independent window and timer");
        System.out.println("PASS: immutable meme snapshots, real 5 Hz gates/timer, source selection, scaled map, ordinary non-topmost window, pause/hide/reopen/close.");
        System.out.println("Synthetic neural telemetry preview: "+preview.toAbsolutePath());
    }
}
