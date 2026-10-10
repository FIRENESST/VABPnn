package cn.vabpnn.mod.client.scope;

import javax.swing.JPanel;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import static cn.vabpnn.mod.client.scope.ScopeSnapshot.*;

/** AtomSpike-inspired dark scope; all drawn activity and connections come from observed GPU state. */
public final class ScopePanel extends JPanel {
    public static final int LOGICAL_WIDTH=768,LOGICAL_HEIGHT=1238;
    private static final Color BG=new Color(8,11,16),PANEL=new Color(14,19,27),LINE=new Color(32,44,58);
    private static final Color TEXT=new Color(226,234,244),MUTED=new Color(122,138,158);
    private static final Color CYAN=new Color(90,228,255),AMBER=new Color(255,186,74);
    private static final Color GREEN=new Color(86,232,164),PINK=new Color(255,92,176);
    private static final Color[] EDGE_COLORS={CYAN,AMBER,GREEN,PINK};
    private static final Font SMALL=new Font("Microsoft YaHei",Font.PLAIN,12);
    private static final Font BODY=new Font("Microsoft YaHei",Font.PLAIN,13);
    private static final Font TITLE=new Font("Microsoft YaHei",Font.BOLD,20);
    private static final Font DELTA=new Font("Consolas",Font.BOLD,17);
    private static final int HISTORY=180;
    private final AtomicReference<ScopeSnapshot> latest;
    private final AtomicReference<NetworkMonitor.Status> status;
    private final AtomicReference<MouseMotionTracker.Sample> motion;
    private final AtomicInteger inspectedTile;
    private final BufferedImage overview=new BufferedImage(120,64,BufferedImage.TYPE_INT_RGB);
    private final BufferedImage raster=new BufferedImage(HISTORY,64,BufferedImage.TYPE_INT_RGB);
    private final float[] glowPositive=new float[CELLS],glowNegative=new float[CELLS],timing=new float[HISTORY];
    private ScopeSnapshot shown;
    private long lastAdvance=System.nanoTime(),fpsStart=lastAdvance,lastFrame;
    private int cursor,samples,hover=-1,selected=-1,frames;
    private float positive,negative,fps;
    private int mapView;
    private final AtomicReference<NetworkMonitor.Hierarchy> hierarchy;

    public ScopePanel(AtomicReference<ScopeSnapshot> latest,AtomicReference<NetworkMonitor.Status> status,AtomicInteger tile) {
        this(latest,status,tile,new AtomicReference<>(MouseMotionTracker.EMPTY));
    }
    public ScopePanel(AtomicReference<ScopeSnapshot> latest,AtomicReference<NetworkMonitor.Status> status,
                      AtomicInteger tile,AtomicReference<MouseMotionTracker.Sample> motion) {
        this(latest,status,tile,motion,new AtomicReference<>(NetworkMonitor.Hierarchy.from(new long[14])));
    }
    public ScopePanel(AtomicReference<ScopeSnapshot> latest,AtomicReference<NetworkMonitor.Status> status,
                      AtomicInteger tile,AtomicReference<MouseMotionTracker.Sample> motion,
                      AtomicReference<NetworkMonitor.Hierarchy> hierarchy) {
        this.latest=latest; this.status=status; inspectedTile=tile; this.motion=motion; this.hierarchy=hierarchy;
        setBackground(BG); setPreferredSize(new Dimension(LOGICAL_WIDTH,LOGICAL_HEIGHT));
        MouseAdapter mouse=new MouseAdapter() {
            @Override public void mouseMoved(MouseEvent e) {
                var point=logical(e); hover=localAt(point); repaint();
            }
            @Override public void mouseExited(MouseEvent e) { hover=-1; repaint(); }
            @Override public void mouseClicked(MouseEvent e) {
                var point=logical(e); int tileIndex=tileAt(point);
                if(tileIndex>=0) { inspectedTile.set(tileIndex); selected=-1; }
                int node=localAt(point); if(node>=0) selected=selected==node ? -1:node;
                if(point.x>=562 && point.x<=742 && point.y>=104 && point.y<=129) {
                    mapView=(mapView+1)%3; rebuildMap();
                }
                repaint();
            }
        };
        addMouseListener(mouse); addMouseMotionListener(mouse);
    }
    private Point logical(MouseEvent event) {
        double scale=Math.min(getWidth()/(double)LOGICAL_WIDTH,getHeight()/(double)LOGICAL_HEIGHT);
        return new Point((int)((event.getX()-(getWidth()-LOGICAL_WIDTH*scale)/2)/scale),
                         (int)((event.getY()-(getHeight()-LOGICAL_HEIGHT*scale)/2)/scale));
    }
    public static int tileAt(Point p) {
        if(p.x<24 || p.x>=384 || p.y<138 || p.y>=330) return -1;
        return (p.y-138)/3*120+(p.x-24)/3;
    }
    public static int localAt(Point p) {
        if(p.x<24 || p.x>=168 || p.y<395 || p.y>=539) return -1;
        return (p.y-395)/9*16+(p.x-24)/9;
    }
    public void advance(long now) {
        float decay=(float)Math.exp(-Math.max(0,(now-lastAdvance)*1e-9)/.12); lastAdvance=now;
        for(int i=0;i<CELLS;++i) { glowPositive[i]*=decay; glowNegative[i]*=decay; }
        ScopeSnapshot next=latest.get();
        if(next!=null && next.frame()!=lastFrame) {
            if(shown!=null && shown.tile()!=next.tile()) {
                java.util.Arrays.fill(glowPositive,0); java.util.Arrays.fill(glowNegative,0); hover=selected=-1;
            }
            shown=next; lastFrame=next.frame(); samples=Math.min(HISTORY,samples+1);
            positive=negative=0;
            for(int t=0;t<TILES;++t) { positive+=next.value(MAP_POS+t); negative+=next.value(MAP_NEG+t); }
            positive/=TILES; negative/=TILES;
            for(int i=0;i<CELLS;++i) {
                int spike=next.discrete[LOCAL_SPIKE+i];
                if(spike>0) glowPositive[i]=1; if(spike<0) glowNegative[i]=1;
            }
            for(int row=0;row<64;++row) {
                float pos=0,neg=0;
                for(int j=0;j<4;++j) { int s=next.discrete[LOCAL_SPIKE+row*4+j]; pos+=s>0 ? .25f:0; neg+=s<0 ? .25f:0; }
                raster.setRGB(cursor,row,pulseColor(pos,neg).getRGB());
            }
            timing[cursor]=next.meta[2]/1000.0f; cursor=(cursor+1)%HISTORY; rebuildMap();
        }
        frames++;
        if(now-fpsStart>=500_000_000L) { fps=frames*1e9f/(now-fpsStart); frames=0; fpsStart=now; }
    }
    private void rebuildMap() {
        if(shown==null) return;
        int[] pixels=((DataBufferInt)overview.getRaster().getDataBuffer()).getData();
        for(int i=0;i<TILES;++i) {
            pixels[i]=mapView==1 ? pulseColor(shown.value(MAP_POS+i),shown.value(MAP_NEG+i)).getRGB()
                : mapView==2 ? membraneColor(shown.value(BLOCK_SCORE+i),.2f).getRGB()
                : membraneColor(shown.value(MAP_MEM+i),shown.value(0)).getRGB();
        }
    }
    private static Color mix(Color a,Color b,float t) {
        t=clamp(t); return new Color((int)(a.getRed()+(b.getRed()-a.getRed())*t),
            (int)(a.getGreen()+(b.getGreen()-a.getGreen())*t),(int)(a.getBlue()+(b.getBlue()-a.getBlue())*t));
    }
    private static Color membraneColor(float value,float threshold) {
        return mix(new Color(21,20,38),AMBER,(float)Math.pow(clamp(value/Math.max(.01f,threshold)),.85));
    }
    private static Color pulseColor(float positive,float negative) {
        Color base=new Color(10,18,26);
        return mix(mix(base,CYAN,clamp(positive)*.95f),PINK,clamp(negative)*.95f);
    }
    private static String fmt(String pattern,Object... values) { return String.format(Locale.ROOT,pattern,values); }
    private static void text(Graphics2D g,String value,int x,int y,Color color) {
        g.setColor(color); g.drawString(value,x,y);
    }
    private static void panel(Graphics2D g,int x,int y,int w,int h) {
        g.setColor(PANEL); g.fillRoundRect(x,y,w,h,12,12); g.setColor(LINE); g.drawRoundRect(x,y,w,h,12,12);
    }
    private static void meter(Graphics2D g,int x,int y,int w,int h,float fraction,Color color) {
        g.setColor(BG); g.fillRoundRect(x,y,w,h,5,5); g.setColor(color);
        g.fillRoundRect(x,y,(int)(w*clamp(fraction)),h,5,5);
    }
    @Override protected void paintComponent(Graphics original) {
        super.paintComponent(original);
        var g=(Graphics2D)original.create();
        double scale=Math.min(getWidth()/(double)LOGICAL_WIDTH,getHeight()/(double)LOGICAL_HEIGHT);
        g.translate((getWidth()-LOGICAL_WIDTH*scale)/2,(getHeight()-LOGICAL_HEIGHT*scale)/2); g.scale(scale,scale);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
        header(g); overview(g); local(g); history(g); actions(g); mouseMotion(g); hierarchy(g); learning(g);
        g.setFont(SMALL);
        text(g,"F7 显示/隐藏  ·  F8 切换控制  ·  F9 暂停  ·  F10 保存",24,LOGICAL_HEIGHT-9,MUTED);
        text(g,"GPU 状态采样 10 Hz  ·  窗口 "+fmt("%.0f",fps)+(fps>0 ? " fps":" 待计时"),460,LOGICAL_HEIGHT-9,MUTED);
        g.dispose();
    }
    private void header(Graphics2D g) {
        panel(g,12,12,744,76); g.setFont(TITLE); text(g,"VABPnn  ·  脉冲网络",24,42,TEXT);
        var state=status.get();
        String label=state==null ? "等待数据":!state.capturing() ? "采集暂停":state.modelControl() ? "模型控制":"人工示范";
        Color color=state!=null && state.capturing() ? GREEN:AMBER;
        if(shown==null) { label="等待 GPU 数据"; color=AMBER; }
        else if(!shown.valid()) { label="数据无效"; color=PINK; }
        else if(shown.stale(System.nanoTime()) && state!=null && state.capturing()) { label="等待新帧"; color=AMBER; }
        if(state!=null && state.note().contains("测试数据")) { label="测试数据预览"; color=AMBER; }
        g.setColor(color); g.fillOval(578,28,8,8); g.setFont(BODY); text(g,label,598,38,color);
        g.setFont(SMALL);
        String frame=shown==null ? "—":Long.toString(shown.frame());
        String latency=shown==null ? "—":fmt("%.2f ms",shown.meta[2]/1000.0);
        text(g,"1,966,080 神经元    正脉冲 "+fmt("%.1f%%",positive*100)+" / 负脉冲 "+fmt("%.1f%%",negative*100),24,66,MUTED);
        text(g,"帧 "+frame+"   GPU "+latency,475,66,MUTED);
    }
    private void overview(Graphics2D g) {
        panel(g,12,98,744,248); g.setFont(BODY);
        text(g,"全网 "+(mapView==1 ? "正负脉冲":mapView==2 ? "高层评分":"膜电位")+"  ·  每格一个 16×16 块",24,122,CYAN);
        g.setColor(new Color(25,34,47)); g.fillRoundRect(562,104,180,24,6,6);
        g.setFont(SMALL); text(g,mapView==0 ? "点击切换：正负脉冲":mapView==1 ? "点击切换：高层评分":"点击切换：膜电位",574,121,MUTED);
        g.setColor(BG); g.fillRect(24,138,360,192);
        if(shown!=null) {
            g.drawImage(overview,24,138,360,192,null);
            int edges=Math.clamp(shown.discrete[6],0,EDGES);
            g.setStroke(new BasicStroke(.65f));
            for(int i=0;i<edges;i+=4) {
                int src=shown.discrete[EDGE_PAIRS+i*2],encoded=shown.discrete[EDGE_PAIRS+i*2+1],dst=encoded&((1<<21)-1);
                if(src<0 || src>=1966080 || dst>=1966080) continue;
                int kind=(encoded>>>21)&3; Color c=EDGE_COLORS[kind];
                g.setColor(new Color(c.getRed(),c.getGreen(),c.getBlue(),44));
                g.drawLine(24+src%1920*360/1920,138+src/1920*192/1024,
                    24+dst%1920*360/1920,138+dst/1920*192/1024);
            }
            int anchors=Math.clamp((int)shown.meta[11],0,256);
            g.setColor(GREEN);
            for(int i=0;i<anchors;++i) {
                int id=shown.discrete[ANCHORS+i]; if(id<0 || id>=1966080) continue;
                g.fillOval(23+id%1920*360/1920,137+id/1920*192/1024,3,3);
            }
            g.setStroke(new BasicStroke(1.2f)); g.setColor(TEXT);
            int tile=inspectedTile.get(); g.drawRect(24+tile%120*3,138+tile/120*3,3,3);
        }
        g.setFont(BODY);
        text(g,"连续显存 → CSR / LIF → 13 路决策",406,153,TEXT);
        g.setFont(SMALL);
        text(g,"点击左图检查一个块；绿点为高贡献节点",406,179,MUTED);
        text(g,"彩色细线是实际长程边的抽样",406,200,MUTED);
        long local=shown==null ? 0:shown.meta[10],longEdges=shown==null ? 0:shown.meta[12];
        text(g,fmt("块内 %,d / 长程 %,d",local,longEdges),406,227,CYAN);
        text(g,fmt("高贡献 %d / 256  ·  重连 %d 次",shown==null ? 0:shown.meta[11],shown==null ? 0:shown.meta[13]),406,248,MUTED);
        double ms=shown==null ? 0:shown.meta[2]/1000.0;
        text(g,"本帧算子 "+fmt("%.2f",ms)+" / 10 ms",406,275,ms>10 ? PINK:GREEN);
        meter(g,406,286,328,8,(float)(ms/10),ms>10 ? PINK:GREEN);
        text(g,"超预算 "+(shown==null ? 0:shown.meta[4])+" 次  ·  历史最大 "+fmt("%.2f",shown==null ? 0:shown.meta[3]/1000.0)+" ms",406,318,MUTED);
    }
    private void local(Graphics2D g) {
        panel(g,12,358,370,190); g.setFont(BODY);
        int tile=shown==null ? inspectedTile.get():shown.tile();
        text(g,"块 "+tile+" · 16×16 神经元 / CSR",24,382,AMBER);
        int focus=selected>=0 ? selected:hover>=0 ? hover:128;
        g.setColor(BG); g.fillRect(24,395,144,144);
        if(shown!=null) {
            for(int slot=0;slot<4;++slot) {
                int edge=shown.discrete[OUTGOING+focus*4+slot]; if((edge&1024)==0) continue;
                int target=edge&255; g.setColor(EDGE_COLORS[(edge>>>8)&3]); g.setStroke(new BasicStroke(1.6f));
                g.drawLine(28+focus%16*9,399+focus/16*9,28+target%16*9,399+target/16*9);
            }
            for(int i=0;i<CELLS;++i) {
                Color color=membraneColor(shown.value(LOCAL_MEM+i),shown.value(0));
                color=mix(mix(color,CYAN,glowPositive[i]),PINK,glowNegative[i]);
                g.setColor(color); g.fillOval(25+i%16*9,396+i/16*9,6,6);
            }
            g.setColor(TEXT); g.setStroke(new BasicStroke(1));
            g.drawOval(24+focus%16*9,395+focus/16*9,8,8);
        }
        g.setFont(SMALL); int count=0;
        if(shown!=null) for(int slot=0;slot<4;++slot) count+=(shown.discrete[OUTGOING+focus*4+slot]&1024)!=0 ? 1:0;
        text(g,"神经元 "+focus+"  ·  出度 "+count+" / 4",183,411,TEXT);
        text(g,"悬停 / 点击查看它的实际出边",183,433,MUTED);
        String[] kinds={"刺激更容易","刺激更困难","正向更容易","逆向更容易"};
        for(int i=0;i<4;++i) {
            g.setColor(EDGE_COLORS[i]); g.fillRect(184,447+i*20,14,3);
            text(g,kinds[i],207,452+i*20,MUTED);
        }
        if(tile!=inspectedTile.get()) text(g,"等待选中块 "+inspectedTile.get(),183,536,AMBER);
    }
    private void history(Graphics2D g) {
        panel(g,392,358,364,190); g.setFont(BODY); text(g,"脉冲光栅  ·  时间向右",404,382,GREEN);
        g.setColor(BG); g.fillRect(404,395,340,84);
        int split=HISTORY-cursor,x=404+split*340/HISTORY;
        g.drawImage(raster,404,395,x,479,cursor,0,HISTORY,64,null);
        if(cursor>0) g.drawImage(raster,x,395,744,479,0,0,cursor,64,null);
        g.setFont(SMALL); text(g,"青：正向  /  粉：逆向  ·  最近 180 次采样",404,495,MUTED);
        g.setColor(BG); g.fillRect(404,507,340,28);
        g.setColor(AMBER); g.drawLine(404,516,744,516);
        g.setColor(CYAN); int px=404,py=535;
        for(int i=0;i<HISTORY;++i) {
            int index=(cursor+i)%HISTORY,nx=404+i*340/(HISTORY-1),ny=535-(int)(Math.clamp(timing[index]/15,0,1)*28);
            if(i>0) g.drawLine(px,py,nx,ny); px=nx; py=ny;
        }
        text(g,"GPU 耗时 · 橙线 10 ms",526,541,MUTED);
    }
    private void actions(Graphics2D g) {
        panel(g,12,560,744,276); g.setFont(BODY); text(g,"13 路动作  ·  后验概率 / 学习阈值 / 原始票数",24,586,PINK);
        var state=status.get();
        boolean executing=state!=null && state.capturing() && state.modelControl() && state.inputActive()
            && state.appliedFrame()>0;
        g.setFont(SMALL); text(g,executing ? "执行包 #"+state.appliedFrame():state!=null && state.modelControl() && state.inputActive() ? "等待有效指令":"仅显示模型建议",585,586,executing ? GREEN:MUTED);
        text(g,"概率       橙线：阈值 · 灰线：先验",126,605,MUTED); text(g,"状态",504,605,MUTED); text(g,"原始票数",632,605,MUTED);
        for(int k=0;k<ACTIONS;++k) {
            int y=624+k*16;
            float probability=shown==null ? 0:shown.probability(k),threshold=shown==null ? .5f:shown.threshold(k);
            boolean selected=shown!=null && shown.valid() && shown.selected(k),eligible=shown!=null && shown.valid() && shown.eligible(k);
            text(g,(k+1)+" "+ACTION_NAMES[k],24,y,TEXT);
            meter(g,126,y-10,300,10,probability,selected ? GREEN:CYAN);
            g.setColor(MUTED); int prior=126+(int)(300*(shown==null ? .5f:shown.prior(k))); g.drawLine(prior,y-11,prior,y);
            g.setColor(AMBER); int tick=126+(int)(300*threshold); g.drawLine(tick,y-12,tick,y+1);
            text(g,shown==null ? "—":fmt("%.1f%%",probability*100),438,y,selected ? GREEN:TEXT);
            boolean applied=executing && (state.appliedMask() & (1<<k))!=0;
            boolean held=executing && ((state.heldKeys() & (1<<k))!=0
                    || k==7 && (state.heldButtons()&1)!=0 || k==8 && (state.heldButtons()&2)!=0);
            String label=held ? "按住":applied ? (k==5 || k==6 || k==11 ? "许可":"已应用"):selected ? "选中":eligible ? "候选":"—";
            text(g,label,504,y,held || applied ? GREEN:selected ? CYAN:MUTED);
            String votes=shown==null ? "—":fmt("%+,d",shown.discrete[VOTES+k]);
            text(g,votes,731-g.getFontMetrics().stringWidth(votes),y,MUTED);
        }
    }
    private void mouseMotion(Graphics2D g) {
        panel(g,12,848,744,116); g.setFont(BODY);
        text(g,"鼠标位移 · Δx / Δy",24,871,CYAN);
        var sample=motion.get(); var state=status.get();
        boolean fresh=sample!=null && sample.fresh(System.nanoTime()) && state!=null
                && state.capturing() && state.inputActive();
        g.setFont(SMALL);
        text(g,fresh ? fmt("最近 %.1f ms 累计 · 原始输入单位",sample.intervalNanos()/1e6)
                : "暂停 / 等待输入 · 原始输入单位",406,871,MUTED);
        motionRow(g,"人工输入",898,fresh ? sample.humanDx():0,fresh ? sample.humanDy():0,CYAN);
        motionRow(g,"模型指令",921,fresh ? sample.predictedDx():0,fresh ? sample.predictedDy():0,AMBER);
        motionRow(g,"模型执行",944,fresh ? sample.executedDx():0,fresh ? sample.executedDy():0,GREEN);
    }
    private static void motionRow(Graphics2D g,String label,int y,double dx,double dy,Color color) {
        g.setFont(BODY); text(g,label,24,y,color);
        g.setFont(DELTA);
        text(g,fmt("Δx %+.3f",dx==0 ? 0:dx),316,y,color);
        text(g,fmt("Δy %+.3f",dy==0 ? 0:dy),548,y,color);
    }
    private void hierarchy(Graphics2D g) {
        panel(g,12,976,744,128); g.setFont(BODY);
        var h=hierarchy.get();
        String mode=shown==null ? "等待数据":shown.discrete[BLOCK_DATA]==0 ? "关闭":
                switch(shown.discrete[BLOCK_DATA+1]) { case 0 -> "独立"; case 2 -> "竞争"; default -> "合作"; };
        text(g,"7,680 高层控制器  ·  "+mode+"  ·  当前块 "+(shown==null ? "—":shown.tile()),24,999,CYAN);
        g.setFont(SMALL);
        int examples=shown==null ? 0:shown.discrete[BLOCK_DATA+2];
        text(g,fmt("方向一致率 %.1f%%  ·  覆盖 %.1f%%  ·  直接证据贡献 %.4f  ·  示例 %,d",
                shown==null ? 0:shown.value(BLOCK_SELECTED+1)*100,
                shown==null ? 0:shown.value(BLOCK_SELECTED+2)*100,
                shown==null ? 0:shown.value(BLOCK_SELECTED),examples),24,1022,TEXT);
        text(g,fmt("阈值 ×%.3f  ·  极性 %+.4f  ·  膜时间常数 ×%.3f  ·  学习率 ×%.3f",
                shown==null ? 1:shown.value(BLOCK_SELECTED+3),shown==null ? 0:shown.value(BLOCK_SELECTED+4),
                shown==null ? 1:shown.value(BLOCK_SELECTED+5),shown==null ? 1:shown.value(BLOCK_SELECTED+6)),24,1044,MUTED);
        text(g,fmt("策略更新 %,d  ·  评分帧 %,d  ·  排名 %,d  ·  累计变异 %,d  ·  最近处罚 %d",
                h.updates(),h.samples(),h.rankings(),h.mutations(),h.punished()),24,1066,MUTED);
        text(g,examples<32 ? "预热中：前 32 个真实示范样本保持中性，之后开始小幅探索":
                "人工示范评分与学习；F8 模型控制时保持推理，暂停监督更新",24,1089,AMBER);
    }
    private void learning(Graphics2D g) {
        panel(g,12,1116,744,94); g.setFont(BODY);
        text(g,"学习与队列",24,1139,CYAN);
        int k=shown==null ? 0:shown.discrete[2],chosen=shown==null ? 0:shown.discrete[3];
        text(g,"网络 k="+k+"  ·  选中 "+chosen+" 项",548,1139,MUTED);
        g.setFont(SMALL);
        long[] m=shown==null ? new long[META]:shown.meta;
        text(g,fmt("示范 %,d    像素更新 %,d    决策更新 %,d    JEPA %,d    位移训练 %,d",m[15],m[16],m[17],m[18],m[19]),24,1163,TEXT);
        String checkpoint=m[22]==2 ? "已恢复":m[22]==3 ? "已保存":m[22]<0 ? "保存异常":"待保存";
        text(g,fmt("JEPA 损失 %.5f   ·   丢帧 %,d / 跳过 %,d   ·   模型 %.1f MiB   ·   %s",
            shown==null ? 0:shown.value(2),m[8],m[9],m[20]/1048576.0,checkpoint),24,1189,MUTED);
    }
}
