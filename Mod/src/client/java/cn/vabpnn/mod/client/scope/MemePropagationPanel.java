package cn.vabpnn.mod.client.scope;

import javax.swing.JPanel;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import static cn.vabpnn.mod.client.scope.MemeSnapshot.*;

/** Block telemetry only; cached 120 x 64 map, redrawn at 5 Hz without interpolation. */
public final class MemePropagationPanel extends JPanel {
    public static final int WIDTH=1320,HEIGHT=900,MAP_X=24,MAP_Y=196,CELL=6;
    private static final Color BG=new Color(12,18,29),CARD=new Color(20,29,43);
    private static final Color TEXT=new Color(229,238,245),MUTED=new Color(145,165,186);
    private static final Color TEAL=new Color(69,217,181),BLUE=new Color(94,169,255),GOLD=new Color(255,197,94);
    private static final Color RED=new Color(250,120,140);
    private static final String[] ACTIONS={"W","A","S","D","E","鼠标 X","鼠标 Y","左按","右按","左松","右松","滚轮","空格","Ctrl","Shift"};
    private static final String[] CONTEXT={"R 均值","G 均值","B 均值","亮度方差","移动历史","交互历史","鼠标历史","跳跃历史"};
    private final AtomicReference<MemeSnapshot> latest;
    private final AtomicReference<MemePropagationMonitor.Status> status;
    private final AtomicInteger selected;
    private final BufferedImage map=new BufferedImage(120,64,BufferedImage.TYPE_INT_RGB);
    private MemeSnapshot mapped;
    private int mapMode,mappedMode=-1;
    private long now=System.nanoTime();

    public MemePropagationPanel(AtomicReference<MemeSnapshot> latest,
                                AtomicReference<MemePropagationMonitor.Status> status,AtomicInteger selected) {
        this.latest=latest;this.status=status;this.selected=selected;
        setBackground(BG);setDoubleBuffered(true);
        addMouseListener(new MouseAdapter() {
            @Override public void mousePressed(MouseEvent event) { inspectAt(event.getX(),event.getY()); }
        });
    }
    public void advance(long nanos) { now=nanos; }
    /** Shared by the actual listener and scaled-coordinate UI smoke test. */
    public void inspectAt(int sx,int sy) {
        double x=sx*WIDTH/(double)Math.max(1,getWidth()),y=sy*HEIGHT/(double)Math.max(1,getHeight());
        if(x>=MAP_X && x<MAP_X+720 && y>=MAP_Y && y<MAP_Y+384) {
            selected.set((int)((y-MAP_Y)/CELL)*120+(int)((x-MAP_X)/CELL));return;
        }
        if(y>=158 && y<190 && x>=452 && x<744) { mapMode=Math.min(2,(int)((x-452)/98));return; }
        MemeSnapshot sample=latest.get();
        if(sample==null || !sample.selected(selected.get())) return;
        if(y>=608 && y<635) {
            int tile=x>=24 && x<262 ? sample.integer(ORIGIN+sample.tile()):
                    x>=272 && x<516 ? sample.integer(SENDER+sample.tile()):-1;
            if(tile>=0 && tile<TILES) selected.set(tile);
        }
    }
    @Override protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);
        Graphics2D g=(Graphics2D)graphics.create();
        try {
            g.scale(getWidth()/(double)WIDTH,getHeight()/(double)HEIGHT);
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            MemeSnapshot sample=latest.get();
            var state=status.get();
            text(g,"模因传播",24,40,26,TEXT);
            text(g,"VABPnn 1.4.1   /   GPU 观测 · 5 Hz",187,38,14,MUTED);
            boolean running=state.capturing() && sample!=null && !sample.stale(now);
            badge(g,running ? "实时":"等待 / 暂停",1120,18,176,running ? TEAL:GOLD);
            text(g,state.predictionOnly() ? "仅预测：权重冻结，选点 / 连接动态，不注入操作":state.evaluation() ? "连续控制：权重冻结，选点 / 连接 / 包传播继续":"示范模式：在线学习与模因传播",24,68,14,
                    state.evaluation() ? BLUE:TEAL);
            text(g,clip(state.note(),40),632,68,14,MUTED);
            card(g,24,88,300,58);card(g,348,88,300,58);card(g,672,88,300,58);card(g,996,88,300,58);
            metric(g,"本地活跃块",sample==null ? "—":sample.activeLocal()+" / 7680",44,109,TEAL);
            metric(g,"外来包活跃块",sample==null ? "—":sample.activeImported()+" / 7680",368,109,BLUE);
            metric(g,"累计接收 / 传播轮",sample==null ? "—":sample.meta(11)+" / "+sample.meta(10),692,109,GOLD);
            metric(g,"GPU 帧 / 整帧耗时",sample==null ? "—":sample.frame()+" / "+fmt(sample.meta(38)/1000.0)+" ms",1016,109,TEXT);
            text(g,"120 × 64 个块 · 每块 16 × 16 神经元",24,181,15,TEXT);
            for(int i=0;i<3;i++) badge(g,new String[]{"传播强度","原始来源","包年龄"}[i],452+i*98,159,92,
                    mapMode==i ? TEAL:MUTED);
            renderMap(g,sample);
            text(g,mapMode==0 ? "绿：本地发出   蓝：外来包   亮度：有效权重":
                    mapMode==1 ? "颜色：原始来源块   亮度：接收权重":"蓝 → 黄 → 红：包年龄 / 生存期",24,600,13,MUTED);
            if(sample!=null && sample.selected(selected.get())) {
                badge(g,"来源 "+block(sample.integer(ORIGIN+sample.tile())),24,610,238,GOLD);
                badge(g,"上一跳 "+block(sample.integer(SENDER+sample.tile())),272,610,244,BLUE);
                text(g,"点击块 / 来源 / 上一跳可检查",530,629,12,MUTED);
            }
            boolean detail=sample!=null && sample.selected(selected.get()) &&
                    (sample.meta(2)!=0)==state.evaluation();
            text(g,"检查 "+block(selected.get()),768,181,15,TEXT);
            if(detail) {
                packet(g,sample,0,196,"本地知识",TEAL);
                packet(g,sample,1,302,"本轮发出",GOLD);
                packet(g,sample,2,408,"收到的包",BLUE);
                evidence(g,sample);
                context(g,sample);
                rules(g,sample);
            } else {
                card(g,768,196,528,680);card(g,24,646,720,230);
                text(g,sample==null ? "等待第一帧 GPU 数据…":"正在读取所选块 / 当前模式的数据…",794,240,17,MUTED);
                text(g,"/vabpnn start 后开始取样，F6 开关窗口",794,273,14,MUTED);
            }
            if(sample!=null && (!state.capturing() || sample.stale(now) || !sample.enabled())) {
                badge(g,!sample.enabled() ? "模因 / 块模型关闭":sample.stale(now) ? "数据已停更":"流程暂停",294,530,250,GOLD);
            }
            text(g,"F6 开关窗口   ·   关闭窗口停止 GPU 取样   ·   实线：实际最近一跳，虚线：CSR 候选",24,892,12,MUTED);
        } finally { g.dispose(); }
    }
    private void renderMap(Graphics2D g,MemeSnapshot s) {
        if(s!=mapped || mappedMode!=mapMode) {
            for(int tile=0;tile<TILES;tile++) {
                float own=s==null ? 0:s.value(OWN+tile),foreign=s==null ? 0:s.value(IMPORTED+tile);
                int rgb=0x172337;
                if(own>0 || foreign>0) {
                    float power=(float)Math.sqrt(Math.min(1,Math.max(own,foreign)));
                    Color color;
                    if(mapMode==1 && foreign>0) color=Color.getHSBColor((s.integer(ORIGIN+tile)*0.61803398875f)%1,0.58f,1);
                    else if(mapMode==2 && foreign>0) {
                        float age=Math.min(1,s.value(AGE+tile)/Math.max(.001f,s.value(CONFIG+6)));
                        color=age<.5f ? mix(BLUE,GOLD,age*2):mix(GOLD,RED,(age-.5f)*2);
                    } else color=foreign>own ? BLUE:TEAL;
                    rgb=mix(new Color(23,35,55),color,0.18f+.82f*power).getRGB();
                }
                map.setRGB(tile%120,tile/120,rgb);
            }
            mapped=s;mappedMode=mapMode;
        }
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
        g.drawImage(map,MAP_X,MAP_Y,720,384,null);
        int tile=selected.get();
        if(s!=null && s.selected(tile) && s.value(IMPORTED+tile)>0) {
            for(int offset=21;offset<=22;offset++) if(s.meta(offset)>=0 && s.meta(offset)<TILES)
                arrow(g,(int)s.meta(offset),tile,MUTED,true);
            int sender=s.integer(SENDER+tile);
            if(sender>=0) arrow(g,sender,tile,BLUE,false);
            int origin=s.integer(ORIGIN+tile);
            if(origin>=0 && origin<TILES) {
                g.setColor(GOLD);g.setStroke(new BasicStroke(2));
                g.drawOval(MAP_X+(origin%120)*CELL-3,MAP_Y+(origin/120)*CELL-3,12,12);
            }
        }
        g.setColor(TEXT);g.setStroke(new BasicStroke(2));
        g.drawRect(MAP_X+(tile%120)*CELL-1,MAP_Y+(tile/120)*CELL-1,CELL+2,CELL+2);
    }
    private void arrow(Graphics2D g,int from,int to,Color color,boolean dashed) {
        double x0=MAP_X+from%120*CELL+3,y0=MAP_Y+from/120*CELL+3;
        double x1=MAP_X+to%120*CELL+3,y1=MAP_Y+to/120*CELL+3;
        g.setColor(color);g.setStroke(dashed ? new BasicStroke(1,0,0,1,new float[]{4,4},0):new BasicStroke(2));
        g.drawLine((int)x0,(int)y0,(int)x1,(int)y1);
        double a=Math.atan2(y1-y0,x1-x0);
        for(double sign:new double[]{-1,1}) g.drawLine((int)x1,(int)y1,(int)(x1-8*Math.cos(a+sign*.45)),(int)(y1-8*Math.sin(a+sign*.45)));
    }
    private void packet(Graphics2D g,MemeSnapshot s,int slot,int y,String label,Color color) {
        card(g,768,y,528,96);
        int start=DETAIL+slot*PACKET;
        long support=s.meta(slot==0 ? 23:slot==1 ? 26:31);
        long revision=s.meta(slot==0 ? 24:slot==1 ? 27:32);
        long sequence=slot==0 ? 0:s.meta(slot==1 ? 28:33);
        int origin=(int)s.meta(slot==0 ? 25:slot==1 ? 29:34);
        text(g,label+"  ·  "+(support==0 ? "空":block(origin)),784,y+23,15,color);
        text(g,"支持 "+support+"   修订 "+revision+"   序列 "+sequence,784,y+45,12,MUTED);
        text(g,"信度 "+fmt(s.value(start+26))+"   效用 "+fmt(s.value(start+23))+
                "   一致 "+fmt(s.value(start+24))+"   覆盖 "+fmt(s.value(start+25)),784,y+65,12,TEXT);
        if(slot==2) {
            int channel=s.integer(CHANNEL+s.tile())&255;
            String hop=channel==1 ? "网格邻接":channel==2 ? "长程 CSR":"未观测最近一跳";
            text(g,"年龄 "+fmt(s.value(start+27))+" s   跳数 "+s.meta(35)+"   "+hop+" / 帧 "+s.meta(17),784,y+84,12,MUTED);
        } else text(g,slot==0 ? "知识可保留；新观察决定是否重新发出":
                "有效权重 "+fmt(s.value(OWN+s.tile()))+"   活动包年龄 "+fmt(s.value(start+27))+" s",784,y+84,12,MUTED);
    }
    private void evidence(Graphics2D g,MemeSnapshot s) {
        card(g,24,646,720,230);
        text(g,"15 项动作证据  ·  有符号投票比例",40,665,14,TEXT);
        String[] names={"本地知识","本轮发出","收到的包","当前混合","实际动作输入"};
        int[] offsets={DETAIL,DETAIL+PACKET,DETAIL+2*PACKET,FEATURES,ACTION};
        Color[] colors={TEAL,GOLD,BLUE,TEXT,RED};
        for(int col=0;col<5;col++) text(g,names[col],130+col*121,685,12,colors[col]);
        for(int row=0;row<CHANNELS;row++) {
            int y=699+row*11;
            text(g,ACTIONS[row],40,y+5,11,MUTED);
            for(int col=0;col<5;col++) {
                float value=s.value(offsets[col]+row);int center=164+col*121;
                g.setColor(new Color(50,65,84));g.drawLine(center-28,y,center+28,y);
                g.setColor(colors[col]);int dx=(int)(28*Math.max(-1,Math.min(1,value)));
                g.fillRect(center+Math.min(0,dx),y-3,Math.max(1,Math.abs(dx)),6);
                text(g,fmt(value),center+32,y+4,10,colors[col]);
            }
        }
        text(g,"证据 ≠ 概率；实际动作输入保留模型上一轮使用的特征",40,870,11,MUTED);
    }
    private void context(Graphics2D g,MemeSnapshot s) {
        card(g,768,522,528,196);
        text(g,"视觉上下文匹配",784,546,15,TEXT);
        text(g,"当前 / 外来包",1110,546,12,BLUE);
        for(int i=0;i<8;i++) {
            int col=i/4,row=i%4,x=784+col*255,y=572+row*25;
            text(g,CONTEXT[i],x,y,12,MUTED);
            text(g,fmt(s.value(VISUAL+i))+" / "+fmt(s.value(DETAIL+2*PACKET+CHANNELS+i)),x+105,y,12,TEXT);
        }
        text(g,"匹配度 "+fmt(s.value(COMPATIBILITY+s.tile()))+"   外来有效权重 "+fmt(s.value(IMPORTED+s.tile())),784,687,12,BLUE);
        text(g,"收到的包保持原始来源 / 修订 / 序列；衰减随秒递进",784,708,12,MUTED);
    }
    private void rules(Graphics2D g,MemeSnapshot s) {
        card(g,768,730,528,146);
        String mode=s.meta(6)==0 ? "独立":s.meta(6)==2 ? "竞争":"合作";
        text(g,"传播条件 · "+mode+"   /   "+(s.enabled() ? "开启":"关闭"),784,754,15,TEXT);
        text(g,"最少支持 "+(int)s.value(CONFIG)+"   最多跳数 "+(int)s.value(CONFIG+1)+
                "   信度 ≥ "+fmt(s.value(CONFIG+4)),784,779,12,MUTED);
        text(g,"匹配 ≥ "+fmt(s.value(CONFIG+3))+"   生存期 "+fmt(s.value(CONFIG+6))+
                " s   衰减 "+fmt(s.value(CONFIG+7))+" s",784,801,12,MUTED);
        text(g,"外来混合 "+fmt(s.value(CONFIG+5))+"   最近接收权重 "+fmt(s.value(CONFIG+8)),784,823,12,TEXT);
        text(g,"投影更新 "+s.meta(36)+"   累计发出 "+s.meta(13)+"   CSR 选路 "+s.meta(12),784,847,12,MUTED);
        text(g,"取样帧一致；历史完整路径未记录，图中显示实际最近一跳",784,869,11,MUTED);
    }
    private static Color mix(Color a,Color b,float t) {
        float f=Math.max(0,Math.min(1,t));
        return new Color((int)(a.getRed()+(b.getRed()-a.getRed())*f),
                (int)(a.getGreen()+(b.getGreen()-a.getGreen())*f),(int)(a.getBlue()+(b.getBlue()-a.getBlue())*f));
    }
    private static String fmt(double v) { return String.format(Locale.ROOT,"%.2f",v); }
    private static String clip(String value,int count) { return value.length()>count ? value.substring(0,count)+"…":value; }
    private static void text(Graphics2D g,String value,int x,int y,int size,Color color) {
        g.setFont(new Font("Microsoft YaHei UI",Font.PLAIN,size));g.setColor(color);g.drawString(value,x,y);
    }
    private static void card(Graphics2D g,int x,int y,int w,int h) { g.setColor(CARD);g.fillRoundRect(x,y,w,h,12,12); }
    private static void badge(Graphics2D g,String value,int x,int y,int width,Color color) {
        g.setColor(mix(CARD,color,.12f));g.fillRoundRect(x,y,width,26,8,8);text(g,value,x+10,y+18,12,color);
    }
    private static void metric(Graphics2D g,String name,String value,int x,int y,Color color) {
        text(g,name,x,y,12,MUTED);text(g,value,x,y+25,19,color);
    }
}
