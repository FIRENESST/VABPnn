package cn.vabpnn.scorer;
import com.google.gson.*;
import com.sun.net.httpserver.*;
import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

public final class ScorerApp {
 static HttpServer server(Engine engine,int port)throws IOException{
  HttpServer server=HttpServer.create(new InetSocketAddress("127.0.0.1",port),16);
  server.setExecutor(Executors.newFixedThreadPool(2,r->{Thread t=new Thread(r,"scorer-http");t.setDaemon(true);return t;}));
  server.createContext("/",exchange->{
   int status=200;JsonObject result=new JsonObject();
   try {
    String host=exchange.getRequestHeaders().getFirst("Host");
    if(exchange.getRequestHeaders().containsKey("Origin")||host==null||!(host.equals("127.0.0.1:"+server.getAddress().getPort())||host.equals("localhost:"+server.getAddress().getPort())))throw new SecurityException("只接受本地应用请求");
    String path=exchange.getRequestURI().getPath(),method=exchange.getRequestMethod();
    if((path.equals("/v1/telemetry")||path.equals("/v1/game-status")||path.equals("/v1/training-status"))&&method.equals("POST")){
     String type=exchange.getRequestHeaders().getFirst("Content-Type");if(type==null||!type.startsWith("application/json"))throw new IllegalArgumentException("需要 application/json");
     byte[] data=exchange.getRequestBody().readNBytes(2_000_001);if(data.length>2_000_000)throw new IllegalArgumentException("请求超过 2 MB");
     JsonObject body=JsonParser.parseString(new String(data,StandardCharsets.UTF_8)).getAsJsonObject();result=switch(path){case "/v1/telemetry"->engine.accept(body);case "/v1/game-status"->engine.receiveGame(body);default->engine.receiveTraining(body);};
    } else if(method.equals("GET")){
     switch(path){
      case "/health" -> {result.addProperty("service","vabpnn-scorer");result.addProperty("schema",1);result.addProperty("version","0.5.0");result.addProperty("training_protocol","vabpnn-reward-v1");}
      case "/v1/collector-config" -> result=Json.obj(engine.config(),"collector");
      case "/v1/status" -> result=engine.status();
      case "/v1/training" -> result=engine.trainingInfo();
      case "/v1/logs" -> {var query=query(exchange.getRequestURI());result=engine.logsAfter(Long.parseLong(query.getOrDefault("after","0")),"");}
      case "/v1/rewards" -> {var query=query(exchange.getRequestURI());result=engine.rewardAfter(Long.parseLong(query.getOrDefault("after","0")),query.getOrDefault("stream_id",""));}
      default -> {status=404;result.addProperty("error","unknown endpoint");}
     }
    }else{status=405;result.addProperty("error","method not allowed");}
   }catch(SecurityException e){status=403;result.addProperty("error",e.getMessage());}
   catch(IllegalArgumentException|IllegalStateException e){status=400;result.addProperty("error",e.getMessage());}
   catch(Exception e){status=500;result.addProperty("error",e.toString());}
   byte[] bytes=Json.COMPACT.toJson(result).getBytes(StandardCharsets.UTF_8);exchange.getResponseHeaders().set("Content-Type","application/json; charset=utf-8");exchange.getResponseHeaders().set("Cache-Control","no-store");exchange.sendResponseHeaders(status,bytes.length);
   try(OutputStream out=exchange.getResponseBody()){out.write(bytes);}finally{exchange.close();}
  });server.start();return server;
 }
 private static Map<String,String> query(URI uri){
  Map<String,String> values=new HashMap<>();String raw=uri.getRawQuery();if(raw==null||raw.isEmpty())return values;
  for(String part:raw.split("&")){String[] pair=part.split("=",2);String key=URLDecoder.decode(pair[0],StandardCharsets.UTF_8),value=URLDecoder.decode(pair.length==2?pair[1]:"",StandardCharsets.UTF_8);if(values.putIfAbsent(key,value)!=null)throw new IllegalArgumentException("重复查询参数："+key);}return values;
 }
 static JPanel panel(Engine e,LlmAdvisor advisor){
  JPanel root=new JPanel(new BorderLayout(12,12));root.setBorder(BorderFactory.createEmptyBorder(18,20,18,20));
  JPanel heading=new JPanel(new GridLayout(0,1,0,5));JLabel title=new JLabel("VABPnn  ·  Scorer");title.setFont(title.getFont().deriveFont(Font.BOLD,25));heading.add(title);
  JLabel status=new JLabel("等待遥测 · 127.0.0.1:9876");heading.add(status);JLabel modelStatus=new JLabel("本地模型：127.0.0.1:8080");heading.add(modelStatus);root.add(heading,BorderLayout.NORTH);
  JTabbedPane tabs=new JTabbedPane();root.add(tabs);
  DefaultTableModel scores=new DefaultTableModel(new String[]{"评分项","本次有效量","权重","本次分数"},0){public boolean isCellEditable(int r,int c){return false;}};
  JTable scoreTable=new JTable(scores);scoreTable.setRowHeight(26);JScrollPane scoreScroll=new JScrollPane(scoreTable);scoreScroll.setColumnHeaderView(scoreTable.getTableHeader());tabs.addTab("实时评分",scoreScroll);
  String[] columns={"评分项 / ID","生存","资源","铁器","钻石","下界","末地","上限 / 分钟","锁定 LLM"};
  DefaultTableModel weights=new DefaultTableModel(columns,0){public boolean isCellEditable(int r,int c){return c>0;}public Class<?> getColumnClass(int c){return c==8?Boolean.class:c==0?String.class:Double.class;}};
  JTable weightTable=new JTable(weights);weightTable.setRowHeight(26);weightTable.getColumnModel().getColumn(0).setPreferredWidth(240);
  JPanel wp=new JPanel(new BorderLayout(8,8));JScrollPane weightScroll=new JScrollPane(weightTable);weightScroll.setColumnHeaderView(weightTable.getTableHeader());wp.add(weightScroll);JPanel buttons=new JPanel(new FlowLayout(FlowLayout.LEFT));
  JButton load=new JButton("读取当前权重"),save=new JButton("应用表格修改");buttons.add(load);buttons.add(save);buttons.add(new JLabel("正值奖励，负值惩罚；0 关闭。上限按指标有效量计算。"));wp.add(buttons,BorderLayout.SOUTH);tabs.addTab("各阶段权重",wp);
  long[] editingVersion={e.version()};
  Runnable reload=()->{weights.setRowCount(0);JsonObject c=e.config();Set<String> locked=new HashSet<>();for(JsonElement x:Json.arr(c,"locked_metrics"))locked.add(x.getAsString());for(int i=0;i<Config.METRICS.length;i++){String m=Config.METRICS[i];Object[] row=new Object[9];row[0]=Config.LABELS[i]+" / "+m;for(int j=0;j<6;j++)row[j+1]=Json.num(Json.obj(Json.obj(c,"weights"),m),Config.STAGES[j],0);row[7]=Json.num(Json.obj(c,"caps"),m,20);row[8]=locked.contains(m);weights.addRow(row);}editingVersion[0]=e.version();};reload.run();
  load.addActionListener(a->reload.run());save.addActionListener(a->{try{if(weightTable.isEditing())weightTable.getCellEditor().stopCellEditing();if(editingVersion[0]!=e.version())throw new IllegalStateException("期间发生自动调权，请先读取当前权重再编辑");JsonObject c=e.config();JsonArray locked=new JsonArray();for(int i=0;i<Config.METRICS.length;i++){String m=Config.METRICS[i];for(int j=0;j<6;j++)Json.obj(Json.obj(c,"weights"),m).addProperty(Config.STAGES[j],Double.parseDouble(weights.getValueAt(i,j+1).toString()));Json.obj(c,"caps").addProperty(m,Double.parseDouble(weights.getValueAt(i,7).toString()));if(Boolean.TRUE.equals(weights.getValueAt(i,8)))locked.add(m);}c.add("locked_metrics",locked);e.configure(c,"manual-table");reload.run();}catch(Exception x){error(root,x);}});
  tabs.addTab("大模型调权",llmPanel(e,advisor));
  JTextArea config=new JTextArea(Json.G.toJson(e.config()));config.setFont(new Font(Font.MONOSPACED,Font.PLAIN,13));JPanel cp=new JPanel(new BorderLayout(8,8));cp.add(new JScrollPane(config));
  JPanel cb=new JPanel(new FlowLayout(FlowLayout.LEFT));JButton configLoad=new JButton("读取完整参数"),configSave=new JButton("校验并应用参数"),analyze=new JButton("立即请求大模型分析");cb.add(configLoad);cb.add(configSave);cb.add(analyze);cp.add(cb,BorderLayout.SOUTH);
  long[] configVersion={e.version()};configLoad.addActionListener(a->{config.setText(Json.G.toJson(e.config()));configVersion[0]=e.version();});configSave.addActionListener(a->{try{if(configVersion[0]!=e.version())throw new IllegalStateException("配置已变化，请先重新读取");e.configure(JsonParser.parseString(config.getText()).getAsJsonObject(),"manual-json");configVersion[0]=e.version();reload.run();}catch(Exception x){error(root,x);}});analyze.addActionListener(a->advisor.requestNow());tabs.addTab("全部参数",cp);
  JTextArea telemetry=new JTextArea(),inference=new JTextArea();for(JTextArea area:new JTextArea[]{telemetry,inference}){area.setEditable(false);area.setFont(new Font(Font.MONOSPACED,Font.PLAIN,13));}
  tabs.addTab("观测与覆盖范围",new JScrollPane(telemetry));tabs.addTab("推断与使用说明",new JScrollPane(inference));
  inference.setText("Scorer 提供游戏奖励。接入的模型在启用自训练后，根据实际执行动作配对奖励并更新权重；训练范围和启停见模型版本说明。\n\n每项评分 = 观测增量 × 当前阶段权重；总分截断到 reward_clip。\n首次连接、跨世界、跨会话和长时间缺测只建立基线，不补发历史奖励。\n库存用持久化历史最高值；探索、结构、进度、挖掘位置持久去重。\n\n高级行为是带证据的规则推断，不能直接观察模型的意图。\n寻找资源：探索后得到资源或进度。\n行前准备：资源增长并具备食物、健康和工具。\n有效新策略：新行为组合带来实际进展，具有冷却时间。\n\n所有参数可在完整 JSON 中编辑。stage_override=auto 为自动阶段。\nLLM 的 model 为空时调用 /v1/models；可填写明确模型 ID。\n自动调权每 60 秒一次；无新观测或暂停时不调用。\n零权重和锁定项不由 LLM 改动，不允许改变奖励正负号。\n\n可见方块来自有距离上限的视线采样，覆盖率在 telemetry 中明确标记。\n附近注册表和结构来自已加载区块；不强制加载或生成新区块。\n\n日志、权重历史、原始观测、奖励和去重状态：\n"+e.dir.toAbsolutePath());
  javax.swing.Timer timer=new javax.swing.Timer(1000,a->{JsonObject s=e.status(),r=Json.obj(s,"reward");scores.setRowCount(0);for(int i=0;i<Config.METRICS.length;i++){JsonObject c=Json.obj(Json.obj(r,"components"),Config.METRICS[i]);scores.addRow(new Object[]{Config.LABELS[i],String.format(Locale.ROOT,"%.4f",Json.num(c,"amount",0)),String.format(Locale.ROOT,"%.4f",Json.num(c,"weight",0)),String.format(Locale.ROOT,"%.4f",Json.num(c,"score",0))});}if(tabs.getSelectedIndex()==4){int position=telemetry.getCaretPosition();telemetry.setText(Json.G.toJson(s));telemetry.setCaretPosition(Math.min(position,telemetry.getDocument().getLength()));}});
  LivePanels live=new LivePanels(e,advisor,tabs);tabs.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
  timer.addActionListener(a->{JsonObject s=e.status(),runtime=Json.obj(s,"runtime"),model=Json.obj(runtime,"llm");live.refresh(s);status.setText(Json.str(Json.obj(runtime,"game"),"label","")+"   |   阶段 "+e.stage()+"   |   奖励 "+String.format(Locale.ROOT,"%.4f",Json.num(Json.obj(s,"reward"),"reward",0))+"   |   遥测 "+String.format(Locale.ROOT,"%.1f",Json.num(s,"age_seconds",-1))+" 秒前");modelStatus.setText("大模型："+LivePanels.connection(Json.str(model,"connection","unknown"))+" · "+LivePanels.phase(Json.str(model,"phase","idle"))+" · "+Json.str(s,"llm_status",""));});
  for(var listener:timer.getActionListeners())listener.actionPerformed(null);
  timer.start();root.putClientProperty("timer",timer);return root;
 }
 static JPanel llmPanel(Engine e,LlmAdvisor advisor){
  JsonObject llm=Json.obj(e.config(),"llm");JPanel panel=new JPanel(new BorderLayout(12,12)),form=new JPanel(new GridBagLayout());panel.setBorder(BorderFactory.createEmptyBorder(18,18,18,18));
  JTextField url=new JTextField(Json.str(llm,"base_url",""),38),model=new JTextField(Json.str(llm,"model",""),38);JPasswordField key=new JPasswordField(Json.str(llm,"api_key",""),38);
  JCheckBox enabled=new JCheckBox("每分钟根据最近的游戏进展自动调权",Json.bool(llm,"enabled",true));
  JSpinner interval=new JSpinner(new SpinnerNumberModel((int)Json.num(llm,"interval_seconds",60),60,3600,10)),relative=new JSpinner(new SpinnerNumberModel(Json.num(llm,"max_relative_change",.2),0.0,1.0,.05)),absolute=new JSpinner(new SpinnerNumberModel(Json.num(llm,"max_absolute_change",.25),0.0,10.0,.05)),max=new JSpinner(new SpinnerNumberModel(Json.num(llm,"max_weight",20),0.0,100.0,1.0));
  String[] labels={"自动调权","服务地址","模型 ID（空则自动查询）","API Key（可空）","间隔 / 秒","单次相对变化上限","单次绝对变化上限","自动权重绝对值上限"};JComponent[] fields={enabled,url,model,key,interval,relative,absolute,max};
  for(int i=0;i<labels.length;i++){GridBagConstraints g=new GridBagConstraints();g.gridx=0;g.gridy=i;g.anchor=GridBagConstraints.WEST;g.insets=new Insets(7,7,7,16);form.add(new JLabel(labels[i]),g);g.gridx=1;g.weightx=1;g.fill=GridBagConstraints.HORIZONTAL;form.add(fields[i],g);}panel.add(form,BorderLayout.NORTH);
  JTextArea note=new JTextArea("接口：/v1/chat/completions。model 为空时先读取 /v1/models。\n\n仅发送一分钟统计摘要，不发送完整 NBT。模型只能建议已有评分项的权重；\n不能改变奖励正负号、启用零权重项、覆盖锁定项或修改评分规则。\n\n服务未启动、超时或返回无效内容时，继续使用当前权重。\n\n调权理由和实际应用前后的权重保存在 data/llm.jsonl。");note.setEditable(false);note.setLineWrap(true);note.setWrapStyleWord(true);panel.add(note);
  JPanel actions=new JPanel(new FlowLayout(FlowLayout.LEFT));JButton apply=new JButton("保存大模型设置"),now=new JButton("立即分析最近一分钟");actions.add(apply);actions.add(now);panel.add(actions,BorderLayout.SOUTH);
  apply.addActionListener(a->{try{JsonObject c=e.config(),l=Json.obj(c,"llm");l.addProperty("enabled",enabled.isSelected());l.addProperty("base_url",url.getText().trim());l.addProperty("model",model.getText().trim());l.addProperty("api_key",new String(key.getPassword()));l.addProperty("interval_seconds",((Number)interval.getValue()).intValue());l.addProperty("max_relative_change",((Number)relative.getValue()).doubleValue());l.addProperty("max_absolute_change",((Number)absolute.getValue()).doubleValue());l.addProperty("max_weight",((Number)max.getValue()).doubleValue());e.configure(c,"manual-llm");}catch(Exception x){error(panel,x);}});now.addActionListener(a->advisor.requestNow());return panel;
 }
 private static void error(Component p,Exception e){JOptionPane.showMessageDialog(p,e.getMessage(),"参数未应用",JOptionPane.ERROR_MESSAGE);}
 public static void main(String[] args)throws Exception {
  Path dir=Path.of("data");boolean headless=false;int port=9876;for(int i=0;i<args.length;i++){if(args[i].equals("--headless"))headless=true;else if(args[i].equals("--data"))dir=Path.of(args[++i]);else if(args[i].equals("--port"))port=Integer.parseInt(args[++i]);else throw new IllegalArgumentException("未知参数 "+args[i]);}
  Engine engine=new Engine(dir);HttpServer http=server(engine,port);LlmAdvisor advisor=new LlmAdvisor(engine);
  Runtime.getRuntime().addShutdownHook(new Thread(()->{advisor.close();http.stop(0);try{engine.save();}catch(Exception x){x.printStackTrace();}}));
  System.out.println("Scorer ready: http://127.0.0.1:"+port+" ; data="+dir.toAbsolutePath());
  if(!headless)SwingUtilities.invokeLater(()->{try{UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());}catch(Exception ignored){}JFrame f=new JFrame("VABPnn Scorer — 本地训练评分器");f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);f.setContentPane(panel(engine,advisor));f.setSize(1120,800);f.setMinimumSize(new Dimension(900,600));f.setLocationByPlatform(true);f.setVisible(true);});
 }
}

