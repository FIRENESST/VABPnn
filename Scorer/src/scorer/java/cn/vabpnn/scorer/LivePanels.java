package cn.vabpnn.scorer;

import com.google.gson.*;
import javax.swing.*;
import javax.swing.table.*;
import javax.swing.event.*;
import java.awt.*;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.List;

final class LivePanels {
 private final Engine engine;
 private final JTextArea game=new JTextArea(),model=new JTextArea(),training=new JTextArea();
 private final List<LogPanel> logs=new ArrayList<>();
 LivePanels(Engine engine,LlmAdvisor advisor,JTabbedPane tabs){
  this.engine=engine;JPanel dashboard=new JPanel(new GridLayout(1,2,14,0));dashboard.setBorder(BorderFactory.createEmptyBorder(14,10,14,10));
  dashboard.add(card("游戏与采集",game));JPanel llm=card("大模型与调权",model);JButton probe=new JButton("测试模型连接");probe.addActionListener(a->advisor.testConnection());llm.add(probe,BorderLayout.SOUTH);dashboard.add(llm);tabs.addTab("运行状态",dashboard);
  for(String[] entry:new String[][]{{"game","游戏日志"},{"llm","模型通信"},{"behavior","行为与工具日志"}}){LogPanel p=new LogPanel(engine,entry[0]);logs.add(p);tabs.addTab(entry[1],p);}
  tabs.addTab("工具评分设置",toolPanel(engine));
  tabs.addTab("合成与拾取设置",progressionPanel(engine));
  tabs.addTab("模型自训练",card("奖励接入与训练状态",training));LogPanel trainingLogs=new LogPanel(engine,"training");logs.add(trainingLogs);tabs.addTab("训练通信",trainingLogs);
 }
 private static JPanel card(String title,JTextArea text){JPanel p=new JPanel(new BorderLayout(8,12));p.setBorder(BorderFactory.createTitledBorder(title));text.setEditable(false);text.setFont(new Font(Font.DIALOG,Font.PLAIN,15));text.setLineWrap(true);text.setWrapStyleWord(true);p.add(new JScrollPane(text));return p;}
 void refresh(JsonObject status){
  JsonObject runtime=Json.obj(status,"runtime"),g=Json.obj(runtime,"game"),m=Json.obj(runtime,"llm"),t=Json.obj(status,"telemetry"),p=Json.obj(t,"player"),coverage=Json.obj(t,"coverage");
  String last=Json.bool(g,"telemetry_stale",true)?"历史快照（当前未更新）":"实时快照";
  game.setText(Json.str(g,"label","等待游戏连接")+"\n\n"+"心跳年龄："+decimal(g,"heartbeat_age_seconds")+" 秒\n遥测年龄："+decimal(status,"age_seconds")+" 秒\n玩家："+Json.str(g,"player_name","—")+"\n维度："+Json.str(g,"dimension","—")+"\n\n"+last+"\n序号 / tick："+Json.num(t,"seq",0)+" / "+Json.num(t,"tick",0)+"\n生命 / 饥饿："+decimal(p,"health")+" / "+decimal(p,"food")+"\n坐标："+decimal(p,"x")+", "+decimal(p,"y")+", "+decimal(p,"z")+"\n采集耗时："+decimal(coverage,"capture_ms")+" ms\n\n传输："+Json.str(g,"transport_status","等待")+"\n已发送 / 覆盖或丢弃："+(long)Json.num(g,"sent",0)+" / "+(long)Json.num(g,"dropped",0)+"\n日志："+Json.str(Json.obj(g,"log"),"status","等待游戏日志")+"\n日志跳过字节："+(long)Json.num(Json.obj(g,"log"),"skipped_bytes",0));
  JsonObject life=Json.obj(status,"episode");
  game.append("\n\n生命轮次："+((long)Json.num(life,"index",0)+1)+"\n本轮累计奖励："+String.format(Locale.ROOT,"%.4f",Json.num(life,"total_reward",0))+"\n重生时清空本轮基线、自动阶段与记录，保留当前权重。");
  JsonObject c=Json.obj(engine.config(),"llm");
  model.setText("自动调权："+(Json.bool(c,"enabled",true)?"启用，每 "+(int)Json.num(c,"interval_seconds",60)+" 秒":"关闭（仍可手动分析）")+"\n服务："+Json.str(c,"base_url","")+"\n模型："+Json.str(m,"model",Json.str(c,"model","")).replaceAll("^$","自动发现")+"\n\n连接："+connection(Json.str(m,"connection","unknown"))+"\n任务："+phase(Json.str(m,"phase","idle"))+"\n请求编号："+(long)Json.num(m,"request_id",0)+"\n请求等待："+decimal(m,"request_elapsed_seconds")+" 秒\n最近 HTTP："+(m.has("last_http_status")?(int)Json.num(m,"last_http_status",0):"—")+"\n最近耗时："+decimal(m,"latency_ms")+" ms\n最近响应："+(m.has("last_response_ms")?localTime(Instant.ofEpochMilli((long)Json.num(m,"last_response_ms",0)).toString()):"—")+"\n\n"+Json.str(status,"llm_status","")+"\n\n请求、响应正文、错误和实际调权结果见“模型通信”。连接状态来自最近一次通信。");
  model.append("\n\n重生通知待发送："+(long)Json.num(life,"pending_llm_notifications",0)+"\n重生通知状态："+Json.str(Json.obj(life,"notification"),"state","尚无重生")+"\n通知失败会保留事件并重试，不影响新轮次评分。");
  JsonObject tr=Json.obj(status,"training");String trainingState=switch(Json.str(tr,"state","waiting")){case "connected"->"已连接，奖励会话匹配";case "unbound"->"已收到状态，奖励会话尚未匹配";case "offline"->"模型状态上报超时";default->"等待模型接入";};
  training.setText(trainingState+"\n\n模型版本："+Json.str(tr,"model_version","—")+"\n模式："+Json.str(tr,"mode","—")+"\n算法："+Json.str(tr,"algorithm","—")+"\n最近上报："+(Json.num(tr,"age_seconds",-1)<0?"尚未上报":decimal(tr,"age_seconds")+" 秒前")+"\n最近消费奖励："+(long)Json.num(tr,"last_reward_id",0)+"\n已应用奖励："+(long)Json.num(tr,"applied_rewards",0)+"\n实际训练步（模型上报）："+(long)Json.num(tr,"updates",0)+"\n丢弃奖励："+(long)Json.num(tr,"dropped_rewards",0)+"\n等待配对动作："+(long)Json.num(tr,"pending_actions",0)+"\n\n状态原因："+Json.str(tr,"reason","—")+"\n最近错误："+Json.str(tr,"last_error","—")+"\n\n在游戏中使用模型的自训练命令启停训练。模型只使用与已执行动作配对的有效奖励，断线或模式切换后重新配对。\n训练计数由模型上报；评分规则和大模型调权设置继续生效。\n\n配对与训练详情见“训练通信”。");
  logs.forEach(LogPanel::refresh);
 }
 private static String decimal(JsonObject o,String key){return o.has(key)?String.format(Locale.ROOT,"%.1f",Json.num(o,key,0)):"—";}
 static String connection(String v){return switch(v){case "reachable"->"服务可达";case "timeout"->"请求超时";case "unreachable"->"无法连接";default->"尚未通信";};}
 static String phase(String v){return switch(v){case "requesting"->"等待响应";case "analyzing"->"正在分析";case "testing"->"测试连接";case "applied"->"调权已应用";case "error"->"失败，保留原权重";case "ready"->"连接测试成功";case "waiting_game"->"等待游戏遥测";case "respawn_notifying"->"发送重生通知";case "respawn_notified"->"重生通知已送达";default->"等待分析";};}
 static String localTime(String iso){try{return DateTimeFormatter.ofPattern("HH:mm:ss.SSS").format(Instant.parse(iso).atZone(ZoneId.systemDefault()));}catch(Exception e){return iso;}}
 private static JPanel toolPanel(Engine engine){
  JPanel root=new JPanel(new BorderLayout(12,12));root.setBorder(BorderFactory.createEmptyBorder(20,20,20,20));JPanel fields=new JPanel(new GridLayout(0,2,12,12));
  JsonObject rules=Json.obj(engine.config(),"tool_rules");JCheckBox enabled=new JCheckBox("启用挖掘工具适配评分",Json.bool(rules,"enabled",true));fields.add(enabled);fields.add(new JLabel("权重在“各阶段权重”中单独设置"));
  String[] keys={"wrong_tool_amount","insufficient_tier_amount","inefficient_amount","minimum_speed_ratio"},labels={"工具类型错误：每次有效量","等级不足：每次有效量","有更合适工具：每次有效量","手持速度 / 背包最佳速度阈值"};JSpinner[] values=new JSpinner[4];
  for(int i=0;i<keys.length;i++){fields.add(new JLabel(labels[i]));values[i]=new JSpinner(new SpinnerNumberModel(Json.num(rules,keys[i],0),0.0,i==3?1.0:10.0,.05));fields.add(values[i]);}root.add(fields,BorderLayout.NORTH);
  JTextArea note=new JTextArea("工具判断依据当前方块要求和游戏中的工具组件，包含能否掉落、工具类别、挖掘速度与背包可用工具。\n\n空手或斧头挖石头：工具不适配扣分。\n等级不足的镐挖高级矿石：无法获得掉落物，扣分。\n背包有明显更合适的工具：按速度比例阈值判断，可轻扣。\n开局没有斧头时徒手采木：中性。\n\n从开始挖掘起记录尝试，持续挖掘最多每游戏秒计一次，不必等挖掉方块。\n正确工具奖励只在首次有效采集成功时产生；挖自己放置的方块不能刷奖励。\n\n评分范围：当前为挖掘行为。工具不适配 / 正确工具有效采集拥有六阶段独立权重。\n行为与工具日志提供方块、手持工具、原因和评分明细。");note.setEditable(false);note.setLineWrap(true);note.setWrapStyleWord(true);root.add(new JScrollPane(note));
  JPanel buttons=new JPanel(new FlowLayout(FlowLayout.LEFT));JButton load=new JButton("读取工具规则"),save=new JButton("应用工具规则");buttons.add(load);buttons.add(save);root.add(buttons,BorderLayout.SOUTH);long[] version={engine.version()};
  load.addActionListener(a->{JsonObject current=Json.obj(engine.config(),"tool_rules");enabled.setSelected(Json.bool(current,"enabled",true));for(int i=0;i<keys.length;i++)values[i].setValue(Json.num(current,keys[i],0));version[0]=engine.version();});
  save.addActionListener(a->{try{if(version[0]!=engine.version())throw new IllegalStateException("参数已变化，请先读取工具规则");JsonObject config=engine.config(),current=Json.obj(config,"tool_rules");current.addProperty("enabled",enabled.isSelected());for(int i=0;i<keys.length;i++){values[i].commitEdit();current.addProperty(keys[i],((Number)values[i].getValue()).doubleValue());}engine.configure(config,"manual-tools");version[0]=engine.version();}catch(Exception e){JOptionPane.showMessageDialog(root,e.getMessage(),"参数未应用",JOptionPane.ERROR_MESSAGE);}});return root;
 }
 private static JPanel progressionPanel(Engine engine){
  JPanel root=new JPanel(new BorderLayout(12,12));root.setBorder(BorderFactory.createEmptyBorder(18,18,18,18));JPanel fields=new JPanel(new GridLayout(0,2,12,8));
  JCheckBox craft=new JCheckBox("启用关键合成评分"),pickup=new JCheckBox("启用有效拾取评分");fields.add(craft);fields.add(pickup);
  String[] groups={"crafting_rules","crafting_rules","crafting_rules","crafting_rules","crafting_rules","crafting_rules","pickup_rules","pickup_rules"};
  String[] keys={"first_amount","upgrade_amount","replacement_amount","minimum_effective_uses","replacement_window_seconds","replacement_use_seconds","per_item_amount","source_window_seconds"};
  String[] labels={"首次关键合成：有效量","主动升级装备：有效量","耐久耗尽后有效补造：有效量","损坏前至少有效使用次数","损坏到补造时限 / 游戏秒","补造到有效使用时限 / 游戏秒","每个实际拾取物品：有效量","采集掉落物来源保留 / 游戏秒"};double[] mins={0,0,0,1,1,1,0,1},maxs={100,100,100,10000,3600,3600,100,3600};JSpinner[] values=new JSpinner[keys.length];
  for(int i=0;i<keys.length;i++){fields.add(new JLabel(labels[i]));values[i]=new JSpinner(new SpinnerNumberModel(mins[i],mins[i],maxs[i],i<3||i==6?.1:1));fields.add(values[i]);}root.add(fields,BorderLayout.NORTH);
  JTextArea note=new JTextArea("两项评分拥有各阶段独立权重，可在“各阶段权重”中调整。\n\n关键合成：首次制作、升级、补造分开记录原因，同一次取最高值。已有更高等级装备时补做低级装备，不算关键首次。\n补造须有实际损坏统计、损坏前的有效使用、缺少备用装备，并在补造后有效使用才确认奖励；这是用途证据，不能直接读取目标或意图。\n\n拾取：默认每个有效采集物 +0.01。追踪成功采集产生的实际掉落物，进入背包后才给分，部分拾取只算进入的数量。堆叠合并与区块保存保留来源。\n丢下再捡、死亡回收、挖自己放置的方块、未观察到来源的物品不产生这项奖励。\n\n物品类别与默认升级等级在“全部参数” → crafting_rules.equipment 中编辑，可添加模组物品。判断理由和实际分数显示在“行为与工具日志”。");note.setEditable(false);note.setLineWrap(true);note.setWrapStyleWord(true);root.add(new JScrollPane(note));
  JPanel actions=new JPanel(new FlowLayout(FlowLayout.LEFT));JButton load=new JButton("读取合成与拾取规则"),apply=new JButton("应用规则");actions.add(load);actions.add(apply);root.add(actions,BorderLayout.SOUTH);long[] version={0};
  Runnable reload=()->{JsonObject c=engine.config();craft.setSelected(Json.bool(Json.obj(c,"crafting_rules"),"enabled",true));pickup.setSelected(Json.bool(Json.obj(c,"pickup_rules"),"enabled",true));for(int i=0;i<keys.length;i++)values[i].setValue(Json.num(Json.obj(c,groups[i]),keys[i],mins[i]));version[0]=engine.version();};reload.run();load.addActionListener(a->reload.run());
  apply.addActionListener(a->{try{if(version[0]!=engine.version())throw new IllegalStateException("参数已变化，请先读取当前规则");JsonObject c=engine.config();Json.obj(c,"crafting_rules").addProperty("enabled",craft.isSelected());Json.obj(c,"pickup_rules").addProperty("enabled",pickup.isSelected());for(int i=0;i<keys.length;i++){values[i].commitEdit();Json.obj(c,groups[i]).addProperty(keys[i],((Number)values[i].getValue()).doubleValue());}engine.configure(c,"manual-progression");version[0]=engine.version();}catch(Exception e){JOptionPane.showMessageDialog(root,e.getMessage(),"参数未应用",JOptionPane.ERROR_MESSAGE);}});return root;
 }
 private static final class LogPanel extends JPanel {
  private final Engine engine;private final String channel;private final java.util.List<JsonObject> entries=new ArrayList<>(),visible=new ArrayList<>();
  private final DefaultTableModel rows=new DefaultTableModel(new String[]{"时间","级别","记录"},0){public boolean isCellEditable(int r,int c){return false;}};
  private final JTable table=new JTable(rows);private final JTextArea details=new JTextArea();private final JTextField search=new JTextField(22);private final JComboBox<String> level=new JComboBox<>(new String[]{"全部级别","INFO","WARN","ERROR"});
  private final JCheckBox pause=new JCheckBox("暂停刷新"),follow=new JCheckBox("跟随最新",true);private final JLabel count=new JLabel();private long cursor;
  LogPanel(Engine engine,String channel){super(new BorderLayout(8,8));this.engine=engine;this.channel=channel;setBorder(BorderFactory.createEmptyBorder(10,10,10,10));
   JPanel top=new JPanel(new FlowLayout(FlowLayout.LEFT));top.add(level);top.add(new JLabel("搜索"));top.add(search);top.add(pause);top.add(follow);JButton clear=new JButton("清空显示");top.add(clear);add(top,BorderLayout.NORTH);
   table.setRowHeight(24);table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);table.getColumnModel().getColumn(0).setPreferredWidth(110);table.getColumnModel().getColumn(1).setPreferredWidth(65);table.getColumnModel().getColumn(2).setPreferredWidth(700);
   details.setEditable(false);details.setLineWrap(true);details.setWrapStyleWord(true);details.setFont(new Font(Font.MONOSPACED,Font.PLAIN,13));JScrollPane tableScroll=new JScrollPane(table);tableScroll.setColumnHeaderView(table.getTableHeader());JSplitPane split=new JSplitPane(JSplitPane.VERTICAL_SPLIT,tableScroll,new JScrollPane(details));split.setResizeWeight(.6);split.setDividerLocation(320);add(split);add(count,BorderLayout.SOUTH);
   table.getSelectionModel().addListSelectionListener(e->{int i=table.getSelectedRow();if(i>=0&&i<visible.size()){JsonObject row=visible.get(i),detail=Json.obj(row,"details").deepCopy();String body=Json.str(detail,"body","");detail.remove("body");details.setText(Json.str(row,"time","")+"  "+Json.str(row,"level","")+"\n"+Json.str(row,"message","")+"\n\n"+(detail.isEmpty()?"":Json.G.toJson(detail))+(!body.isEmpty()?"\n\n"+body:""));details.setCaretPosition(0);}});level.addActionListener(a->render());search.getDocument().addDocumentListener(new DocumentListener(){public void insertUpdate(DocumentEvent e){render();}public void removeUpdate(DocumentEvent e){render();}public void changedUpdate(DocumentEvent e){render();}});
   clear.addActionListener(a->{entries.clear();cursor=(long)Json.num(engine.logsAfter(0,channel),"latest_id",0);details.setText("");render();});
  }
  void refresh(){if(pause.isSelected())return;JsonObject batch=engine.logsAfter(cursor,channel);boolean changed=false;for(JsonElement value:Json.arr(batch,"logs")){entries.add(value.getAsJsonObject());changed=true;}while(entries.size()>1000)entries.removeFirst();cursor=(long)Json.num(batch,"latest_id",cursor);if(changed)render();}
  private void render(){
   int selected=table.getSelectedRow();long selectedId=selected>=0&&selected<visible.size()?(long)Json.num(visible.get(selected),"id",-1):-1;visible.clear();rows.setRowCount(0);
   String needle=search.getText().toLowerCase(Locale.ROOT),severity=String.valueOf(level.getSelectedItem());
   for(JsonObject row:entries){if(!severity.equals("全部级别")&&!severity.equals(Json.str(row,"level","")))continue;if(!needle.isBlank()&&!Json.COMPACT.toJson(row).toLowerCase(Locale.ROOT).contains(needle))continue;visible.add(row);String message=Json.str(row,"message","");rows.addRow(new Object[]{localTime(Json.str(row,"time","")),Json.str(row,"level",""),message.length()>240?message.substring(0,240)+"…":message});}
   count.setText("显示 "+visible.size()+" 条 · 保留最近 1000 条 · 完整内容点击行查看 · 清空显示不删除日志文件");
   if(follow.isSelected()&&!visible.isEmpty()){int i=visible.size()-1;table.setRowSelectionInterval(i,i);table.scrollRectToVisible(table.getCellRect(i,0,true));}
   else for(int i=0;i<visible.size();i++)if(Json.num(visible.get(i),"id",-1)==selectedId){table.setRowSelectionInterval(i,i);break;}
  }
 }
}
