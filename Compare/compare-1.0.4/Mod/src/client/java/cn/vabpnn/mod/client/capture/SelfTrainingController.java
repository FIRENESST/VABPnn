package cn.vabpnn.mod.client.capture;
import com.google.gson.*;
import java.lang.reflect.Method;
import java.net.URI;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import net.minecraft.client.Minecraft;

/** Main-thread orchestration; the transport performs all HTTP work off the game thread. */
public final class SelfTrainingController implements AutoCloseable {
    private static final String ALGORITHM="on-policy REINFORCE: averaged executed-window scores";
    private static final String VERSION="compare-1.0.4";
    private final RewardLedger ledger=new RewardLedger();
    private final String trainerId=UUID.randomUUID().toString();
    private final long[] proposalData=new long[10],nativeStatus=new long[12];
    private NativeCudaBridge bridge;
    private ScorerRewardTransport transport;
    private CompletableFuture<JsonObject> handshake,rewards;
    private long epoch=1,handshakeEpoch,rewardEpoch,nextHandshake,nextPoll,nextReport;
    private boolean armed,ready,wasActive,closed;
    private float exploration;
    private boolean bootstrap;
    private String reason="disabled",lastError="";
    private Method contextMethod,terminalContextMethod;
    private boolean settlingDeath,episodeCompleted;
    private long deathDeadline,terminalQueuedId;
    private boolean providerChecked;
    private RewardLedger.Context previousContext;
    private RewardLedger.Proposal preparing;
    public boolean armed(){return armed;}
    public boolean ready(){return armed&&ready&&!settlingDeath;}
    public boolean takeEpisodeCompleted(){boolean value=episodeCompleted;episodeCompleted=false;return value;}
    public String reason(){return reason;}
    public float exploration(){return exploration;}
    public boolean bootstrap(){return exploration>0 && bootstrap;}
    public void exploration(float probability,boolean allowSubThreshold) {
        if(!Float.isFinite(probability) || probability<0 || probability>1)
            throw new IllegalArgumentException("Exploration probability must be in 0..1");
        exploration=probability;bootstrap=probability>0 && allowSubThreshold;
        invalidate("exploration_changed",previousContext==null ? 0:previousContext.tick());
    }
    public String explorationText() {
        return "探索 "+Math.round(exploration*100)+"% · "+(bootstrap() ? "起步模式，可尝试未过阈值动作":"保留置信度阈值")
            +"；每次探索最多 2 项 · 累计探索提案 "+nativeStatus[11];
    }
    public long updates(){return nativeStatus[2];}
    public long dropped(){return ledger.dropped()+nativeStatus[4];}
    public long consumed(){return ledger.cursor();}
    public int pending(){return ledger.pending();}
    public String statusText() {
        return "Scorer 自训练："+(armed ? ready ? "运行":"等待":"关闭")+" · "+reason
                +" | 奖励游标 "+consumed()+" / GPU 更新 "+updates()+" / 待配对 "+pending()+" / 丢弃 "+dropped()
                +"；仅动作证据与 k 头，图/PNN/JEPA 冻结 | "+explorationText();
    }
    public void enable() {
        if(closed)throw new IllegalStateException("Self-training controller closed");
        String modules=System.getProperty("vabpnn.selftrain.modules","policy_heads");
        if(!modules.equals("policy_heads"))throw new IllegalArgumentException("Reward training supports policy_heads only");
        if(transport==null)transport=new ScorerRewardTransport(URI.create(System.getProperty("vabpnn.scorer.endpoint","http://127.0.0.1:9876")));
        armed=true;invalidate("waiting_for_scorer",0);
    }
    public void disable(String cause) {armed=false;exploration=0;bootstrap=false;invalidate(cause,previousContext==null ? 0:previousContext.tick());}
    public void boundary(String cause) {
        if(!armed)return; // Explicit disable already invalidated receipts; retain its completion reason.
        Minecraft client=Minecraft.getInstance();
        if(armed && ledger.bound() && canSettleDeath(client) && terminalContextMethod!=null) {
            if(!settlingDeath){settlingDeath=true;deathDeadline=System.nanoTime()+3_000_000_000L;reason="settling_death_window";}
            preparing=null;return;
        }
        invalidate(cause,previousContext==null ? 0:previousContext.tick());
    }
    private boolean canSettleDeath(Minecraft client) {
        var mode=CaptureController.inputState();
        return mode.capturing() && mode.modelControl() && !mode.predictionOnly()
                && client.player!=null && client.level!=null && !client.player.isAlive()
                && client.isWindowActive() && !client.isPaused() && client.hasSingleplayerServer()
                && (client.screen==null || client.screen instanceof net.minecraft.client.gui.screens.DeathScreen);
    }
    private void invalidate(String cause,long tick) {
        ready=false;reason=cause;preparing=null;settlingDeath=false;terminalQueuedId=0;
        ++epoch;ledger.clear(cause,tick);
        if(handshake!=null)handshake.cancel(true);handshake=null;
        if(rewards!=null)rewards.cancel(true);rewards=null;
        if(bridge!=null)bridge.rewardMode(epoch,false);
        nextHandshake=System.nanoTime()+250_000_000L;nextPoll=0;
    }
    public void detach() {invalidate("pipeline_stopped",0);bridge=null;}
    public RewardLedger.Context context(Minecraft client) {
        if(client.player==null || client.level==null || !client.hasSingleplayerServer())return null;
        try {
            if(!providerChecked) {
                providerChecked=true;Class<?> provider=Class.forName("cn.vabpnn.telemetry.TelemetryMod");
                contextMethod=provider.getMethod("trainingContext");
                try {terminalContextMethod=provider.getMethod("trainingTerminalContext");}catch(NoSuchMethodException ignored) {terminalContextMethod=null;}
            }
            if(contextMethod==null)return null;
            String json=(String)contextMethod.invoke(null);
            if(canSettleDeath(client) && terminalContextMethod!=null)json=(String)terminalContextMethod.invoke(null);
            var value=RewardLedger.context(json,System.currentTimeMillis());
            if(value!=null && (!value.player().equals(client.player.getUUID().toString())
                    || !value.dimension().equals(client.level.dimension().identifier().toString())))return null;
            return value;
        }catch(ReflectiveOperationException|IllegalArgumentException error) {
            if(error instanceof NoSuchMethodException || error instanceof ClassNotFoundException)reason="telemetry_0.4.0_required";
            if(error instanceof NoSuchMethodException || error instanceof ClassNotFoundException)contextMethod=null;
            return null;
        }
    }
    public void tick(NativeCudaBridge current,boolean active,RewardLedger.Context context) {
        if(closed)return;
        if(current!=bridge) {if(bridge!=null)detach();bridge=current;invalidate("pipeline_changed",context==null ? 0:context.tick());}
        if(bridge!=null) {
            bridge.readRewardStatus(nativeStatus);
            if(ledger.bound() && !settlingDeath)ledger.policyVersion(nativeStatus[1],context==null ? 0:context.tick());
        }
        long now=System.nanoTime(),millis=System.currentTimeMillis();
        boolean terminalUsable=settlingDeath && canSettleDeath(Minecraft.getInstance()) && context!=null
                && previousContext!=null && previousContext.same(context) && now<=deathDeadline;
        boolean usable=(active || terminalUsable) && context!=null && bridge!=null && bridge.phase()==NativeCudaBridge.READY;
        if(settlingDeath && (!terminalUsable || terminalQueuedId>0 && nativeStatus[5]>=terminalQueuedId)) {
            disable(terminalQueuedId>0 && nativeStatus[5]>=terminalQueuedId ? "terminal_reward_settled":"terminal_window_expired");
            episodeCompleted=true;usable=false;
        }
        if(armed && (!usable || previousContext!=null && !previousContext.same(context))) {
            if(ready || handshake!=null || rewards!=null || wasActive)invalidate(context==null ? "context_unavailable":"input_or_episode_boundary",context==null ? 0:context.tick());
        }
        previousContext=context;wasActive=usable;
        if(armed && usable) {
            try {
                if(handshake!=null && handshake.isDone()) {
                    var value=handshake.join();handshake=null;
                    if(handshakeEpoch==epoch) {
                        if(!RewardLedger.bool(value,"ready") && !settlingDeath) {
                            invalidate("scorer_"+RewardLedger.text(value,"reason"),context.tick());
                        } else if(!ledger.bound()) {
                            ledger.bind(value,context,millis,nativeStatus[1]);ready=true;reason="running";
                            bridge.rewardMode(++epoch,true,exploration,bootstrap);nextPoll=0;
                        } else if(!RewardLedger.text(value,"stream_id").equals(ledger.stream())
                                || !RewardLedger.idsMatch(RewardLedger.object(value,"binding"),context)
                                || !RewardLedger.text(value,"dimension").equals(context.dimension())
                                || !ledger.sameConfiguration(value)) {
                            invalidate("scorer_binding_changed",context.tick());
                        }
                    }
                }
                if(rewards!=null && rewards.isDone()) {
                    var response=rewards.join();rewards=null;
                    if(rewardEpoch==epoch) {
                        if(RewardLedger.bool(response,"gap") || !RewardLedger.text(response,"stream_id").equals(ledger.stream())) {
                            invalidate("reward_stream_gap",context.tick());
                        } else {
                            var values=response.getAsJsonArray("rewards");
                            if(values==null || values.size()>512)throw new IllegalArgumentException("Invalid bounded reward list");
                            for(var element:values) {
                                var reward=element.getAsJsonObject();
                                if(RewardLedger.bool(reward,"terminal") && !settlingDeath) {
                                    disable("terminal_without_active_death_window");episodeCompleted=true;break;
                                }
                                var match=ledger.consume(reward,context,millis);reason=match.reason();
                                if(match.reset()){invalidate(match.reason(),context.tick());break;}
                                if(match.accepted()) {
                                    if(settlingDeath && !match.reason().equals("terminal_reward_matched"))continue;
                                    if(!bridge.reward(epoch,match.id(),match.startTick(),match.endTick(),match.credit(),match.frames()))
                                        reason="native_receipt_rejected";
                                    else if(match.reason().equals("terminal_reward_matched")){terminalQueuedId=match.id();reason="awaiting_terminal_gpu_receipt";break;}
                                }
                            }
                        }
                    }
                }
                if(handshake==null && now>=nextHandshake) {
                    handshakeEpoch=epoch;handshake=transport.handshake();nextHandshake=now+1_000_000_000L;
                }
                if(ready && ledger.bound() && terminalQueuedId==0 && rewards==null && now>=nextPoll) {
                    rewardEpoch=epoch;rewards=transport.rewards(ledger.cursor(),ledger.stream());nextPoll=now+250_000_000L;
                }
            }catch(RuntimeException error) {
                lastError=error.getMessage()==null ? error.getClass().getSimpleName():error.getMessage();
                if(lastError.length()>200)lastError=lastError.substring(0,200);
                invalidate("scorer_unavailable_or_invalid_response",context.tick());nextHandshake=now+1_000_000_000L;
            }
        }
        if(transport!=null && now>=nextReport) {
            nextReport=now+1_000_000_000L;transport.report(report()).exceptionally(error->null);
        }
    }
    public boolean prepare(long[] command,long now,RewardLedger.Context context) {
        preparing=null;
        if(!ready() || bridge==null || context==null || !ledger.binding().same(context))return false;
        bridge.readRewardProposal(command[0],proposalData);
        if(proposalData[8]!=1 || proposalData[0]!=command[0] || proposalData[1]!=command[1]
                || proposalData[2]!=epoch || proposalData[3]!=ledger.policyVersion() || proposalData[4]!=command[2])return false;
        var proposal=new RewardLedger.Proposal(proposalData[0],proposalData[1],proposalData[2],proposalData[3],
                (int)proposalData[4],(int)proposalData[5],(int)proposalData[6],(int)proposalData[7],
                Float.intBitsToFloat((int)proposalData[9]));
        if(proposal.captured()>now || now-proposal.captured()>ActionExecutor.COMMAND_LEASE_NANOS
                || !Float.isFinite(proposal.logProbability()))return false;
        preparing=proposal;return true;
    }
    public void executed(long frame,int mask,long commandsBefore,long commandsAfter,long now,
                          RewardLedger.Context before,RewardLedger.Context after) {
        var proposal=preparing;preparing=null;
        if(proposal==null || frame!=proposal.frame() || mask!=proposal.mask() || commandsAfter!=commandsBefore+1
                || before==null || !before.same(after) || !ready())return;
        if(bridge.rewardExecution(epoch,frame,proposal.captured(),mask,before.tick(),after.tick(),now,proposal.policyVersion()))
            ledger.record(proposal,before,after,epoch,now);
        else reason="native_execution_rejected";
    }
    private JsonObject report() {
        var value=new JsonObject();value.addProperty("schema",1);value.addProperty("trainer_id",trainerId);
        value.addProperty("stream_id",ledger.stream());value.addProperty("model_version",VERSION);
        value.addProperty("mode",armed ? ready ? "self_training":"waiting":"off");
        value.addProperty("algorithm",ALGORITHM);value.addProperty("connection",ready ? "connected":"disconnected");
        value.addProperty("exploration_probability",exploration);value.addProperty("exploration_bootstrap",bootstrap());
        value.addProperty("exploration_proposals",nativeStatus[11]);
        value.addProperty("reason",reason);value.addProperty("last_reward_id",ledger.cursor());
        value.addProperty("applied_rewards",nativeStatus[3]);value.addProperty("updates",nativeStatus[2]);
        value.addProperty("dropped_rewards",dropped());value.addProperty("pending_actions",ledger.pending());
        value.addProperty("last_error",lastError);
        JsonObject binding=new JsonObject();
        var context=ledger.binding()!=null ? ledger.binding():previousContext;
        if(context!=null)binding=context.binding();
        else for(String field:new String[]{"world_id","player_id","session_id"})binding.addProperty(field,"");
        value.add("binding",binding);return value;
    }
    @Override public void close() {
        if(closed)return;disable("closed");closed=true;
        if(transport!=null)transport.close();transport=null;
    }
}
