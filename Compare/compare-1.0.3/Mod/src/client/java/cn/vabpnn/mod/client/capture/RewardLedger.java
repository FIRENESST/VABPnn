package cn.vabpnn.mod.client.capture;
import com.google.gson.*;
import java.util.*;

/** Pure protocol/credit ledger. It never reads pixels, computes imitation labels or updates weights. */
public final class RewardLedger {
    public static final int CAPACITY=512;
    public static final String CLOCK="integrated_server_tick";
    public record Context(String world,String player,String session,String dimension,long tick,long observedMillis,String gameMode) {
        public boolean same(Context other) {
            return other!=null && world.equals(other.world) && player.equals(other.player)
                    && session.equals(other.session) && dimension.equals(other.dimension);
        }
        public JsonObject binding() {
            var value=new JsonObject();value.addProperty("world_id",world);value.addProperty("player_id",player);
            value.addProperty("session_id",session);return value;
        }
    }
    public record Proposal(long frame,long captured,long epoch,long policyVersion,int mask,int eligible,int k,int count,float logProbability) {}
    public record Execution(Proposal proposal,Context before,Context after,long appliedNanos) {}
    public record Match(long id,long startTick,long endTick,float credit,long[] frames,String reason,boolean reset) {
        public boolean accepted(){return frames.length>0;}
    }
    private final NavigableMap<Long,Execution> pending=new TreeMap<>();
    private Context binding;
    private String stream="",reason="disabled";
    private long cursor,baselineTick,lastEnd,policyVersion,configVersion=-1,lastTelemetrySeq,accepted,dropped;
    private double rewardClip=20,maxGap=10;
    public String stream(){return stream;}public long cursor(){return cursor;}
    public long accepted(){return accepted;}public long dropped(){return dropped;}
    public int pending(){return pending.size();}public String reason(){return reason;}
    public Context binding(){return binding;}
    public long policyVersion(){return policyVersion;}
    public boolean sameConfiguration(JsonObject handshake) {
        return integer(handshake,"config_version")==configVersion
                && number(handshake,"reward_clip")==rewardClip && number(handshake,"max_gap_seconds")==maxGap;
    }
    public boolean bound(){return binding!=null&&!stream.isEmpty();}
    public void clear(String cause,long currentTick) {
        pending.clear();binding=null;stream="";reason=cause;baselineTick=currentTick;lastEnd=0;
        configVersion=-1;lastTelemetrySeq=0;
    }
    public void policyVersion(long value,long tick) {
        if(value!=policyVersion){pending.clear();baselineTick=tick;reason="policy_version_changed";policyVersion=value;}
    }
    public void bind(JsonObject handshake,Context context,long nowMillis,long version) {
        if(context==null || !bool(handshake,"ready") || integer(handshake,"schema")!=1
                || !text(handshake,"protocol").equals("vabpnn-reward-v1") || !text(handshake,"clock").equals(CLOCK))
            throw new IllegalArgumentException("Scorer handshake is not ready or protocol is unsupported");
        var ids=object(handshake,"binding");
        if(!idsMatch(ids,context) || !text(handshake,"dimension").equals(context.dimension))
            throw new IllegalArgumentException("Scorer and current game identities differ");
        double age=number(handshake,"age_seconds"),gap=number(handshake,"max_gap_seconds"),clip=number(handshake,"reward_clip");
        if(age<0 || age>Math.min(5,gap) || gap<=0 || gap>120 || clip<=0 || clip>1e6)
            throw new IllegalArgumentException("Scorer handshake is stale or limits invalid");
        binding=context;stream=text(handshake,"stream_id");
        if(stream.isBlank() || stream.length()>160)throw new IllegalArgumentException("Invalid reward stream");
        cursor=integer(handshake,"latest_id");baselineTick=context.tick;lastEnd=0;lastTelemetrySeq=0;
        policyVersion=version;maxGap=gap;rewardClip=clip;pending.clear();
        configVersion=handshake.has("config_version") ? integer(handshake,"config_version"):-1;
        reason="running";
    }
    public boolean record(Proposal proposal,Context before,Context after,long epoch,long nowNanos) {
        boolean ok=bound() && binding.same(before) && before.same(after)
            && proposal.epoch==epoch && proposal.policyVersion==policyVersion && proposal.frame>0
            && proposal.captured<=nowNanos && nowNanos-proposal.captured<=ActionExecutor.COMMAND_LEASE_NANOS
            && before.tick<=after.tick && after.tick-before.tick<=1 && before.tick>=baselineTick
            && proposal.mask>=0 && (proposal.mask&~proposal.eligible)==0
            && (proposal.eligible&~ActionLayout.KNOWN_MASK)==0 && Float.isFinite(proposal.logProbability)
            && proposal.logProbability<=1e-5 && !pending.containsKey(proposal.frame);
        if(!ok){++dropped;reason="execution_proof_invalid";return false;}
        if(pending.size()>=CAPACITY) {
            baselineTick=Math.max(baselineTick,pending.pollFirstEntry().getValue().after.tick);reason="cache_overflow";
        }
        pending.put(proposal.frame,new Execution(proposal,before,after,nowNanos));return true;
    }
    public Match consume(JsonObject reward,Context current,long nowMillis) {
        long id=integer(reward,"reward_id");
        if(id<=cursor){++dropped;return reject(id,"duplicate_reward",false);}
        cursor=id;
        if(!bound() || !binding.same(current) || !idsMatch(reward,current)
                || !text(reward,"stream_id").equals(stream)){++dropped;return reject(id,"binding_or_stream_changed",true);}
        long start=integer(reward,"start_tick"),end=integer(reward,"tick");
        boolean overlap=start<lastEnd;lastEnd=Math.max(lastEnd,end);
        if(!bool(reward,"valid")) {expire(end);++dropped;return reject(id,"invalid_reward",false);}
        if(!text(reward,"clock").equals(CLOCK) || start>=end || end>current.tick || start<baselineTick || overlap
                || end-start>maxGap*20 || Math.abs(number(reward,"interval_seconds")-(end-start)/20.0)>1e-6) {
            expire(end);++dropped;return reject(id,"reward_window_invalid",false);
        }
        long received=integer(reward,"received_at_ms");
        if(received>nowMillis+1000 || nowMillis-received>Math.min(10,maxGap)*1000) {
            expire(end);++dropped;return reject(id,"stale_reward",false);
        }
        if(!text(reward,"start_dimension").equals(current.dimension) || !text(reward,"dimension").equals(current.dimension)) {
            pending.clear();++dropped;return reject(id,"dimension_boundary",true);
        }
        boolean terminal=bool(reward,"terminal");
        if(terminal && !text(reward,"terminal_reason").equals("death")) {
            pending.clear();++dropped;return reject(id,"terminal_dimension_boundary",true);
        }
        long version=integer(reward,"config_version"),sequence=integer(reward,"telemetry_seq");
        if(configVersion<0)configVersion=version;
        if(version!=configVersion){pending.clear();++dropped;return reject(id,"scorer_config_changed",true);}
        if(sequence<=lastTelemetrySeq){expire(end);++dropped;return reject(id,"telemetry_replay",false);}
        lastTelemetrySeq=sequence;
        double value=number(reward,"reward");
        if(Math.abs(value)>rewardClip+1e-6) {expire(end);++dropped;return reject(id,"unclipped_reward_rejected",false);}
        var frames=new ArrayList<Long>();
        boolean oldPolicy=false;
        for(var entry:pending.entrySet()) {
            var action=entry.getValue();
            if(action.before.tick>start && action.after.tick<=end) {
                if(action.proposal.policyVersion!=policyVersion)oldPolicy=true;else frames.add(entry.getKey());
            }
        }
        expire(end);
        if(oldPolicy || frames.isEmpty()){++dropped;return reject(id,oldPolicy ? "old_policy":"no_executed_action",false);}
        ++accepted;reason=terminal ? "terminal_reward_matched":value==0 ? "zero_reward":"reward_matched";
        return new Match(id,start,end,(float)Math.max(-1,Math.min(1,value/rewardClip)),
                frames.stream().mapToLong(Long::longValue).toArray(),reason,false);
    }
    private void expire(long tick) {pending.entrySet().removeIf(e->e.getValue().after.tick<=tick);}
    private Match reject(long id,String cause,boolean reset) {
        reason=cause;return new Match(id,0,0,0,new long[0],cause,reset);
    }
    public static Context context(String json,long nowMillis) {
        JsonObject value=JsonParser.parseString(json).getAsJsonObject();
        if(value.size()==0)return null;
        if(integer(value,"schema")!=1 || !text(value,"authority").equals("singleplayer_integrated_server")
                || !text(value,"clock").equals(CLOCK))throw new IllegalArgumentException("Server tick authority missing");
        long observed=integer(value,"observed_at_ms");
        if(observed>nowMillis+1000 || nowMillis-observed>500)throw new IllegalArgumentException("Local server context stale");
        return new Context(identifier(value,"world_id"),identifier(value,"player_id"),identifier(value,"session_id"),
                identifier(value,"dimension"),integer(value,"tick"),observed,identifier(value,"game_mode"));
    }
    static boolean idsMatch(JsonObject value,Context context) {
        return context!=null && text(value,"world_id").equals(context.world)
                && text(value,"player_id").equals(context.player) && text(value,"session_id").equals(context.session);
    }
    static JsonObject object(JsonObject value,String key) {
        if(!value.has(key) || !value.get(key).isJsonObject())throw new IllegalArgumentException("Missing object "+key);
        return value.getAsJsonObject(key);
    }
    static String text(JsonObject value,String key) {
        if(!value.has(key) || !value.get(key).isJsonPrimitive() || !value.getAsJsonPrimitive(key).isString())
            throw new IllegalArgumentException("Missing string "+key);
        return value.get(key).getAsString();
    }
    static String identifier(JsonObject value,String key) {
        String id=text(value,key);if(id.isBlank() || id.length()>160)throw new IllegalArgumentException("Invalid "+key);return id;
    }
    static long integer(JsonObject value,String key) {
        double number=number(value,key);
        if(number<0 || number>9e15 || number!=Math.rint(number))throw new IllegalArgumentException("Invalid integer "+key);
        return (long)number;
    }
    static double number(JsonObject value,String key) {
        if(!value.has(key) || !value.get(key).isJsonPrimitive() || !value.getAsJsonPrimitive(key).isNumber())
            throw new IllegalArgumentException("Missing number "+key);
        double number=value.get(key).getAsDouble();if(!Double.isFinite(number))throw new IllegalArgumentException("Nonfinite "+key);return number;
    }
    static boolean bool(JsonObject value,String key) {
        if(!value.has(key) || !value.get(key).isJsonPrimitive() || !value.getAsJsonPrimitive(key).isBoolean())
            throw new IllegalArgumentException("Missing boolean "+key);
        return value.get(key).getAsBoolean();
    }
}
