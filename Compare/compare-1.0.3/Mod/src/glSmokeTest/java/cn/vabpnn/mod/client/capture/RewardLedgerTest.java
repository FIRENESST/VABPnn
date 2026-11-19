package cn.vabpnn.mod.client.capture;
import com.google.gson.*;
import java.util.Arrays;
/** Genuine interval/identity/execution protocol checks; never opens a production model. */
public final class RewardLedgerTest {
    private static final long NOW=1_790_000_000_000L, CAPTURE=1_000_000_000L, EPOCH=7;
    private static void require(boolean value,String message){if(!value)throw new AssertionError(message);}
    private static RewardLedger.Context context(long tick){return new RewardLedger.Context("world","player","session","minecraft:overworld",tick,NOW,"survival");}
    private static JsonObject handshake(){
        var j=new JsonObject();j.addProperty("schema",1);j.addProperty("ready",true);j.addProperty("protocol","vabpnn-reward-v1");
        j.addProperty("clock",RewardLedger.CLOCK);j.add("binding",context(100).binding());j.addProperty("dimension","minecraft:overworld");
        j.addProperty("stream_id","stream");j.addProperty("latest_id",10);j.addProperty("age_seconds",0.1);
        j.addProperty("max_gap_seconds",10);j.addProperty("reward_clip",20);j.addProperty("config_version",3);return j;
    }
    private static RewardLedger ledger(){var l=new RewardLedger();l.bind(handshake(),context(100),NOW,0);return l;}
    private static RewardLedger.Proposal proposal(long frame,long version,int mask){return new RewardLedger.Proposal(frame,CAPTURE,EPOCH,version,mask,3,1,Integer.bitCount(mask),-.7f);}
    private static void record(RewardLedger l,long frame,long tick){require(l.record(proposal(frame,0,1),context(tick),context(tick),EPOCH,CAPTURE+1),"valid actual execution");}
    private static JsonObject reward(long id,long start,long end,double amount){
        var j=context(end).binding();j.addProperty("reward_id",id);j.addProperty("stream_id","stream");j.addProperty("start_tick",start);
        j.addProperty("tick",end);j.addProperty("valid",true);j.addProperty("clock",RewardLedger.CLOCK);j.addProperty("interval_seconds",(end-start)/20.0);
        j.addProperty("received_at_ms",NOW);j.addProperty("start_dimension","minecraft:overworld");j.addProperty("dimension","minecraft:overworld");
        j.addProperty("terminal",false);j.addProperty("terminal_reason","");j.addProperty("config_version",3);j.addProperty("telemetry_seq",id);
        j.addProperty("reward",amount);return j;
    }
    public static void run(){
        for(double credit:new double[]{20,-20,0}){
            var l=ledger();record(l,1,101);record(l,2,120);
            var r=reward(11,100,120,credit);var match=l.consume(r,context(120),NOW);
            require(match.accepted() && Arrays.equals(match.frames(),new long[]{1,2}) && match.credit()==credit/20,"positive/negative/zero normalization");
            require(l.pending()==0 && !l.consume(r,context(120),NOW).accepted(),"consume exactly once");
        }
        var l=ledger();record(l,1,100);record(l,2,101);
        require(Arrays.equals(l.consume(reward(11,100,120,1),context(120),NOW).frames(),new long[]{2}),"open interval excludes start-tick action");
        l=ledger();require(!l.consume(reward(11,100,120,1),context(120),NOW).accepted(),"unexecuted proposals never earn credit");
        l=ledger();record(l,1,101);require(!l.record(proposal(1,0,1),context(101),context(101),EPOCH,CAPTURE+1),"duplicate execution");
        require(!l.record(proposal(2,1,1),context(101),context(101),EPOCH,CAPTURE+1),"old policy");
        require(!l.record(proposal(2,0,4),context(101),context(101),EPOCH,CAPTURE+1),"ineligible actual action");
        require(!l.record(proposal(2,0,1),context(101),context(104),EPOCH,CAPTURE+1),"tick gap");
        require(!l.record(proposal(2,0,1),context(101),context(101),EPOCH,CAPTURE+600_000_000L),"expired packet");
        l.policyVersion(1,120);require(l.pending()==0 && !l.consume(reward(11,100,120,1),context(120),NOW).accepted(),"policy change clears old window");
        for(String field:new String[]{"stream_id","world_id","player_id","session_id","dimension","start_dimension","clock"}){
            l=ledger();record(l,1,101);var r=reward(11,100,120,1);r.addProperty(field,"wrong");
            require(!l.consume(r,context(120),NOW).accepted(),"identity/dimension/clock mismatch "+field);
        }
        l=ledger();record(l,1,101);var r=reward(11,100,120,1);r.addProperty("config_version",4);
        require(l.consume(r,context(120),NOW).reset(),"configuration boundary");
        var h=handshake();h.addProperty("reward_clip",10);require(!ledger().sameConfiguration(h),"handshake clip change");
        l=ledger();record(l,1,101);r=reward(11,100,120,21);require(!l.consume(r,context(120),NOW).accepted(),"unclipped reward");
        l=ledger();record(l,1,101);r=reward(11,100,120,1);r.addProperty("received_at_ms",NOW-11000);
        require(!l.consume(r,context(120),NOW).accepted(),"stale reward");
        l=ledger();record(l,1,101);r=reward(11,100,120,-5);r.addProperty("terminal",true);r.addProperty("terminal_reason","death");
        var terminal=l.consume(r,context(120),NOW);require(terminal.accepted()&&!terminal.reset()&&terminal.reason().equals("terminal_reward_matched"),"exact death window accepted before episode close");
        l=ledger();record(l,1,101);r=reward(11,100,120,-5);r.addProperty("terminal",true);r.addProperty("terminal_reason","dimension_changed");
        require(l.consume(r,context(120),NOW).reset(),"dimension terminal rejects credit");
        l=ledger();record(l,1,101);l.clear("freeze",120);require(l.pending()==0&&!l.bound(),"F8/predict/gap/pause/world boundary clear");
        var c=context(100).binding();c.addProperty("schema",1);c.addProperty("authority","singleplayer_integrated_server");
        c.addProperty("clock",RewardLedger.CLOCK);c.addProperty("dimension","minecraft:overworld");c.addProperty("tick",100);
        c.addProperty("observed_at_ms",NOW);c.addProperty("game_mode","survival");
        require(RewardLedger.context(c.toString(),NOW).tick()==100 && RewardLedger.context("{}",NOW)==null,"authoritative local context");
        boolean stale=false;try{RewardLedger.context(c.toString(),NOW+501);}catch(IllegalArgumentException expected){stale=true;}
        require(stale && RewardLedger.context(c.toString(),NOW).tick()==100,"temporary stale snapshot can recover");
        System.out.println("PASS: reward ledger execution proof, exact open intervals, positive/negative/zero, duplicates, policy/config/identity/clock/stale boundaries and terminal death.");
    }
}
