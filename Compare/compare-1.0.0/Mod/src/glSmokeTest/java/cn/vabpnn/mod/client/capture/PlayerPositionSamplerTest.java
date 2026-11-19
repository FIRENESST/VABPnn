package cn.vabpnn.mod.client.capture;

final class PlayerPositionSamplerTest {
    private static void require(boolean ok,String text) { if(!ok) throw new AssertionError(text); }
    static void run() {
        var sampler=new PlayerPositionSampler();
        Object world=new Object(),player=new Object(); long origin=1_000_000_000L;
        for(int frame=0;frame<600;frame++) {
            long now=origin+frame*1_000_000_000L/60;
            if(sampler.poll(world,player,true,now)) sampler.coordinates(10+frame/60.0,64,-20);
        }
        require(sampler.samples()==300,"60 FPS render callbacks produce exactly 300 samples over 10 seconds");
        long before=sampler.samples();
        require(sampler.poll(world,player,true,origin+20_000_000_000L),"resume after dropped render callbacks");
        sampler.coordinates(20,64,-20);
        require(sampler.samples()==before+1,"delayed callback produces one sample, with no catch-up burst");
        require(!sampler.poll(world,player,true,origin+20_000_000_001L),"same render interval cannot oversample");
        long epoch=sampler.epoch();
        require(sampler.poll(world,new Object(),true,origin+20_000_000_002L),"respawn gets a fresh sample immediately");
        sampler.coordinates(0,64,0);
        require(sampler.epoch()>epoch,"respawn separates motion history");
        require(sampler.poll(null,null,false,origin+20_000_000_003L) && sampler.nanos()==0,"disconnect publishes invalidation");
        require(!sampler.poll(null,null,false,origin+20_000_000_004L),"standby performs no polling work");
        sampler.poll(world,player,true,origin+20_000_000_005L); sampler.coordinates(Double.NaN,64,0);
        require(sampler.nanos()==0,"invalid coordinate is never forwarded");
        epoch=sampler.epoch(); sampler.poll(world,player,true,origin); sampler.coordinates(0,64,0);
        require(sampler.epoch()>epoch,"clock reversal creates a new motion epoch");
        sampler.reset(); require(sampler.nanos()==0,"stop resets the coordinate observation");
        System.out.println("PASS: 30 Hz phase-preserving position sampling; no catch-up bursts; respawn/disconnect/invalid state/clock reset.");
    }
}
