#include "FrameOperator.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "NetworkScope.h"
#include "MemeState.h"
#include "RewardPolicy.h"
#include "LongTermMemory.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
bool RewardStress=false;
bool PositionEnabled=true,BlocksEnabled=true,BlockStress=false,MemeStress=false,MemesEnabled=true,MemeScopeEnabled=false,MinuteStress=false;
void check(cudaError_t code) { if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
__global__ void pattern(unsigned char* image,std::size_t pitch,int width,int height) {
    const int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width || y>=height) return;
    auto* pixel=image+y*pitch+x*4;
    pixel[0]=(x*3+y*7+91)&255; pixel[1]=(x+y*11+43)&255;
    pixel[2]=(x*13+y+29)&255; pixel[3]=255;
}
// Stress capacity, independently of how long the human model has been trained.
__global__ void fullCapacity(vabpnn::SpikePixelStatistics* stats) {
    const auto pixel=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pixel>=vabpnn::NeuronCount) return;
    const int x=pixel%1920,y=pixel/1920;
    if(x%16==8 && y%16==8) stats[pixel]={.33f,.33f,.34f,1e10f,0};
}
// Synthetic maturity windows force ranking/mutation cost; this is a stress fixture, not a learned score.
__global__ void matureRankFixture(vabpnn::BlockPersistent* p) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x; if(tile>=vabpnn::BlockCount) return;
    auto& state=p->state[tile]; const float score=tile<1530 ? -.8f:.2f;
    state.examples=state.examples<1024 ? 1024:state.examples;
    state.windowExamples=512; state.windowSum=512*score; state.windowSquares=512*score*score;
    state.windowCoverage=512; state.previousRankScore=score+.01f; state.protectedAction=0; state.cooldownUntil=0;
}
// Synthetic learned prototypes activate dispatch, long routes and projection learning before timing.
// Reuse preceding visual context; knowledge content/weights stay nonzero throughout the stress run.
__global__ void matureMemeFixture(vabpnn::DeviceMemeView v,std::uint64_t sequence) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;if(tile>=vabpnn::MemeBlocks)return;
    auto& prototype=v.persistent->prototypes[tile];
    for(int d=0;d<vabpnn::MemeContext;++d)prototype.context[d]=v.contexts[tile*vabpnn::MemeContext+d];
    for(int k=0;k<vabpnn::VoteCount;++k)prototype.evidence[k]=k==0 ? .8f:-.6f;
    prototype.origin=tile;prototype.support=1024;prototype.revision=23;prototype.confidence=.9f;
    prototype.utility=.4f;prototype.agreement=1;prototype.coverage=1;
    auto packet=prototype;packet.sequence=sequence;packet.hops=0;packet.age=0;
    v.packets[tile*2]=packet;
    for(int at=0;at<vabpnn::MemeWeights;++at)
        if(v.persistent->policy[tile].optimizerSteps==0)v.persistent->policy[tile].weights[at]=.025f;
}
// Mature supported experience index; fixture preparation is outside all timed events.
__global__ void matureLongMemoryFixture(vabpnn::DeviceLongMemoryView v,std::uint64_t wallMillis) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=vabpnn::LongMemoryCapacity)return;
    auto& e=v.entries[id];e={};
    for(int d=0;d<vabpnn::LongMemoryDimension;++d)e.key[d]=v.trace->key[d];
    for(int a=0;a<vabpnn::VoteCount;++a)e.action[a]=a==0 ? .8f:.1f;
    for(int k=0;k<vabpnn::LongMemoryKChoices;++k)e.kMass[k]=k==1 ? .8f:.2f/15;
    e.worldHi=11;e.worldLo=22;e.sessionHi=33;e.sessionLo=44;e.ordinal=id+1;
    e.wallMillis=e.lastSeenMillis=wallMillis-id*400;e.observedSeconds=3600-id*.4;
    e.seconds=420-id*.4f;e.minutes=e.seconds/60;e.labels=64;e.quality=.8f;e.valid=1;
    if(id==0){v.catalog->used=vabpnn::LongMemoryCapacity;v.catalog->ordinal=vabpnn::LongMemoryCapacity+1;
        v.catalog->observedSeconds=3600;
        for(int a=0;a<vabpnn::VoteCount;++a)v.persistent->parameters.evidenceGain[a]=.1f;
        for(int k=0;k<vabpnn::LongMemoryKChoices;++k)v.persistent->parameters.kGain[k]=.05f;}
}
// Reuse the previous completed visual key to keep all 8192 entries above the cosine cutoff,
// including during JEPA updates. Frozen profiles seed only before measurement;
// every measured frozen frame keeps the entire index unchanged.
__global__ void refreshLongMemoryFixture(vabpnn::DeviceLongMemoryView v) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<vabpnn::LongMemoryCapacity*vabpnn::LongMemoryDimension)
        v.entries[i/vabpnn::LongMemoryDimension].key[i%vabpnn::LongMemoryDimension]=v.trace->key[i%vabpnn::LongMemoryDimension];
}
struct Sample { double copy,op,total,wall; int jepa,pixel,topology,scope,actor,ranking,mutation,meme; };
void report(const char* label,const std::vector<Sample>& values) {
    if(values.empty()) return;
    std::vector<double> times; double copy=0,wall=0,total=0; int over10=0; double maximumTotal=0;
    for(const auto& s:values) { times.push_back(s.op); copy+=s.copy; wall+=s.wall; total+=s.total; over10+=s.total>10; maximumTotal=std::max(maximumTotal,s.total); }
    std::sort(times.begin(),times.end());
    auto percentile=[&](double p) { return times[static_cast<std::size_t>((times.size()-1)*p)]; };
    const double average=std::accumulate(times.begin(),times.end(),0.0)/times.size();
    std::printf("%s n=%zu op_mean=%.4f op_p50=%.4f op_p95=%.4f op_p99=%.4f op_max=%.4f copy_mean=%.4f copy_op_control_mean=%.4f worker_wall_mean=%.4f ms\n",
        label,times.size(),average,percentile(.5),percentile(.95),percentile(.99),times.back(),
        copy/times.size(),total/times.size(),wall/times.size());
    std::printf(" budget_over10=%d total_max=%.4f ms\n",over10,maximumTotal);
}
void profile(int mode,int selfSupervised,const char* label,bool overlap=false,bool forceRanking=false,bool trainRewards=false) {
    constexpr int Width=1920,Height=1024,Warmup=64,Count=512;
    cudaStream_t stream=nullptr; unsigned char* source=nullptr; unsigned char* image=nullptr; float* rgb=nullptr;
    std::size_t sourcePitch=0,pitch=0; cudaEvent_t events[4]{};
    VabpnnNeuronModel* neuron=nullptr; VabpnnDecisionModel* decision=nullptr;
    VabpnnJepaModel* jepa=nullptr; VabpnnOnlineModel* online=nullptr; VabpnnSpikeModel* spike=nullptr;
    VabpnnRewardPolicy* reward=nullptr;vabpnn::RewardBatch batch{};
    VabpnnLongMemory* memory=nullptr;
    vabpnn::ScopeTelemetry* scopeDevice=nullptr;
    vabpnn::MemeScopeTelemetry* memeScopeDevice=nullptr;vabpnn::MemeRelayTrace* memeTrace=nullptr;
    struct HostStatus { vabpnn::LongMemoryStatus memory;vabpnn::RewardProposal reward;vabpnn::ControlCommand command; vabpnn::DecisionStatus decision; vabpnn::OnlineOutput online; vabpnn::JepaOutput jepa; vabpnn::SpikeCounters spike; vabpnn::BlockCounters blocks; vabpnn::MemeCounters memes; vabpnn::ScopeTelemetry scope; vabpnn::MemeScopeTelemetry memeScope; };
    HostStatus* status=nullptr;
    try {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaMallocPitch(reinterpret_cast<void**>(&source),&sourcePitch,Width*4,Height));
        check(cudaMallocPitch(reinterpret_cast<void**>(&image),&pitch,Width*4,Height));
        check(cudaMalloc(reinterpret_cast<void**>(&rgb),static_cast<std::size_t>(Width)*Height*3*sizeof(float)));
        check(cudaMallocHost(reinterpret_cast<void**>(&status),sizeof(HostStatus))); *status=HostStatus{};
        check(cudaMalloc(reinterpret_cast<void**>(&scopeDevice),sizeof(vabpnn::ScopeTelemetry)));
        if(MemeScopeEnabled) {
            check(cudaMalloc(reinterpret_cast<void**>(&memeScopeDevice),sizeof(vabpnn::MemeScopeTelemetry)));
            check(cudaMalloc(reinterpret_cast<void**>(&memeTrace),vabpnn::MemeBlocks*sizeof(vabpnn::MemeRelayTrace)));
            check(cudaMemsetAsync(memeTrace,0,vabpnn::MemeBlocks*sizeof(vabpnn::MemeRelayTrace),stream));
        }
        for(auto& event:events) check(cudaEventCreate(&event));
        pattern<<<dim3(120,64),dim3(16,16),0,stream>>>(source,sourcePitch,Width,Height);
        neuron=vabpnnCreateNeuronModel(stream); decision=vabpnnCreateDecisionModel(stream);
        memory=vabpnnCreateLongMemory(stream);
        jepa=vabpnnCreateJepaModel(stream); online=vabpnnCreateOnlineModel(stream); spike=vabpnnCreateSpikeModel(stream);
        auto blockConfig=vabpnn::BlockConfig{}; blockConfig.enabled=BlocksEnabled;
        if(forceRanking) { blockConfig.warmup=0; blockConfig.rankEvery=20; blockConfig.poorWindows=1; }
        auto* blockModel=vabpnnSpikeBlockModel(spike); vabpnnConfigureBlockModel(blockModel,blockConfig,stream);
        auto* memeModel=vabpnnBlockMemeModel(blockModel);
        vabpnnConfigureMemeEnabled(memeModel,MemesEnabled ? 1:0,stream);
        fullCapacity<<<(vabpnn::NeuronCount+255)/256,256,0,stream>>>(vabpnnMutableSpikeStatistics(spike,stream));
        auto config=vabpnn::JepaConfig{}; config.selfSupervisedEnabled=selfSupervised;
        if(overlap) config.trainEveryFrames=1;
        check(cudaMemcpyAsync(vabpnnMutableJepaConfig(jepa,stream),&config,sizeof(config),cudaMemcpyHostToDevice,stream));
        if(mode) {
            // Freeze an already connected model, rather than benchmark an empty new long graph.
            check(cudaMemcpy2DAsync(image,pitch,source,sourcePitch,Width*4,Height,cudaMemcpyDeviceToDevice,stream));
            VabpnnGpuFrame initial{image,pitch,rgb,Width,Height,0,1'000'000'000,stream,neuron,decision,jepa,online,spike};
            for(int t=0;t<6;++t) vabpnnLaunchSpikeForward(initial);
            vabpnnResetSpikeHistory(spike,stream);
            initial.evaluationMode=true;initial.frameId=1;
            vabpnnLaunchOperator(initial);
            initial.frameId++;initial.captureNanos++;
            vabpnnLaunchOperator(initial); // Seed knowledge with a real completed previous decision context.
            if(MemeStress && MemesEnabled) matureMemeFixture<<<30,256,0,stream>>>(vabpnnMemeModelView(memeModel),1);
        }
        if(MinuteStress) {
            // Fill eight actually observed minute bins before timing, including their
            // complete causal attention cost after the short-seconds cache has rolled.
            std::int32_t* seedVotes=nullptr;
            check(cudaMalloc(reinterpret_cast<void**>(&seedVotes),vabpnn::VoteCount*sizeof(std::int32_t)));
            check(cudaMemsetAsync(seedVotes,0,vabpnn::VoteCount*sizeof(std::int32_t),stream));
            vabpnnResetDecisionState(decision,stream);
            for(int t=0;t<vabpnn::MinuteHistory;++t) {
                VabpnnGpuFrame seed{};seed.stream=stream;seed.frameId=t+1;
                seed.captureNanos=1'000'000'000ULL+t*vabpnn::MinuteNanos;seed.evaluationMode=true;
                vabpnnLaunchTemporalDecision(seed,seedVotes,decision);
            }
            check(cudaStreamSynchronize(stream));check(cudaFree(seedVotes));
        }
        // Warmup supplies the first actual query key before any measured frames. Do not
        // advance the operator here: doing so shifts the existing worst-case overlap schedules.
        matureLongMemoryFixture<<<32,256,0,stream>>>(vabpnnLongMemoryView(memory),1790000000000ull);
        if(trainRewards) {
            reward=vabpnnCreateRewardPolicy(stream);vabpnnConfigureRewardPolicy(reward,1,true,stream,1,true);
            batch.epoch=1;batch.credit=1;
        }
        check(cudaStreamSynchronize(stream));
        std::vector<Sample> all,ordinary,jepaFrames,pixelFrames,both,topologyFrames,topologyPixel,scopeFrames,allOverlap;
        std::vector<Sample> actorFrames,rankingFrames,mutationFrames,completeOverlap,memeFrames,completeMemeOverlap;
        std::uint64_t previousActors=0,previousRanks=0,previousMutations=0,previousMeme=0;
        std::uint64_t previousTopology=0;
        std::uint64_t previousJepa=0,previousPixel=0;
        vabpnn::DemoSample demo; demo.active=1; demo.epoch=1; demo.modelControl=mode; demo.pointerMode=1; demo.heldKeys=1;
        std::array<vabpnn::RewardProposal,vabpnn::RewardCacheSlots> receipts{};
        for(int i=0;i<Warmup+Count;++i) {
            const auto id=static_cast<std::uint64_t>(i+1);
            const auto minuteOffset=MinuteStress ? (vabpnn::MinuteHistory-1)*vabpnn::MinuteNanos:0;
            const auto nanos=1'000'000'000ULL+minuteOffset+id*16'666'667ULL;
            demo.mouseX+=1; demo.keyHeldNanos[0]+=16'666'667ULL;
            VabpnnGpuFrame frame{image,pitch,rgb,Width,Height,id,nanos,stream,neuron,decision,jepa,online,spike};
            frame.longMemory=memory;frame.longMemoryContext={11,22,33,44,nanos,1790000000000ull+id*17};
            const auto sampleId=(id-1)/2*2+1; // 30 Hz observations, 60 Hz predictions.
            if(PositionEnabled) frame.position={1'000'000'000ULL+minuteOffset+sampleId*16'666'667ULL,1,
                125.0+sampleId/30.0,64.0,-82.0+sampleId/60.0};
            if(forceRanking && i>0 && i%20==0)
                matureRankFixture<<<30,256,0,stream>>>(vabpnnMutableBlockPersistent(blockModel,stream));
            if(MemeStress && MemesEnabled)
                matureMemeFixture<<<30,256,0,stream>>>(vabpnnMemeModelView(memeModel),id);
            if(trainRewards && i>0 && i%vabpnn::RewardCacheSlots==0) {
                batch.rewardId++;batch.count=0;batch.credit=batch.rewardId%2 ? 1:-1;
                for(unsigned j=0;j<vabpnn::RewardCacheSlots;++j){
                    const auto fid=id-vabpnn::RewardCacheSlots+j;const auto& receipt=receipts[fid%vabpnn::RewardCacheSlots];
                    if(receipt.valid && receipt.frameId==fid){batch.frames[batch.count]=fid;batch.masks[batch.count++]=receipt.mask;}
                }
                if(batch.count!=vabpnn::RewardCacheSlots)throw std::runtime_error("Full reward cache was not actually filled");
            }
            if(i>0 && (!mode || i==Warmup-1))refreshLongMemoryFixture<<<2048,256,0,stream>>>(vabpnnLongMemoryView(memory));
            const auto begin=std::chrono::steady_clock::now();
            check(cudaEventRecord(events[0],stream));
            check(cudaMemcpy2DAsync(image,pitch,source,sourcePitch,Width*4,Height,cudaMemcpyDeviceToDevice,stream));
            check(cudaEventRecord(events[1],stream));
            frame.evaluationMode=demo.modelControl!=0;frame.memeTrace=memeTrace;
            if(trainRewards && batch.count){
                vabpnnApplyRewardBatch(reward,vabpnnMutableDecisionParameters(decision,stream),batch,stream);batch.count=0;
            }
            vabpnnSetDemoSample(online,&demo,stream); vabpnnLaunchOperator(frame);
            if(trainRewards) {
                const auto v=vabpnnDecisionModelView(decision);
                vabpnnProposeRewardAction(reward,v.parameters,v.constraints,
                    const_cast<vabpnn::DecisionOutput*>(v.output),vabpnnJepaModelView(jepa).output,
                    vabpnnMutableMotionProposal(decision,stream),stream);
            }
            const bool sampleScope=forceRanking || i%6==0;
            if(sampleScope) vabpnnLaunchNetworkScope(frame,32*120+60,scopeDevice);
            const bool sampleMeme=MemeScopeEnabled && i%12==0; // 5 Hz at simulated 60 Hz.
            if(sampleMeme) vabpnnLaunchMemeScope(frame,32*120+60,memeScopeDevice);
            check(cudaEventRecord(events[2],stream));
            check(cudaMemcpyAsync(&status->memory,&vabpnnLongMemoryView(memory).output->status,sizeof(status->memory),cudaMemcpyDeviceToHost,stream));
            if(trainRewards)check(cudaMemcpyAsync(&status->reward,vabpnnRewardPolicyView(reward).proposal,sizeof(status->reward),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->command,vabpnnDecisionModelView(decision).command,sizeof(status->command),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->decision,&vabpnnDecisionModelView(decision).output->status,sizeof(status->decision),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->online,vabpnnOnlineOutput(online),sizeof(status->online),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->jepa,vabpnnJepaModelView(jepa).output,sizeof(status->jepa),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->spike,vabpnnSpikeModelView(spike).counters,sizeof(status->spike),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->blocks,vabpnnBlockModelView(blockModel).counters,sizeof(status->blocks),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->memes,&vabpnnMemeModelView(memeModel).persistent->counters,sizeof(status->memes),cudaMemcpyDeviceToHost,stream));
            if(sampleScope) check(cudaMemcpyAsync(&status->scope,scopeDevice,sizeof(vabpnn::ScopeTelemetry),cudaMemcpyDeviceToHost,stream));
            if(sampleMeme) check(cudaMemcpyAsync(&status->memeScope,memeScopeDevice,sizeof(vabpnn::MemeScopeTelemetry),cudaMemcpyDeviceToHost,stream));
            check(cudaEventRecord(events[3],stream)); check(cudaEventSynchronize(events[3]));
            if(trainRewards)receipts[id%vabpnn::RewardCacheSlots]=status->reward;
            const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            float copy=0,op=0,total=0;
            check(cudaEventElapsedTime(&copy,events[0],events[1]));
            check(cudaEventElapsedTime(&op,events[1],events[2]));
            check(cudaEventElapsedTime(&total,events[0],events[3]));
            const bool learnedJepa=status->jepa.selfSupervisedSteps>previousJepa;
            const bool learnedPixel=status->online.pixelSteps>previousPixel;
            const bool topology=status->spike.topologyUpdates>previousTopology; previousTopology=status->spike.topologyUpdates;
            previousJepa=status->jepa.selfSupervisedSteps; previousPixel=status->online.pixelSteps;
            const bool actor=status->blocks.actorUpdates>previousActors,rank=status->blocks.rankings>previousRanks,
                mutation=status->blocks.mutations>previousMutations;
            previousActors=status->blocks.actorUpdates; previousRanks=status->blocks.rankings; previousMutations=status->blocks.mutations;
            const bool meme=status->memes.projectionUpdates>previousMeme;previousMeme=status->memes.projectionUpdates;
            if(MinuteStress && status->decision.minuteBuckets!=vabpnn::MinuteHistory)
                throw std::runtime_error("Minute stress must retain all eight observed bins");
            if(i>=Warmup && (status->memory.used!=vabpnn::LongMemoryCapacity || status->memory.retrieved!=vabpnn::LongMemoryTopK || status->memory.strength<=0)) {
                std::fprintf(stderr,"Memory fixture frame=%d valid=%u enabled=%u used=%u top=%u sim=%.5f strength=%.5f\n",i,status->memory.valid,status->memory.enabled,status->memory.used,status->memory.retrieved,status->memory.similarity,status->memory.strength);
                throw std::runtime_error("Full supported memory retrieval, Bayesian fusion and gates were not actually active");
            }
            if(i>=Warmup) {
                Sample sample{copy,op,total,wall,learnedJepa,learnedPixel,topology,sampleScope,actor,rank,mutation,meme};
                if(meme)memeFrames.push_back(sample);
                if(meme && actor && rank && mutation && sampleScope && topology && learnedJepa && learnedPixel)completeMemeOverlap.push_back(sample);
                if(actor) actorFrames.push_back(sample); if(rank) rankingFrames.push_back(sample); if(mutation) mutationFrames.push_back(sample);
                if(actor && rank && mutation && sampleScope && topology && learnedJepa && learnedPixel) completeOverlap.push_back(sample); all.push_back(sample);
                if(sampleScope) scopeFrames.push_back(sample);
                if(sampleScope && topology && learnedJepa && learnedPixel) allOverlap.push_back(sample);
                if(!learnedJepa && !learnedPixel && !topology) ordinary.push_back(sample);
                if(topology) topologyFrames.push_back(sample);
                if(topology && learnedPixel) topologyPixel.push_back(sample);
                if(learnedJepa && !learnedPixel) jepaFrames.push_back(sample);
                if(learnedPixel && !learnedJepa) pixelFrames.push_back(sample);
                if(learnedPixel && learnedJepa) both.push_back(sample);
            }
        }
        if(mode && (status->jepa.selfSupervisedSteps || status->jepa.motionSteps || status->online.examples
            || status->blocks.actorUpdates || status->blocks.mutations || status->memes.supervisedFrames
            || status->spike.topologyUpdates<=1)) throw std::runtime_error("Evaluation must freeze neural learning while continuing dynamic topology");
        std::printf("PROFILE %s selfSteps=%llu demoExamples=%llu mouseSteps=%llu modelBytes=%llu\n",label,
            static_cast<unsigned long long>(status->jepa.selfSupervisedSteps),
            static_cast<unsigned long long>(status->online.examples),
            static_cast<unsigned long long>(status->jepa.motionSteps),
            static_cast<unsigned long long>(vabpnnLongMemoryBytes(memory)+vabpnnNeuronModelBytes(neuron)+vabpnnDecisionModelBytes(decision)+vabpnnJepaModelBytes(jepa)+vabpnnOnlineModelBytes(online)+vabpnnSpikeModelBytes(spike)+sizeof(vabpnn::ScopeTelemetry)+(MemeScopeEnabled ? sizeof(vabpnn::MemeScopeTelemetry)+vabpnn::MemeBlocks*sizeof(vabpnn::MemeRelayTrace):0)));
        std::printf("MEMORY_STRESS used=%u retrieved=%u queries=%llu supervised=%llu gate_steps=%llu seconds=%.2f minutes=%.2f age_seconds=%.2f strength=%.5f bytes=%llu\n",
            status->memory.used,status->memory.retrieved,(unsigned long long)status->memory.queries,
            (unsigned long long)status->memory.supervised,(unsigned long long)status->memory.steps,
            status->memory.seconds,status->memory.minutes,status->memory.ageSeconds,status->memory.strength,
            (unsigned long long)vabpnnLongMemoryBytes(memory));
        report("all",all); report("ordinary",ordinary); report("JEPA_update",jepaFrames);
        report("pixel_update",pixelFrames); report("JEPA_and_pixel",both);
        report("topology_update",topologyFrames); report("topology_and_pixels",topologyPixel);
        report("block_actor_update",actorFrames); report("block_ranking",rankingFrames); report("block_mutation",mutationFrames);
        report("scope_topology_pixels_JEPA_actor_rank_mutation",completeOverlap);
        report("meme_projection_update",memeFrames);
        report("scope_topology_pixels_JEPA_actor_rank_mutation_MEME",completeMemeOverlap);
        std::printf(" meme_fixture=%d enabled=%u observed=%llu supervised=%llu projection_updates=%llu dispatches=%llu receipts=%llu long_selections=%llu local=%u imports=%u\n",
            MemeStress,status->memes.enabled,(unsigned long long)status->memes.observedFrames,(unsigned long long)status->memes.supervisedFrames,
            (unsigned long long)status->memes.projectionUpdates,(unsigned long long)status->memes.dispatches,
            (unsigned long long)status->memes.receipts,(unsigned long long)status->memes.longSelections,
            status->memes.localActive,status->memes.importActive);
        if(MemeStress && MemesEnabled && (status->memes.receipts==0 || status->memes.longSelections==0 || status->memes.localActive==0))
            throw std::runtime_error("Mature meme transport/CSR fixture not actually active");
        if(MemeStress && MemesEnabled && forceRanking && (memeFrames.empty() || completeMemeOverlap.empty()))
            throw std::runtime_error("Meme projection and complete overlap not actually measured");
        report("scope_sample",scopeFrames); report("scope_topology_pixels_JEPA",allOverlap);
        std::printf(" anchor_count=%u long_edges=%u topology_updates=%llu local_edges=%llu\n",status->spike.anchors,status->spike.longEdges,
            static_cast<unsigned long long>(status->spike.topologyUpdates),static_cast<unsigned long long>(status->spike.localEdges));
        std::printf(" block_samples=%llu actor_updates=%llu rankings=%llu mutations=%llu controls=%llu broadcasts=%llu stress_fixture=%d\n",
            (unsigned long long)status->blocks.scoredFrames,(unsigned long long)status->blocks.actorUpdates,
            (unsigned long long)status->blocks.rankings,(unsigned long long)status->blocks.mutations,
            (unsigned long long)status->blocks.controlUpdates,(unsigned long long)status->blocks.broadcasts,forceRanking);
        if(forceRanking && (rankingFrames.empty() || mutationFrames.empty() || completeOverlap.empty()))
            throw std::runtime_error("controller sorting/mutation/full overlap was not actually measured");
        if(status->spike.anchors!=256 || status->spike.longEdges<80000) throw std::runtime_error("full-capacity long graph was not measured");
        if(trainRewards){vabpnn::RewardStatus rewardStatus{};
            check(cudaMemcpy(&rewardStatus,vabpnnRewardPolicyView(reward).status,sizeof(rewardStatus),cudaMemcpyDeviceToHost));
            if(rewardStatus.updates<1 || rewardStatus.proposals<Count)throw std::runtime_error("Reward stress did not exercise actual GPU learning");
            std::printf("REWARD_STRESS updates=%llu proposals=%llu full-cache=512 weights=1022\n",
                static_cast<unsigned long long>(rewardStatus.updates),static_cast<unsigned long long>(rewardStatus.proposals));}
    } catch(...) {
        if(stream) cudaStreamSynchronize(stream);
        vabpnnDestroyRewardPolicy(reward);vabpnnDestroyLongMemory(memory);
        if(memeScopeDevice) cudaFree(memeScopeDevice);if(memeTrace) cudaFree(memeTrace);
        if(scopeDevice) cudaFree(scopeDevice); vabpnnDestroySpikeModel(spike); vabpnnDestroyOnlineModel(online); vabpnnDestroyJepaModel(jepa);
        vabpnnDestroyDecisionModel(decision); vabpnnDestroyNeuronModel(neuron);
        if(source) cudaFree(source); if(image) cudaFree(image); if(rgb) cudaFree(rgb); if(status) cudaFreeHost(status);
        for(auto event:events) if(event) cudaEventDestroy(event); if(stream) cudaStreamDestroy(stream);
        throw;
    }
    check(cudaStreamSynchronize(stream));
    vabpnnDestroyRewardPolicy(reward);vabpnnDestroyLongMemory(memory);
    if(memeScopeDevice) cudaFree(memeScopeDevice);if(memeTrace) cudaFree(memeTrace);
        if(scopeDevice) cudaFree(scopeDevice); vabpnnDestroySpikeModel(spike); vabpnnDestroyOnlineModel(online); vabpnnDestroyJepaModel(jepa);
    vabpnnDestroyDecisionModel(decision); vabpnnDestroyNeuronModel(neuron);
    cudaFree(source); cudaFree(image); cudaFree(rgb); cudaFreeHost(status);
    for(auto event:events) cudaEventDestroy(event); cudaStreamDestroy(stream);
}
int main(int argc,char** argv) {
    for(int i=1;i<argc;i++) {
        const std::string arg=argv[i];
        if(arg=="--reward-stress") RewardStress=true;
        if(arg=="--without-position") PositionEnabled=false;
        if(arg=="--without-blocks") BlocksEnabled=false;
        if(arg=="--block-stress") BlockStress=true;
        if(arg=="--meme-stress") MemeStress=true;
        if(arg=="--meme-scope") MemeScopeEnabled=true;
        if(arg=="--minute-stress") MinuteStress=true;
        if(arg=="--without-memes") MemesEnabled=false;
    }
    std::printf("LONG_MEMORY=8192 supported vectors / 64 dimensions / exact top-8 / active learned gates and Bayesian fusion (mature stress fixture)\n");
    std::printf("MINUTE_HISTORY=%s (observed eight-bin causal history)\n",MinuteStress ? "mature stress":"normal");
    std::printf("MEME_SCOPE=%s (5 Hz telemetry + continuous optional relay trace)\n",MemeScopeEnabled ? "enabled":"hidden");
    std::printf("MEMES=%s mature_fixture=%d\n",MemesEnabled ? "enabled":"disabled",MemeStress);
    std::printf("BLOCK_CONTROLLERS=%s\n",BlocksEnabled ? "enabled":"disabled");
    std::printf("POSITION_FEATURES=%s (same GPU pipeline, 30 Hz position metadata / 60 Hz predictions)\n",PositionEnabled ? "enabled":"disabled");
    try {
        cudaDeviceProp gpu{}; check(cudaGetDeviceProperties(&gpu,0));
        std::printf("GPU=%s frame=1920x1024 warmup=64 measured=512 mode=release/O3; no GL or Minecraft render included\n",gpu.name);
        profile(1,0,"inference_only");
        profile(1,1,"evaluation_weights_frozen_dynamic_topology");
        profile(0,1,"human_demo_plus_JEPA");
        profile(0,1,"stress_topology_pixels_JEPA_every_frame",true);
        if(BlockStress) profile(0,1,"stress_hierarchy_rank_mutation_fixture",true,true);
        if(RewardStress)profile(1,0,"selftrain_bootstrap_full_reward_cache",false,false,true);
        return 0;
    } catch(const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}

