#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "FrameOperator.h"
#include "SpikeConnections.h"
#include "OnlineLearning.h"
#include "MemeState.h"
#include "model_checkpoint.h"
#include <memory>
#include <vector>
#include <cmath>
#include <cstdio>
using namespace vabpnn;
void check(cudaError_t c) { if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c)); }
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
template<class T> T read(const T* p) { T v; check(cudaMemcpy(&v,p,sizeof(v),cudaMemcpyDeviceToHost)); return v; }
template<class T> void put(T* p,const T& v,cudaStream_t s) {
    check(cudaMemcpyAsync(p,&v,sizeof(v),cudaMemcpyHostToDevice,s)); check(cudaStreamSynchronize(s));
}
unsigned hash(unsigned x) { x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16); }
double noise(unsigned seed) {
    double u=(double(hash(seed)&0xffffffu)+1)/16777217.0;
    double v=double(hash(seed+97)&0xffffffu)/16777216.0;
    return std::sqrt(-2*std::log(u))*std::cos(6.28318530718*v);
}
__global__ void rankFixture(BlockPersistent* p,int kind) {
    int tile=blockIdx.x*blockDim.x+threadIdx.x; if(tile>=BlockCount) return;
    auto& state=p->state[tile]; state=BlockState{};
    state.examples=64; state.windowExamples=64; state.windowCoverage=64;
    const float score=kind==0 ? 0:tile<BlockCount/5 ? -.8f:.2f;
    state.windowSum=64*score; state.windowSquares=64*score*score; state.previousRankScore=kind==4 ? score:score+.03f;
    state.weights[0]=.1f; state.first[0]=.04f; state.second[0]=.005f; state.optimizerSteps=90;
    if(kind==2) { if(tile==0) state.protectedAction=1; if(tile==1) state.cooldownUntil=p->counters.scoredFrames+100; }
    if(kind==3 && tile==2) { state.windowExamples=1; state.windowSum=-.8f; state.windowSquares=.64f; state.windowCoverage=1; }
    if(kind==3 && tile==3) state.windowCoverage=0;
}
void controlsBounded(DeviceBlockView b) {
    std::vector<BlockControl> controls(BlockCount); check(cudaMemcpy(controls.data(),b.controls,controls.size()*sizeof(BlockControl),cudaMemcpyDeviceToHost));
    for(auto c:controls) require(std::isfinite(c.thresholdScale) && c.thresholdScale>=.8f && c.thresholdScale<=1.2f
        && c.polarityBias>=-.020001f && c.polarityBias<=.020001f && c.memoryScale>=.84999f && c.memoryScale<=1.15001f
        && c.learningScale>=.79999f && c.learningScale<=1.20001f,"all four control attributes stay bounded");
}
int main(int argc,char** argv) {
    cudaStream_t stream=nullptr; VabpnnNeuronModel* neuron=nullptr; VabpnnDecisionModel* decision=nullptr;
    VabpnnJepaModel* jepa=nullptr; VabpnnOnlineModel* online=nullptr; VabpnnSpikeModel* spike=nullptr;
    unsigned char* image=nullptr; float* rgb=nullptr; int* valid=nullptr; unsigned* label=nullptr; int result=0;
    try {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        neuron=vabpnnCreateNeuronModel(stream); decision=vabpnnCreateDecisionModel(stream);
        jepa=vabpnnCreateJepaModel(stream); online=vabpnnCreateOnlineModel(stream); spike=vabpnnCreateSpikeModel(stream);
        check(cudaMalloc(reinterpret_cast<void**>(&image),16)); check(cudaMemsetAsync(image,128,16,stream));
        check(cudaMalloc(reinterpret_cast<void**>(&rgb),12*sizeof(float)));
        check(cudaMalloc(reinterpret_cast<void**>(&valid),sizeof(int)));
        check(cudaMalloc(reinterpret_cast<void**>(&label),sizeof(unsigned))); put(valid,1,stream); put(label,1u,stream);
        VabpnnGpuFrame f{image,8,rgb,2,2,123,1'000'000'000ULL,stream,neuron,decision,jepa,online,spike};
        auto* model=vabpnnSpikeBlockModel(spike);
        vabpnnConfigureMemeEnabled(vabpnnBlockMemeModel(model),0,stream); // Isolate original policy CPU reference; new branch has its own numerical suite.
        auto b=vabpnnBlockModelView(model);
        auto* persistent=vabpnnMutableBlockPersistent(model,stream);
        auto nv=vabpnnNeuronModelView(neuron); auto dv=vabpnnDecisionModelView(decision); auto sv=vabpnnSpikeModelView(spike);
        check(cudaStreamSynchronize(stream));
        require(read(b.controls).thresholdScale==1 && read(b.controls).memoryScale==1,"neutral initialization");
        auto config=BlockConfig{}; auto invalid=config; invalid.radius=3;
        bool rejected=false; try { vabpnnConfigureBlockModel(model,invalid,stream); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"unbounded communication rejected");
        // Independent reference: the action was generated from the OLD zero movement context.
        config.warmup=0; config.trainEvery=1; config.communication=0; config.rankEvery=65536;
        vabpnnConfigureBlockModel(model,config,stream);
        DemoSample demo; demo.active=1; demo.epoch=1; put(const_cast<DemoSample*>(vabpnnOnlineDemoSample(online)),demo,stream);
        DecisionOutput context{}; context.valid=1; context.posterior[12]=.4f;
        put(const_cast<DecisionOutput*>(dv.output),context,stream);
        check(cudaMemsetAsync(const_cast<unsigned*>(sv.championIds),0xff,13*sizeof(unsigned),stream));
        check(cudaMemsetAsync(const_cast<int*>(nv.blockVotes),0,13*BlockCount*sizeof(int),stream));
        put(const_cast<int*>(nv.blockVotes),255,stream);
        auto params=read(dv.parameters); params.rawVoteWeight[0]=1; put(vabpnnMutableDecisionParameters(decision,stream),params,stream);
        vabpnnPrepareBlockControllers(f); check(cudaStreamSynchronize(stream));
        auto control=read(b.controls);
        const double epsilon=noise(unsigned(f.frameId)^hash(1));
        require(std::fabs(control.thresholdScale-(1+.2*std::tanh(config.exploration*epsilon)))<2e-6,
                "actual latent Gaussian action agrees with CPU reference");
        auto output=context; output.frameId=f.frameId; for(auto& p:output.posterior) p=.8f;
        put(const_cast<DecisionOutput*>(dv.output),output,stream);
        f.frameId++; vabpnnLearnBlockControllers(f,valid,label); check(cudaStreamSynchronize(stream));
        auto state=read(b.state);
        const double expected=config.learningRate*(epsilon>0 ? 1:-1);
        require(state.optimizerSteps==1 && std::fabs(state.weights[0]-expected)<3e-6,"first actor AdamW step uses cached latent Gaussian");
        require(state.first[1]==0 && state.weights[1]==0,"new context is not substituted for cached action features");
        auto counters=read(b.counters); require(counters.scoredFrames==1 && counters.actorUpdates==BlockCount,"true feedback trains real independent actors");
        vabpnnLearnBlockControllers(f,valid,label); check(cudaStreamSynchronize(stream));
        require(read(b.counters).scoredFrames==1,"one prediction cannot be scored twice");
        controlsBounded(b);
        // Invalid calls exactly at the old host-period boundary must not suppress rankings.
        config.warmup=999999; config.rankEvery=4; config.minimumExamples=0; config.minimumWindow=2; config.poorWindows=1;
        vabpnnConfigureBlockModel(model,config,stream);
        check(cudaMemsetAsync(const_cast<int*>(nv.blockVotes),0,13*BlockCount*sizeof(int),stream));
        const auto before=read(b.counters);
        for(int call=1;call<=12;++call) {
            f.frameId++; vabpnnPrepareBlockControllers(f);
            output.frameId=f.frameId; put(const_cast<DecisionOutput*>(dv.output),output,stream);
            put(valid,call%4==0 ? 0:1,stream); f.frameId++;
            vabpnnLearnBlockControllers(f,valid,label);
        }
        check(cudaStreamSynchronize(stream)); counters=read(b.counters);
        require(counters.scoredFrames==before.scoredFrames+9 && counters.rankings==before.rankings+2,
                "rankings count valid scored frames across periodic invalid feedback");
        require(counters.mutations==0,"uniform or low-coverage population is not arbitrarily punished");
        put(valid,1,stream); config.rankEvery=1; config.minimumWindow=32; vabpnnConfigureBlockModel(model,config,stream);
        auto feedback=[&]() {
            f.frameId++; vabpnnPrepareBlockControllers(f); output.frameId=f.frameId;
            put(const_cast<DecisionOutput*>(dv.output),output,stream); f.frameId++;
            vabpnnLearnBlockControllers(f,valid,label); check(cudaStreamSynchronize(stream));
        };
        rankFixture<<<30,256,0,stream>>>(persistent,0); feedback();
        require(read(b.counters).punished==0,"tied mature scores cannot be punished by ID order");
        rankFixture<<<30,256,0,stream>>>(persistent,4); feedback();
        require(read(b.counters).punished==0,"bottom-ranked but improving controllers must not be punished");
        rankFixture<<<30,256,0,stream>>>(persistent,2); feedback(); counters=read(b.counters);
        std::printf("RANK_FIXTURE eligible=%u candidates=%u punished=%u\n",counters.eligible,counters.candidates,counters.punished);
        require(counters.punished>0 && counters.punished<=counters.eligible/5,"strict bounded bottom fifth penalty");
        require(read(b.state).mutations==0 && read(b.state+1).mutations==0,"rare action champion and cooldown preserved");
        state=read(b.state+2); require(state.mutations==1 && state.optimizerSteps==0 && state.first[0]==0 && state.second[0]==0,
                "mutation resets both Adam moments and bias-correction step, while retaining policy");
        rankFixture<<<30,256,0,stream>>>(persistent,3); feedback();
        require(read(b.state+2).windowExamples==2 && read(b.state+3).windowExamples==65,
                "unmatured and insufficient-coverage windows survive global rankings");
        // Double-buffered neighbor messages: no instant feedback, no flattened row wrap.
        config.rankEvery=65536; config.minimumCoverage=.05f; config.communication=1; config.radius=1;
        vabpnnConfigureBlockModel(model,config,stream);
        check(cudaMemsetAsync(persistent->state,0,BlockCount*sizeof(BlockState),stream));
        BlockState sender; sender.weights[0]=.25f; sender.coverage=1; put(persistent->state+119,sender,stream);
        f.frameId++; vabpnnPrepareBlockControllers(f); check(cudaStreamSynchronize(stream));
        require(read(b.controls+118).thresholdScale==1 && read(b.controls+120).thresholdScale==1,"first message tick reads previous buffer only");
        for(int i=0;i<5;++i) { f.frameId++; vabpnnPrepareBlockControllers(f); }
        check(cudaStreamSynchronize(stream));
        require(read(b.controls+118).thresholdScale>1 && read(b.controls+120).thresholdScale==1,
                "bounded Manhattan broadcast crosses actual neighbors, never a wrapped row");
        vabpnnConfigureBlockMode(model,-1,0,stream); f.frameId++; vabpnnPrepareBlockControllers(f);
        check(cudaStreamSynchronize(stream)); require(read(b.controls+118).thresholdScale==1,"independent mode cannot inherit neighbor influence");
        vabpnnConfigureBlockMode(model,0,-1,stream); f.frameId++; vabpnnPrepareBlockControllers(f);
        check(cudaStreamSynchronize(stream)); require(read(b.controls+119).thresholdScale==1,"off mode restores neutral controls");
        // Independent numeric LIF checks: same stimulus, seconds-based decay and different signed release.
        vabpnnResetSpikeHistory(spike,stream);
        check(cudaMemsetAsync(vabpnnMutableSpikeOutgoing(spike,stream),0,NeuronCount*4*sizeof(unsigned short),stream));
        vabpnnRebuildSpikeLocalCsr(spike,stream);
        SpikeConfig lif; lif.maxAnchors=0; lif.threshold=1.5f; lif.refractorySeconds=0;
        put(vabpnnMutableSpikeConfig(spike,stream),lif,stream);
        f.captureNanos+=20'000'000; vabpnnLaunchSpikeForward(f); check(cudaStreamSynchronize(stream));
        put(const_cast<float*>(sv.membrane),.2f,stream); put(const_cast<float*>(sv.membrane)+16,.2f,stream);
        BlockControl longer; longer.memoryScale=1.15f; put(const_cast<BlockControl*>(b.controls),longer,stream);
        f.captureNanos+=20'000'000; vabpnnLaunchSpikeForward(f); check(cudaStreamSynchronize(stream));
        const double decay=std::exp(-.02/(double(lif.membraneTau)*1.15));
        const double oldDecay=std::exp(-.02/double(lif.membraneTau));
        require(std::fabs(read(sv.membrane)-(decay*.2+(1-decay)*128/255.0))<2e-6
                && std::fabs(read(sv.membrane+16)-(oldDecay*.2+(1-oldDecay)*128/255.0))<2e-6,
                "seconds-based membrane scaling agrees with independent CPU reference and neutral block");
        vabpnnResetSpikeHistory(spike,stream); lif.membraneTau=.001f; lif.threshold=.6f; lif.negativeMargin=.01f;
        put(vabpnnMutableSpikeConfig(spike,stream),lif,stream);
        f.captureNanos+=20'000'000; vabpnnLaunchSpikeForward(f);
        BlockControl positive; positive.thresholdScale=.8f; positive.polarityBias=.02f;
        BlockControl negative=positive; negative.polarityBias=-.02f;
        put(const_cast<BlockControl*>(b.controls),positive,stream); put(const_cast<BlockControl*>(b.controls)+2,negative,stream);
        f.captureNanos+=20'000'000; vabpnnLaunchSpikeForward(f); check(cudaStreamSynchronize(stream));
        require(read(sv.spikes)==1 && read(sv.spikes+16)==0 && read(sv.spikes+32)==-1,
                "actual threshold and polarity modulation changes positive/quiet/negative spike release");
        put(vabpnnMutableSpikeConfig(spike,stream),SpikeConfig{},stream);
        // Full integrated path: local LR affects the actual pixel latent update; model mode stops feedback.
        config=BlockConfig{}; config.warmup=999999; config.communication=0; vabpnnConfigureBlockModel(model,config,stream);
        check(cudaMemsetAsync(persistent->state,0,BlockCount*sizeof(BlockState),stream));
        BlockState local; local.weights[3*BlockFeatures]=.5f; local.weights[0]=-.5f; local.weights[2*BlockFeatures]=.5f;
        put(persistent->state,local,stream);
        auto jc=read(vabpnnJepaModelView(jepa).config); jc.selfSupervisedEnabled=0;
        put(vabpnnMutableJepaConfig(jepa,stream),jc,stream);
        vabpnnResetDecisionState(decision,stream); vabpnnResetOnlineHistory(online,stream); vabpnnResetSpikeHistory(spike,stream);
        check(cudaMemsetAsync(vabpnnMutableNeuronWeights(neuron,stream),0,ParameterCount,stream));
        ModelPart onlineParts[8]; vabpnnOnlineCheckpointParts(online,onlineParts);
        check(cudaMemsetAsync(onlineParts[0].device,0,onlineParts[0].bytes,stream));
        demo.heldKeys=1; demo.epoch=17;
        auto run=[&]() { f.frameId++; f.captureNanos+=20'000'000; demo.keyHeldNanos[0]+=20'000'000;
            vabpnnSetDemoSample(online,&demo,stream); vabpnnLaunchOperator(f); check(cudaStreamSynchronize(stream)); };
        const auto examplesBefore=read(vabpnnOnlineOutput(online)).examples;
        for(int i=0;i<5;++i) run();
        auto onlineStatus=read(vabpnnOnlineOutput(online));
        require(onlineStatus.examples==examplesBefore+4,"actual online pipeline carries true previous-frame block feedback");
        const float master=read(static_cast<float*>(onlineParts[0].device));
        control=read(b.controls);
        require(control.learningScale>1 && std::fabs(master-.02f*control.learningScale*.5f*(128.0f/255))<2e-6,
                "controller learning scale changes the actual pixel SGD step");
        require(control.thresholdScale<1 && control.memoryScale>1,"nonneutral threshold and second-based time constant reach spike forward");
        const auto supervised=read(b.counters).scoredFrames;
        demo.modelControl=1; demo.epoch++; vabpnnResetOnlineHistory(online,stream); vabpnnResetSpikeHistory(spike,stream);
        run(); run(); require(read(b.counters).scoredFrames==supervised,"model actions are never hierarchical supervision");
        // Save version4 in an isolated fixture, preserve learned state, discard pending credit on resume.
        auto path=(std::filesystem::current_path()/L"block-controllers-test-v4.bin").wstring();
        std::vector<ModelPart> part{vabpnnBlockCheckpointPart(model)};
        const auto saved=read(b.state); saveModelCheckpoint(path,part,4);
        check(cudaMemset(persistent->state,0,BlockCount*sizeof(BlockState)));
        require(loadModelCheckpoint(path,part,false,4),"version4 hash/layout roundtrip");
        vabpnnResumeBlockModel(model,stream); check(cudaStreamSynchronize(stream));
        state=read(b.state); require(state.weights[0]==saved.weights[0] && state.examples==saved.examples,"resume retains learned policy and scores");
        const auto restored=read(b.counters).scoredFrames;
        vabpnnLearnBlockControllers(f,valid,label); check(cudaStreamSynchronize(stream));
        require(read(b.counters).scoredFrames==restored,"restore clears old action credit instead of replaying labels");
        std::filesystem::remove(path);
        if(argc>1) {
            // Read-only migration check against a real v3 model supplied by the caller.
            std::vector<ModelPart> oldParts{
                {vabpnnMutableNeuronWeights(neuron,stream),ParameterCount},
                {vabpnnMutableDecisionParameters(decision,stream),sizeof(DecisionParameters)}};
            ModelPart extra[8];
            auto append=[&](std::size_t count) { oldParts.insert(oldParts.end(),extra,extra+count); };
            append(vabpnnJepaCheckpointParts(jepa,extra)); append(vabpnnOnlineCheckpointParts(online,extra));
            append(vabpnnSpikeCheckpointParts(spike,extra)); append(vabpnnDecisionPositionCheckpointParts(decision,extra));
            require(oldParts.size()==15,"legacy v3 retains the original fifteen parts");
            const auto legacy=std::filesystem::u8path(argv[1]).wstring();
            require(loadModelCheckpoint(legacy,oldParts,false,3),"actual legacy v3 model loads read-only");
            check(cudaMemsetAsync(persistent->state,0,BlockCount*sizeof(BlockState),stream));
            auto neutralConfig=BlockConfig{}; vabpnnConfigureBlockModel(model,neutralConfig,stream);
            check(cudaStreamSynchronize(stream));
            vabpnnResumeSpikeModel(spike,stream); vabpnnResumeJepaOptimizer(jepa,stream);
            jc=read(vabpnnJepaModelView(jepa).config); jc.selfSupervisedEnabled=0;
            put(vabpnnMutableJepaConfig(jepa,stream),jc,stream);
            demo.modelControl=1; demo.active=1;
            auto prediction=[&](int enabled) {
                vabpnnConfigureBlockMode(model,enabled,0,stream);
                vabpnnResetDecisionState(decision,stream); vabpnnResetJepaHistory(jepa,stream);
                vabpnnResetOnlineHistory(online,stream); vabpnnResetSpikeHistory(spike,stream);
                for(int i=0;i<4;++i) run();
                return read(dv.output);
            };
            const auto disabled=prediction(0),neutralEnabled=prediction(1);
            require(disabled.selectedMask==neutralEnabled.selectedMask && disabled.requestedK==neutralEnabled.requestedK,
                    "legacy-trained control selection preserved by neutral preview controllers");
            for(int k=0;k<VoteCount;++k)
                require(disabled.normalizedVotes[k]==neutralEnabled.normalizedVotes[k]
                    && std::fabs(disabled.posterior[k]-neutralEnabled.posterior[k])<1e-6
                    && disabled.threshold[k]==neutralEnabled.threshold[k],
                    "real legacy model votes/posteriors/thresholds match disabled baseline");
            std::puts("PASS: actual v3 read-only restore, fifteen-part layout, neutral controllers preserve legacy inference.");
            if(argc>2) {
                oldParts.push_back(vabpnnBlockCheckpointPart(model));
                require(oldParts.size()==16,"preview v4 contains exactly sixteen persistent parts");
                require(loadModelCheckpoint(std::filesystem::u8path(argv[2]).wstring(),oldParts,false,4),
                        "actual full sixteen-part Minecraft v4 model loads");
                const auto savedController=read(b.state);
                const auto savedCounts=read(b.counters);
                const auto savedConfig=read(b.config);
                require(savedCounts.scoredFrames>0 && savedCounts.actorUpdates>0 && savedController.examples>0,
                        "full Minecraft checkpoint contains learned high-level state");
                vabpnnResumeSpikeModel(spike,stream); vabpnnResumeJepaOptimizer(jepa,stream);
                vabpnnResetDecisionState(decision,stream); vabpnnResetOnlineHistory(online,stream);
                check(cudaStreamSynchronize(stream));
                const auto resumed=read(b.state); const auto resumedConfig=read(b.config);
                require(std::memcmp(resumed.weights,savedController.weights,sizeof(resumed.weights))==0
                        && std::memcmp(resumed.first,savedController.first,sizeof(resumed.first))==0
                        && std::memcmp(resumed.second,savedController.second,sizeof(resumed.second))==0
                        && resumed.optimizerSteps==savedController.optimizerSteps
                        && resumed.examples==savedController.examples
                        && resumed.windowExamples==0 && resumed.poorStreak==0
                        && resumedConfig.enabled==savedConfig.enabled && resumedConfig.communication==savedConfig.communication,
                        "full v4 resume retains policy, AdamW and configuration while clearing transient windows");
                vabpnnLearnBlockControllers(f,valid,label); check(cudaStreamSynchronize(stream));
                require(read(b.counters).scoredFrames==savedCounts.scoredFrames,
                        "full v4 resume cannot replay a stale action-label pair");
                demo.modelControl=1; run(); run();
                const auto after=read(b.counters);
                require(after.scoredFrames==savedCounts.scoredFrames && after.actorUpdates==savedCounts.actorUpdates,
                        "cold-restored high-level model retains supervised counters in model-control mode");
                controlsBounded(b);
                std::printf("PASS: actual full v4 cold restore; samples=%llu actorUpdates=%llu rankings=%llu; learned policy/AdamW/config preserved, stale credit cleared.\n",
                        (unsigned long long)after.scoredFrames,(unsigned long long)after.actorUpdates,(unsigned long long)after.rankings);
            }
        }
        std::puts("PASS: neutral/bounded controls, cached action CPU AdamW, duplicate/invalid feedback, true-label rank schedule, ties/bottom-fifth/champion/cooldown, sparse windows, bounded double-buffer communication, actual pixel LR and F8 gating, isolated v4 persistence.");
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); result=1; }
    if(stream) cudaStreamSynchronize(stream);
    vabpnnDestroySpikeModel(spike); vabpnnDestroyOnlineModel(online); vabpnnDestroyJepaModel(jepa);
    vabpnnDestroyDecisionModel(decision); vabpnnDestroyNeuronModel(neuron);
    if(label) cudaFree(label); if(valid) cudaFree(valid); if(rgb) cudaFree(rgb); if(image) cudaFree(image);
    if(stream) cudaStreamDestroy(stream); return result;
}
