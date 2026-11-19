#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "FrameOperator.h"
#include "MemeState.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "NetworkScope.h"
#include "model_checkpoint.h"
#include "action_checkpoint_migration.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
using namespace vabpnn;
void check(cudaError_t c) { if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c)); }
void require(bool ok,const char* msg) {if(!ok) throw std::runtime_error(msg);}
template<class T> T read(const T* p) {T v;check(cudaMemcpy(&v,p,sizeof(v),cudaMemcpyDeviceToHost));return v;}
template<class T> void put(T* p,const T& v,cudaStream_t s) {
    check(cudaMemcpyAsync(p,&v,sizeof(v),cudaMemcpyHostToDevice,s));check(cudaStreamSynchronize(s));
}
struct Fixture {
    cudaStream_t stream{}; VabpnnNeuronModel* neuron{};VabpnnDecisionModel* decision{};
    VabpnnOnlineModel* online{};VabpnnSpikeModel* spike{};VabpnnJepaModel* jepa{};
    unsigned char* image{};float* rgb{};int* valid{};unsigned* label{};
    VabpnnBlockModel* blocks{};VabpnnMemeModel* memes{};MemePersistent* p{};BlockPersistent* bp{};
    VabpnnGpuFrame frame{};DemoSample demo{};int width=7,height=5;
    std::vector<unsigned char> pixels;
    Fixture() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        neuron=vabpnnCreateNeuronModel(stream);decision=vabpnnCreateDecisionModel(stream);
        online=vabpnnCreateOnlineModel(stream);spike=vabpnnCreateSpikeModel(stream);jepa=vabpnnCreateJepaModel(stream);
        blocks=vabpnnSpikeBlockModel(spike);memes=vabpnnBlockMemeModel(blocks);
        p=vabpnnMutableMemePersistent(memes,stream);bp=vabpnnMutableBlockPersistent(blocks,stream);
        check(cudaMalloc(reinterpret_cast<void**>(&image),width*height*4));
        check(cudaMalloc(reinterpret_cast<void**>(&rgb),width*height*3*sizeof(float)));
        check(cudaMalloc(reinterpret_cast<void**>(&valid),sizeof(int)));check(cudaMalloc(reinterpret_cast<void**>(&label),sizeof(unsigned)));
        put(valid,1,stream);put(label,1u,stream);
        pixels.resize(width*height*4);
        for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
            int at=(y*width+x)*4;pixels[at]=33+x*27;pixels[at+1]=45+y*37;pixels[at+2]=31+x*11+y*13;pixels[at+3]=255;
        }
        check(cudaMemcpyAsync(image,pixels.data(),pixels.size(),cudaMemcpyHostToDevice,stream));
        // RGB tensor deliberately contains stale unrelated values: context must use the current RGBA.
        check(cudaMemsetAsync(rgb,0xff,width*height*3*sizeof(float),stream));
        frame={image,static_cast<std::size_t>(width*4),rgb,width,height,1,1'000'000'000ULL,stream,neuron,decision,jepa,online,spike};
        demo.active=1;demo.epoch=1;demo.pointerMode=1;vabpnnSetDemoSample(online,&demo,stream);
        auto bc=BlockConfig{};bc.warmup=0;bc.trainEvery=1;bc.rankEvery=65536;bc.communication=1;
        vabpnnConfigureBlockModel(blocks,bc,stream);sync();
    }
    ~Fixture() {
        if(stream) cudaStreamSynchronize(stream);
        vabpnnDestroySpikeModel(spike);vabpnnDestroyOnlineModel(online);vabpnnDestroyJepaModel(jepa);
        vabpnnDestroyDecisionModel(decision);vabpnnDestroyNeuronModel(neuron);
        if(image) cudaFree(image);if(rgb) cudaFree(rgb);if(valid) cudaFree(valid);if(label) cudaFree(label);
        if(stream) cudaStreamDestroy(stream);
    }
    void sync() {check(cudaStreamSynchronize(stream));}
    DeviceMemeView view() {return vabpnnMemeModelView(memes);}
    void next(double dt=.02) {++frame.frameId;frame.captureNanos+=static_cast<unsigned long long>(dt*1e9);}
    void prepare() {vabpnnPrepareMemes(frame,true);sync();}
    void clear(int mode=1) {
        auto bc=read(&bp->config);bc.communication=mode;vabpnnConfigureBlockModel(blocks,bc,stream);
        check(cudaMemsetAsync(p->prototypes,0,MemeBlocks*sizeof(MemePacket),stream));
        check(cudaMemsetAsync(p->policy,0,MemeBlocks*sizeof(MemePolicy),stream));
        auto sv=vabpnnSpikeModelView(spike);auto c=read(sv.counters);c.longEdges=0;
        put(const_cast<SpikeCounters*>(sv.counters),c,stream);next();prepare();
    }
    MemePacket learned(int tile,float confidence=.9f,float utility=.7f) {
        auto v=view();MemePacket packet;check(cudaMemcpy(packet.context,v.contexts+tile*MemeContext,sizeof(packet.context),cudaMemcpyDeviceToHost));
        for(int k=0;k<MemeChannels;++k) packet.evidence[k]=k==0 ? .8f:-.6f;
        packet.support=64;packet.revision=7;packet.origin=tile;packet.confidence=confidence;
        packet.utility=utility;packet.agreement=1;packet.coverage=1;return packet;
    }
    void inject(int tile,float confidence=.9f,float utility=.7f) {
        auto packet=learned(tile,confidence,utility);packet.sequence=frame.frameId;packet.age=0;
        put(view().packets+tile*2,packet,stream);
    }
    void csr(const std::vector<int>& sources) {
        std::vector<unsigned> rows(NeuronCount+1);std::vector<unsigned> columns;
        for(unsigned pixel=0;pixel<NeuronCount;++pixel) {
            rows[pixel]=static_cast<unsigned>(columns.size());
            if(pixel/1920<16 && pixel%1920<16)
                for(int source:sources) columns.push_back(spikePixel(source,73));
        }
        rows[NeuronCount]=static_cast<unsigned>(columns.size());
        auto v=vabpnnSpikeModelView(spike);
        check(cudaMemcpyAsync(const_cast<unsigned*>(v.longRows),rows.data(),rows.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(const_cast<unsigned*>(v.longColumns),columns.data(),columns.size()*4,cudaMemcpyHostToDevice,stream));
        auto c=read(v.counters);c.longEdges=static_cast<unsigned>(columns.size());put(const_cast<SpikeCounters*>(v.counters),c,stream);
    }
    void feedback(int tile,float advantage=.4f) {
        DecisionOutput out{};out.valid=1;out.frameId=frame.frameId;put(const_cast<DecisionOutput*>(vabpnnDecisionModelView(decision).output),out,stream);
        MemeActorCredit credit;credit.frame=frame.frameId;credit.valid=1;credit.exploring=1;
        credit.advantage=advantage;credit.contribution=.3f;credit.coverage=1;credit.agreement=1;
        credit.noise[0]=.75f;credit.noise[1]=-.5f;credit.noise[2]=1;credit.noise[3]=-.25f;
        put(view().credit+tile,credit,stream);
        vabpnnLearnMemes(frame,valid,label);sync();
    }
    std::vector<ModelPart> parts(bool meme=true) {
        std::vector<ModelPart> result{{vabpnnMutableNeuronWeights(neuron,stream),ParameterCount},
            {vabpnnMutableDecisionParameters(decision,stream),sizeof(DecisionParameters)}};
        ModelPart extra[8];auto append=[&](std::size_t n){result.insert(result.end(),extra,extra+n);};
        append(vabpnnJepaCheckpointParts(jepa,extra));append(vabpnnOnlineCheckpointParts(online,extra));
        append(vabpnnSpikeCheckpointParts(spike,extra));append(vabpnnDecisionPositionCheckpointParts(decision,extra));
        result.push_back(vabpnnBlockCheckpointPart(blocks));if(meme)result.push_back(vabpnnMemeCheckpointPart(memes));return result;
    }
};
void contextAndParticipation(Fixture& f) {
    f.clear();auto v=f.view();
    for(int tile:{0,17,68,119,120,38*120,51*120,32*120+61,7679}) {
        double key[4]{},lumSum=0;
        for(unsigned tid=0;tid<256;++tid) {
            const unsigned pixel=spikePixel(tile,tid);
            // Independent CPU reference includes odd source size and OpenGL bottom origin.
            const int x=static_cast<unsigned long long>(2*(pixel%1920)+1)*f.width/(2*1920);
            const int y=static_cast<unsigned long long>(2*(pixel/1920)+1)*f.height/(2*1024);
            const auto* rgba=f.pixels.data()+((f.height-1-y)*f.width+x)*4;
            double r=rgba[0]/255.,g=rgba[1]/255.,b=rgba[2]/255.,lum=.2126*r+.7152*g+.0722*b;
            key[0]+=r/256;key[1]+=g/256;key[2]+=b/256;key[3]+=lum*lum/256;lumSum+=lum/256;
        }
        key[3]-=lumSum*lumSum;
        for(int d=0;d<4;++d) require(std::abs(read(v.contexts+tile*8+d)-key[d])<2e-6,"odd/scaled source center sampling and bottom origin match neuron context");
    }
    auto weights=vabpnnMutableNeuronWeights(f.neuron,f.stream);
    check(cudaMemsetAsync(weights,0,ParameterCount,f.stream));
    auto* signals=vabpnnMutableSpikeSignals(f.spike,f.stream);
    check(cudaMemsetAsync(signals,0,NeuronCount,f.stream));
    const unsigned nonChampion=spikePixel(0,37);
    put(signals+nonChampion,static_cast<signed char>(1),f.stream);
    put(weights+weightIndex(nonChampion,0,0),TernaryWeight::Positive,f.stream);
    check(cudaMemsetAsync(const_cast<unsigned*>(vabpnnSpikeModelView(f.spike).championIds),0xff,VoteCount*sizeof(unsigned),f.stream));
    vabpnnLaunchTernaryVotes(f.frame,f.neuron);vabpnnObserveMemes(f.frame);f.sync();
    require(read(v.observations).evidence[0]==1/256.f,"ordinary nonchampion neuron contributes exact raw vote/256");
    put(signals+nonChampion,static_cast<signed char>(-1),f.stream);
    f.next();f.prepare();vabpnnLaunchTernaryVotes(f.frame,f.neuron);vabpnnObserveMemes(f.frame);f.sync();
    require(read(v.observations).evidence[0]==-1/256.f,"ordinary neuron sign change changes local evidence without any champion");
    f.feedback(0);require(read(f.p->prototypes).support==1,"real-label ordinary pixel evidence creates learned knowledge");
    auto old=read(&f.p->counters);f.feedback(0);
    require(read(&f.p->counters).supervisedFrames==old.supervisedFrames,"duplicate feedback cannot create support or optimizer step");
    std::puts("PASS: odd/scale context, current RGBA causality, all-neuron nonchampion vote participation and feedback dedup.");
}
void propagation(Fixture& f) {
    f.clear();auto a=f.learned(0);put(f.p->prototypes,a,f.stream);f.next();f.prepare();
    require(read(f.view().packets).origin==0 && read(f.view().packets).support==64,"origin dispatch retains learned support");
    require(read(f.view().packets+1*2+1).support==0,"double buffering prevents same-frame neighbor relay");
    f.next();f.prepare();auto imported=read(f.view().packets+1*2+1);
    require(imported.origin==0 && imported.hops==1 && imported.support==64 && imported.revision==7,"one-hop receiving preserves identity and support");
    require(read(f.view().packets+120*2+1).support==64 && read(f.view().packets+119*2+1).support==0,
        "grid boundary does not wrap left/right");
    auto support=read(f.p->prototypes).support;auto counts=read(&f.p->counters);
    f.demo.modelControl=1;vabpnnSetDemoSample(f.online,&f.demo,f.stream);
    for(int i=0;i<6;++i){f.next(.6);f.prepare();vabpnnObserveMemes(f.frame);f.feedback(0);}
    require(read(f.p->prototypes).support==support && read(&f.p->counters).supervisedFrames==counts.supervisedFrames,
        "F8 does not mint supervision/support but still observes");
    require(read(f.view().packets).support==64 && read(f.view().packets).age==0 && read(f.view().packets).sequence==f.frame.frameId,
        "new matching observations reactivate learned knowledge after old TTL in inference");
    f.demo.modelControl=0;vabpnnSetDemoSample(f.online,&f.demo,f.stream);
    // Remove origin knowledge after dispatch. Imports may circulate but may not rejuvenate.
    put(f.p->prototypes,MemePacket{},f.stream);
    for(int i=0;i<6;++i){f.next(.6);f.prepare();}
    std::vector<MemePacket> packets(MemeBlocks*2);
    check(cudaMemcpy(packets.data(),f.view().packets,packets.size()*sizeof(MemePacket),cudaMemcpyDeviceToHost));
    for(auto p:packets) require(p.support==0,"pure relaying of stopped origin expires rather than refreshing TTL");
    f.clear();auto positive=f.learned(0),negative=f.learned(1);
    for(auto& x:positive.evidence)x=.9f;for(auto& x:negative.evidence)x=-.9f;
    put(f.p->prototypes,positive,f.stream);put(f.p->prototypes+1,negative,f.stream);
    f.next();f.prepare();f.next();f.prepare();
    require(read(f.view().features)>0 && read(f.view().features+MemeChannels)<0,"local anchors preserve distinct knowledge instead of a network-wide average");
    f.clear();f.inject(1);f.next();f.prepare();
    auto held=read(f.view().packets+1);auto replay=held;replay.hops=0;replay.age=0;
    put(f.view().packets+1*2,replay,f.stream);f.next(.4);f.prepare();
    require(std::abs(read(f.view().packets+1).age-(held.age+.4f))<1e-5,"same origin/revision/sequence does not rejuvenate held age on rebroadcast");
    f.clear(0);f.inject(240);f.csr({240});f.next();f.prepare();
    require(read(f.view().routes)==NoNeuron && read(f.view().packets+1).support==0,"independent mode isolates local and long channels");
    f.clear();f.inject(240,.95f);f.inject(360,.8f);f.inject(480,.65f);f.csr({480,240,360,240});
    f.next();f.prepare();
    require(read(f.view().routes)==240 && read(f.view().routes+1)==360,"actual GPU top2 CSR excludes duplicate source tiles across all 256 destination rows");
    require(read(f.view().packets+1).origin==240,"best long-range packet reaches target through actual CSR");
    f.clear();f.inject(1,.99f);f.inject(240,.8f);f.csr({0,1,240});f.next();f.prepare();
    require(read(f.view().routes)==240 && read(f.view().routes+1)==NoNeuron,"self and four-neighbor long edges do not double count");
    f.clear(2);a=f.learned(0,.9f,.5f);put(f.p->prototypes,a,f.stream);
    f.inject(1,.95f,.49f);f.inject(120,.6f,.7f);f.next();f.prepare();
    require(read(f.view().packets+1).origin==120,"competition filters ineligible higher-weight neighbor before argmax");
    f.clear(2);a=f.learned(0,.9f,.5f);put(f.p->prototypes,a,f.stream);
    f.inject(240,.99f,.49f);f.inject(360,.95f,.49f);f.inject(480,.6f,.7f);f.csr({240,360,480});
    f.next();f.prepare();
    require(read(f.view().routes)==480 && read(f.view().routes+1)==NoNeuron && read(f.view().packets+1).origin==480,
        "competition filters long sources before top2 and retains qualified third source");
    f.clear();a=f.learned(0);for(auto& value:a.context)value=1-value;put(f.p->prototypes,a,f.stream);f.next();f.prepare();
    require(read(f.view().packets).support==0,"context mismatch rejects unrelated knowledge");
    f.clear();f.inject(1);f.next();f.prepare();
    auto feature=read(f.view().features);require(std::abs(feature)<=MemeInputScale*.2f,"foreign-only input never renormalizes weak support upward");
    auto before=read(&f.p->counters);vabpnnConfigureMemeEnabled(f.memes,0,f.stream);vabpnnResetBlockHistory(f.blocks,f.stream);f.sync();
    auto after=read(&f.p->counters);
    require(after.localActive==0 && after.importActive==0 && after.dispatches==before.dispatches,"off clears active state without erasing accumulated counts");
    vabpnnConfigureMemeEnabled(f.memes,1,f.stream);f.sync();
    std::puts("PASS: double-buffer transport, grid boundaries, origin/support/hops/TTL, F8 knowledge reactivation, independent isolation, real GPU CSR top2, qualified competition and no weak-import amplification.");
}
void projection(Fixture& f) {
    f.clear();auto learned=f.learned(0);put(f.p->prototypes,learned,f.stream);
    auto config=read(&f.bp->config);config.communication=0;config.warmup=0;config.trainEvery=1;
    vabpnnConfigureBlockModel(f.blocks,config,f.stream);
    DecisionOutput out{};out.valid=1;for(auto& probability:out.posterior)probability=.5f;
    put(const_cast<DecisionOutput*>(vabpnnDecisionModelView(f.decision).output),out,f.stream);
    // Rebind learned key to the same old model context used by current control.
    f.next();f.prepare();learned=f.learned(0);put(f.p->prototypes,learned,f.stream);
    vabpnnResetBlockHistory(f.blocks,f.stream);f.next();vabpnnPrepareBlockControllers(f.frame);f.sync();
    auto action=read(f.view().actions);require(action.active && action.feature[0]!=0,"zero projection receives nonzero actual cached feature");
    for(int k=0;k<VoteCount;++k) put(const_cast<int*>(vabpnnNeuronModelView(f.neuron).blockVotes)+k*BlockCount,k==0 ? 256:-256,f.stream);
    vabpnnObserveMemes(f.frame);
    out.frameId=f.frame.frameId;put(const_cast<DecisionOutput*>(vabpnnDecisionModelView(f.decision).output),out,f.stream);
    f.next();vabpnnLearnBlockControllers(f.frame,f.valid,f.label);f.sync();
    const auto credit=read(f.view().credit);const auto policy=read(f.p->policy);const auto bc=read(&f.bp->config);
    require(credit.valid && credit.exploring && policy.optimizerSteps==1,"projection has own actual update step rather than global/base count");
    for(int at=0;at<MemeWeights;++at) {
        double gradient=std::clamp(-double(credit.advantage)*credit.noise[at/MemeChannels]*action.feature[at%MemeChannels]/credit.exploration,-1.,1.);
        double first=.1*gradient,second=.001*gradient*gradient;
        // Float implementation's bias corrections use float pow; use these exact single-step coefficients.
        double expected=-bc.learningRate*(first/(1-std::pow(.9f,1.f)))/(std::sqrt(second/(1-std::pow(.999f,1.f)))+1e-8);
        require(std::abs(policy.first[at]-first)<2e-7 && std::abs(policy.second[at]-second)<2e-7
            && std::abs(policy.weights[at]-expected)<2e-6,"independent CPU old Gaussian score-function / AdamW reference");
    }
    require(policy.weights[0]!=0,"zero-initialized projection learns instead of dead zero derivative");
    // During five-frame held action, a changed observation must not overwrite its cached input.
    auto original=read(f.view().actions);auto flipped=read(f.p->prototypes);
    for(auto& value:flipped.evidence)value=-value;put(f.p->prototypes,flipped,f.stream);
    for(int i=0;i<3;++i){f.next();vabpnnPrepareBlockControllers(f.frame);}
    f.sync();require(std::memcmp(original.feature,read(f.view().actions).feature,sizeof(original.feature))==0,
        "held action keeps feature actually used rather than future neighbor context");
    f.next();f.prepare();require(read(f.view().features)<0,"new projected candidate differs from held positive feature");
    vabpnnPrepareBlockControllers(f.frame);f.sync();
    require(read(f.view().actions).feature[0]==original.feature[0],"even a new imported candidate cannot overwrite held action credit");
    vabpnnResetMemeHistory(f.memes,f.stream);f.sync();
    require(read(f.view().actions).batch==0 && read(f.p->policy).optimizerSteps==1,"reset clears partial gradients but preserves learned optimizer");
    // Fixed input scaling applied exactly once before the Gaussian/tanh.
    f.clear();check(cudaMemsetAsync(f.bp->state,0,BlockCount*sizeof(BlockState),f.stream));
    config=read(&f.bp->config);config.warmup=1000000;config.communication=0;vabpnnConfigureBlockModel(f.blocks,config,f.stream);
    f.next();f.prepare();learned=f.learned(0);put(f.p->prototypes,learned,f.stream);
    auto one=MemePolicy{};one.weights[0]=1;put(f.p->policy,one,f.stream);
    vabpnnResetBlockHistory(f.blocks,f.stream);f.next();vabpnnPrepareBlockControllers(f.frame);f.sync();
    const float input=read(f.view().actions).feature[0];const auto control=read(vabpnnBlockModelView(f.blocks).controls);
    require(std::abs(control.thresholdScale-(1+.2f*std::tanh(input)))<1e-6,"learned meme projection actually changes bounded control with exactly-once scaling");
    vabpnnConfigureMemeEnabled(f.memes,0,f.stream);vabpnnResetBlockHistory(f.blocks,f.stream);f.next();vabpnnPrepareBlockControllers(f.frame);f.sync();
    auto disabled=read(vabpnnBlockModelView(f.blocks).controls);
    require(disabled.thresholdScale==1 && disabled.polarityBias==0 && disabled.memoryScale==1 && disabled.learningScale==1,"disabled branch returns original neutral control exactly");
    vabpnnConfigureMemeEnabled(f.memes,1,f.stream);
    std::puts("PASS: actual cached old Gaussian credit, independent CPU AdamW, non-dead zero initialization, held features, reset and bounded/disabled original-control equivalence.");
}
__global__ void jointRankFixture(BlockPersistent* p,DeviceMemeView memes) {
    int tile=blockIdx.x*blockDim.x+threadIdx.x;if(tile>=BlockCount)return;
    auto& state=p->state[tile];state=BlockState{};state.bestScore=-1e20f;
    float score=tile<BlockCount/5 ? -.8f:.2f;
    state.examples=64;state.windowExamples=64;state.windowCoverage=64;
    state.windowSum=64*score;state.windowSquares=64*score*score;state.previousRankScore=score+.03f;
    auto& policy=memes.persistent->policy[tile];
    for(int at=0;at<MemeWeights;++at) {
        policy.weights[at]=.27f;policy.first[at]=.03f;policy.second[at]=.004f;
        memes.actions[tile].gradient[at]=.2f;
    }
    policy.optimizerSteps=42;memes.actions[tile].batch=3;memes.actions[tile].active=1;
}
void jointEvolution(Fixture& f) {
    auto config=read(&f.bp->config);config.communication=0;config.rankEvery=65536;
    config.minimumExamples=64;config.minimumWindow=32;config.poorWindows=1;
    vabpnnConfigureBlockModel(f.blocks,config,f.stream);f.next();vabpnnPrepareBlockControllers(f.frame);
    vabpnnObserveMemes(f.frame);DecisionOutput out{};out.valid=1;out.frameId=f.frame.frameId;
    for(auto& probability:out.posterior)probability=.5f;
    put(const_cast<DecisionOutput*>(vabpnnDecisionModelView(f.decision).output),out,f.stream);
    f.next();vabpnnLearnBlockControllers(f.frame,f.valid,f.label);f.sync();
    config.rankEvery=1;put(&f.bp->config,config,f.stream);
    jointRankFixture<<<30,256,0,f.stream>>>(f.bp,f.view());
    auto before=read(&f.p->counters);vabpnnRankBlockControllers(f.blocks,f.stream);f.sync();
    const auto ranked=read(vabpnnBlockModelView(f.blocks).counters);
    const auto changed=read(f.p->policy);const auto action=read(f.view().actions);
    require(ranked.punished==BlockCount/5 && changed.optimizerSteps==0 && action.batch==0 && !action.active,
        "base mutation synchronously resets corresponding meme optimizer/partial action");
    for(int at=0;at<MemeWeights;++at)
        require(changed.bestWeights[at]==.27f && std::abs(changed.weights[at]-.9f*.27f)<=.010001f
            && changed.first[at]==0 && changed.second[at]==0 && action.gradient[at]==0,
            "best/evolve copies matching meme branch and clears its AdamW moments");
    require(read(&f.p->counters).projectionUpdates==before.projectionUpdates,"mutation is not counted as gradient supervision");
    std::puts("PASS: joint best snapshot, actual bottom-fifth evolution and branch AdamW/gradient reset.");
}
void persistence(Fixture& f,const char* legacy) {
    static_assert(sizeof(BlockPersistent)==8663184,"Old v4 block bytes unchanged");
    require(f.parts(false).size()==16 && f.parts().size()==17,"legacy parts remain byte-identical plus one appended meme part");
    if(legacy) {
        const auto original=std::filesystem::u8path(legacy).wstring();
        require(loadLegacyActionCheckpoint(original,f.parts(),4),"real pre-v1 file is loaded read-only");
        const auto saved=read(f.bp->state);require(saved.examples>0,"actual legacy learned block state loaded");
        // Mimic fresh constructor state for the newly appended part during migration.
        check(cudaMemsetAsync(f.p->prototypes,0,MemeBlocks*sizeof(MemePacket),f.stream));
        check(cudaMemsetAsync(f.p->policy,0,MemeBlocks*sizeof(MemePolicy),f.stream));f.sync();
        vabpnnResumeSpikeModel(f.spike,f.stream);f.sync();
        require(read(f.p->policy).weights[0]==0 && read(f.bp->state).optimizerSteps==saved.optimizerSteps,
            "read-only v4 migration keeps old policy/optimizer and neutral new projection");
        std::puts("PASS: actual learned sixteen-part pre-v1 migration; old controller policy and AdamW preserved.");
    }
    auto config=MemeConfig{};vabpnnConfigureMemeModel(f.memes,config,f.stream);f.next();f.prepare();
    auto packet=f.learned(3);packet.support=1234;put(f.p->prototypes+3,packet,f.stream);
    MemePolicy policy;policy.weights[0]=.27f;policy.first[0]=.03f;policy.second[0]=.004f;policy.bestWeights[0]=.21f;policy.optimizerSteps=42;
    put(f.p->policy+3,policy,f.stream);f.next();f.prepare();
    auto path=std::filesystem::absolute("meme-test/full-v6.bin").wstring();
    saveModelCheckpoint(path,f.parts(),6);
    CheckpointHeaderV5 header;std::ifstream file(std::filesystem::path(path),std::ios::binary);file.read(reinterpret_cast<char*>(&header),sizeof(header));file.close();
    require(header.version==6 && header.count==17 && header.sizes[16]==sizeof(MemePersistent)
        && std::filesystem::file_size(path)==256+header.bytes,"actual full v6 header and appended part size");
    put(f.p->policy+3,MemePolicy{},f.stream);put(f.p->prototypes+3,MemePacket{},f.stream);
    require(loadModelCheckpoint(path,f.parts(),false,6),"full v6 restores actual learned state");
    vabpnnResumeSpikeModel(f.spike,f.stream);f.sync();
    const auto restored=read(f.p->policy+3);
    require(std::memcmp(&policy,&restored,sizeof(policy))==0,"cold restore preserves own moments/best weights/step");
    require(read(f.p->prototypes+3).support==1234 && read(f.view().packets+3*2).support==0
        && read(f.view().actions+3).batch==0 && read(&f.p->counters).localActive==0,"knowledge restored, imported dispatch/pending feedback not persisted");
    std::fstream corrupt(std::filesystem::path(path),std::ios::binary|std::ios::in|std::ios::out);
    corrupt.seekg(-1,std::ios::end);char value;corrupt.read(&value,1);value^=1;
    corrupt.seekp(-1,std::ios::end);corrupt.write(&value,1);corrupt.close();
    auto sentinel=read(f.p->policy+3);sentinel.weights[0]=.543f;put(f.p->policy+3,sentinel,f.stream);
    auto* firstWeight=vabpnnMutableNeuronWeights(f.neuron,f.stream);
    auto savedWeight=read(firstWeight);auto weightSentinel=savedWeight==TernaryWeight::Negative ? TernaryWeight::Positive:TernaryWeight::Negative;
    put(firstWeight,weightSentinel,f.stream);
    bool rejected=false;try{loadModelCheckpoint(path,f.parts(),false,5);}catch(const std::runtime_error&){rejected=true;}
    require(rejected && read(f.p->policy+3).weights[0]==.543f && read(firstWeight)==weightSentinel,"whole-body SHA rejection precedes any GPU overwrite");
    std::puts("PASS: actual full seventeen-part v6 roundtrip, cold restore preserves knowledge/AdamW/best, clears transients; corrupt checkpoint rejected before GPU writes.");
}
int main(int argc,char** argv) {
    try {Fixture f;contextAndParticipation(f);propagation(f);projection(f);jointEvolution(f);persistence(f,argc>1 ? argv[1]:nullptr);return 0;}
    catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}

