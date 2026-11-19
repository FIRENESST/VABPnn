#include "FrameOperator.h"
#include "PnnJepa.h"
#include "FullSelfLearning.h"
#include "full_learning_reference.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace vabpnn;
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }

template<class T> T read(const T* pointer) {
    T value{};
    check(cudaMemcpy(&value, pointer, sizeof(value), cudaMemcpyDeviceToHost));
    return value;
}
template<class T> void write(T* pointer, const T& value) {
    check(cudaMemcpy(pointer, &value, sizeof(value), cudaMemcpyHostToDevice));
}
template<class T> T* deviceArray(std::size_t elements) {
    T* pointer=nullptr;
    check(cudaMalloc(&pointer,elements*sizeof(T)));
    return pointer;
}

__global__ void paint(unsigned char* rgba, int frame) {
    const int at=blockIdx.x*blockDim.x+threadIdx.x;
    if (at>=NeuronCount) return;
    const int x=at%NeuronWidth, y=at/NeuronWidth;
    rgba[4*at]=static_cast<unsigned char>(128+(x/64+y/32*3+frame*5)%128);
    rgba[4*at+1]=static_cast<unsigned char>(100+(x/32*7+y/64+frame*3)%128);
    rgba[4*at+2]=static_cast<unsigned char>(110+(x/16*13+y/16*7)%128);
    rgba[4*at+3]=255;
}
__global__ void seedPnn(TernaryWeight* weights, float* master) {
    const int at=blockIdx.x*blockDim.x+threadIdx.x;
    if (at>=NeuronCount) return;
    const int tileX=at%NeuronWidth/16, tileY=at/NeuronWidth/16;
    for (int a=0;a<VoteCount;++a)
        for (int c=0;c<3;++c) {
            const int vote=(tileX*(a+1)+3*tileY+c+a)%3-1;
            const auto position=weightIndex(at,c,a);
            master[position]=.051f*vote;
            weights[position]=static_cast<TernaryWeight>(vote);
        }
}
__global__ void fingerprint(const unsigned int* data,std::size_t words,unsigned long long* output) {
    unsigned long long hash=0;
    for (std::size_t at=blockIdx.x*blockDim.x+threadIdx.x; at<words; at+=gridDim.x*blockDim.x) {
        auto x=static_cast<unsigned long long>(data[at])^(at*0x9e3779b97f4a7c15ull);
        x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;
        hash^=(x^(x>>27))*0x94d049bb133111ebull;
    }
    atomicXor(output,hash);
}
unsigned long long hash(const void* data,std::size_t bytes) {
    auto* output=deviceArray<unsigned long long>(1);
    check(cudaMemset(output,0,sizeof(*output)));
    fingerprint<<<192,256>>>(static_cast<const unsigned int*>(data),bytes/4,output);
    const auto result=read(output);
    check(cudaFree(output));
    return result;
}

void verifyGradient(const VabpnnGpuFrame& frame) {
    const auto decision=vabpnnDecisionModelView(frame.decisionModel);
    const auto output=read(decision.output);
    auto state=read(decision.learningState);
    FullRewardRoots roots{};
    for (int a=0;a<VoteCount;++a)
        for (int d=0;d<DecisionWidth;++d)
            roots.hidden[a][d]=.03f*std::sin(.7f*static_cast<float>(a*DecisionWidth+d));
    auto* rootsDevice=deviceArray<FullRewardRoots>(1);
    auto* gradient=deviceArray<float>(FullDecisionParameters);
    auto* spatial=deviceArray<float>(VoteCount*DecisionWidth);
    void* scratch=nullptr;
    check(cudaMalloc(&scratch,vabpnnSecondsScratchBytes()));
    write(rootsDevice,roots);
    vabpnnSecondsRewardScore(frame.decisionModel,rootsDevice,gradient,nullptr,scratch,nullptr,spatial);
    const auto parameters=read(decision.parameters);
    require(state.latentDecoder && state.count==DecisionHistory,"causal JEPA context is the decoder memory");
    float maximum=0, largest=0;
    for (int region=0;region<VoteCount;region+=2)
        for (int d=0;d<DecisionWidth;d+=3) {
            float& value=state.historyLatents[state.head][region][d];
            const float before=value;
            value=before+.001f;
            const auto positive=full_reference::secondsObjective(parameters,state,roots,output);
            const float plus=value;
            value=before-.001f;
            const auto negative=full_reference::secondsObjective(parameters,state,roots,output);
            const float numerical=static_cast<float>((positive-negative)/(plus-value));
            value=before;
            const float analytic=read(spatial+region*DecisionWidth+d);
            require(std::isfinite(analytic),"nonfinite spatial cross-attention gradient");
            maximum=std::max(maximum,std::abs(analytic-numerical));
            largest=std::max(largest,std::abs(numerical));
        }
    require(largest>.00001f && maximum<.00005f,"decoder must depend on individual JEPA spatial tokens with matching gradients");
    std::printf("PASS: decoder gradients on 48 individual JEPA spatial features match FP64 finite differences; max error %.4g.\n",maximum);
    check(cudaFree(scratch)); check(cudaFree(spatial)); check(cudaFree(gradient)); check(cudaFree(rootsDevice));
}

void verifyObservation(const VabpnnGpuFrame& frame) {
    auto state=read(vabpnnDecisionModelView(frame.decisionModel).learningState);
    const auto encoded=read(vabpnnJepaDecoderInput(frame.jepaModel));
    require(state.latentDecoder && state.historyVotes[state.head][2]==encoded.votes[2],
        "action scalars originate from JEPA");
    require(std::memcmp(state.historyLatents[state.head],encoded.spatial,sizeof(encoded.spatial))==0,
        "cross-attention uses the current JEPA spatial tensor");
    require(std::abs(encoded.spatial[0][0]-encoded.spatial[4][0])>1e-5f,"JEPA retained spatial information");
    const auto votes=vabpnnNeuronModelView(frame.neuronModel);
    const auto* patches=vabpnnJepaModelView(frame.jepaModel).currentPatchFeatures;
    int sum=0;
    for (int x=0;x<JepaCells;++x) sum+=read(votes.blockVotes+x);
    const float expected=static_cast<float>(sum)/(JepaCells*256);
    require(std::abs(read(patches)-expected)<1e-7f,"JEPA input is constructed from signed PNN block votes");
}
}

int main() try {
    using namespace vabpnn;
    VabpnnGpuFrame frame{};
    frame.width=NeuronWidth; frame.height=NeuronHeight; frame.rgbaPitch=NeuronWidth*4;
    frame.pnnJepa=true; frame.selfLearning=true;
    auto* rgba=deviceArray<unsigned char>(NeuronCount*4);
    frame.rgba=rgba; frame.rgbNchw=deviceArray<float>(NeuronCount*3);
    frame.neuronModel=vabpnnCreateNeuronModel(nullptr);
    frame.spikeModel=vabpnnCreateSpikeModel(nullptr);
    frame.jepaModel=vabpnnCreateJepaModel(nullptr);
    frame.decisionModel=vabpnnCreateDecisionModel(nullptr);
    frame.onlineModel=vabpnnCreateOnlineModel(nullptr);
    frame.longMemory=vabpnnCreateLongMemory(nullptr);
    frame.fullLearning=vabpnnCreateFullLearning(nullptr);
    auto* policy=vabpnnCreateRewardPolicy(nullptr);
    vabpnnEnableFullLearning(frame.fullLearning,nullptr,true);
    vabpnnConfigureRewardPolicy(policy,19,true,nullptr);
    auto* weights=vabpnnMutableNeuronWeights(frame.neuronModel,nullptr);
    auto* master=vabpnnOnlineMasterWeights(frame.onlineModel);
    seedPnn<<<(NeuronCount+255)/256,256>>>(weights,master);
    auto* parameters=vabpnnMutableDecisionParameters(frame.decisionModel,nullptr);
    auto action=read(parameters);
    for (int a=0;a<VoteCount;++a) {
        action.thresholdBias[a]=-2;
        action.rawVoteWeight[a]=.25f;
        for (int d=0;d<DecisionWidth;++d)
            action.logLikelihoodWeight[a][d]=.07f*std::sin(static_cast<float>(a*7+d+1));
    }
    for (int k=0;k<DecisionKChoices;++k) {
        action.kBias[k]=k==3 ? 4.f : 0.f;
        for (int d=0;d<DecisionKFeatures;++d)
            action.kWeight[k][d]=.04f*std::cos(static_cast<float>(k*17+d+1));
    }
    write(parameters,action);
    DemoSample demo{};
    demo.active=1; demo.modelControl=1; demo.pointerMode=1; demo.epoch=1;
    vabpnnSetDemoSample(frame.onlineModel,&demo,nullptr);
    const auto d=vabpnnDecisionModelView(frame.decisionModel);
    const auto j=vabpnnJepaModelView(frame.jepaModel);
    auto full=vabpnnFullLearningView(frame.fullLearning);
    auto reward=vabpnnRewardPolicyView(policy);
    frame.longMemoryContext={11,22,33,44,0,1795000000000ull};
    auto next=[&] {
        ++frame.frameId;
        frame.captureNanos=1'000'000'000ull+33'333'333ull*frame.frameId;
        frame.longMemoryContext.sampleNanos=frame.captureNanos;
        frame.longMemoryContext.wallMillis+=33;
        frame.position={frame.captureNanos,1,2,64,3+.02*frame.frameId};
        paint<<<(NeuronCount+255)/256,256>>>(rgba,static_cast<int>(frame.frameId));
    };
    RewardBatch batch{};
    batch.epoch=19; batch.rewardId=1; batch.credit=.75f;
    int visual=0, actions=0;
    cudaEvent_t start=nullptr,finish=nullptr;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&finish));
    std::vector<float> frameTimes;
    for (int index=0;index<40;++index) {
        next();
        check(cudaEventRecord(start));
        vabpnnLaunchOperator(frame);
        vabpnnProposeRewardAction(policy,d.parameters,d.constraints,const_cast<DecisionOutput*>(d.output),
            j.output,vabpnnMutableMotionProposal(frame.decisionModel,nullptr),nullptr);
        const auto slot=frame.frameId%RewardCacheSlots;
        vabpnnCacheFullReward(frame,frame.fullLearning,reward.cache+slot);
        check(cudaEventRecord(finish)); check(cudaEventSynchronize(finish));
        float elapsed=0; check(cudaEventElapsedTime(&elapsed,start,finish));
        if (index>=12) frameTimes.push_back(elapsed);
        const auto observation=read(full.cache+slot);
        require(observation.proposal.valid && observation.proposal.frameId==frame.frameId,"valid action proposal from PNN/JEPA");
        require((observation.proposal.mask&~observation.proposal.eligibleMask)==0,"threshold mode selects only eligible actions");
        for (int a=0;a<VoteCount;++a)
            if (observation.proposal.mask&(1u<<a))
                require((observation.proposal.mask&read(d.constraints).conflictMask[a])==0,"action conflicts retained");
        if (index>0) {
            batch.frames[batch.count]=frame.frameId;
            batch.masks[batch.count++]=observation.proposal.mask;
        }
        actions+=observation.proposal.mask!=0;
        visual+=observation.visual && observation.pnnVisualSupport==3;
    }
    require(visual>0 && actions>0,"real spatial pulse snapshots and execution candidates");
    verifyObservation(frame);
    verifyGradient(frame);
    const auto beforeReward=read(parameters);
    const auto beforePnn=hash(master,ParameterCount*sizeof(float));
    check(cudaEventRecord(start));
    vabpnnApplyRewardBatch(policy,parameters,batch,nullptr,&frame);
    check(cudaEventRecord(finish)); check(cudaEventSynchronize(finish));
    float rewardMs=0; check(cudaEventElapsedTime(&rewardMs,start,finish));
    const auto afterReward=read(parameters);
    require(std::memcmp(beforeReward.query,afterReward.query,sizeof(afterReward.query))!=0
        && std::memcmp(beforeReward.value,afterReward.value,sizeof(afterReward.value))!=0,
        "real reward trains decoder attention");
    require(hash(master,ParameterCount*sizeof(float))!=beforePnn,"reward trains the alpha PNN spatial encoder");
    require(read(&full.persistent->pixelSteps)==1 && read(&full.persistent->secondsSteps)==1,
        "one update reached decoder and pixel branch");
    const auto ssl=read(j.output).selfSupervisedSteps;
    require(ssl>0 && read(vabpnnOnlineOutput(frame.onlineModel)).examples==0,
        "JEPA temporal supervision learns while model executions are never human labels");
    const auto policyAfter=read(reward.persistent);
    vabpnnApplyRewardBatch(policy,parameters,batch,nullptr,&frame);
    require(read(reward.persistent).policyVersion==policyAfter.policyVersion,"duplicate receipt cannot update Alpha");
    std::puts("PASS: PNN spatial features -> JEPA SSL -> 15 atomic permissions, strict constraints, real GPU decoder/PNN reward steps, duplicate protection.");

    vabpnnConfigureRewardPolicy(policy,20,false,nullptr);
    frame.selfLearning=false;
    demo.modelControl=0; demo.epoch=2; demo.heldKeys=actionBit(Action::W);
    vabpnnSetDemoSample(frame.onlineModel,&demo,nullptr);
    const auto beforeDemo=read(parameters);
    for (int i=0;i<8;++i) { next(); vabpnnLaunchOperator(frame); }
    const auto afterDemo=read(parameters);
    require(read(vabpnnOnlineOutput(frame.onlineModel)).examples>0
        && std::memcmp(beforeDemo.feedForwardUp,afterDemo.feedForwardUp,sizeof(afterDemo.feedForwardUp))!=0,
        "human demonstrations train the decoder body");
    const auto memory=vabpnnLongMemoryView(frame.longMemory);
    const auto frozenMemory=hash(memory.entries,LongMemoryCapacity*sizeof(LongMemoryEntry))^hash(memory.catalog,sizeof(LongMemoryCatalog));
    const auto frozenJepa=hash(j.parameters,sizeof(JepaParameters))^hash(j.targetEncoder,sizeof(JepaEncoder));
    const auto frozenDecoder=hash(parameters,sizeof(DecisionParameters));
    frame.evaluationMode=true;
    demo.modelControl=1; demo.epoch=3;
    vabpnnSetDemoSample(frame.onlineModel,&demo,nullptr);
    for (int i=0;i<8;++i) { next(); vabpnnLaunchOperator(frame); }
    require(hash(parameters,sizeof(DecisionParameters))==frozenDecoder
        && (hash(j.parameters,sizeof(JepaParameters))^hash(j.targetEncoder,sizeof(JepaEncoder)))==frozenJepa,
        "F8 freezes the decoder, encoder/predictor, and EMA");
    require((hash(memory.entries,LongMemoryCapacity*sizeof(LongMemoryEntry))^hash(memory.catalog,sizeof(LongMemoryCatalog)))==frozenMemory,
        "F8 keeps memory store immutable");
    vabpnnResetDecisionState(frame.decisionModel,nullptr);
    vabpnnResetJepaHistory(frame.jepaModel,nullptr);
    next(); vabpnnLaunchOperator(frame);
    require(read(d.learningState).count==1 && !read(j.output).pairReady,"fresh life does not use stale latents or JEPA pairs");
    std::puts("PASS: demo learning, F8 neural/memory freeze and new-life history reset.");

    std::sort(frameTimes.begin(),frameTimes.end());
    std::printf("ALPHA_TIMING frames=%zu mean=%.3f p95=%.3f max=%.3f reward_batch=%.3f ms.\n",
        frameTimes.size(),std::accumulate(frameTimes.begin(),frameTimes.end(),0.f)/frameTimes.size(),
        frameTimes[static_cast<std::size_t>(.95*(frameTimes.size()-1))],frameTimes.back(),rewardMs);
    check(cudaEventDestroy(start)); check(cudaEventDestroy(finish));
    vabpnnDestroyRewardPolicy(policy); vabpnnDestroyFullLearning(frame.fullLearning);
    vabpnnDestroyLongMemory(frame.longMemory); vabpnnDestroyOnlineModel(frame.onlineModel);
    vabpnnDestroyDecisionModel(frame.decisionModel); vabpnnDestroyJepaModel(frame.jepaModel);
    vabpnnDestroySpikeModel(frame.spikeModel); vabpnnDestroyNeuronModel(frame.neuronModel);
    check(cudaFree(frame.rgbNchw)); check(cudaFree(rgba));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr,"FAIL: %s\n",error.what());
    return 1;
}
