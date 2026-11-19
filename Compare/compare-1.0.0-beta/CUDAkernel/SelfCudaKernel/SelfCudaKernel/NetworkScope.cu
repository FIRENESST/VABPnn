#include "NetworkScope.h"
#include "JepaMotion.h"
#include "MemeState.h"
#include <stdexcept>
#include <string>
namespace {
using namespace vabpnn;
__global__ void scopeBlocks(DeviceSpikeView view,DeviceBlockView blocks,DeviceMemeView memes,ScopeTelemetry* output,int selectedTile) {
    const int tile=blockIdx.x,tid=threadIdx.x;
    const unsigned pixel=spikePixel(tile,tid);
    const float membrane=view.membrane[pixel];
    const int spike=view.spikes[pixel];
    __shared__ float mem[256],positive[256],negative[256];
    mem[tid]=membrane; positive[tid]=spike>0; negative[tid]=spike<0;
    if(tile==selectedTile) {
        output->values[ScopeLocalMem+tid]=membrane;
        output->discrete[ScopeLocalSpike+tid]=spike;
        for(int slot=0;slot<4;++slot)
            output->discrete[ScopeOutgoing+tid*4+slot]=view.localOutgoing[(tile*256+tid)*4+slot];
    }
    __syncthreads();
    for(int stride=128;stride;stride/=2) {
        if(tid<stride) { mem[tid]+=mem[tid+stride]; positive[tid]+=positive[tid+stride]; negative[tid]+=negative[tid+stride]; }
        __syncthreads();
    }
    if(tid==0) {
        output->values[ScopeMapMem+tile]=mem[0]/256;
        output->values[ScopeBlockScore+tile]=blocks.state[tile].score;
        const auto* key=memes.contexts+tile*MemeContext;
        output->values[ScopeMemeMap+tile]=fmaxf(
            memePacketWeight(memes.packets[tile*2],key,memes.persistent->config),
            memePacketWeight(memes.packets[tile*2+1],key,memes.persistent->config));
        output->values[ScopeMapPositive+tile]=positive[0]/256;
        output->values[ScopeMapNegative+tile]=negative[0]/256;
    }
}
__global__ void scopeEdges(DeviceSpikeView view,ScopeTelemetry* output) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<SpikeAnchorLimit) output->discrete[ScopeAnchors+i]=static_cast<int>(view.anchorIds[i]);
    if(i>=ScopeEdges) return;
    const auto count=view.counters->longEdges;
    if(!count) { output->discrete[ScopeEdgePairs+2*i]=-1; output->discrete[ScopeEdgePairs+2*i+1]=-1; return; }
    const unsigned edge=static_cast<unsigned long long>(i)*count/ScopeEdges;
    // Upper bound in destination CSR: empty rows must never be mistaken for the destination.
    unsigned lo=0,hi=static_cast<unsigned>(NeuronCount)+1;
    while(lo<hi) {
        const unsigned mid=lo+(hi-lo)/2;
        if(view.longRows[mid]<=edge) lo=mid+1; else hi=mid;
    }
    const unsigned destination=lo-1,encoded=view.longColumns[edge];
    output->discrete[ScopeEdgePairs+2*i]=encoded&SpikeSourceMask;
    output->discrete[ScopeEdgePairs+2*i+1]=destination|(encoded&~SpikeSourceMask);
}
__global__ void scopeHeads(std::uint64_t id,std::uint64_t nanos,DeviceSpikeView spike,DeviceBlockView blocks,
    DeviceMemeView memes,const DecisionOutput* decision,const std::int32_t* votes,const JepaOutput* jepa,
    ScopeTelemetry* output,int tile) {
    output->frameId=id; output->captureNanos=nanos;
    const auto& state=blocks.state[tile]; const auto& control=blocks.controls[tile];
    const float selected[8]={state.contribution,state.agreement,state.coverage,control.thresholdScale,
        control.polarityBias,control.memoryScale,control.learningScale,state.progress};
    for(int i=0;i<8;++i) output->values[ScopeBlockSelected+i]=selected[i];
    const int blockData[8]={blocks.config->enabled,blocks.config->communication,
        static_cast<int>(state.examples>2147483647 ? 2147483647:state.examples),
        static_cast<int>(state.optimizerSteps>2147483647 ? 2147483647:state.optimizerSteps),
        static_cast<int>(state.poorStreak),static_cast<int>(state.mutations>2147483647 ? 2147483647:state.mutations),
        static_cast<int>(state.protectedAction),static_cast<int>(state.windowExamples>2147483647 ? 2147483647:state.windowExamples)};
    for(int i=0;i<8;++i) output->discrete[ScopeBlockData+i]=blockData[i];
    const auto& knowledge=memes.persistent->prototypes[tile];
    const auto& own=memes.packets[tile*2];const auto& imported=memes.packets[tile*2+1];
    const auto& counts=memes.persistent->counters;
    const float memeData[8]={own.confidence,imported.confidence,imported.age,knowledge.utility,
        memeCompatibility(knowledge.context,memes.contexts+tile*MemeContext,memes.persistent->config),
        knowledge.coverage,knowledge.age,imported.utility};
    for(int i=0;i<8;++i) output->values[ScopeMemeSelected+i]=memeData[i];
    const int memeInts[8]={static_cast<int>(counts.enabled),imported.support ? static_cast<int>(imported.origin):-1,
        static_cast<int>(imported.hops),static_cast<int>(knowledge.support>2147483647 ? 2147483647:knowledge.support),
        static_cast<int>(memes.persistent->policy[tile].optimizerSteps>2147483647 ? 2147483647:memes.persistent->policy[tile].optimizerSteps),
        static_cast<int>(counts.localActive),static_cast<int>(counts.importActive),
        static_cast<int>(knowledge.revision>2147483647 ? 2147483647:knowledge.revision)};
    for(int i=0;i<8;++i) output->discrete[ScopeMemeData+i]=memeInts[i];
    for(int k=0;k<VoteCount;++k)
        output->values[ScopeMemeEvidence+k]=memes.actions[tile].feature[k]/MemeInputScale;
    const int data[8]={decision->valid,tile,decision->requestedK,decision->selectedCount,
        static_cast<int>(decision->eligibleMask),static_cast<int>(decision->selectedMask),
        spike.counters->longEdges ? ScopeEdges:0,static_cast<int>(spike.counters->invalidConfig)};
    for(int i=0;i<8;++i) output->discrete[i]=data[i];
    const float metrics[8]={spike.config->threshold,decision->deltaSeconds,jepa->jepaLoss,jepa->motionLoss,
        jepa->rateX,jepa->rateY,decision->command.wheelSteps,jepa->gradientNorm};
    for(int i=0;i<8;++i) output->values[i]=metrics[i];
    for(int k=0;k<VoteCount;++k) {
        output->values[ScopePosterior+k]=decision->posterior[k];
        output->values[ScopeThreshold+k]=decision->threshold[k];
        output->values[ScopePrior+k]=decision->prior[k];
        output->discrete[ScopeVotes+k]=votes[k];
    }
}
}
void vabpnnLaunchNetworkScope(const VabpnnGpuFrame& frame,int tile,vabpnn::ScopeTelemetry* destination) {
    if(!destination || !frame.spikeModel || !frame.decisionModel || !frame.neuronModel || !frame.jepaModel
        || tile<0 || tile>=vabpnn::SpikeTiles) throw std::invalid_argument("Invalid network scope model/tile");
    const auto view=vabpnnSpikeModelView(frame.spikeModel);
    const auto blocks=vabpnnBlockModelView(vabpnnSpikeBlockModel(frame.spikeModel));
    const auto memes=vabpnnMemeModelView(vabpnnBlockMemeModel(vabpnnSpikeBlockModel(frame.spikeModel)));
    scopeBlocks<<<vabpnn::SpikeTiles,256,0,frame.stream>>>(view,blocks,memes,destination,tile);
    scopeEdges<<<2,256,0,frame.stream>>>(view,destination);
    scopeHeads<<<1,1,0,frame.stream>>>(frame.frameId,frame.captureNanos,view,blocks,memes,
        vabpnnDecisionModelView(frame.decisionModel).output,vabpnnNeuronModelView(frame.neuronModel).output,
        vabpnnJepaModelView(frame.jepaModel).output,destination,tile);
    auto code=cudaGetLastError();
    if(code!=cudaSuccess) throw std::runtime_error(std::string("Network scope: ")+cudaGetErrorString(code));
}


namespace {
__global__ void memeScopeMap(DeviceMemeView memes,const MemeRelayTrace* trace,MemeScopeTelemetry* output) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile>=MemeBlocks) return;
    const auto& config=memes.persistent->config;
    const auto& own=memes.packets[tile*2];const auto& imported=memes.packets[tile*2+1];
    const auto* key=memes.contexts+tile*MemeContext;
    output->values[MemeScopeOwn+tile]=config.enabled ? memePacketWeight(own,key,config):0;
    output->values[MemeScopeImported+tile]=config.enabled ? memePacketWeight(imported,key,config):0;
    output->values[MemeScopeAge+tile]=imported.age;
    output->values[MemeScopeConfidence+tile]=imported.confidence;
    output->values[MemeScopeKnowledge+tile]=memes.persistent->prototypes[tile].confidence;
    output->values[MemeScopeCompatibility+tile]=imported.support ? memeCompatibility(imported.context,key,config):0;
    const bool valid=trace && memeTraceMatches(trace[tile],imported);
    output->discrete[MemeScopeOrigin+tile]=imported.support ? static_cast<int>(imported.origin):-1;
    output->discrete[MemeScopeHops+tile]=imported.hops;
    output->discrete[MemeScopeSender+tile]=valid ? static_cast<int>(trace[tile].sourceTile):-1;
    output->discrete[MemeScopeChannel+tile]=valid ? static_cast<int>(trace[tile].channel|(trace[tile].sourceSlot<<8)):0;
    output->identities[tile]=imported.sequence;
    output->identities[MemeBlocks+tile]=valid ? trace[tile].frame:0;
}
__device__ void memeScopePacket(float* out,const MemePacket& packet) {
    for(int i=0;i<MemeChannels;++i) out[i]=packet.evidence[i];
    for(int i=0;i<MemeContext;++i) out[MemeChannels+i]=packet.context[i];
    out[23]=packet.utility;out[24]=packet.agreement;out[25]=packet.coverage;
    out[26]=packet.confidence;out[27]=packet.age;
}
__global__ void memeScopeDetail(VabpnnGpuFrame frame,DeviceBlockView blocks,DeviceMemeView memes,
                                MemeScopeTelemetry* output,int tile) {
    const auto& p=*memes.persistent;
    const auto& knowledge=p.prototypes[tile];
    const auto& own=memes.packets[tile*2];const auto& imported=memes.packets[tile*2+1];
    const bool valid=frame.memeTrace && memeTraceMatches(frame.memeTrace[tile],imported);
    const MemeRelayTrace relay=valid ? frame.memeTrace[tile]:MemeRelayTrace{};
    const auto source=valid ? static_cast<std::uint64_t>(relay.sourceTile):~0ULL;
    const auto route0=memes.routes[tile*2]<MemeBlocks ? static_cast<std::uint64_t>(memes.routes[tile*2]):~0ULL;
    const auto route1=memes.routes[tile*2+1]<MemeBlocks ? static_cast<std::uint64_t>(memes.routes[tile*2+1]):~0ULL;
    const std::uint64_t metadata[MemeScopeMeta]={
        frame.frameId,frame.captureNanos,frame.evaluationMode ? 1ULL:0ULL,static_cast<std::uint64_t>(tile),
        static_cast<std::uint64_t>(p.config.enabled),static_cast<std::uint64_t>(blocks.config->enabled),
        static_cast<std::uint64_t>(blocks.config->communication),
        p.counters.observedFrames,p.counters.supervisedFrames,p.counters.projectionUpdates,p.counters.propagations,
        p.counters.receipts,p.counters.longSelections,p.counters.dispatches,
        p.counters.localActive,p.counters.importActive,p.counters.invalidConfig,
        relay.frame,source,relay.channel,relay.sourceSlot,route0,route1,
        knowledge.support,knowledge.revision,knowledge.origin,
        own.support,own.revision,own.sequence,own.origin,own.hops,
        imported.support,imported.revision,imported.sequence,imported.origin,imported.hops,
        p.policy[tile].optimizerSteps,relay.captureNanos,0,1};
    for(int i=0;i<MemeScopeMeta;++i) output->meta[i]=metadata[i];
    memeScopePacket(output->values+MemeScopeDetail,knowledge);
    memeScopePacket(output->values+MemeScopeDetail+MemeScopePacketFloats,own);
    memeScopePacket(output->values+MemeScopeDetail+2*MemeScopePacketFloats,imported);
    for(int i=0;i<MemeContext;++i) output->values[MemeScopeVisual+i]=memes.contexts[tile*MemeContext+i];
    for(int k=0;k<MemeChannels;++k) {
        output->values[MemeScopeFeatures+k]=memes.features[tile*MemeChannels+k]/MemeInputScale;
        output->values[MemeScopeAction+k]=memes.actions[tile].feature[k]/MemeInputScale;
    }
    const float cfg[10]={static_cast<float>(p.config.minimumSupport),static_cast<float>(p.config.maximumHops),
        p.config.contextSigma,p.config.minimumCompatibility,p.config.minimumConfidence,p.config.foreignMix,
        p.config.packetLifetimeSeconds,p.config.packetDecaySeconds,relay.sourceWeight,
        static_cast<float>(memes.actions[tile].active)};
    for(int i=0;i<10;++i) output->values[MemeScopeConfig+i]=cfg[i];
}
}
void vabpnnLaunchMemeScope(const VabpnnGpuFrame& frame,int tile,vabpnn::MemeScopeTelemetry* destination) {
    if(!destination || !frame.spikeModel || tile<0 || tile>=vabpnn::MemeBlocks)
        throw std::invalid_argument("Invalid meme scope model/tile");
    const auto blocks=vabpnnBlockModelView(vabpnnSpikeBlockModel(frame.spikeModel));
    const auto memes=vabpnnMemeModelView(vabpnnBlockMemeModel(vabpnnSpikeBlockModel(frame.spikeModel)));
    memeScopeMap<<<(vabpnn::MemeBlocks+255)/256,256,0,frame.stream>>>(memes,frame.memeTrace,destination);
    memeScopeDetail<<<1,1,0,frame.stream>>>(frame,blocks,memes,destination,tile);
    const auto code=cudaGetLastError();
    if(code!=cudaSuccess) throw std::runtime_error(std::string("Meme scope: ")+cudaGetErrorString(code));
}
