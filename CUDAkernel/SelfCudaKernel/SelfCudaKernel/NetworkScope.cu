#include "NetworkScope.h"
#include "JepaMotion.h"
#include <stdexcept>
#include <string>
namespace {
using namespace vabpnn;
__global__ void scopeBlocks(DeviceSpikeView view,DeviceBlockView blocks,ScopeTelemetry* output,int selectedTile) {
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
    const DecisionOutput* decision,const std::int32_t* votes,const JepaOutput* jepa,
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
    scopeBlocks<<<vabpnn::SpikeTiles,256,0,frame.stream>>>(view,blocks,destination,tile);
    scopeEdges<<<2,256,0,frame.stream>>>(view,destination);
    scopeHeads<<<1,1,0,frame.stream>>>(frame.frameId,frame.captureNanos,view,blocks,
        vabpnnDecisionModelView(frame.decisionModel).output,vabpnnNeuronModelView(frame.neuronModel).output,
        vabpnnJepaModelView(frame.jepaModel).output,destination,tile);
    auto code=cudaGetLastError();
    if(code!=cudaSuccess) throw std::runtime_error(std::string("Network scope: ")+cudaGetErrorString(code));
}
