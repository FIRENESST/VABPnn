#include "NetworkScope.h"
#include "JepaMotion.h"
#include "MemeState.h"
#include <vector>
#include <memory>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace vabpnn;
void check(cudaError_t c) { if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c)); }
void require(bool ok,const char* msg) { if(!ok) throw std::runtime_error(msg); }
template<class T> T read(const T* p) { T v; check(cudaMemcpy(&v,p,sizeof(T),cudaMemcpyDeviceToHost)); return v; }
template<class T> void put(T* p,const T& value,cudaStream_t stream) {
    check(cudaMemcpyAsync(p,&value,sizeof(value),cudaMemcpyHostToDevice,stream)); check(cudaStreamSynchronize(stream));
}
__global__ void neuralFixture(float* membrane,signed char* spikes) {
    const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p<NeuronCount) { membrane[p]=static_cast<float>((p%1920+p/1920)%16)/16; spikes[p]=static_cast<int>(p%3)-1; }
}
int main() {
    cudaStream_t stream=nullptr; ScopeTelemetry* device=nullptr; unsigned char* image=nullptr;
    VabpnnNeuronModel* neuron=nullptr; VabpnnDecisionModel* decision=nullptr;
    VabpnnJepaModel* jepa=nullptr; VabpnnSpikeModel* spike=nullptr; int result=0;
    try {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        neuron=vabpnnCreateNeuronModel(stream); decision=vabpnnCreateDecisionModel(stream);
        jepa=vabpnnCreateJepaModel(stream); spike=vabpnnCreateSpikeModel(stream);
        check(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(ScopeTelemetry)));
        check(cudaMalloc(reinterpret_cast<void**>(&image),16)); check(cudaMemsetAsync(image,128,16,stream));
        VabpnnGpuFrame frame{image,8,nullptr,2,2,77,1'000'000'000ULL,stream,neuron,decision,jepa,nullptr,spike};
        const unsigned ids[2]={15*1920+15,16*1920+16};
        SpikeConfig config; config.maxAnchors=2; put(vabpnnMutableSpikeConfig(spike,stream),config,stream);
        for(auto id:ids) {
            SpikePixelStatistics stat{.3f,.3f,.4f,10,1};
            put(vabpnnMutableSpikeStatistics(spike,stream)+id,stat,stream);
        }
        for(int i=0;i<6;++i) { frame.frameId++; frame.captureNanos+=20'000'000; vabpnnLaunchSpikeForward(frame); }
        auto view=vabpnnSpikeModelView(spike);
        neuralFixture<<<(NeuronCount+255)/256,256,0,stream>>>(
            const_cast<float*>(view.membrane),vabpnnMutableSpikeSignals(spike,stream));
        int totals[VoteCount]; for(int k=0;k<VoteCount;++k) totals[k]=(k%2 ? -1:1)*(k+1)*70'000;
        check(cudaMemcpyAsync(const_cast<std::int32_t*>(vabpnnNeuronModelView(neuron).output),totals,sizeof(totals),cudaMemcpyHostToDevice,stream));
        auto parameters=std::make_unique<DecisionParameters>();
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(parameters.get(),vabpnnDecisionModelView(decision).parameters,sizeof(*parameters),cudaMemcpyDeviceToHost));
        parameters->logLikelihoodBias[0]=2; parameters->thresholdBias[0]=-1; parameters->kBias[2]=2;
        put(vabpnnMutableDecisionParameters(decision,stream),*parameters,stream);
        vabpnnLaunchTemporalDecision(frame,vabpnnNeuronModelView(neuron).output,decision);
        check(cudaStreamSynchronize(stream)); const auto before=read(view.counters);
        const int inspected=32*120+60;
        auto block=vabpnnBlockModelView(vabpnnSpikeBlockModel(spike));
        auto* persistent=vabpnnMutableBlockPersistent(vabpnnSpikeBlockModel(spike),stream);
        BlockState controller; controller.score=.17f; controller.contribution=.08f; controller.agreement=.73f;
        controller.coverage=.9f; controller.examples=100; controller.optimizerSteps=9;
        put(persistent->state+inspected,controller,stream);
        BlockControl control; control.memoryScale=1.1f;
        put(const_cast<BlockControl*>(block.controls)+inspected,control,stream);
        auto* memes=vabpnnBlockMemeModel(vabpnnSpikeBlockModel(spike));
        auto mv=vabpnnMemeModelView(memes);
        auto* mp=vabpnnMutableMemePersistent(memes,stream);
        MemePacket knowledge;knowledge.support=123;knowledge.revision=18;knowledge.confidence=.7f;knowledge.utility=.13f;
        put(mp->prototypes+inspected,knowledge,stream);
        MemeAction action;action.feature[0]=MemeInputScale*.37f;action.feature[1]=-MemeInputScale*.2f;
        put(mv.actions+inspected,action,stream);
        MemePolicy policy;policy.optimizerSteps=9;put(mp->policy+inspected,policy,stream);
        vabpnnLaunchNetworkScope(frame,inspected,device); check(cudaStreamSynchronize(stream));
        auto host=std::make_unique<ScopeTelemetry>();
        check(cudaMemcpy(host.get(),device,sizeof(*host),cudaMemcpyDeviceToHost));
        require(host->frameId==frame.frameId && host->captureNanos==frame.captureNanos,"matching completed frame/time");
        require(std::fabs(host->values[ScopeBlockScore+inspected]-.17f)<1e-6
            && std::fabs(host->values[ScopeBlockSelected]-.08f)<1e-6
            && std::fabs(host->values[ScopeBlockSelected+5]-1.1f)<1e-6
            && host->discrete[ScopeBlockData+2]==100 && host->discrete[ScopeBlockData+3]==9,
            "actual controller score/control/counters share the exact GPU snapshot");
        require(host->discrete[ScopeMemeData+3]==123 && host->discrete[ScopeMemeData+4]==9
            && std::fabs(host->values[ScopeMemeEvidence]-.37f)<1e-6
            && std::fabs(host->values[ScopeMemeEvidence+1]+.2f)<1e-6,"actual cached projection feature/support/own step are exported readonly");
        require(host->discrete[0]==1 && host->discrete[1]==inspected && host->discrete[2]==2
            && host->discrete[3]==1 && host->discrete[5]==1,"probabilities/k/eligibility describe the same decision");
        for(int tile=0;tile<SpikeTiles;++tile) {
            float sum=0; int positive=0,negative=0;
            for(int tid=0;tid<256;++tid) {
                unsigned p=spikePixel(tile,tid); sum+=static_cast<float>((p%1920+p/1920)%16)/16;
                int signal=static_cast<int>(p%3)-1; positive+=signal>0; negative+=signal<0;
                if(tile==inspected) {
                    require(std::fabs(host->values[ScopeLocalMem+tid]-static_cast<float>((p%1920+p/1920)%16)/16)<1e-6,"local membranes exact");
                    require(host->discrete[ScopeLocalSpike+tid]==signal,"signed local spikes exact");
                    for(int k=0;k<4;++k)
                        require(host->discrete[ScopeOutgoing+tid*4+k]==read(view.localOutgoing+(tile*256+tid)*4+k),"actual four-kind adjacency");
                }
            }
            require(std::fabs(host->values[ScopeMapMem+tile]-sum/256)<1e-6,"all block membrane averages exact");
            require(std::fabs(host->values[ScopeMapPositive+tile]-positive/256.0f)<1e-6
                && std::fabs(host->values[ScopeMapNegative+tile]-negative/256.0f)<1e-6,"all signed firing rates exact");
        }
        require(std::fabs(host->values[ScopePosterior]-.880797f)<1e-5
            && std::fabs(host->values[ScopeThreshold]-.268941f)<1e-5,"posterior and learned threshold, not a dummy UI probability");
        for(int k=0;k<VoteCount;++k) require(host->discrete[ScopeVotes+k]==totals[k],"raw signed totals preserved");
        for(int i=0;i<ScopeEdges;++i) {
            const unsigned source=host->discrete[ScopeEdgePairs+2*i];
            const unsigned encoded=host->discrete[ScopeEdgePairs+2*i+1],target=encoded&SpikeSourceMask,kind=encoded>>21;
            require(source<NeuronCount && target<NeuronCount && kind<4,"bounded edge endpoints and four kinds");
            const unsigned begin=read(view.longRows+target),end=read(view.longRows+target+1);
            bool found=false;
            for(unsigned e=begin;e<end;++e) found|=read(view.longColumns+e)==(source|(kind<<21));
            require(found,"visualized long edge exists in destination CSR, including empty row boundaries");
        }
        const auto after=read(view.counters);
        require(before.predictions==after.predictions && before.topologyUpdates==after.topologyUpdates
            && before.localEdges==after.localEdges && before.longEdges==after.longEdges,"observing does not change network topology/state");
        bool rejected=false;
        try { vabpnnLaunchNetworkScope(frame,-1,device); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid inspected tile rejected before GPU access");
        std::printf("PASS: live scope matches GPU membranes/signed rates/local CSR/long CSR/posterior/threshold/raw votes; readonly snapshot %zu bytes.\n",sizeof(ScopeTelemetry));
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); result=1; }
    if(stream) cudaStreamSynchronize(stream);
    vabpnnDestroySpikeModel(spike); vabpnnDestroyJepaModel(jepa); vabpnnDestroyDecisionModel(decision); vabpnnDestroyNeuronModel(neuron);
    if(device) cudaFree(device); if(image) cudaFree(image); if(stream) cudaStreamDestroy(stream); return result;
}

