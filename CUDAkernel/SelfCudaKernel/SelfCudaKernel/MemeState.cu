#include "MemeState.h"
#include "SpikeConnections.h"
#include "OnlineLearning.h"
#include <stdexcept>
#include <string>
namespace {
using namespace vabpnn;
struct Runtime {
    std::uint64_t frame=0,nanos=0,observedFrame=0,learnedFrame=0;
    float dt=0;
    int fresh=0,feedback=0,width=0,height=0;
};
void check(cudaError_t result,const char* operation) {
    if(result!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(result));
}
__host__ __device__ bool validConfig(const MemeConfig& c) {
    return c.enabled>=0 && c.enabled<=1 && c.minimumSupport>0 && c.minimumSupport<=1000000
        && c.maximumHops>=0 && c.maximumHops<=16 && isfinite(c.contextSigma) && c.contextSigma>0
        && isfinite(c.minimumCompatibility) && c.minimumCompatibility>0 && c.minimumCompatibility<=1
        && isfinite(c.minimumConfidence) && c.minimumConfidence>0 && c.minimumConfidence<=1
        && isfinite(c.foreignMix) && c.foreignMix>=0 && c.foreignMix<=0.2f
        && isfinite(c.packetLifetimeSeconds) && c.packetLifetimeSeconds>0 && c.packetLifetimeSeconds<=60
        && isfinite(c.packetDecaySeconds) && c.packetDecaySeconds>0 && c.packetDecaySeconds<=60;
}
__global__ void initialize(MemePersistent* p) {
    if(blockIdx.x==0 && threadIdx.x==0) {
        p->config=MemeConfig{};p->counters=MemeCounters{};
    }
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile<MemeBlocks) p->prototypes[tile].origin=tile;
}
__global__ void beginFrame(MemePersistent* p,Runtime* clock,const BlockConfig* blocks,VabpnnGpuFrame frame) {
    const bool configValid=validConfig(p->config);
    p->counters.enabled=p->config.enabled && blocks->enabled;
    p->counters.invalidConfig=!configValid;
    const bool observationValid=frame.rgba && frame.width>0 && frame.height>0
        && frame.rgbaPitch>=static_cast<std::size_t>(frame.width)*4 && frame.captureNanos>0;
    const bool monotonic=observationValid && (!clock->nanos
        || (frame.frameId>clock->frame && frame.captureNanos>clock->nanos));
    clock->dt=monotonic && clock->nanos
        ? fminf(3600.0f,static_cast<float>((frame.captureNanos-clock->nanos)*1e-9)):0;
    clock->fresh=configValid && p->counters.enabled && monotonic;
    if(monotonic) {
        clock->frame=frame.frameId;clock->nanos=frame.captureNanos;
        clock->width=frame.width;clock->height=frame.height;
    }
}
__global__ void agePackets(MemePersistent* p,MemePacket* current,MemePacket* other,const Runtime* clock) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile>=MemeBlocks) return;
    // Learned knowledge does not lose its support during inference. Activity packets do expire.
    p->prototypes[tile].age=fminf(86400.0f,p->prototypes[tile].age+clock->dt);
    for(int slot=0;slot<MemePacketSlots;++slot) {
        current[tile*2+slot].age=fminf(61.0f,current[tile*2+slot].age+clock->dt);
        other[tile*2+slot].age=fminf(61.0f,other[tile*2+slot].age+clock->dt);
    }
}
__global__ void context(VabpnnGpuFrame frame,const Runtime* clock,const DecisionOutput* decision,float* keys) {
    if(!clock->fresh) return;
    const int tile=blockIdx.x,tid=threadIdx.x;
    const unsigned pixel=spikePixel(tile,tid);
    const int x=static_cast<int>((2*(pixel%NeuronWidth)+1)*static_cast<std::size_t>(frame.width)/(2*NeuronWidth));
    const int y=static_cast<int>((2*(pixel/NeuronWidth)+1)*static_cast<std::size_t>(frame.height)/(2*NeuronHeight));
    const auto* rgba=frame.rgba+static_cast<std::size_t>(frame.height-1-y)*frame.rgbaPitch+x*4;
    float value[4]={rgba[0]/255.0f,rgba[1]/255.0f,rgba[2]/255.0f,0};
    const float luminance=.2126f*value[0]+.7152f*value[1]+.0722f*value[2];
    value[3]=luminance*luminance;
    for(int step=16;step;step/=2) for(float& item:value)
        item+=__shfl_down_sync(0xffffffffu,item,step);
    __shared__ float sums[4][8];
    if((tid&31)==0) for(int d=0;d<4;++d) sums[d][tid/32]=value[d];
    __syncthreads();
    if(tid!=0) return;
    auto* key=keys+tile*MemeContext;
    for(int d=0;d<4;++d) {
        float sum=0;for(int warp=0;warp<8;++warp) sum+=sums[d][warp];
        key[d]=sum/256;
    }
    const float mean=.2126f*key[0]+.7152f*key[1]+.0722f*key[2];
    key[3]=fmaxf(0,key[3]-mean*mean);
    for(int d=4;d<MemeContext;++d) key[d]=0;
    if(decision && decision->valid) {
        key[4]=(decision->posterior[0]+decision->posterior[1]+decision->posterior[2]+decision->posterior[3])*.25f;
        key[5]=(decision->posterior[4]+decision->posterior[7]+decision->posterior[8]+decision->posterior[11])*.25f;
        key[6]=(decision->posterior[5]+decision->posterior[6])*.5f;key[7]=decision->posterior[12];
    }
}
__device__ bool samePacket(const MemePacket& a,const MemePacket& b) {
    return a.support && b.support && a.origin==b.origin && a.revision==b.revision && a.sequence==b.sequence;
}
__device__ bool competitionEligible(const MemePacket& packet,const MemePacket& own,float ownWeight,int mode) {
    return mode!=2 || ownWeight<=0 || packet.utility>own.utility+.002f;
}
__device__ float sourceWeight(const MemePacket& packet,const float* key,const MemeConfig& c,int tile) {
    return packet.origin==static_cast<unsigned>(tile) || packet.hops>=static_cast<unsigned>(c.maximumHops)
        ? 0:memePacketWeight(packet,key,c);
}
__global__ void selectRoutes(MemePersistent* p,const Runtime* clock,const BlockConfig* blocks,
        const MemePacket* old,const float* keys,DeviceSpikeView spike,unsigned* routes) {
    const int tile=blockIdx.x,tid=threadIdx.x;
    __shared__ unsigned long long best[2];
    if(tid<2) { best[tid]=~0ULL;routes[tile*2+tid]=NoNeuron; }
    __syncthreads();
    if(!clock->fresh || blocks->communication==0 || !spike.counters->longEdges) return;
    __shared__ float ownWeight;
    if(tid==0) {
        auto own=p->prototypes[tile];own.age=0;own.sequence=clock->frame;own.hops=0;
        ownWeight=memePacketWeight(own,keys+tile*MemeContext,p->config);
    }
    __syncthreads();
    const auto& own=p->prototypes[tile]; // Competition uses only the unchanged utility.
    // Sparse anchor rows can contain hundreds of edges. Assign a warp to a row
    // so its lanes coalesce edge reads, instead of one lane scanning it serially.
    // Visit all 256 actual CSR rows, including ordinary non-anchor pixels.
    const int lane=tid&31,warp=tid>>5;
    for(int local=warp;local<SpikeTileSize;local+=8) {
        const unsigned pixel=spikePixel(tile,local),start=spike.longRows[pixel],end=spike.longRows[pixel+1];
        if(end>=start && end<=spike.counters->longEdges) for(unsigned edge=start+lane;edge<end;edge+=32) {
            const unsigned source=spike.longColumns[edge]&SpikeSourceMask;
            if(source>=NeuronCount) continue;
            const int other=source/NeuronWidth/16*120+source%NeuronWidth/16;
            const int distance=abs(other%120-tile%120)+abs(other/120-tile/120);
            if(distance<=1) continue;
            float weight=0;
            for(int slot=0;slot<2;++slot) {
                const auto& packet=old[other*2+slot];
                if(competitionEligible(packet,own,ownWeight,blocks->communication))
                    weight=fmaxf(weight,sourceWeight(packet,keys+tile*MemeContext,p->config,tile));
            }
            if(weight<=0) continue;
            const unsigned rank=static_cast<unsigned>(fminf(1.0f,weight)*16777215.0f);
            const unsigned long long candidate=(static_cast<unsigned long long>(0xffffffffu-rank)<<32)|other;
            const auto previous=atomicMin(&best[0],candidate);
            if(previous!=candidate) atomicMin(&best[1],previous>candidate ? previous:candidate);
        }
    }
    __syncthreads();
    if(tid<2 && best[tid]!=~0ULL) routes[tile*2+tid]=static_cast<unsigned>(best[tid]);
    if(tid==0) {
        unsigned count=(best[0]!=~0ULL)+(best[1]!=~0ULL);
        atomicAdd(reinterpret_cast<unsigned long long*>(&p->counters.longSelections),static_cast<unsigned long long>(count));
    }
}
__global__ void beginPropagation(MemePersistent* p,const Runtime* clock) {
    p->counters.localActive=p->counters.importActive=0;
    if(!clock->fresh) return;
    ++p->counters.propagations;
}
__global__ void propagate(MemePersistent* p,const Runtime* clock,const BlockConfig* blocks,
        const MemePacket* old,MemePacket* next,const float* keys,const unsigned* routes,float* features,MemeRelayTrace* trace) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile>=MemeBlocks) return;
    const auto& config=p->config; const auto* key=keys+tile*MemeContext;
    MemePacket own{},received{};float ownWeight=0,receivedWeight=0;
    int sender=-1,senderSlot=-1,channel=0;
    if(clock->fresh) {
        // A real new observation may reactivate learned knowledge without a new training label.
        own=p->prototypes[tile];own.age=0;own.sequence=clock->frame;own.hops=0;
        ownWeight=memePacketWeight(own,key,config);
        if(ownWeight<=0) own=MemePacket{};
        else atomicAdd(reinterpret_cast<unsigned long long*>(&p->counters.dispatches),1ULL);
        if(blocks->communication!=0) {
            const auto& held=old[tile*2+1];
            receivedWeight=memePacketWeight(held,key,config);
            if(held.origin==static_cast<unsigned>(tile) || !competitionEligible(held,own,ownWeight,blocks->communication)) receivedWeight=0;
            if(receivedWeight>0) received=held;
            const int x=tile%120,y=tile/120;
            const int neighbors[6]={x>0 ? tile-1:-1,x<119 ? tile+1:-1,
                y>0 ? tile-120:-1,y<63 ? tile+120:-1,
                routes[tile*2]<MemeBlocks ? static_cast<int>(routes[tile*2]):-1,
                routes[tile*2+1]<MemeBlocks ? static_cast<int>(routes[tile*2+1]):-1};
            for(int other:neighbors) if(other>=0) for(int slot=0;slot<2;++slot) {
                const auto& candidate=old[other*2+slot];
                if(!competitionEligible(candidate,own,ownWeight,blocks->communication)
                    || sourceWeight(candidate,key,config,tile)<=0 || samePacket(candidate,held)) continue;
                auto forwarded=candidate;++forwarded.hops;
                const float weight=memePacketWeight(forwarded,key,config);
                if(weight>receivedWeight+1e-7f) {
                    received=forwarded;receivedWeight=weight;
                    if(trace) {
                        sender=other;senderSlot=slot;
                        const int distance=abs(other%120-x)+abs(other/120-y);
                        channel=distance==1 ? 1:2;
                    }
                }
            }
            if(receivedWeight>0 && !samePacket(received,held))
                atomicAdd(reinterpret_cast<unsigned long long*>(&p->counters.receipts),1ULL);
        }
    }
    if(trace) {
        if(receivedWeight>0 && sender>=0) {
            trace[tile]={clock->frame,clock->nanos,received.revision,received.sequence,received.origin,
                static_cast<unsigned>(sender),static_cast<unsigned>(senderSlot),static_cast<unsigned>(channel),receivedWeight,0};
        } else if(receivedWeight<=0 || !memeTraceMatches(trace[tile],received)) trace[tile]=MemeRelayTrace{};
    }
    next[tile*2]=own;next[tile*2+1]=received;
    if(ownWeight>0) atomicAdd(&p->counters.localActive,1u);
    if(receivedWeight>0) atomicAdd(&p->counters.importActive,1u);
    const float fraction=receivedWeight>0 ? config.foreignMix:0;
    for(int k=0;k<MemeChannels;++k) features[tile*MemeChannels+k]=MemeInputScale*
        ((1-fraction)*ownWeight*own.evidence[k]+fraction*receivedWeight*received.evidence[k]);
}
__global__ void observe(const MemePersistent* p,const Runtime* clock,const int* votes,const float* keys,
                       MemeObservation* observations) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile>=MemeBlocks || !clock->fresh || !p->counters.enabled) return;
    auto& observation=observations[tile];
    for(int k=0;k<MemeChannels;++k) observation.evidence[k]=votes[k*MemeBlocks+tile]/256.0f;
    for(int d=0;d<MemeContext;++d) observation.context[d]=keys[tile*MemeContext+d];
}
__global__ void stampObservation(MemePersistent* p,Runtime* clock) {
    clock->observedFrame=clock->fresh ? clock->frame:0;
    if(clock->fresh && p->counters.enabled) ++p->counters.observedFrames;
}
__global__ void beginFeedback(MemePersistent* p,Runtime* clock,const int* valid,const DecisionOutput* out,const DemoSample* demo) {
    clock->feedback=validConfig(p->config) && p->config.enabled && p->counters.enabled && *valid && out->valid
        && demo && demo->active && !demo->modelControl
        && clock->observedFrame==out->frameId && clock->observedFrame!=clock->learnedFrame;
    if(clock->feedback) {clock->learnedFrame=clock->observedFrame;++p->counters.supervisedFrames;}
}
__global__ void learn(MemePersistent* p,const Runtime* clock,const MemeObservation* observations,
        const MemeActorCredit* credits,MemeAction* actions,const unsigned* label,const BlockConfig* blockConfig) {
    if(!clock->feedback) return;
    const int tile=blockIdx.x,tid=threadIdx.x; const auto& credit=credits[tile];
    if(!credit.valid || credit.frame!=clock->observedFrame) return;
    auto& prototype=p->prototypes[tile];const auto& observation=observations[tile];
    auto& policy=p->policy[tile];auto& action=actions[tile];
    __shared__ int update;__shared__ float bias1,bias2;
    if(tid==0) {
        float total=0,participates=0,correct=0;
        for(int k=0;k<MemeChannels;++k) {
            const bool active=(*label>>k)&1u;const float weight=active ? 1.0f:.25f;total+=weight;
            if(observation.evidence[k]!=0) {
                participates+=weight;if((observation.evidence[k]>0)==active) correct+=weight;
            }
        }
        const float coverage=participates/fmaxf(1e-6f,total);
        if(coverage>=blockConfig->minimumCoverage) {
            const bool reset=prototype.support==0
                || memeCompatibility(prototype.context,observation.context,p->config)<p->config.minimumCompatibility;
            const float alpha=reset ? 1.0f:.1f;
            if(reset) {prototype.support=0;prototype.agreement=prototype.coverage=prototype.utility=0;}
            for(int k=0;k<MemeChannels;++k)
                prototype.evidence[k]=(1-alpha)*prototype.evidence[k]+alpha*observation.evidence[k];
            for(int d=0;d<MemeContext;++d)
                prototype.context[d]=(1-alpha)*prototype.context[d]+alpha*observation.context[d];
            prototype.agreement=(1-alpha)*prototype.agreement+alpha*correct/fmaxf(1e-6f,participates);
            prototype.coverage=(1-alpha)*prototype.coverage+alpha*coverage;
            prototype.utility=(1-alpha)*prototype.utility+alpha*credit.contribution;
            ++prototype.support;++prototype.revision;prototype.sequence=0;prototype.origin=tile;prototype.hops=0;prototype.age=0;
            prototype.confidence=fmaxf(0,2*prototype.agreement-1)*prototype.coverage
                *static_cast<float>(prototype.support)/(prototype.support+p->config.minimumSupport);
        }
        update=0;
        if(action.active && credit.exploring && ++action.batch>=static_cast<unsigned>(blockConfig->trainEvery)) {
            const float step=static_cast<float>(++policy.optimizerSteps);
            bias1=1-powf(.9f,step);bias2=1-powf(.999f,step);update=1;
            atomicAdd(reinterpret_cast<unsigned long long*>(&p->counters.projectionUpdates),1ULL);
        }
    }
    __syncthreads();
    for(int at=tid;at<MemeWeights;at+=32) {
        if(action.active && credit.exploring) {
            const float gradient=-credit.advantage*credit.noise[at/MemeChannels]*action.feature[at%MemeChannels]/credit.exploration;
            action.gradient[at]+=fminf(1.0f,fmaxf(-1.0f,gradient));
        }
        if(update) {
            const float gradient=action.gradient[at]/blockConfig->trainEvery;
            policy.first[at]=.9f*policy.first[at]+.1f*gradient;
            policy.second[at]=.999f*policy.second[at]+.001f*gradient*gradient;
            const float next=policy.weights[at]*(1-blockConfig->learningRate*.01f)
                -blockConfig->learningRate*(policy.first[at]/bias1)/(sqrtf(policy.second[at]/bias2)+1e-8f);
            policy.weights[at]=fminf(1.5f,fmaxf(-1.5f,next));action.gradient[at]=0;
        }
    }
    __syncthreads();if(tid==0 && update) action.batch=0;
}
__global__ void resetActivityCounters(MemePersistent* p) {
    p->counters.localActive=p->counters.importActive=0;
    p->counters.enabled=p->config.enabled;
}
} // namespace

struct VabpnnMemeModel {
    vabpnn::MemePersistent* persistent=nullptr;
    vabpnn::MemePacket* packets[2]{};
    vabpnn::MemeAction* actions=nullptr;
    vabpnn::MemeActorCredit* credit=nullptr;
    vabpnn::MemeObservation* observations=nullptr;
    float* contexts=nullptr;
    float* features=nullptr;
    unsigned* routes=nullptr;
    Runtime* runtime=nullptr;
    cudaEvent_t initialized=nullptr;
    cudaStream_t initializationStream=nullptr;
    vabpnn::MemeConfig hostConfig{};
    std::size_t bytes=0;
    int current=0;
};
namespace {
template<class T> void allocate(T*& pointer,std::size_t count,VabpnnMemeModel* model,const char* operation) {
    const auto bytes=count*sizeof(T);
    check(cudaMalloc(reinterpret_cast<void**>(&pointer),bytes),operation);model->bytes+=bytes;
    check(cudaMemsetAsync(pointer,0,bytes,model->initializationStream),"clear meme allocation");
}
void ready(VabpnnMemeModel* model,cudaStream_t stream) {
    if(!model) throw std::invalid_argument("Meme model missing");
    check(cudaStreamWaitEvent(stream,model->initialized,0),"wait for meme initialization");
}
VabpnnMemeModel* from(const VabpnnGpuFrame& frame) {
    return frame.spikeModel ? vabpnnBlockMemeModel(vabpnnSpikeBlockModel(frame.spikeModel)):nullptr;
}
}
VabpnnMemeModel* vabpnnCreateMemeModel(cudaStream_t stream) {
    auto* model=new VabpnnMemeModel();model->initializationStream=stream;
    try {
        allocate(model->persistent,1,model,"allocate learned context/evidence prototypes and projection optimizer");
        for(auto& packet:model->packets) allocate(packet,vabpnn::MemeBlocks*2,model,"allocate synchronous activity packets");
        allocate(model->actions,vabpnn::MemeBlocks,model,"allocate actual meme action features and gradients");
        allocate(model->credit,vabpnn::MemeBlocks,model,"allocate old-action score-function credit");
        allocate(model->observations,vabpnn::MemeBlocks,model,"allocate all-neuron evidence candidates");
        allocate(model->contexts,vabpnn::MemeBlocks*vabpnn::MemeContext,model,"allocate current visual contexts");
        allocate(model->features,vabpnn::MemeBlocks*vabpnn::MemeChannels,model,"allocate bounded projection input");
        allocate(model->routes,vabpnn::MemeBlocks*2,model,"allocate two actual long-range source tiles");
        allocate(model->runtime,1,model,"allocate meme causal clock");
        check(cudaMemsetAsync(model->routes,0xff,vabpnn::MemeBlocks*2*sizeof(unsigned),stream),"initialize absent meme routes");
        initialize<<<(vabpnn::MemeBlocks+255)/256,256,0,stream>>>(model->persistent);
        check(cudaEventCreateWithFlags(&model->initialized,cudaEventDisableTiming),"create meme initialization event");
        check(cudaEventRecord(model->initialized,stream),"record meme initialization");
        check(cudaGetLastError(),"initialize bounded meme state");
        return model;
    } catch(...) {vabpnnDestroyMemeModel(model);throw;}
}
void vabpnnDestroyMemeModel(VabpnnMemeModel* model) noexcept {
    if(!model) return;if(model->initialized) cudaEventSynchronize(model->initialized);
    void* allocations[]={model->persistent,model->packets[0],model->packets[1],model->actions,model->credit,
        model->observations,model->contexts,model->features,model->routes,model->runtime};
    for(void* allocation:allocations) if(allocation) cudaFree(allocation);
    if(model->initialized) cudaEventDestroy(model->initialized);delete model;
}
std::uint64_t vabpnnMemeModelBytes(const VabpnnMemeModel* model) noexcept {return model ? model->bytes:0;}
vabpnn::DeviceMemeView vabpnnMemeModelView(const VabpnnMemeModel* model) {
    if(!model) throw std::invalid_argument("Meme model missing");
    return {model->persistent,model->packets[model->current],model->features,model->actions,model->credit,
        model->observations,model->contexts,model->routes};
}
vabpnn::MemePersistent* vabpnnMutableMemePersistent(VabpnnMemeModel* model,cudaStream_t stream) {
    ready(model,stream);return model->persistent;
}
void vabpnnConfigureMemeModel(VabpnnMemeModel* model,const vabpnn::MemeConfig& config,cudaStream_t stream) {
    if(!validConfig(config)) throw std::invalid_argument("Invalid meme configuration");
    ready(model,stream);model->hostConfig=config;
    check(cudaMemcpyAsync(&model->persistent->config,&model->hostConfig,sizeof(config),cudaMemcpyHostToDevice,stream),"configure bounded meme communication");
    vabpnnResetMemeHistory(model,stream);
}
void vabpnnConfigureMemeEnabled(VabpnnMemeModel* model,int enabled,cudaStream_t stream) {
    if(!model) throw std::invalid_argument("Meme model missing");
    auto config=model->hostConfig;config.enabled=enabled;vabpnnConfigureMemeModel(model,config,stream);
}
void vabpnnPrepareMemes(const VabpnnGpuFrame& frame,bool generateControl) {
    auto* model=from(frame);if(!model) return;ready(model,frame.stream);
    const auto blocks=vabpnnBlockModelView(vabpnnSpikeBlockModel(frame.spikeModel));
    beginFrame<<<1,1,0,frame.stream>>>(model->persistent,model->runtime,blocks.config,frame);
    if(model->hostConfig.enabled) {
        agePackets<<<(vabpnn::MemeBlocks+255)/256,256,0,frame.stream>>>(
            model->persistent,model->packets[model->current],model->packets[1-model->current],model->runtime);
        const auto* decision=frame.decisionModel ? vabpnnDecisionModelView(frame.decisionModel).output:nullptr;
        context<<<vabpnn::MemeBlocks,256,0,frame.stream>>>(frame,model->runtime,decision,model->contexts);
        if(generateControl) {
            const int next=1-model->current;
            selectRoutes<<<vabpnn::MemeBlocks,256,0,frame.stream>>>(model->persistent,model->runtime,blocks.config,
                model->packets[model->current],model->contexts,vabpnnSpikeModelView(frame.spikeModel),model->routes);
            beginPropagation<<<1,1,0,frame.stream>>>(model->persistent,model->runtime);
            propagate<<<(vabpnn::MemeBlocks+255)/256,256,0,frame.stream>>>(model->persistent,model->runtime,
                blocks.config,model->packets[model->current],model->packets[next],model->contexts,model->routes,model->features,frame.memeTrace);
            model->current=next;
        }
    }
    check(cudaGetLastError(),"prepare context-gated synchronous meme packets");
}
void vabpnnObserveMemes(const VabpnnGpuFrame& frame) {
    auto* model=from(frame);if(!model || !frame.neuronModel || !model->hostConfig.enabled) return;
    observe<<<(vabpnn::MemeBlocks+255)/256,256,0,frame.stream>>>(model->persistent,model->runtime,
        vabpnnNeuronModelView(frame.neuronModel).blockVotes,model->contexts,model->observations);
    stampObservation<<<1,1,0,frame.stream>>>(model->persistent,model->runtime);
    check(cudaGetLastError(),"observe all-neuron signed evidence without image readback");
}
void vabpnnLearnMemes(const VabpnnGpuFrame& frame,const int* valid,const std::uint32_t* label) {
    auto* model=from(frame);if(frame.evaluationMode || !model || !frame.decisionModel || !model->hostConfig.enabled) return;
    const auto blocks=vabpnnBlockModelView(vabpnnSpikeBlockModel(frame.spikeModel));
    beginFeedback<<<1,1,0,frame.stream>>>(model->persistent,model->runtime,valid,vabpnnDecisionModelView(frame.decisionModel).output,vabpnnOnlineDemoSample(frame.onlineModel));
    learn<<<vabpnn::MemeBlocks,32,0,frame.stream>>>(model->persistent,model->runtime,model->observations,
        model->credit,model->actions,label,blocks.config);
    check(cudaGetLastError(),"learn only old visual memes and actual cached projection actions");
}
void vabpnnResetMemeHistory(VabpnnMemeModel* model,cudaStream_t stream) {
    ready(model,stream);
    for(auto packet:model->packets)
        check(cudaMemsetAsync(packet,0,vabpnn::MemeBlocks*2*sizeof(vabpnn::MemePacket),stream),"clear in-flight meme activity packets");
    for(auto item:{vabpnn::ModelPart{model->actions,vabpnn::MemeBlocks*sizeof(vabpnn::MemeAction)},
        {model->credit,vabpnn::MemeBlocks*sizeof(vabpnn::MemeActorCredit)},
        {model->observations,vabpnn::MemeBlocks*sizeof(vabpnn::MemeObservation)},
        {model->contexts,vabpnn::MemeBlocks*vabpnn::MemeContext*sizeof(float)},
        {model->features,vabpnn::MemeBlocks*vabpnn::MemeChannels*sizeof(float)},
        {model->runtime,sizeof(Runtime)}})
        check(cudaMemsetAsync(item.device,0,item.bytes,stream),"clear transient meme action/candidate/context history");
    check(cudaMemsetAsync(model->routes,0xff,vabpnn::MemeBlocks*2*sizeof(unsigned),stream),"clear transient long-range routes");
    resetActivityCounters<<<1,1,0,stream>>>(model->persistent);
    check(cudaGetLastError(),"reset active meme status");
    model->current=0;
}
void vabpnnResumeMemeModel(VabpnnMemeModel* model,cudaStream_t stream) {
    ready(model,stream);
    check(cudaMemcpy(&model->hostConfig,&model->persistent->config,sizeof(model->hostConfig),cudaMemcpyDeviceToHost),"restore meme scheduler configuration");
    if(!validConfig(model->hostConfig)) throw std::runtime_error("Invalid saved meme configuration");
    vabpnnResetMemeHistory(model,stream);
}
vabpnn::ModelPart vabpnnMemeCheckpointPart(VabpnnMemeModel* model) {
    return {model->persistent,sizeof(vabpnn::MemePersistent)};
}


