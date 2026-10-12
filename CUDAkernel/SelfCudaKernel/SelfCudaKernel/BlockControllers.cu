#include "BlockControllers.h"
#include "SpikeConnections.h"
#include "OnlineLearning.h"
#include "MemeState.h"
#include <cub/block/block_radix_sort.cuh>
#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <string>
namespace {
using namespace vabpnn;
struct Pending {
    float feature[BlockFeatures],noise[BlockOutputs],gradient[BlockWeights];
    std::uint32_t batch,exploring;
};
struct Message { float score,coverage,threshold,polarity,memory,learning; };
struct Runtime { std::uint64_t appliedFrame,lastScoredFrame,lastRankScoredFrames; int ready,feedback,rank; };
void check(cudaError_t result,const char* operation) {
    if(result!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(result));
}
__host__ __device__ unsigned blockHash(unsigned x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
__device__ float gaussian(unsigned seed) {
    const float u=(static_cast<float>(blockHash(seed)&0xffffffu)+1)/16777217.0f;
    const float v=static_cast<float>(blockHash(seed+97)&0xffffffu)/16777216.0f;
    return sqrtf(-2*logf(u))*cosf(6.28318530718f*v);
}
__host__ __device__ bool validConfig(const BlockConfig& c) {
    return c.enabled>=0 && c.enabled<=1 && c.communication>=0 && c.communication<=2 && c.radius>=0 && c.radius<=2
        && c.warmup>=0 && c.trainEvery>0 && c.trainEvery<=64 && c.rankEvery>0 && c.rankEvery<=65536
        && c.minimumExamples>=0 && c.minimumWindow>0 && c.poorWindows>0 && c.poorWindows<=32 && c.cooldown>=0
        && isfinite(c.exploration) && c.exploration>0 && c.exploration<=0.25f
        && isfinite(c.learningRate) && c.learningRate>0 && c.learningRate<=0.01f
        && isfinite(c.minimumCoverage) && c.minimumCoverage>=0 && c.minimumCoverage<=1
        && isfinite(c.rankMargin) && c.rankMargin>=0 && isfinite(c.mutation) && c.mutation>=0 && c.mutation<=0.1f;
}
__global__ void initialize(BlockPersistent* persistent,BlockControl* controls) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile==0) { persistent->config=BlockConfig{}; persistent->counters=BlockCounters{}; }
    if(tile<BlockCount) { persistent->state[tile].bestScore=-1e20f; controls[tile]=BlockControl{}; }
}
__global__ void neutral(BlockControl* controls) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile<BlockCount) controls[tile]=BlockControl{};
}
__global__ void beginFeedback(BlockPersistent* p,Runtime* runtime,const DecisionOutput* decision,
                              const int* valid) {
    runtime->feedback=validConfig(p->config) && p->config.enabled && *valid && decision->valid
        && runtime->ready && runtime->appliedFrame==decision->frameId && runtime->lastScoredFrame!=decision->frameId;
    if(runtime->feedback) { ++p->counters.scoredFrames; runtime->lastScoredFrame=decision->frameId; }
}
__device__ float directGain(float probability,float label,float delta) {
    probability=fminf(1-1e-6f,fmaxf(1e-6f,probability));
    if(fabsf(delta)<0.001f)
        return (label-probability)*delta+0.5f*probability*(1-probability)*delta*delta;
    return log1pf(probability*expm1f(-delta))+label*delta;
}
__global__ void learn(BlockPersistent* p,Pending* pending,const Runtime* runtime,const int* votes,
                      const DecisionOutput* decision,const DecisionParameters* parameters,
                      const unsigned* label,const unsigned* champions,DeviceMemeView meme) {
    const int tile=blockIdx.x,tid=threadIdx.x;
    if(!runtime->feedback) {if(tid==0) meme.credit[tile].valid=0;return;}
    auto& s=p->state[tile]; auto& cache=pending[tile]; const auto& config=p->config;
    __shared__ float gain[32],agreement[32],coverage[32],weight[32],advantage,bias1,bias2;
    __shared__ int update;
    float g=0,hit=0,participates=0,w=0;
    if(tid<VoteCount) {
        const int vote=votes[tid*BlockCount+tile];
        const float y=(*label>>tid)&1u;
        w=y ? 1.0f:0.25f; // Active operations keep weight when inactive operations dominate.
        const float delta=parameters->rawVoteWeight[tid]*vote/NeuronCount;
        g=directGain(decision->posterior[tid],y,delta)*BlockCount*w;
        participates=vote!=0 ? w:0;
        hit=vote!=0 && ((vote>0)==(y>0)) ? w:0;
    }
    gain[tid]=g; agreement[tid]=hit; coverage[tid]=participates; weight[tid]=w;
    __syncthreads();
    for(int step=16;step;step/=2) {
        if(tid<step) { gain[tid]+=gain[tid+step]; agreement[tid]+=agreement[tid+step];
            coverage[tid]+=coverage[tid+step]; weight[tid]+=weight[tid+step]; }
        __syncthreads();
    }
    if(tid==0) {
        const float contribution=fminf(2.0f,fmaxf(-2.0f,gain[0]/fmaxf(1e-6f,weight[0])));
        const float participation=coverage[0]/fmaxf(1e-6f,weight[0]);
        const float consistency=coverage[0]>0 ? agreement[0]/coverage[0]:0;
        const float progress=fminf(.25f,fmaxf(-.25f,contribution-s.contribution));
        const float score=.75f*contribution+.20f*(consistency-.5f)*participation+.05f*progress;
        const float residual=score-s.baseline;
        advantage=fminf(1.0f,fmaxf(-1.0f,residual/fmaxf(.05f,sqrtf(s.variance+1e-6f))));
        s.variance=.97f*s.variance+.03f*residual*residual;
        s.baseline=.97f*s.baseline+.03f*score;
        s.score=.95f*s.score+.05f*score;
        s.contribution=.95f*s.contribution+.05f*contribution;
        s.agreement=.95f*s.agreement+.05f*consistency;
        s.coverage=.95f*s.coverage+.05f*participation;
        s.progress=.95f*s.progress+.05f*progress;
        auto& credit=meme.credit[tile];credit.frame=decision->frameId;credit.advantage=advantage;
        credit.contribution=contribution;credit.agreement=consistency;credit.coverage=participation;
        credit.exploration=config.exploration;credit.valid=1;credit.exploring=cache.exploring;
        for(int d=0;d<BlockOutputs;++d) credit.noise[d]=cache.noise[d];
        s.windowSum+=score; s.windowSquares+=score*score; s.windowCoverage+=participation;
        ++s.windowExamples; ++s.examples;
        for(int k=0;k<VoteCount;++k) {
            const unsigned id=champions[k];
            if(id<NeuronCount && id/NeuronWidth/16*120+id%NeuronWidth/16==tile) s.protectedAction=1;
        }
        update=0;
        if(cache.exploring && ++cache.batch>=static_cast<unsigned>(config.trainEvery)) {
            const float step=static_cast<float>(++s.optimizerSteps);
            bias1=1-powf(.9f,step); bias2=1-powf(.999f,step); update=1;
            atomicAdd(reinterpret_cast<unsigned long long*>(&p->counters.actorUpdates),1ULL);
        }
    }
    __syncthreads();
    for(int at=tid;at<BlockWeights;at+=32) {
        if(cache.exploring) {
            const int output=at/BlockFeatures,feature=at%BlockFeatures;
            // Score-function of the cached *latent* Gaussian; no clipped-action noise reconstruction.
            const float gradient=-advantage*cache.noise[output]*cache.feature[feature]/config.exploration;
            cache.gradient[at]+=fminf(1.0f,fmaxf(-1.0f,gradient));
        }
        if(update) {
            const float gradient=cache.gradient[at]/config.trainEvery;
            s.first[at]=.9f*s.first[at]+.1f*gradient;
            s.second[at]=.999f*s.second[at]+.001f*gradient*gradient;
            const float decay=(at%BlockFeatures)==0 ? 1.0f:1-config.learningRate*.01f;
            const float next=s.weights[at]*decay-config.learningRate*(s.first[at]/bias1)/(sqrtf(s.second[at]/bias2)+1e-8f);
            s.weights[at]=fminf(1.5f,fmaxf(-1.5f,next)); cache.gradient[at]=0;
        }
    }
    __syncthreads();
    if(tid==0 && update) cache.batch=0;
}
__global__ void prepare(VabpnnGpuFrame frame,BlockPersistent* p,Pending* pending,BlockControl* controls,
                        const Message* oldMessages,Message* nextMessages,const float* membrane,
                        const signed char* spikes,const float* threshold,const DemoSample* demo,
                        const DecisionOutput* decision,DeviceMemeView meme) {
    const int tile=blockIdx.x,tid=threadIdx.x;
    const auto& config=p->config; auto& state=p->state[tile]; auto& cache=pending[tile];
    if(!validConfig(config) || !config.enabled) {
        if(tid==0) { controls[tile]=BlockControl{}; nextMessages[tile]=Message{}; cache.exploring=0; }
        return;
    }
    const unsigned pixel=spikePixel(tile,tid);
    float positive=spikes[pixel]>0,negative=spikes[pixel]<0,mem=membrane[pixel];
    for(int offset=16;offset;offset/=2) {
        positive+=__shfl_down_sync(0xffffffffu,positive,offset);
        negative+=__shfl_down_sync(0xffffffffu,negative,offset);
        mem+=__shfl_down_sync(0xffffffffu,mem,offset);
    }
    __shared__ float sums[3][8];
    if((tid&31)==0) { sums[0][tid/32]=positive; sums[1][tid/32]=negative; sums[2][tid/32]=mem; }
    __syncthreads();
    if(tid!=0) return;
    float feature[BlockFeatures]={1};
    if(decision && decision->valid) {
        feature[1]=(decision->posterior[0]+decision->posterior[1]+decision->posterior[2]+decision->posterior[3])*.25f;
        feature[2]=(decision->posterior[4]+decision->posterior[7]+decision->posterior[8]+decision->posterior[11])*.25f;
        feature[3]=(decision->posterior[5]+decision->posterior[6])*.5f;
        feature[4]=decision->posterior[12]; feature[5]=decision->selectedCount/13.0f;
        for(int k=0;k<VoteCount;++k) feature[6]+=fabsf(decision->normalizedVotes[k])/VoteCount;
    }
    for(int warp=0;warp<8;++warp) { feature[7]+=sums[0][warp]/256; feature[8]+=sums[1][warp]/256;
        feature[9]+=sums[2][warp]/256/fmaxf(.1f,*threshold); }
    feature[10]=state.agreement; feature[11]=state.coverage; feature[12]=state.score;
    float neighborThreshold=0,neighborPolarity=0,neighborScore=0;
    int neighbors=0; const int x=tile%120,y=tile/120;
    // Each iteration is bounded by a radius <=2; never reads a newly-written message.
    if(config.communication!=0) {
        for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx) {
            const int distance=abs(dx)+abs(dy),nx=x+dx,ny=y+dy;
            if(!distance || distance>config.radius || nx<0 || nx>=120 || ny<0 || ny>=64) continue;
            const auto& message=oldMessages[ny*120+nx];
            if(message.coverage<config.minimumCoverage) continue;
            neighborThreshold+=message.threshold; neighborPolarity+=message.polarity;
            neighborScore+=message.score; ++neighbors;
        }
    }
    if(neighbors) { neighborThreshold/=neighbors; neighborPolarity/=neighbors; neighborScore/=neighbors; }
    feature[13]=neighborThreshold; feature[14]=neighborPolarity; feature[15]=neighborScore;
    for(int d=0;d<BlockFeatures;++d) cache.feature[d]=isfinite(feature[d]) ? fminf(2.0f,fmaxf(-2.0f,feature[d])):0;
    cache.exploring=!frame.evaluationMode && demo && demo->active && !demo->modelControl && state.examples>=static_cast<std::uint64_t>(config.warmup);
    float latent[BlockOutputs]{};
    auto& memeAction=meme.actions[tile];float memeMagnitude=0;
    for(int k=0;k<MemeChannels;++k) {
        const float feature=meme.persistent->config.enabled ? meme.features[tile*MemeChannels+k]:0;
        memeAction.feature[k]=feature;memeMagnitude+=fabsf(feature);
    }
    memeAction.active=cache.exploring && memeMagnitude>1e-8f;
    for(int output=0;output<BlockOutputs;++output) {
        float mean=0;
        for(int d=0;d<BlockFeatures;++d) mean+=state.weights[output*BlockFeatures+d]*cache.feature[d];
        for(int k=0;k<MemeChannels;++k) mean+=meme.persistent->policy[tile].weights[output*MemeChannels+k]*memeAction.feature[k];
        cache.noise[output]=cache.exploring ? gaussian(static_cast<unsigned>(frame.frameId)^blockHash(tile*17+output*43+1)):0;
        latent[output]=tanhf(mean+config.exploration*cache.noise[output]);
    }
    const float influence=config.communication==1 ? .08f*neighborThreshold:
        config.communication==2 && neighbors ? -.04f*tanhf(neighborScore-state.score):0;
    BlockControl control;
    control.thresholdScale=fminf(1.2f,fmaxf(.8f,1+.2f*latent[0]+influence));
    control.polarityBias=fminf(.02f,fmaxf(-.02f,.02f*latent[1]+(config.communication==1 ? .002f*neighborPolarity:0)));
    control.memoryScale=1+.15f*latent[2]; control.learningScale=1+.2f*latent[3];
    if(state.cooldownUntil>p->counters.scoredFrames) control.learningScale=fmaxf(.8f,control.learningScale*.9f);
    controls[tile]=control;
    nextMessages[tile]={state.score,state.coverage,control.thresholdScale-1,control.polarityBias/.02f,
        control.memoryScale-1,control.learningScale-1};
    if(config.communication!=0 && neighbors)
        atomicAdd(reinterpret_cast<unsigned long long*>(&p->counters.broadcasts),static_cast<unsigned long long>(neighbors));
}
__global__ void stamp(BlockPersistent* p,Runtime* runtime,std::uint64_t frame,bool generated) {
    auto& count=p->counters; const auto& config=p->config;
    count.enabled=config.enabled; count.communication=config.communication; count.radius=config.radius;
    count.invalidConfig=!validConfig(config);
    runtime->appliedFrame=frame; runtime->ready=!count.invalidConfig && config.enabled;
    if(generated && runtime->ready) ++count.controlUpdates;
}
__global__ void beginRank(BlockPersistent* p,Runtime* runtime) {
    runtime->rank=runtime->feedback
        && p->counters.scoredFrames-runtime->lastRankScoredFrames>=static_cast<unsigned>(p->config.rankEvery);
    if(runtime->rank) {
        runtime->lastRankScoredFrames=p->counters.scoredFrames; ++p->counters.rankings;
        p->counters.eligible=p->counters.candidates=p->counters.punished=0;
    }
}
__global__ void rankKeys(BlockPersistent* p,const Runtime* runtime,float* keys,unsigned* ids,unsigned char* punish,DeviceMemeView meme) {
    if(!runtime->rank) return;
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile>=BlockCount) return;
    auto& state=p->state[tile]; const auto& config=p->config;
    ids[tile]=tile; keys[tile]=FLT_MAX; punish[tile]=0;
    const bool eligible=state.examples>=static_cast<unsigned>(config.minimumExamples)
        && state.windowExamples>=static_cast<unsigned>(config.minimumWindow)
        && state.windowCoverage/fmaxf(1.0f,state.windowExamples)>=config.minimumCoverage
        && !state.protectedAction && state.cooldownUntil<=p->counters.scoredFrames;
    if(!eligible) return;
    state.rankScore=state.windowSum/state.windowExamples;
    if(!isfinite(state.rankScore)) return;
    keys[tile]=state.rankScore;
    atomicAdd(&p->counters.eligible,1u);
    if(state.rankScore>state.bestScore+config.rankMargin) {
        state.bestScore=state.rankScore;
        for(int i=0;i<BlockWeights;++i) state.bestWeights[i]=state.weights[i];
        for(int i=0;i<MemeWeights;++i) meme.persistent->policy[tile].bestWeights[i]=meme.persistent->config.enabled ? meme.persistent->policy[tile].weights[i]:0;
    }
}
__global__ void sortPopulation(const Runtime* runtime,const float* keys,const unsigned* ids,
                               float* sortedKeys,unsigned* sortedIds) {
    if(!runtime->rank) return; // No actual sort on ordinary frames.
    using Sort=cub::BlockRadixSort<float,256,32,unsigned>;
    __shared__ Sort::TempStorage storage;
    float items[32]; unsigned index[32];
    for(int i=0;i<32;++i) {
        const int at=threadIdx.x*32+i;
        items[i]=at<BlockCount ? keys[at]:FLT_MAX;
        index[i]=at<BlockCount ? ids[at]:NoNeuron;
    }
    Sort(storage).Sort(items,index);
    for(int i=0;i<32;++i) {
        const int at=threadIdx.x*32+i;
        if(at<BlockCount) { sortedKeys[at]=items[i]; sortedIds[at]=index[i]; }
    }
}
__global__ void markRank(BlockPersistent* p,const Runtime* runtime,const float* sortedKeys,
                         const unsigned* sortedIds,unsigned char* punish) {
    if(!runtime->rank) return;
    const int rank=blockIdx.x*blockDim.x+threadIdx.x;
    if(rank>=BlockCount || sortedIds[rank]>=BlockCount) return;
    const int tile=sortedIds[rank]; auto& state=p->state[tile]; const auto& config=p->config;
    punish[tile]=0;
    if(!runtime->rank) return;
    const unsigned eligible=p->counters.eligible,budget=eligible/5;
    bool low=false;
    if(eligible>=5 && rank<static_cast<int>(budget)) {
        const float mean=state.rankScore;
        const float variance=fmaxf(0,state.windowSquares/fmaxf(1.0f,state.windowExamples)-mean*mean);
        const float uncertainty=2*sqrtf(variance/fmaxf(1.0f,state.windowExamples));
        // Equal/borderline scores must not be punished by a tile-index tiebreak.
        low=mean+uncertainty+config.rankMargin<sortedKeys[budget]
            && mean+uncertainty+config.rankMargin<sortedKeys[eligible/2]
            && mean<=state.previousRankScore+config.rankMargin;
    }
    if(rank<static_cast<int>(eligible)) {
        state.poorStreak=low ? state.poorStreak+1:0;
        state.previousRankScore=state.rankScore;
        if(low) atomicAdd(&p->counters.candidates,1u);
        if(low && state.poorStreak>=static_cast<unsigned>(config.poorWindows)) {
            punish[tile]=1; atomicAdd(&p->counters.punished,1u);
        }
    } else state.poorStreak=0;
}
__global__ void evolve(BlockPersistent* p,Pending* pending,const Runtime* runtime,const unsigned char* punish,DeviceMemeView meme) {
    const int tile=blockIdx.x,tid=threadIdx.x;
    if(!runtime->rank || !punish[tile]) return;
    auto& state=p->state[tile]; const auto& config=p->config;
    int teacher=tile;
    if(config.communication==1) {
        const int x=tile%120,y=tile/120;
        for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx) {
            const int distance=abs(dx)+abs(dy),nx=x+dx,ny=y+dy;
            if(!distance || distance>config.radius || nx<0 || nx>=120 || ny<0 || ny>=64) continue;
            const int other=ny*120+nx;
            if(p->state[other].bestScore>p->state[teacher].bestScore
                && p->state[other].examples>=static_cast<unsigned>(config.minimumExamples)) teacher=other;
        }
    }
    for(int at=tid;at<BlockWeights;at+=32) {
        const float base=p->state[teacher].bestScore>-1e10f ? p->state[teacher].bestWeights[at]:state.weights[at];
        const unsigned seed=blockHash(tile*BlockWeights+at)^static_cast<unsigned>(p->counters.scoredFrames);
        const float variation=config.mutation*(2*static_cast<float>(blockHash(seed)&0xffffu)/65535-1);
        state.weights[at]=fminf(1.5f,fmaxf(-1.5f,.9f*base+variation));
        state.first[at]=state.second[at]=pending[tile].gradient[at]=0;
    }
    if(meme.persistent->config.enabled) {
        auto& policy=meme.persistent->policy[tile];
        for(int at=tid;at<MemeWeights;at+=32) {
            const float base=p->state[teacher].bestScore>-1e10f
                ? meme.persistent->policy[teacher].bestWeights[at]:policy.weights[at];
            const unsigned seed=blockHash(tile*MemeWeights+at+7919)^static_cast<unsigned>(p->counters.scoredFrames);
            const float variation=config.mutation*(2*static_cast<float>(blockHash(seed)&0xffffu)/65535-1);
            policy.weights[at]=fminf(1.5f,fmaxf(-1.5f,.9f*base+variation));
            policy.first[at]=policy.second[at]=meme.actions[tile].gradient[at]=0;
        }
        if(tid==0) {policy.optimizerSteps=0;meme.actions[tile].batch=meme.actions[tile].active=0;}
    }
    if(tid==0) {
        state.optimizerSteps=0;
        ++state.mutations; state.poorStreak=0; state.cooldownUntil=p->counters.scoredFrames+config.cooldown;
        pending[tile].batch=0; pending[tile].exploring=0;
        atomicAdd(reinterpret_cast<unsigned long long*>(&p->counters.mutations),1ULL);
    }
}
__global__ void clearRankWindows(BlockPersistent* p,const Runtime* runtime) {
    if(!runtime->rank) return;
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile<BlockCount) {
        auto& state=p->state[tile];
        const auto& config=p->config;
        if(state.examples>=static_cast<unsigned>(config.minimumExamples)
            && state.windowExamples>=static_cast<unsigned>(config.minimumWindow)
            && state.windowCoverage/fmaxf(1.0f,state.windowExamples)>=config.minimumCoverage) {
            state.windowSum=state.windowSquares=state.windowCoverage=0;
            state.windowExamples=state.protectedAction=0;
        }
    }
}
__global__ void resetWindows(BlockPersistent* p,Runtime* runtime) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile==0) {
        p->counters.eligible=p->counters.candidates=p->counters.punished=0;
        runtime->lastRankScoredFrames=p->counters.scoredFrames;
    }
    if(tile<BlockCount) {
        auto& state=p->state[tile];
        state.windowSum=state.windowSquares=state.windowCoverage=0;
        state.windowExamples=state.poorStreak=state.protectedAction=0;
    }
}
}
struct VabpnnBlockModel {
    VabpnnMemeModel* memes=nullptr;
    vabpnn::BlockPersistent* persistent=nullptr;
    vabpnn::BlockControl* controls=nullptr;
    Pending* pending=nullptr;
    Message* messages[2]{};
    Runtime* runtime=nullptr;
    float *keys=nullptr,*sortedKeys=nullptr;
    unsigned *ids=nullptr,*sortedIds=nullptr;
    unsigned char* punish=nullptr;
    std::size_t bytes=0;
    cudaEvent_t initialized=nullptr;
    vabpnn::BlockConfig hostConfig{};
    std::uint64_t hostControls=0,hostFeedbacks=0;
    int current=0;
    bool forceControl=true;
    cudaStream_t initializationStream=nullptr;
};
namespace {
template<class T> void allocate(T*& pointer,std::size_t count,VabpnnBlockModel* model,const char* operation) {
    const auto bytes=count*sizeof(T);
    check(cudaMalloc(reinterpret_cast<void**>(&pointer),bytes),operation); model->bytes+=bytes;
    check(cudaMemsetAsync(pointer,0,bytes,model->initializationStream),"clear block allocation");
}
void ready(VabpnnBlockModel* model,cudaStream_t stream) {
    if(!model) throw std::invalid_argument("Block model missing");
    check(cudaStreamWaitEvent(stream,model->initialized,0),"wait for block initialization");
}
}
VabpnnBlockModel* vabpnnCreateBlockModel(cudaStream_t stream) {
    auto* model=new VabpnnBlockModel();
    model->initializationStream=stream;
    try {
        allocate(model->persistent,1,model,"allocate block policy, AdamW and scores");
        allocate(model->controls,vabpnn::BlockCount,model,"allocate block controls");
        allocate(model->pending,vabpnn::BlockCount,model,"allocate block action credit cache");
        for(auto& messages:model->messages) allocate(messages,vabpnn::BlockCount,model,"allocate bounded message buffer");
        allocate(model->runtime,1,model,"allocate block runtime");
        allocate(model->keys,vabpnn::BlockCount,model,"allocate block rank keys");
        allocate(model->sortedKeys,vabpnn::BlockCount,model,"allocate sorted block keys");
        allocate(model->ids,vabpnn::BlockCount,model,"allocate block IDs");
        allocate(model->sortedIds,vabpnn::BlockCount,model,"allocate sorted block IDs");
        allocate(model->punish,vabpnn::BlockCount,model,"allocate penalty candidates");
        model->memes=vabpnnCreateMemeModel(stream);model->bytes+=vabpnnMemeModelBytes(model->memes);
        initialize<<<(vabpnn::BlockCount+255)/256,256,0,stream>>>(model->persistent,model->controls);
        check(cudaEventCreateWithFlags(&model->initialized,cudaEventDisableTiming),"create block model event");
        check(cudaEventRecord(model->initialized,stream),"record block model event");
        check(cudaGetLastError(),"initialize bounded block controllers");
        return model;
    } catch(...) { vabpnnDestroyBlockModel(model); throw; }
}
void vabpnnDestroyBlockModel(VabpnnBlockModel* model) noexcept {
    if(!model) return;
    if(model->initialized) cudaEventSynchronize(model->initialized);
    void* allocations[]={model->persistent,model->controls,model->pending,model->messages[0],model->messages[1],
        model->runtime,model->keys,model->sortedKeys,model->ids,model->sortedIds,model->punish};
    vabpnnDestroyMemeModel(model->memes);
    for(void* allocation:allocations) if(allocation) cudaFree(allocation);
    if(model->initialized) cudaEventDestroy(model->initialized);
    delete model;
}
VabpnnMemeModel* vabpnnBlockMemeModel(const VabpnnBlockModel* model) {return model ? model->memes:nullptr;}
std::uint64_t vabpnnBlockModelBytes(const VabpnnBlockModel* model) noexcept { return model ? model->bytes:0; }
vabpnn::DeviceBlockView vabpnnBlockModelView(const VabpnnBlockModel* model) {
    if(!model) throw std::invalid_argument("Block model missing");
    return {&model->persistent->config,&model->persistent->counters,model->persistent->state,model->controls};
}
vabpnn::BlockPersistent* vabpnnMutableBlockPersistent(VabpnnBlockModel* model,cudaStream_t stream) {
    ready(model,stream); return model->persistent;
}
void vabpnnConfigureBlockModel(VabpnnBlockModel* model,const vabpnn::BlockConfig& config,cudaStream_t stream) {
    if(!validConfig(config)) throw std::invalid_argument("Invalid block controller configuration");
    ready(model,stream); model->hostConfig=config;
    check(cudaMemcpyAsync(&model->persistent->config,&model->hostConfig,sizeof(config),cudaMemcpyHostToDevice,stream),"set bounded block configuration");
    vabpnnResetBlockHistory(model,stream);
}
void vabpnnConfigureBlockMode(VabpnnBlockModel* model,int enabled,int communication,cudaStream_t stream) {
    auto config=model->hostConfig; if(enabled>=0) config.enabled=enabled; if(communication>=0) config.communication=communication;
    vabpnnConfigureBlockModel(model,config,stream);
}
void vabpnnPrepareBlockControllers(const VabpnnGpuFrame& frame) {
    if(!frame.spikeModel) return;
    auto* model=vabpnnSpikeBlockModel(frame.spikeModel); ready(model,frame.stream);
    const bool generated=model->hostControls++%vabpnn::BlockControlEvery==0 || model->forceControl;
    model->forceControl=false;
    vabpnnPrepareMemes(frame,generated);
    if(generated) {
        const auto spike=vabpnnSpikeModelView(frame.spikeModel);
        const auto* decision=frame.decisionModel ? vabpnnDecisionModelView(frame.decisionModel).output:nullptr;
        const auto* demo=frame.onlineModel ? vabpnnOnlineDemoSample(frame.onlineModel):nullptr;
        const int next=1-model->current;
        prepare<<<vabpnn::BlockCount,256,0,frame.stream>>>(frame,model->persistent,model->pending,model->controls,
            model->messages[model->current],model->messages[next],spike.membrane,spike.spikes,&spike.config->threshold,demo,decision,vabpnnMemeModelView(model->memes));
        model->current=next;
    }
    stamp<<<1,1,0,frame.stream>>>(model->persistent,model->runtime,frame.frameId,generated);
    check(cudaGetLastError(),"prepare contextual block controls before pixel learning");
}
void vabpnnLearnBlockControllers(const VabpnnGpuFrame& frame,const int* valid,const std::uint32_t* label) {
    if(frame.evaluationMode || !frame.spikeModel || !frame.neuronModel || !frame.decisionModel) return;
    auto* model=vabpnnSpikeBlockModel(frame.spikeModel); ready(model,frame.stream);
    const auto decision=vabpnnDecisionModelView(frame.decisionModel);
    beginFeedback<<<1,1,0,frame.stream>>>(model->persistent,model->runtime,decision.output,valid);
    learn<<<vabpnn::BlockCount,32,0,frame.stream>>>(model->persistent,model->pending,model->runtime,
        vabpnnNeuronModelView(frame.neuronModel).blockVotes,decision.output,decision.parameters,label,
        vabpnnSpikeModelView(frame.spikeModel).championIds,vabpnnMemeModelView(model->memes));
    vabpnnLearnMemes(frame,valid,label);
    vabpnnRankBlockControllers(model,frame.stream);
    check(cudaGetLastError(),"learn cached block actions from previous true-label feedback");
}
void vabpnnRankBlockControllers(VabpnnBlockModel* model,cudaStream_t stream) {
    ready(model,stream);
    beginRank<<<1,1,0,stream>>>(model->persistent,model->runtime);
    rankKeys<<<(vabpnn::BlockCount+255)/256,256,0,stream>>>(model->persistent,model->runtime,model->keys,model->ids,model->punish,vabpnnMemeModelView(model->memes));
    sortPopulation<<<1,256,0,stream>>>(model->runtime,model->keys,model->ids,model->sortedKeys,model->sortedIds);
    markRank<<<(vabpnn::BlockCount+255)/256,256,0,stream>>>(model->persistent,model->runtime,model->sortedKeys,model->sortedIds,model->punish);
    evolve<<<vabpnn::BlockCount,32,0,stream>>>(model->persistent,model->pending,model->runtime,model->punish,vabpnnMemeModelView(model->memes));
    clearRankWindows<<<(vabpnn::BlockCount+255)/256,256,0,stream>>>(model->persistent,model->runtime);
    check(cudaGetLastError(),"controlled bottom-fifth policy mutation");
}
void vabpnnResetBlockHistory(VabpnnBlockModel* model,cudaStream_t stream) {
    ready(model,stream);
    check(cudaMemsetAsync(model->pending,0,vabpnn::BlockCount*sizeof(Pending),stream),"clear action/feature/noise/partial-gradient history");
    for(auto message:model->messages)
        check(cudaMemsetAsync(message,0,vabpnn::BlockCount*sizeof(Message),stream),"clear transient neighbor broadcasts");
    check(cudaMemsetAsync(model->runtime,0,sizeof(Runtime),stream),"clear block frame credit");
    resetWindows<<<(vabpnn::BlockCount+255)/256,256,0,stream>>>(model->persistent,model->runtime);
    neutral<<<(vabpnn::BlockCount+255)/256,256,0,stream>>>(model->controls);
    vabpnnResetMemeHistory(model->memes,stream);
    model->current=0; model->hostControls=model->hostFeedbacks=0; model->forceControl=true;
    check(cudaGetLastError(),"reset transient block history while preserving learned policies");
}
void vabpnnResumeBlockModel(VabpnnBlockModel* model,cudaStream_t stream) {
    ready(model,stream);
    // Restore/resize lifecycle only, never a per-frame readback.
    check(cudaMemcpy(&model->hostConfig,&model->persistent->config,sizeof(model->hostConfig),cudaMemcpyDeviceToHost),"restore block scheduler configuration");
    if(!validConfig(model->hostConfig)) throw std::runtime_error("Invalid saved block controller configuration");
    vabpnnResetBlockHistory(model,stream);
    vabpnnResumeMemeModel(model->memes,stream);
}
vabpnn::ModelPart vabpnnBlockCheckpointPart(VabpnnBlockModel* model) {
    return {model->persistent,sizeof(vabpnn::BlockPersistent)};
}
