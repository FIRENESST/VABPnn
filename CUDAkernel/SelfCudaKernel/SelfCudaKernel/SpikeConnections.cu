#include "SpikeConnections.h"
#include <cub/device/device_scan.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
namespace {
using namespace vabpnn;
constexpr int NearStride=SpikeAnchorLimit-1+SpikeNearCount;
struct Clock { std::uint64_t lastNanos; float dt,decay; int width,height,valid; };
__constant__ int2 nearby[SpikeNearCount];
void check(cudaError_t code,const char* name) {
    if(code!=cudaSuccess) throw std::runtime_error(std::string(name)+": "+cudaGetErrorString(code));
}
__host__ __device__ unsigned hash(unsigned x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
__device__ float importance(const SpikePixelStatistics& s) {
    const float sum=s.positive+s.negative+s.zero;
    float h=0,squares=0;
    const float values[3]={s.positive,s.negative,s.zero};
    for(int k=0;k<3;++k) {
        const float p=sum>0 ? values[k]/sum:0;
        if(p>0) h-=p*logf(p);
        squares+=p*p;
    }
    const float shannon=h*0.9102392266f,simpson=sum>0 ? (1-squares)*1.5f:0;
    return fmaxf(0,s.crossEntropy)*(0.5f+0.25f*shannon+0.25f*simpson)+0.25f*fmaxf(0,s.hit);
}
__device__ unsigned tileOf(unsigned pixel) { return pixel/NeuronWidth/16*120+pixel%NeuronWidth/16; }
__device__ unsigned localOf(unsigned pixel) { return pixel/NeuronWidth%16*16+pixel%NeuronWidth%16; }
__global__ void initializeConfig(SpikeConfig* config) { *config=SpikeConfig{}; }
__global__ void localOutgoing(std::uint16_t* out,const float* scores,
                              const SpikeConfig* config,unsigned epoch,bool initial) {
    const int tile=blockIdx.x,local=threadIdx.x;
    unsigned seed=hash(config->seed ^ spikePixel(tile,local) ^ hash(epoch+1));
    const int degree=initial ? 4:static_cast<int>(seed%5);
    int targets[4]={-1,-1,-1,-1};
    for(int slot=0;slot<4;++slot) {
        unsigned short encoded=0;
        if(slot<degree) {
            seed=hash(seed+slot+37);
            int target=static_cast<int>(seed&255);
            const int alternate=static_cast<int>((seed>>8)&255);
            if(!initial && scores[spikePixel(tile,alternate)]>scores[spikePixel(tile,target)]) target=alternate;
            // Bounded deterministic rejection: at most four exclusions in a 256-node tile.
            for(int attempt=0;attempt<6;++attempt) {
                bool duplicate=target==local;
                for(int k=0;k<slot;++k) duplicate|=targets[k]==target;
                if(!duplicate) break;
                target=(target+1)&255;
            }
            targets[slot]=target;
            encoded=static_cast<unsigned short>(target|((seed>>16)&3)<<8|0x400u);
        }
        out[(tile*SpikeTileSize+local)*4+slot]=encoded;
    }
}
__global__ void localCsr(const std::uint16_t* outgoing,std::uint16_t* rows,
                         std::uint16_t* columns,SpikeCounters* counters) {
    using BlockScan=cub::BlockScan<unsigned,256>;
    __shared__ BlockScan::TempStorage scanStorage;
    __shared__ unsigned counts[256],cursor[256];
    const int tile=blockIdx.x,source=threadIdx.x;
    counts[source]=0; __syncthreads();
    for(int k=0;k<4;++k) {
        const auto edge=outgoing[(tile*256+source)*4+k];
        if(edge&0x400u) atomicAdd(&counts[edge&255],1u);
    }
    __syncthreads();
    unsigned start,total;
    BlockScan(scanStorage).ExclusiveSum(counts[source],start,total);
    rows[tile*257+source]=static_cast<unsigned short>(start);
    cursor[source]=start;
    if(source==255) {
        rows[tile*257+256]=static_cast<unsigned short>(total);
        atomicAdd(reinterpret_cast<unsigned long long*>(&counters->localEdges),static_cast<unsigned long long>(total));
    }
    __syncthreads();
    for(int k=0;k<4;++k) {
        const auto edge=outgoing[(tile*256+source)*4+k];
        if(edge&0x400u) {
            const auto position=atomicAdd(&cursor[edge&255],1u);
            columns[tile*1024+position]=static_cast<unsigned short>(source|(edge&0x300u));
        }
    }
}
__global__ void resetLocalCount(SpikeCounters* counters) { counters->localEdges=0; }
__global__ void refreshScores(const SpikePixelStatistics* stats,float* scores) {
    const unsigned pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel<NeuronCount) scores[pixel]=importance(stats[pixel]);
}
__global__ void candidates(const float* scores,float* keys,unsigned* ids) {
    __shared__ float values[256];
    __shared__ unsigned index[256];
    const int tile=blockIdx.x,tid=threadIdx.x; const auto pixel=spikePixel(tile,tid);
    values[tid]=scores[pixel]; index[tid]=pixel; __syncthreads();
    for(int step=128;step;step/=2) {
        if(tid<step && (values[tid+step]>values[tid]
            || (values[tid+step]==values[tid] && index[tid+step]<index[tid]))) {
            values[tid]=values[tid+step]; index[tid]=index[tid+step];
        }
        __syncthreads();
    }
    if(tid==0) { keys[tile]=values[0]; ids[tile]=index[0]; }
}
__global__ void reduceChampions(const unsigned long long* blockChampions,unsigned* champions) {
    const int action=blockIdx.x,tid=threadIdx.x;
    unsigned long long best=0;
    for(int b=tid;b<SpikeTiles;b+=256) best=max(best,blockChampions[b*VoteCount+action]);
    __shared__ unsigned long long values[256]; values[tid]=best; __syncthreads();
    for(int step=128;step;step/=2) {
        if(tid<step) values[tid]=max(values[tid],values[tid+step]);
        __syncthreads();
    }
    if(tid==0) champions[action]=values[0] ? NoNeuron-static_cast<unsigned>(values[0]):NoNeuron;
}
__global__ void selectAnchors(const float* keys,const unsigned* ids,const unsigned* champions,
                              const SpikeConfig* config,const float* scores,
                              unsigned* anchors,SpikeCounters* counters) {
    int count=0;
    const int limit=max(0,min(SpikeAnchorLimit,config->maxAnchors));
    for(int k=0;k<VoteCount && count<limit;++k) {
        const unsigned id=champions[k];
        if(id>=NeuronCount || scores[id]<=config->minimumScore) continue;
        bool duplicate=false; for(int j=0;j<count;++j) duplicate|=anchors[j]==id;
        if(!duplicate) anchors[count++]=id;
    }
    for(int k=0;k<SpikeTiles && count<limit;++k) {
        if(!(keys[k]>config->minimumScore)) break;
        const unsigned id=ids[k];
        bool duplicate=false; for(int j=0;j<count;++j) duplicate|=anchors[j]==id;
        if(!duplicate) anchors[count++]=id;
    }
    for(int j=count;j<SpikeAnchorLimit;++j) anchors[j]=NoNeuron;
    counters->anchors=count; ++counters->topologyUpdates;
}
__global__ void indexAnchors(const unsigned* anchors,unsigned short* index,const SpikeCounters* counters) {
    const unsigned k=threadIdx.x;
    if(k<counters->anchors) index[anchors[k]]=static_cast<unsigned short>(k);
}
__device__ bool longTarget(unsigned sourceSlot,int candidate,const unsigned* anchors,
                           const unsigned short* membership,const SpikeCounters* counters,unsigned& target) {
    if(sourceSlot>=counters->anchors) return false;
    if(candidate<SpikeAnchorLimit-1) {
        const unsigned targetSlot=candidate+(candidate>=static_cast<int>(sourceSlot));
        if(targetSlot>=counters->anchors) return false;
        target=anchors[targetSlot]; return true;
    }
    const auto source=anchors[sourceSlot];
    const auto d=nearby[candidate-(SpikeAnchorLimit-1)];
    const int x=source%NeuronWidth+d.x,y=source/NeuronWidth+d.y;
    if(x<0 || x>=NeuronWidth || y<0 || y>=NeuronHeight) return false;
    target=y*NeuronWidth+x;
    return membership[target]==0xffffu; // Anchor pairs already have a clique edge.
}
__global__ void longDegree(unsigned* degrees,const unsigned* anchors,const unsigned short* membership,
                           const SpikeCounters* counters) {
    const int edge=blockIdx.x*blockDim.x+threadIdx.x;
    if(edge>=SpikeLongCapacity) return;
    unsigned target;
    if(longTarget(edge/NearStride,edge%NearStride,anchors,membership,counters,target)) atomicAdd(&degrees[target],1u);
}
__global__ void longScatter(const unsigned* rows,unsigned* cursor,unsigned* columns,const unsigned* anchors,
                            const unsigned short* membership,const SpikeCounters* counters,const SpikeConfig* config) {
    const int edge=blockIdx.x*blockDim.x+threadIdx.x;
    if(edge>=SpikeLongCapacity) return;
    const unsigned sourceSlot=edge/NearStride; unsigned target;
    if(!longTarget(sourceSlot,edge%NearStride,anchors,membership,counters,target)) return;
    const unsigned source=anchors[sourceSlot];
    const auto at=rows[target]+atomicAdd(&cursor[target],1u);
    const unsigned kind=hash(source ^ hash(target) ^ static_cast<unsigned>(counters->topologyUpdates) ^ config->seed)&3;
    columns[at]=source|(kind<<21);
}
__global__ void longCount(const unsigned* rows,SpikeCounters* counters) { counters->longEdges=rows[NeuronCount]; }
__global__ void beginTick(Clock* clock,SpikeCounters* counters,const SpikeConfig* c,std::uint64_t nanos,int width,int height) {
    const bool finite=isfinite(c->membraneTau) && c->membraneTau>0 && isfinite(c->threshold)
        && c->threshold>0 && isfinite(c->minimumThreshold) && c->minimumThreshold>0
        && isfinite(c->maximumThreshold) && c->maximumThreshold>=c->minimumThreshold
        && isfinite(c->stimulationGain) && c->stimulationGain>=0 && isfinite(c->polarityGain) && c->polarityGain>=0
        && isfinite(c->negativeMargin) && c->negativeMargin>=0 && isfinite(c->refractorySeconds) && c->refractorySeconds>=0
        && isfinite(c->statisticsDecay) && c->statisticsDecay>=0 && c->statisticsDecay<1
        && isfinite(c->maxGapSeconds) && c->maxGapSeconds>0 && c->maxAnchors>=0 && c->maxAnchors<=SpikeAnchorLimit
        && isfinite(c->minimumScore) && c->minimumScore>=0;
    const float dt=clock->lastNanos && nanos>clock->lastNanos ? static_cast<float>((nanos-clock->lastNanos)*1e-9):0;
    clock->valid=finite;
    clock->dt=dt>0 && dt<=c->maxGapSeconds && width==clock->width && height==clock->height ? dt:0;
    clock->decay=finite && clock->dt>0 ? expf(-clock->dt/c->membraneTau):0;
    clock->lastNanos=nanos; clock->width=width; clock->height=height;
    counters->invalidConfig=!finite; ++counters->predictions;
}
__global__ void advance(VabpnnGpuFrame frame,const SpikeConfig* config,const Clock* clock,
                        float* membrane,float* refractory,const signed char* previous,signed char* next,
                        const unsigned short* localRows,const unsigned short* localColumns,
                        const unsigned* longRows,const unsigned* longColumns,const BlockControl* controls) {
    __shared__ signed char tileSignals[256];
    __shared__ float blockThreshold,blockPolarity,blockDecay;
    const int tile=blockIdx.x,tid=threadIdx.x; const unsigned pixel=spikePixel(tile,tid);
    tileSignals[tid]=previous[pixel];
    if(tid==0) {
        const auto control=controls[tile];
        blockThreshold=config->threshold*control.thresholdScale; blockPolarity=control.polarityBias;
        // One exp per tile, preserving seconds-based tau. Neutral controls retain the old decay bit for bit.
        blockDecay=control.memoryScale==1 ? clock->decay:expf(-clock->dt/(config->membraneTau*control.memoryScale));
    }
    __syncthreads();
    if(!clock->valid) { next[pixel]=0; membrane[pixel]=refractory[pixel]=0; return; }
    if(!config->enabled) { next[pixel]=1; membrane[pixel]=refractory[pixel]=0; return; }
    if(clock->dt==0) { next[pixel]=0; membrane[pixel]=refractory[pixel]=0; return; }
    float effect[4]={};
    const unsigned localBegin=localRows[tile*257+tid],localEnd=localRows[tile*257+tid+1];
    for(unsigned e=localBegin;e<localEnd;++e) {
        const auto edge=localColumns[tile*1024+e];
        effect[(edge>>8)&3]+=abs(static_cast<int>(tileSignals[edge&255]));
    }
    const unsigned globalBegin=longRows[pixel],globalEnd=longRows[pixel+1];
    for(unsigned e=globalBegin;e<globalEnd;++e) {
        const auto edge=longColumns[e]; effect[(edge>>21)&3]+=abs(static_cast<int>(previous[edge&SpikeSourceMask]));
    }
    const float normalization=1.0f/fmaxf(1.0f,static_cast<float>(localEnd-localBegin+globalEnd-globalBegin));
    const float stimulation=(effect[0]-effect[1])*normalization*config->stimulationGain;
    const float positiveThreshold=fminf(config->maximumThreshold,fmaxf(config->minimumThreshold,
        blockThreshold-stimulation-effect[2]*normalization*config->polarityGain-blockPolarity));
    const float negativeThreshold=fminf(config->maximumThreshold,fmaxf(config->minimumThreshold,
        blockThreshold+config->negativeMargin-stimulation-effect[3]*normalization*config->polarityGain+blockPolarity));
    const int x=pixel%NeuronWidth,y=pixel/NeuronWidth;
    const int sx=static_cast<std::size_t>(2*x+1)*frame.width/(2*NeuronWidth);
    const int sy=static_cast<std::size_t>(2*y+1)*frame.height/(2*NeuronHeight);
    const auto rgba=frame.rgba+static_cast<std::size_t>(frame.height-1-sy)*frame.rgbaPitch+sx*4;
    const float stimulus=(0.2126f*rgba[0]+0.7152f*rgba[1]+0.0722f*rgba[2])/255.0f;
    float wait=fmaxf(0,refractory[pixel]-clock->dt),v=0; signed char signal=0;
    if(wait==0) {
        v=blockDecay*membrane[pixel]+(1-blockDecay)*stimulus;
        if(v>=fminf(positiveThreshold,negativeThreshold)) {
            signal=negativeThreshold<positiveThreshold ? -1:1; v=0; wait=config->refractorySeconds;
        }
    }
    membrane[pixel]=v; refractory[pixel]=wait; next[pixel]=signal;
}
__global__ void recordVotes(const unsigned* packed,SpikePixelStatistics* statistics,const SpikeConfig* config,float* scores) {
    const unsigned pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=NeuronCount) return;
    if(!isfinite(config->statisticsDecay) || config->statisticsDecay<0 || config->statisticsDecay>=1) return;
    unsigned values=packed[pixel]; int positive=0,negative=0,zero=0;
    for(int k=0;k<VoteCount;++k) {
        const unsigned v=(values>>(k*2))&3;
        positive+=v==2; negative+=v==0; zero+=v==1;
    }
    auto& s=statistics[pixel]; const float decay=config->statisticsDecay,alpha=1-decay;
    s.positive=decay*s.positive+alpha*positive/VoteCount;
    s.negative=decay*s.negative+alpha*negative/VoteCount;
    s.zero=decay*s.zero+alpha*zero/VoteCount;
    scores[pixel]=importance(s);
}
__device__ float crossEntropyGain(float probability,float y,float delta) {
    // Stable leave-one-neuron-out change for the direct raw-vote evidence term.
    // For tiny delta the second-order formula avoids catastrophic cancellation.
    if(fabsf(delta)<0.001f)
        return (y-probability)*delta+0.5f*probability*(1-probability)*delta*delta;
    return log1pf(probability*expm1f(-delta))+y*delta;
}
__global__ void contributions(const unsigned* packed,SpikePixelStatistics* statistics,const SpikeConfig* config,
                              const DecisionOutput* decision,const DecisionParameters* parameters,
                              const int* valid,const unsigned* labels,unsigned long long* blockChampions,
                              SpikeCounters* counters,float* scores) {
    if(!isfinite(config->statisticsDecay) || config->statisticsDecay<0 || config->statisticsDecay>=1) return;
    if(!*valid || !decision->valid) return; // Uniform block condition; no barrier divergence.
    const int tile=blockIdx.x,tid=threadIdx.x; const unsigned pixel=spikePixel(tile,tid);
    const unsigned encoded=packed[pixel],label=*labels;
    const float inverse=1.0f/static_cast<float>(NeuronCount);
    float ce=0,hit=0;
    __shared__ unsigned long long warpBest[13][8];
    for(int k=0;k<VoteCount;++k) {
        const int vote=static_cast<int>((encoded>>(2*k))&3)-1;
        const float y=(label>>k)&1u;
        const float delta=parameters->rawVoteWeight[k]*vote*inverse;
        const float gain=crossEntropyGain(decision->posterior[k],y,delta)*NeuronCount;
        ce+=gain;
        const bool correctHit=(decision->selectedMask & label & (1u<<k)) && vote>0 && gain>0;
        const float value=correctHit ? gain:0;
        if(correctHit) hit=fmaxf(hit,value);
        unsigned long long best=value>0 ? (static_cast<unsigned long long>(__float_as_uint(value))<<32)
            |(NoNeuron-pixel):0;
        for(int offset=16;offset;offset/=2) best=max(best,__shfl_down_sync(0xffffffffu,best,offset));
        if((tid&31)==0) warpBest[k][tid/32]=best;
    }
    auto& s=statistics[pixel]; const float a=1-config->statisticsDecay;
    s.crossEntropy=config->statisticsDecay*s.crossEntropy+a*ce/VoteCount;
    s.hit=config->statisticsDecay*s.hit+a*hit;
    scores[pixel]=importance(s);
    __syncthreads();
    if(tid<VoteCount) {
        unsigned long long best=0;
        for(int w=0;w<8;++w) best=max(best,warpBest[tid][w]);
        blockChampions[tile*VoteCount+tid]=best;
    }
    if(tile==0 && tid==0) ++counters->contributionUpdates;
}
}
struct VabpnnSpikeModel {
    VabpnnBlockModel* blocks=nullptr;
    vabpnn::SpikeConfig* config=nullptr;
    vabpnn::SpikePixelStatistics* statistics=nullptr;
    vabpnn::SpikeCounters* counters=nullptr;
    Clock* clock=nullptr;
    float* scores=nullptr;
    float* membrane=nullptr;
    float* refractory=nullptr;
    signed char* signals[2]{};
    unsigned* packedVotes=nullptr;
    unsigned short *outgoing=nullptr,*localRows=nullptr,*localColumns=nullptr,*membership=nullptr;
    unsigned *longRows=nullptr,*cursor=nullptr,*longColumns=nullptr,*anchors=nullptr,*champions=nullptr;
    float *keys=nullptr,*sortedKeys=nullptr;
    unsigned *ids=nullptr,*sortedIds=nullptr;
    unsigned long long* blockChampions=nullptr;
    void* temporary=nullptr;
    std::size_t scanBytes=0,sortBytes=0,bytes=0;
    std::uint64_t hostPredictions=0;
    int current=0;
    cudaEvent_t initialized=nullptr;
};
namespace {
template<class T> void allocate(T*& pointer,std::size_t count,VabpnnSpikeModel* m,const char* name) {
    const auto bytes=count*sizeof(T); check(cudaMalloc(reinterpret_cast<void**>(&pointer),bytes),name); m->bytes+=bytes;
}
void ready(VabpnnSpikeModel* m,cudaStream_t stream) {
    if(!m) throw std::invalid_argument("Spike model missing");
    check(cudaStreamWaitEvent(stream,m->initialized,0),"wait for spike initialization");
}
void buildLong(VabpnnSpikeModel* m,cudaStream_t stream) {
    check(cudaMemsetAsync(m->membership,0xff,NeuronCount*sizeof(unsigned short),stream),"clear anchor membership");
    indexAnchors<<<1,256,0,stream>>>(m->anchors,m->membership,m->counters);
    check(cudaMemsetAsync(m->cursor,0,(NeuronCount+1)*sizeof(unsigned),stream),"clear CSR degrees");
    longDegree<<<(SpikeLongCapacity+255)/256,256,0,stream>>>(m->cursor,m->anchors,m->membership,m->counters);
    check(cub::DeviceScan::ExclusiveSum(m->temporary,m->scanBytes,m->cursor,m->longRows,NeuronCount+1,stream),"scan long CSR");
    check(cudaMemsetAsync(m->cursor,0,(NeuronCount+1)*sizeof(unsigned),stream),"clear CSR cursors");
    longScatter<<<(SpikeLongCapacity+255)/256,256,0,stream>>>(m->longRows,m->cursor,m->longColumns,
        m->anchors,m->membership,m->counters,m->config);
    longCount<<<1,1,0,stream>>>(m->longRows,m->counters);
}
void rebuildTopology(VabpnnSpikeModel* m,cudaStream_t stream,unsigned epoch) {
    refreshScores<<<(NeuronCount+255)/256,256,0,stream>>>(m->statistics,m->scores);
    localOutgoing<<<SpikeTiles,256,0,stream>>>(m->outgoing,m->scores,m->config,epoch,false);
    vabpnnRebuildSpikeLocalCsr(m,stream);
    candidates<<<SpikeTiles,256,0,stream>>>(m->scores,m->keys,m->ids);
    reduceChampions<<<VoteCount,256,0,stream>>>(m->blockChampions,m->champions);
    check(cub::DeviceRadixSort::SortPairsDescending(m->temporary,m->sortBytes,m->keys,m->sortedKeys,
        m->ids,m->sortedIds,SpikeTiles,0,32,stream),"sort high-contribution candidates");
    selectAnchors<<<1,1,0,stream>>>(m->sortedKeys,m->sortedIds,m->champions,m->config,m->scores,m->anchors,m->counters);
    buildLong(m,stream);
    check(cudaGetLastError(),"rebuild bounded spike topology");
}
}
VabpnnSpikeModel* vabpnnCreateSpikeModel(cudaStream_t stream) {
    auto* m=new VabpnnSpikeModel();
    try {
        allocate(m->config,1,m,"allocate spike config"); allocate(m->counters,1,m,"allocate spike counters");
        allocate(m->clock,1,m,"allocate spike clock"); allocate(m->statistics,NeuronCount,m,"allocate contribution statistics");
        allocate(m->scores,NeuronCount,m,"allocate cached contribution scores");
        allocate(m->membrane,NeuronCount,m,"allocate membrane"); allocate(m->refractory,NeuronCount,m,"allocate refractory");
        allocate(m->signals[0],NeuronCount,m,"allocate synchronous signals"); allocate(m->signals[1],NeuronCount,m,"allocate synchronous signals");
        allocate(m->packedVotes,NeuronCount,m,"allocate packed neuron votes");
        allocate(m->outgoing,NeuronCount*4,m,"allocate bounded outgoing");
        allocate(m->localRows,SpikeTiles*257,m,"allocate local CSR rows");
        allocate(m->localColumns,SpikeTiles*1024,m,"allocate local CSR columns");
        allocate(m->membership,NeuronCount,m,"allocate anchor membership");
        allocate(m->longRows,NeuronCount+1,m,"allocate long CSR rows");
        allocate(m->cursor,NeuronCount+1,m,"allocate long CSR workspace");
        allocate(m->longColumns,SpikeLongCapacity,m,"allocate bounded long edges");
        allocate(m->anchors,SpikeAnchorLimit,m,"allocate anchor IDs"); allocate(m->champions,VoteCount,m,"allocate operation champions");
        allocate(m->keys,SpikeTiles,m,"allocate candidate keys"); allocate(m->sortedKeys,SpikeTiles,m,"allocate sorted keys");
        allocate(m->ids,SpikeTiles,m,"allocate candidate IDs"); allocate(m->sortedIds,SpikeTiles,m,"allocate sorted IDs");
        allocate(m->blockChampions,SpikeTiles*VoteCount,m,"allocate block champions");
        check(cub::DeviceScan::ExclusiveSum(nullptr,m->scanBytes,m->cursor,m->longRows,NeuronCount+1,stream),"size CSR scan workspace");
        check(cub::DeviceRadixSort::SortPairsDescending(nullptr,m->sortBytes,m->keys,m->sortedKeys,m->ids,m->sortedIds,
            SpikeTiles,0,32,stream),"size sorting workspace");
        const auto scratch=std::max(m->scanBytes,m->sortBytes);
        check(cudaMalloc(&m->temporary,scratch),"allocate reusable CUB scratch"); m->bytes+=scratch;
        for(auto item:{vabpnn::ModelPart{m->statistics,NeuronCount*sizeof(SpikePixelStatistics)},
            {m->scores,NeuronCount*sizeof(float)},{m->membrane,NeuronCount*sizeof(float)},{m->refractory,NeuronCount*sizeof(float)},
            {m->signals[0],NeuronCount},{m->signals[1],NeuronCount},{m->packedVotes,NeuronCount*sizeof(unsigned)},
            {m->clock,sizeof(Clock)},{m->counters,sizeof(SpikeCounters)},{m->blockChampions,SpikeTiles*VoteCount*sizeof(unsigned long long)},
            {m->longRows,(NeuronCount+1)*sizeof(unsigned)}})
            check(cudaMemsetAsync(item.device,0,item.bytes,stream),"zero spike state");
        check(cudaMemsetAsync(m->membership,0xff,NeuronCount*sizeof(unsigned short),stream),"initialize membership");
        check(cudaMemsetAsync(m->anchors,0xff,SpikeAnchorLimit*sizeof(unsigned),stream),"initialize anchors");
        check(cudaMemsetAsync(m->champions,0xff,VoteCount*sizeof(unsigned),stream),"initialize champions");
        check(cudaMemsetAsync(m->packedVotes,0x55,NeuronCount*sizeof(unsigned),stream),"initialize neutral votes");
        initializeConfig<<<1,1,0,stream>>>(m->config);
        std::array<int2,SpikeNearCount> offsets; int at=0;
        for(int y=-SpikeRadius;y<=SpikeRadius;++y) for(int x=-SpikeRadius;x<=SpikeRadius;++x)
            if(std::abs(x)+std::abs(y)>0 && std::abs(x)+std::abs(y)<=SpikeRadius) offsets[at++]={x,y};
        check(cudaMemcpyToSymbolAsync(nearby,offsets.data(),sizeof(offsets),0,cudaMemcpyHostToDevice,stream),"initialize Manhattan offsets");
        localOutgoing<<<SpikeTiles,256,0,stream>>>(m->outgoing,m->scores,m->config,0,true);
        resetLocalCount<<<1,1,0,stream>>>(m->counters);
        localCsr<<<SpikeTiles,256,0,stream>>>(m->outgoing,m->localRows,m->localColumns,m->counters);
        m->blocks=vabpnnCreateBlockModel(stream); m->bytes+=vabpnnBlockModelBytes(m->blocks);
        check(cudaEventCreateWithFlags(&m->initialized,cudaEventDisableTiming),"create spike ready event");
        check(cudaEventRecord(m->initialized,stream),"record spike readiness");
        check(cudaGetLastError(),"initialize spiking connections");
        return m;
    } catch(...) { vabpnnDestroySpikeModel(m); throw; }
}
void vabpnnDestroySpikeModel(VabpnnSpikeModel* m) noexcept {
    if(!m) return; if(m->initialized) cudaEventSynchronize(m->initialized);
    void* allocations[]={m->config,m->counters,m->clock,m->statistics,m->membrane,m->refractory,
        m->signals[0],m->signals[1],m->packedVotes,m->outgoing,m->localRows,m->localColumns,m->membership,
        m->longRows,m->cursor,m->longColumns,m->anchors,m->champions,m->keys,m->sortedKeys,m->ids,m->sortedIds,
        m->blockChampions,m->temporary,m->scores};
    vabpnnDestroyBlockModel(m->blocks);
    for(void* pointer:allocations) if(pointer) cudaFree(pointer);
    if(m->initialized) cudaEventDestroy(m->initialized); delete m;
}
std::uint64_t vabpnnSpikeModelBytes(const VabpnnSpikeModel* m) noexcept { return m ? m->bytes:0; }
vabpnn::DeviceSpikeView vabpnnSpikeModelView(const VabpnnSpikeModel* m) {
    return {m->config,m->counters,m->signals[m->current],m->membrane,m->refractory,m->statistics,m->packedVotes,
        m->outgoing,m->localRows,m->localColumns,m->longRows,m->longColumns,m->anchors,m->membership,m->champions,
        vabpnnBlockModelView(m->blocks).controls};
}
VabpnnBlockModel* vabpnnSpikeBlockModel(const VabpnnSpikeModel* m) { return m ? m->blocks:nullptr; }
vabpnn::SpikeConfig* vabpnnMutableSpikeConfig(VabpnnSpikeModel* m,cudaStream_t s) { ready(m,s); return m->config; }
vabpnn::SpikePixelStatistics* vabpnnMutableSpikeStatistics(VabpnnSpikeModel* m,cudaStream_t s) { ready(m,s); return m->statistics; }
std::int8_t* vabpnnMutableSpikeSignals(VabpnnSpikeModel* m,cudaStream_t s) { ready(m,s); return m->signals[m->current]; }
std::uint16_t* vabpnnMutableSpikeOutgoing(VabpnnSpikeModel* m,cudaStream_t s) { ready(m,s); return m->outgoing; }
std::uint32_t* vabpnnMutableSpikeVoteCache(VabpnnSpikeModel* m,cudaStream_t s) { ready(m,s); return m->packedVotes; }
void vabpnnRebuildSpikeLocalCsr(VabpnnSpikeModel* m,cudaStream_t s) {
    resetLocalCount<<<1,1,0,s>>>(m->counters);
    localCsr<<<SpikeTiles,256,0,s>>>(m->outgoing,m->localRows,m->localColumns,m->counters);
    check(cudaGetLastError(),"rebuild incoming local CSR");
}
void vabpnnLaunchSpikeForward(const VabpnnGpuFrame& frame) {
    auto* m=frame.spikeModel; ready(m,frame.stream);
    if(m->hostPredictions>0 && m->hostPredictions%5==0) rebuildTopology(m,frame.stream,static_cast<unsigned>(m->hostPredictions/5));
    beginTick<<<1,1,0,frame.stream>>>(m->clock,m->counters,m->config,frame.captureNanos,frame.width,frame.height);
    const int next=1-m->current;
    advance<<<SpikeTiles,256,0,frame.stream>>>(frame,m->config,m->clock,m->membrane,m->refractory,
        m->signals[m->current],m->signals[next],m->localRows,m->localColumns,m->longRows,m->longColumns,
        vabpnnBlockModelView(m->blocks).controls);
    m->current=next; ++m->hostPredictions;
    check(cudaGetLastError(),"dense synchronous CSR/LIF spike tick");
}
void vabpnnRecordSpikeVotes(VabpnnSpikeModel* m,cudaStream_t s) {
    recordVotes<<<(NeuronCount+255)/256,256,0,s>>>(m->packedVotes,m->statistics,m->config,m->scores);
    check(cudaGetLastError(),"record Shannon-Simpson vote distribution");
}
void vabpnnObserveSpikeContribution(VabpnnSpikeModel* m,const vabpnn::DecisionOutput* decision,
    const vabpnn::DecisionParameters* parameters,const int* valid,const std::uint32_t* label,cudaStream_t s) {
    if(!m) return;
    contributions<<<SpikeTiles,256,0,s>>>(m->packedVotes,m->statistics,m->config,decision,parameters,valid,label,m->blockChampions,m->counters,m->scores);
    check(cudaGetLastError(),"observe supervised cross-entropy contributions and operation hits");
}
void vabpnnResetSpikeHistory(VabpnnSpikeModel* m,cudaStream_t s) {
    ready(m,s);
    check(cudaMemsetAsync(m->clock,0,sizeof(Clock),s),"reset spike clock");
    check(cudaMemsetAsync(m->membrane,0,NeuronCount*sizeof(float),s),"reset membrane");
    check(cudaMemsetAsync(m->refractory,0,NeuronCount*sizeof(float),s),"reset refractory");
    for(auto signal:m->signals) check(cudaMemsetAsync(signal,0,NeuronCount,s),"reset spikes");
    check(cudaMemsetAsync(m->packedVotes,0x55,NeuronCount*sizeof(unsigned),s),"clear cached neutral votes");
    m->current=0; m->hostPredictions=0;
    vabpnnResetBlockHistory(m->blocks,s);
}
std::size_t vabpnnSpikeCheckpointParts(VabpnnSpikeModel* m,vabpnn::ModelPart* parts) {
    parts[0]={m->outgoing,NeuronCount*4*sizeof(unsigned short)};
    parts[1]={m->statistics,NeuronCount*sizeof(SpikePixelStatistics)};
    parts[2]={m->config,sizeof(SpikeConfig)}; parts[3]={m->counters,sizeof(SpikeCounters)};
    parts[4]={m->anchors,SpikeAnchorLimit*sizeof(unsigned)}; return 5;
}
void vabpnnResumeSpikeModel(VabpnnSpikeModel* m,cudaStream_t s) {
    vabpnnResetSpikeHistory(m,s); vabpnnResumeBlockModel(m->blocks,s);
    vabpnnRebuildSpikeLocalCsr(m,s); buildLong(m,s);
    refreshScores<<<(NeuronCount+255)/256,256,0,s>>>(m->statistics,m->scores);
    check(cudaGetLastError(),"resume topology without cached activations");
}
