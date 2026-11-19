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
static_assert(SpikeTiles%4==0,"Four complete scout tiles per block.");
struct Clock { std::uint64_t lastNanos; float dt,decay; int width,height,valid; };
// Transient probe details never enter a checkpoint; even sampled RGB stays GPU-only.
struct PlacementProbe { unsigned pixel,packed,rgb; float contrast,novelty,proxy,alpha; };
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
__global__ void localOutgoing(std::uint16_t* out,float* scores,const SpikePixelStatistics* statistics,
                              const SpikeConfig* config,unsigned epoch,bool initial) {
    const int tile=blockIdx.x,local=threadIdx.x;
    __shared__ float localScore[256];
    if(!initial) {
        localScore[local]=importance(statistics[spikePixel(tile,local)]);
        scores[spikePixel(tile,local)]=localScore[local];
    }
    __syncthreads(); // Fused refresh preserves external statistics edits and exact target scores.
    unsigned seed=hash(config->seed ^ spikePixel(tile,local) ^ hash(epoch+1));
    const int degree=initial ? 4:static_cast<int>(seed%5);
    int targets[4]={-1,-1,-1,-1};
    for(int slot=0;slot<4;++slot) {
        unsigned short encoded=0;
        if(slot<degree) {
            seed=hash(seed+slot+37);
            int target=static_cast<int>(seed&255);
            const int alternate=static_cast<int>((seed>>8)&255);
            if(!initial && localScore[alternate]>localScore[target]) target=alternate;
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
__device__ float placementIndex(const SpikePlacementArm& arm) {
    const float variance=fmaxf(0,arm.secondMoment-arm.utility*arm.utility);
    return arm.utility+0.12f*sqrtf((variance+0.0025f)/(1+arm.visits));
}
__global__ void candidates(const float* scores,const SpikePlacementTile* placement,float* keys,unsigned* ids) {
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
    if(tid==0) {
        float best=values[0]; unsigned id=index[0];
        for(const auto& arm:placement[tile].arms) {
            if(arm.pixel>=NeuronCount || !(arm.visits>0)) continue;
            const float value=placementIndex(arm);
            if(isfinite(value) && value>best) { best=value;id=arm.pixel; }
        }
        keys[tile]=best;ids[tile]=id;
    }
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
__device__ bool insertAnchor(unsigned id,bool balanced,unsigned* seen,unsigned short* regions,
                               unsigned* anchors,int& count) {
    if(id>=NeuronCount) return false;
    unsigned slot=hash(id)&511u;
    while(seen[slot]!=NoNeuron && seen[slot]!=id) slot=(slot+1)&511u;
    if(seen[slot]==id) return false;
    const unsigned region=id/NeuronWidth/64*30+id%NeuronWidth/64;
    if(balanced && regions[region]>=4) return false;
    seen[slot]=id;++regions[region];anchors[count++]=id;return true;
}
__global__ void selectAnchors(const float* keys,const unsigned* ids,const unsigned* champions,
                              const SpikeConfig* config,const float* scores,
                              unsigned* anchors,SpikeCounters* counters) {
    // The chosen sequence is identical to the quadratic reference. A bounded
    // half-full hash set and 480 region counters avoid repeated ID/division scans.
    __shared__ unsigned seen[512];
    __shared__ unsigned short regions[30*16];
    for(int j=threadIdx.x;j<512;j+=blockDim.x) seen[j]=NoNeuron;
    for(int j=threadIdx.x;j<30*16;j+=blockDim.x) regions[j]=0;
    __syncthreads();
    if(threadIdx.x) return;
    int count=0;
    const int limit=max(0,min(SpikeAnchorLimit,config->maxAnchors));
    for(int k=0;k<VoteCount && count<limit;++k) {
        const unsigned id=champions[k];
        if(id>=NeuronCount || scores[id]<=config->minimumScore) continue;
        insertAnchor(id,false,seen,regions,anchors,count);
    }
    const int exploration=limit/8;
    for(int k=0;k<exploration && count<limit;++k) {
        const int at=hash(config->seed ^ hash(static_cast<unsigned>(counters->topologyUpdates)+1) ^ hash(k+1))%SpikeTiles;
        const unsigned id=ids[at];
        if(id>=NeuronCount || !(keys[at]>config->minimumScore)) continue;
        insertAnchor(id,false,seen,regions,anchors,count);
    }
    for(int k=0;k<SpikeTiles && count<limit;++k) {
        if(!(keys[k]>config->minimumScore)) break;
        insertAnchor(ids[k],limit>=8,seen,regions,anchors,count);
    }
    for(int j=count;j<SpikeAnchorLimit;++j) anchors[j]=NoNeuron;
    counters->anchors=count;++counters->topologyUpdates;
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
                              SpikeCounters* counters,float* scores,bool inferred) {
    if(!isfinite(config->statisticsDecay) || config->statisticsDecay<0 || config->statisticsDecay>=1) return;
    if((!inferred && (!valid || !*valid)) || !decision->valid) return; // Uniform condition.
    const int tile=blockIdx.x,tid=threadIdx.x; const unsigned pixel=spikePixel(tile,tid);
    const unsigned encoded=packed[pixel],label=inferred ? 0:*labels;
    const float inverse=1.0f/static_cast<float>(NeuronCount);
    float ce=0,hit=0;
    const unsigned hitMask=inferred ? decision->selectedMask:decision->selectedMask & label;
    __shared__ unsigned long long warpBest[VoteCount][8];
    for(int k=0;k<VoteCount;++k) {
        const int vote=static_cast<int>((encoded>>(2*k))&3)-1;
        const float y=(label>>k)&1u;
        const float delta=parameters->rawVoteWeight[k]*vote*inverse;
        const float gain=inferred ? fabsf(parameters->rawVoteWeight[k]*vote)*decision->posterior[k]*(1-decision->posterior[k])
            :crossEntropyGain(decision->posterior[k],y,delta)*NeuronCount;
        ce+=gain;
        const bool correctHit=inferred ? (hitMask & (1u<<k)) && delta>0 && gain>0
            :(decision->selectedMask & label & (1u<<k)) && vote>0 && gain>0;
        const float value=correctHit ? gain:0;
        if(correctHit) hit=fmaxf(hit,value);
        // Eligibility is uniform across the block; non-hit actions have exactly
        // zero champions. Keep gain/EMA arithmetic and hit tie-breaking unchanged.
        unsigned long long best=0;
        if(hitMask & (1u<<k)) {
            best=value>0 ? (static_cast<unsigned long long>(__float_as_uint(value))<<32)
                |(NoNeuron-pixel):0;
            for(int offset=16;offset;offset/=2) best=max(best,__shfl_down_sync(0xffffffffu,best,offset));
        }
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
__global__ void initializePlacement(SpikePlacementTile* placement) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile>=SpikeTiles) return;
    auto& state=placement[tile];state=SpikePlacementTile{};
    unsigned used[SpikePlacementScouts];
    for(int a=0;a<SpikePlacementScouts;++a) {
        unsigned local=hash(0x7a6b9d31u ^ hash(tile+1) ^ hash(a+1))&255u;
        for(int attempt=0;attempt<SpikePlacementScouts;++attempt) {
            bool duplicate=false;for(int j=0;j<a;++j) duplicate|=used[j]==local;
            if(!duplicate) break;local=(local+1)&255u;
        }
        used[a]=local;state.arms[a].pixel=spikePixel(tile,local);
    }
}
__device__ unsigned placementRgb(const VabpnnGpuFrame& frame,int x,int y) {
    x=max(0,min(NeuronWidth-1,x));y=max(0,min(NeuronHeight-1,y));
    const int sx=static_cast<std::size_t>(2*x+1)*frame.width/(2*NeuronWidth);
    const int sy=static_cast<std::size_t>(2*y+1)*frame.height/(2*NeuronHeight);
    const auto* c=frame.rgba+static_cast<std::size_t>(frame.height-1-sy)*frame.rgbaPitch+sx*4;
    return c[0]|(static_cast<unsigned>(c[1])<<8)|(static_cast<unsigned>(c[2])<<16);
}
__device__ float relativeRgbDifference(unsigned a,unsigned b) {
    int difference=0,total=1;
    for(int c=0;c<3;++c) {
        const int x=(a>>(8*c))&255,y=(b>>(8*c))&255;
        difference+=abs(x-y);total+=x+y;
    }
    return static_cast<float>(difference)/total; // Contrast, never raw luminance.
}
__device__ float placementDiversity(unsigned packed) {
    int count[3]={};
    for(int k=0;k<VoteCount;++k) {
        const unsigned v=(packed>>(2*k))&3;
        if(v<3) ++count[v];
    }
    float entropy=0,squares=0;
    for(int c=0;c<3;++c) {
        const float p=static_cast<float>(count[c])/VoteCount;
        if(p>0) entropy-=p*logf(p);
        squares+=p*p;
    }
    return 0.5f+0.25f*entropy*0.9102392266f+0.375f*(1-squares);
}
__global__ void probePlacement(VabpnnGpuFrame frame,SpikePlacementTile* placement,PlacementProbe* probes,
    const SpikeConfig* config,const TernaryWeight* weights,const DecisionOutput* decision,
    const DecisionParameters* parameters) {
    if(!decision->valid) return; // Uniform whole-warp condition.
    const int tile=blockIdx.x*4+(threadIdx.x>>5),a=threadIdx.x&31;
    auto& state=placement[tile];
    if(a==0 && state.rounds>0) {
        int best=0,worst=1;
        for(int j=1;j<SpikePlacementScouts;++j)
            if(placementIndex(state.arms[j])>placementIndex(state.arms[best])) best=j;
        worst=best==0 ? 1:0;
        for(int j=0;j<SpikePlacementScouts;++j)
            if(j!=best && state.arms[j].utility<state.arms[worst].utility) worst=j;
        unsigned seed=hash(config->seed ^ hash(tile+1) ^ hash(static_cast<unsigned>(state.rounds)+1));
        unsigned pixel;
        if(state.rounds&1) {
            const auto origin=state.arms[best].pixel<NeuronCount ? state.arms[best].pixel:spikePixel(tile,0);
            const auto offset=nearby[seed%SpikeNearCount];
            const int x=max(0,min(NeuronWidth-1,static_cast<int>(origin%NeuronWidth)+offset.x));
            const int y=max(0,min(NeuronHeight-1,static_cast<int>(origin/NeuronWidth)+offset.y));
            pixel=y*NeuronWidth+x;
        } else pixel=spikePixel(tile,seed&255u);
        for(int attempt=0;attempt<SpikePlacementScouts;++attempt) {
            bool duplicate=false;
            for(int j=0;j<SpikePlacementScouts;++j) duplicate|=j!=worst && state.arms[j].pixel==pixel;
            if(!duplicate) break;
            pixel=spikePixel(tile,(seed+attempt+1)&255u);
        }
        state.arms[worst]=SpikePlacementArm{pixel,0,0,0};
    }
    __syncthreads();
    if(a>=SpikePlacementScouts) return;
    auto& arm=state.arms[a];auto& probe=probes[tile*SpikePlacementScouts+a];
    if(arm.pixel>=NeuronCount) arm=SpikePlacementArm{spikePixel(tile,hash(tile+a)&255u),0,0,0};
    const int x=arm.pixel%NeuronWidth,y=arm.pixel/NeuronWidth;
    const unsigned rgb=placementRgb(frame,x,y);
    float contrast=0;
    for(int side=0;side<4;++side)
        contrast+=relativeRgbDifference(rgb,placementRgb(frame,x+(side==0)-(side==1),y+(side==2)-(side==3)));
    contrast*=0.25f;
    const float novelty=arm.visits>0 && probe.pixel==arm.pixel ? relativeRgbDifference(rgb,probe.rgb):0;
    const int r=rgb&255,g=(rgb>>8)&255,b=(rgb>>16)&255;
    unsigned packed=0;float sensitivity=0,selectedSupport=0;
    for(int k=0;k<VoteCount;++k) {
        const int v=r*static_cast<int>(weights[weightIndex(arm.pixel,0,k)])
            +g*static_cast<int>(weights[weightIndex(arm.pixel,1,k)])
            +b*static_cast<int>(weights[weightIndex(arm.pixel,2,k)]);
        const int vote=(v>0)-(v<0); // Ungated counterfactual: dark neurons get observed too.
        packed|=static_cast<unsigned>(vote+1)<<(k*2);
        const float delta=parameters->rawVoteWeight[k]*vote;
        const float impact=fabsf(delta)*decision->posterior[k]*(1-decision->posterior[k]);
        sensitivity+=impact;
        if((decision->selectedMask & (1u<<k)) && delta>0) selectedSupport=fmaxf(selectedSupport,impact);
    }
    const float reward=fminf(4,sensitivity/VoteCount*placementDiversity(packed)+0.1f*selectedSupport
        +0.025f*contrast+0.015f*novelty);
    const float alpha=1.0f/fminf(16,1+arm.visits);
    arm.utility+=(reward-arm.utility)*alpha;
    arm.secondMoment+=(reward*reward-arm.secondMoment)*alpha;
    arm.visits=fminf(1'000'000,1+arm.visits);
    probe=PlacementProbe{arm.pixel,packed,rgb,contrast,novelty,reward,alpha};
    if(a==0) ++state.rounds;
}
__global__ void supervisePlacement(SpikePlacementTile* placement,const PlacementProbe* probes,
    const DecisionOutput* decision,const DecisionParameters* parameters,const int* valid,const unsigned* label) {
    if(!valid || !*valid || !decision->valid) return;
    const int tile=blockIdx.x*4+(threadIdx.x>>5),a=threadIdx.x&31;
    if(a>=SpikePlacementScouts) return;
    auto& arm=placement[tile].arms[a];const auto& probe=probes[tile*SpikePlacementScouts+a];
    if(arm.pixel!=probe.pixel || !arm.visits) return;
    float gain=0,hit=0;
    for(int k=0;k<VoteCount;++k) {
        const int vote=static_cast<int>((probe.packed>>(2*k))&3)-1;
        const float delta=parameters->rawVoteWeight[k]*vote/NeuronCount;
        const float ce=crossEntropyGain(decision->posterior[k],static_cast<float>((*label>>k)&1u),delta)*NeuronCount;
        gain+=ce;
        if((decision->selectedMask & *label & (1u<<k)) && vote>0) hit=fmaxf(hit,ce);
    }
    const float reward=fminf(4,fmaxf(-4,gain/VoteCount*placementDiversity(probe.packed)+0.25f*hit
        +0.025f*probe.contrast+0.015f*probe.novelty));
    // Replace this observation's proxy with actual causal human feedback. No
    // second visit, invented label, neural update or extra optimizer step.
    arm.utility+=probe.alpha*(reward-probe.proxy);
    arm.secondMoment=fmaxf(0,arm.secondMoment+probe.alpha*(reward*reward-probe.proxy*probe.proxy));
    if(a==0) ++placement[tile].supervisedRounds;
}

}
struct VabpnnSpikeModel {
    VabpnnBlockModel* blocks=nullptr;
    vabpnn::SpikeConfig* config=nullptr;
    vabpnn::SpikePixelStatistics* statistics=nullptr;
    vabpnn::SpikeCounters* counters=nullptr;
    Clock* clock=nullptr;
    SpikePlacementTile* placement=nullptr;
    PlacementProbe* probes=nullptr;
    bool placementPending=false;
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
    localOutgoing<<<SpikeTiles,256,0,stream>>>(m->outgoing,m->scores,m->statistics,m->config,epoch,false);
    vabpnnRebuildSpikeLocalCsr(m,stream);
    candidates<<<SpikeTiles,256,0,stream>>>(m->scores,m->placement,m->keys,m->ids);
    reduceChampions<<<VoteCount,256,0,stream>>>(m->blockChampions,m->champions);
    check(cub::DeviceRadixSort::SortPairsDescending(m->temporary,m->sortBytes,m->keys,m->sortedKeys,
        m->ids,m->sortedIds,SpikeTiles,0,32,stream),"sort high-contribution candidates");
    selectAnchors<<<1,128,0,stream>>>(m->sortedKeys,m->sortedIds,m->champions,m->config,m->scores,m->anchors,m->counters);
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
        allocate(m->placement,SpikeTiles,m,"allocate persistent placement learners");
        allocate(m->probes,SpikeTiles*SpikePlacementScouts,m,"allocate GPU-only placement probes");
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
        check(cudaMemsetAsync(m->probes,0xff,SpikeTiles*SpikePlacementScouts*sizeof(PlacementProbe),stream),"invalidate placement probes");
        initializePlacement<<<(SpikeTiles+255)/256,256,0,stream>>>(m->placement);
        initializeConfig<<<1,1,0,stream>>>(m->config);
        std::array<int2,SpikeNearCount> offsets; int at=0;
        for(int y=-SpikeRadius;y<=SpikeRadius;++y) for(int x=-SpikeRadius;x<=SpikeRadius;++x)
            if(std::abs(x)+std::abs(y)>0 && std::abs(x)+std::abs(y)<=SpikeRadius) offsets[at++]={x,y};
        check(cudaMemcpyToSymbolAsync(nearby,offsets.data(),sizeof(offsets),0,cudaMemcpyHostToDevice,stream),"initialize Manhattan offsets");
        localOutgoing<<<SpikeTiles,256,0,stream>>>(m->outgoing,m->scores,m->statistics,m->config,0,true);
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
        m->blockChampions,m->temporary,m->scores,m->placement,m->probes};
    vabpnnDestroyBlockModel(m->blocks);
    for(void* pointer:allocations) if(pointer) cudaFree(pointer);
    if(m->initialized) cudaEventDestroy(m->initialized); delete m;
}
std::uint64_t vabpnnSpikeModelBytes(const VabpnnSpikeModel* m) noexcept { return m ? m->bytes:0; }
vabpnn::DeviceSpikeView vabpnnSpikeModelView(const VabpnnSpikeModel* m) {
    return {m->config,m->counters,m->signals[m->current],m->membrane,m->refractory,m->statistics,m->packedVotes,
        m->outgoing,m->localRows,m->localColumns,m->longRows,m->longColumns,m->anchors,m->membership,m->champions,
        vabpnnBlockModelView(m->blocks).controls,m->placement};
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
    contributions<<<SpikeTiles,256,0,s>>>(m->packedVotes,m->statistics,m->config,decision,parameters,valid,label,m->blockChampions,m->counters,m->scores,false);
    if(m->placementPending) {
        supervisePlacement<<<SpikeTiles/4,128,0,s>>>(m->placement,m->probes,decision,parameters,valid,label);
        m->placementPending=false;
    }
    check(cudaGetLastError(),"observe supervised cross-entropy contributions and operation hits");
}

void vabpnnObserveSpikePrediction(const VabpnnGpuFrame& frame) {
    auto* m=frame.spikeModel;
    if(!m || !frame.decisionModel || !frame.neuronModel) return;
    const auto decision=vabpnnDecisionModelView(frame.decisionModel);
    if(frame.evaluationMode || !frame.onlineModel)
        contributions<<<SpikeTiles,256,0,frame.stream>>>(m->packedVotes,m->statistics,m->config,
            decision.output,decision.parameters,nullptr,nullptr,m->blockChampions,m->counters,m->scores,true);
    if(m->hostPredictions && m->hostPredictions%5==0) {
        probePlacement<<<SpikeTiles/4,128,0,frame.stream>>>(frame,m->placement,m->probes,m->config,
            vabpnnNeuronModelView(frame.neuronModel).weights,decision.output,decision.parameters);
        m->placementPending=true;
    }
    check(cudaGetLastError(),"update dynamic salience and ungated placement scouts");
}
vabpnn::ModelPart vabpnnSpikePlacementCheckpointPart(VabpnnSpikeModel* m) {
    if(!m) throw std::invalid_argument("Spike placement model missing");
    return {m->placement,SpikeTiles*sizeof(SpikePlacementTile)};
}

void vabpnnResetSpikeHistory(VabpnnSpikeModel* m,cudaStream_t s) {
    ready(m,s);
    check(cudaMemsetAsync(m->clock,0,sizeof(Clock),s),"reset spike clock");
    check(cudaMemsetAsync(m->membrane,0,NeuronCount*sizeof(float),s),"reset membrane");
    check(cudaMemsetAsync(m->refractory,0,NeuronCount*sizeof(float),s),"reset refractory");
    for(auto signal:m->signals) check(cudaMemsetAsync(signal,0,NeuronCount,s),"reset spikes");
    check(cudaMemsetAsync(m->packedVotes,0x55,NeuronCount*sizeof(unsigned),s),"clear cached neutral votes");
    m->current=0; m->hostPredictions=0;m->placementPending=false;
    check(cudaMemsetAsync(m->probes,0xff,SpikeTiles*SpikePlacementScouts*sizeof(PlacementProbe),s),"invalidate transient placement probes");
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

