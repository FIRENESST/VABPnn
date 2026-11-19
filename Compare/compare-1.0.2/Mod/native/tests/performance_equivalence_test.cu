// Frozen 1.0.1 reference kernels: verify optimization equivalence on synthetic GPU data only.
#include "FrameOperator.h"
#include "SpikeConnections.h"
#include "MemeState.h"
#include <vector>
#include <array>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <stdexcept>
using namespace vabpnn;
namespace {
constexpr int TestTileCount=7680,TestWarpCount=8;
struct ReferenceRuntime {
    std::uint64_t frame=0,nanos=0,observedFrame=0,learnedFrame=0;
    float dt=0;int fresh=0,feedback=0,width=0,height=0;
};
__device__ __forceinline__ int referenceWarpSum(int value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// Every model tile is complete, so every warp has 32 active lanes.
__global__ __launch_bounds__(vabpnn::ThreadsPerBlock)
void referenceNeuronVotes(VabpnnGpuFrame frame, const vabpnn::TernaryWeight* weights,
                 std::int32_t* blockVotes,const std::int8_t* spikes,std::uint32_t* packedVotes) {
    __shared__ int warpVotes[vabpnn::VoteCount][TestWarpCount];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const std::size_t pixel = static_cast<std::size_t>(y) * vabpnn::NeuronWidth + x;

    // Center-based nearest sampling; keep neurons in a fixed top-left-origin grid.
    const int sourceX = static_cast<std::size_t>(2 * x + 1) * frame.width / (2 * vabpnn::NeuronWidth);
    const int sourceY = static_cast<std::size_t>(2 * y + 1) * frame.height / (2 * vabpnn::NeuronHeight);
    const unsigned char* rgba = frame.rgba
        + static_cast<std::size_t>(frame.height - 1 - sourceY) * frame.rgbaPitch
        + static_cast<std::size_t>(sourceX) * 4;
    const int r = rgba[0], g = rgba[1], b = rgba[2];
    const int pulse=spikes ? spikes[pixel]:1;
    std::uint32_t encoded=0;
    // Byte RGB gives the same mathematical sign as RGB/255 and preserves exact ties.
    #pragma unroll
    for (int k = 0; k < vabpnn::VoteCount; ++k) {
        const int score = r * static_cast<int>(weights[vabpnn::weightIndex(pixel, 0, k)])
                        + g * static_cast<int>(weights[vabpnn::weightIndex(pixel, 1, k)])
                        + b * static_cast<int>(weights[vabpnn::weightIndex(pixel, 2, k)]);
        const int vote = ((score > 0) - (score < 0))*pulse;
        encoded |= static_cast<std::uint32_t>(vote+1)<<(k*2);
        const int sum = referenceWarpSum(vote);
        if (lane == 0) warpVotes[k][warp] = sum;
    }
    if(packedVotes) packedVotes[pixel]=encoded;
    __syncthreads();
    if (tid < vabpnn::VoteCount) {
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < TestWarpCount; ++w) sum += warpVotes[tid][w];
        const int block = blockIdx.y * gridDim.x + blockIdx.x;
        blockVotes[static_cast<std::size_t>(tid) * TestTileCount + block] = sum;
    }
}

// One block per vote item, reducing 7680 tile sums to one signed int32 total.
__global__ __launch_bounds__(vabpnn::ThreadsPerBlock)
void referenceFinishVotes(const std::int32_t* blockVotes, std::int32_t* totals) {
    __shared__ int warpVotes[TestWarpCount];
    const int k = blockIdx.x;
    const int tid = threadIdx.x;
    int sum = 0;
    for (int block = tid; block < TestTileCount; block += blockDim.x)
        sum += blockVotes[static_cast<std::size_t>(k) * TestTileCount + block];
    sum = referenceWarpSum(sum);
    if ((tid & 31) == 0) warpVotes[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) {
        int total = 0;
        #pragma unroll
        for (int w = 0; w < TestWarpCount; ++w) total += warpVotes[w];
        totals[k] = total;
    }
}
__device__ float referenceImportance(const SpikePixelStatistics& s) {
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

__device__ float referenceCrossEntropyGain(float probability,float y,float delta) {
    // Stable leave-one-neuron-out change for the direct raw-vote evidence term.
    // For tiny delta the second-order formula avoids catastrophic cancellation.
    if(fabsf(delta)<0.001f)
        return (y-probability)*delta+0.5f*probability*(1-probability)*delta*delta;
    return log1pf(probability*expm1f(-delta))+y*delta;
}
__global__ void referenceContributions(const unsigned* packed,SpikePixelStatistics* statistics,const SpikeConfig* config,
                              const DecisionOutput* decision,const DecisionParameters* parameters,
                              const int* valid,const unsigned* labels,unsigned long long* blockChampions,
                              SpikeCounters* counters,float* scores) {
    if(!isfinite(config->statisticsDecay) || config->statisticsDecay<0 || config->statisticsDecay>=1) return;
    if(!*valid || !decision->valid) return; // Uniform block condition; no barrier divergence.
    const int tile=blockIdx.x,tid=threadIdx.x; const unsigned pixel=spikePixel(tile,tid);
    const unsigned encoded=packed[pixel],label=*labels;
    const float inverse=1.0f/static_cast<float>(NeuronCount);
    float ce=0,hit=0;
    __shared__ unsigned long long warpBest[VoteCount][8];
    for(int k=0;k<VoteCount;++k) {
        const int vote=static_cast<int>((encoded>>(2*k))&3)-1;
        const float y=(label>>k)&1u;
        const float delta=parameters->rawVoteWeight[k]*vote*inverse;
        const float gain=referenceCrossEntropyGain(decision->posterior[k],y,delta)*NeuronCount;
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
    scores[pixel]=referenceImportance(s);
    __syncthreads();
    if(tid<VoteCount) {
        unsigned long long best=0;
        for(int w=0;w<8;++w) best=max(best,warpBest[tid][w]);
        blockChampions[tile*VoteCount+tid]=best;
    }
    if(tile==0 && tid==0) ++counters->contributionUpdates;
}
__device__ bool referenceCompetitionEligible(const MemePacket& packet,const MemePacket& own,float ownWeight,int mode) {
    return mode!=2 || ownWeight<=0 || packet.utility>own.utility+.002f;
}
__device__ float referenceSourceWeight(const MemePacket& packet,const float* key,const MemeConfig& c,int tile) {
    return packet.origin==static_cast<unsigned>(tile) || packet.hops>=static_cast<unsigned>(c.maximumHops)
        ? 0:memePacketWeight(packet,key,c);
}
__global__ void referenceSelectRoutes(MemePersistent* p,const ReferenceRuntime* clock,const BlockConfig* blocks,
        const MemePacket* old,const float* keys,DeviceSpikeView spike,unsigned* routes) {
    const int tile=blockIdx.x,tid=threadIdx.x;
    __shared__ unsigned long long best[2];
    if(tid<2) { best[tid]=~0ULL;routes[tile*2+tid]=NoNeuron; }
    __syncthreads();
    if(!clock->fresh || blocks->communication==0 || !spike.counters->longEdges) return;
    auto own=p->prototypes[tile];own.age=0;own.sequence=clock->frame;own.hops=0;
    const float ownWeight=memePacketWeight(own,keys+tile*MemeContext,p->config);
    const unsigned pixel=spikePixel(tile,tid),start=spike.longRows[pixel],end=spike.longRows[pixel+1];
    if(end>=start && end<=spike.counters->longEdges) for(unsigned edge=start;edge<end;++edge) {
        const unsigned source=spike.longColumns[edge]&SpikeSourceMask;
        if(source>=NeuronCount) continue;
        const int other=source/NeuronWidth/16*120+source%NeuronWidth/16;
        const int distance=abs(other%120-tile%120)+abs(other/120-tile/120);
        if(distance<=1) continue; // Self and regular four-neighbor channels cannot be counted twice.
        float weight=0;
        for(int slot=0;slot<2;++slot) {
            const auto& packet=old[other*2+slot];
            if(referenceCompetitionEligible(packet,own,ownWeight,blocks->communication))
                weight=fmaxf(weight,referenceSourceWeight(packet,keys+tile*MemeContext,p->config,tile));
        }
        if(weight<=0) continue;
        const unsigned rank=static_cast<unsigned>(fminf(1.0f,weight)*16777215.0f);
        const unsigned long long candidate=(static_cast<unsigned long long>(0xffffffffu-rank)<<32)|other;
        const auto previous=atomicMin(&best[0],candidate);
        if(previous!=candidate) atomicMin(&best[1],previous>candidate ? previous:candidate);
    }
    __syncthreads();
    if(tid<2 && best[tid]!=~0ULL) routes[tile*2+tid]=static_cast<unsigned>(best[tid]);
    if(tid==0) {
        unsigned count=(best[0]!=~0ULL)+(best[1]!=~0ULL);
        atomicAdd(reinterpret_cast<unsigned long long*>(&p->counters.longSelections),static_cast<unsigned long long>(count));
    }
}

void check(cudaError_t c) {if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
void require(bool ok,const char* text) {if(!ok)throw std::runtime_error(text);}
struct Fixture {
    cudaStream_t stream{};VabpnnNeuronModel* neuron{};VabpnnSpikeModel* spike{};
    std::vector<void*> allocations;unsigned char* rgba{};float* rgb{};std::size_t pitch{};
    Fixture() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        neuron=vabpnnCreateNeuronModel(stream);spike=vabpnnCreateSpikeModel(stream);
        check(cudaMallocPitch(reinterpret_cast<void**>(&rgba),&pitch,1920*4,1024));allocations.push_back(rgba);
        rgb=alloc<float>(NeuronCount*3);check(cudaStreamSynchronize(stream));
    }
    ~Fixture() {
        cudaStreamSynchronize(stream);vabpnnDestroySpikeModel(spike);vabpnnDestroyNeuronModel(neuron);
        for(auto p:allocations)cudaFree(p);cudaStreamDestroy(stream);
    }
    template<class T>T* alloc(std::size_t count=1) {T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));allocations.push_back(p);return p;}
    template<class T>void put(T* p,const T& value) {check(cudaMemcpyAsync(p,&value,sizeof(value),cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));}
    template<class T>T read(const T* p) {T value;check(cudaMemcpy(&value,p,sizeof(value),cudaMemcpyDeviceToHost));return value;}
    void same(const void* a,const void* b,std::size_t bytes,const char* what) {
        // Bounded parameter/fixture verification only. No production frame readback.
        constexpr std::size_t Chunk=1024*1024;std::vector<unsigned char> x(Chunk),y(Chunk);
        for(std::size_t at=0;at<bytes;at+=Chunk) {
            const auto n=std::min(Chunk,bytes-at);
            check(cudaMemcpy(x.data(),static_cast<const unsigned char*>(a)+at,n,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(y.data(),static_cast<const unsigned char*>(b)+at,n,cudaMemcpyDeviceToHost));
            require(std::memcmp(x.data(),y.data(),n)==0,what);
        }
    }
    VabpnnGpuFrame frame(int w=37,int h=23) {return {rgba,pitch,rgb,w,h,1,1'000'000'000ULL,stream,neuron,nullptr,nullptr,nullptr,spike};}
};
__global__ void voteFixture(unsigned char* image,std::size_t pitch,int width,int height,
        TernaryWeight* weights,signed char* signals,int mode) {
    const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=NeuronCount)return;
    signals[p]=mode==0 ? 0:mode==1 ? 1:mode==2 ? static_cast<signed char>(static_cast<int>(p%3)-1):
        static_cast<signed char>(p%131==0 ? (p&1 ? -1:1):0);
    for(int c=0;c<3;++c)for(int k=0;k<VoteCount;++k)
        weights[weightIndex(p,c,k)]=static_cast<TernaryWeight>(static_cast<int>((p*7+c*11+k*5)%3)-1);
    if(p<static_cast<unsigned>(width*height)) {
        const int x=p%width,y=p/width;auto* q=image+y*pitch+x*4;
        q[0]=(x*3+y*7+1)%100;q[1]=(x*5+y*2+1)%100;
        q[2]=q[0]+q[1]+((x+y)%3-1);q[3]=255;
    }
}
void voteEquivalence(Fixture& f) {
    auto* blocks=f.alloc<int>(VoteCount*TestTileCount);auto* sums=f.alloc<int>(VoteCount);
    auto* packed=f.alloc<unsigned>(NeuronCount);
    for(int mode=0;mode<4;++mode) {
        auto frame=f.frame(mode%2 ? 1920:37,mode%2 ? 1024:23);
        voteFixture<<<(NeuronCount+255)/256,256,0,f.stream>>>(f.rgba,f.pitch,frame.width,frame.height,
            vabpnnMutableNeuronWeights(f.neuron,f.stream),vabpnnMutableSpikeSignals(f.spike,f.stream),mode);
        const auto view=vabpnnNeuronModelView(f.neuron);const auto sv=vabpnnSpikeModelView(f.spike);
        referenceNeuronVotes<<<gridShape(1920,1024),blockShape(),0,f.stream>>>(frame,view.weights,blocks,sv.spikes,packed);
        referenceFinishVotes<<<VoteCount,256,0,f.stream>>>(blocks,sums);
        vabpnnLaunchTernaryVotes(frame,f.neuron);check(cudaStreamSynchronize(f.stream));
        f.same(blocks,view.blockVotes,VoteCount*TestTileCount*sizeof(int),"per-tile votes differ from 1.0.1");
        f.same(sums,view.output,VoteCount*sizeof(int),"global votes differ from 1.0.1");
        f.same(packed,sv.packedVotes,NeuronCount*sizeof(unsigned),"packed votes differ from 1.0.1");
    }
    std::puts("PASS: exact 1.0.1 global/tile/packed votes for silent, dense, mixed signed and sparse warps; full/odd pitched source.");
}
__global__ void plasticFixture(unsigned* packed,SpikePixelStatistics* stats) {
    const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=NeuronCount)return;
    unsigned bits=0;for(int k=0;k<VoteCount;++k)bits|=((p*13+k*7)%3)<<(k*2);
    packed[p]=bits;stats[p]={.2f,.3f,.5f,.1f,.05f};
}
void contributionEquivalence(Fixture& f) {
    auto* stats=f.alloc<SpikePixelStatistics>(NeuronCount);auto* scores=f.alloc<float>(NeuronCount);
    auto* champions=f.alloc<unsigned long long>(VoteCount*TestTileCount);auto* counters=f.alloc<SpikeCounters>();
    auto* out=f.alloc<DecisionOutput>();auto* param=f.alloc<DecisionParameters>();auto* valid=f.alloc<int>();auto* label=f.alloc<unsigned>();
    auto* actualStats=vabpnnMutableSpikeStatistics(f.spike,f.stream);
    for(int mode=0;mode<3;++mode) {
        plasticFixture<<<(NeuronCount+255)/256,256,0,f.stream>>>(vabpnnMutableSpikeVoteCache(f.spike,f.stream),actualStats);
        check(cudaMemcpyAsync(stats,actualStats,NeuronCount*sizeof(SpikePixelStatistics),cudaMemcpyDeviceToDevice,f.stream));
        DecisionOutput d{};DecisionParameters p{};d.valid=1;d.selectedMask=mode==0 ? 0:mode==1 ? 0x6021:0x7fff;
        for(int k=0;k<VoteCount;++k) {d.posterior[k]=.05f+.06f*k;p.rawVoteWeight[k]=(k%2 ? -1:1)*(k%3 ? 2.0f:10000.0f);}
        const unsigned labels=mode==2 ? 0x7fff:0x6333;
        f.put(out,d);f.put(param,p);f.put(valid,1);f.put(label,labels);f.put(counters,SpikeCounters{});
        const auto sv=vabpnnSpikeModelView(f.spike);
        referenceContributions<<<TestTileCount,256,0,f.stream>>>(sv.packedVotes,stats,sv.config,out,param,valid,label,champions,counters,scores);
        vabpnnObserveSpikeContribution(f.spike,out,param,valid,label,f.stream);check(cudaStreamSynchronize(f.stream));
        f.same(stats,actualStats,NeuronCount*sizeof(SpikePixelStatistics),"contribution EMA differs from 1.0.1");
        std::vector<unsigned long long> best(VoteCount*TestTileCount);
        check(cudaMemcpy(best.data(),champions,best.size()*sizeof(best[0]),cudaMemcpyDeviceToHost));
        std::array<unsigned,VoteCount> expected{};
        for(int k=0;k<VoteCount;++k) {unsigned long long maximum=0;for(int tile=0;tile<TestTileCount;++tile)maximum=std::max(maximum,best[tile*VoteCount+k]);
            expected[k]=maximum ? NoNeuron-static_cast<unsigned>(maximum):NoNeuron;}
        auto frame=f.frame();vabpnnResetSpikeHistory(f.spike,f.stream);
        for(int i=0;i<6;++i) {++frame.frameId;frame.captureNanos+=20'000'000;vabpnnLaunchSpikeForward(frame);}
        std::array<unsigned,VoteCount> actual{};check(cudaMemcpyAsync(actual.data(),vabpnnSpikeModelView(f.spike).championIds,sizeof(actual),cudaMemcpyDeviceToHost,f.stream));
        check(cudaStreamSynchronize(f.stream));require(actual==expected,"global hit winners differ from 1.0.1");
    }
    std::puts("PASS: exact 1.0.1 contribution/hit EMA and champion IDs, including no hits, partial/all hits and both gain branches.");
}
__global__ void packetFixture(DeviceMemeView v,std::uint64_t serial) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x;if(tile>=MemeBlocks)return;
    MemePacket own{};own.support=1000;own.revision=17;own.confidence=.8f;own.utility=.4f;own.origin=tile;
    for(int d=0;d<MemeContext;++d)own.context[d]=v.contexts[tile*MemeContext+d];
    for(int k=0;k<MemeChannels;++k)own.evidence[k]=k==0 ? .8f:-.6f;
    v.persistent->prototypes[tile]=own;
    for(int slot=0;slot<2;++slot) {auto p=own;p.sequence=serial;p.confidence=.4f+.004f*(tile%128);
        p.utility=(tile%7)*.12f;p.age=tile%19==0 ? 2.1f:0;p.hops=tile%17==0 ? 8:slot;
        v.packets[tile*2+slot]=p;}
}
void routeEquivalence(Fixture& f) {
    auto* memes=vabpnnBlockMemeModel(vabpnnSpikeBlockModel(f.spike));auto* refPersistent=f.alloc<MemePersistent>();
    auto* clock=f.alloc<ReferenceRuntime>();auto* routes=f.alloc<unsigned>(MemeBlocks*2);
    std::vector<unsigned> rows(NeuronCount+1),columns;
    for(unsigned pixel=0;pixel<NeuronCount;++pixel) {
        rows[pixel]=static_cast<unsigned>(columns.size());
        const unsigned tile=pixel/1920/16*120+pixel%1920/16,local=pixel/1920%16*16+pixel%1920%16;
        int count=local==73 && tile%37==0 ? 256:local%61==0 && tile%113==0 ? 30:0;
        for(int i=0;i<count;++i) {
            const unsigned other=(i%11==0 ? 2:(i*29+tile*17)%MemeBlocks);
            const unsigned source=i%47==0 ? SpikeSourceMask:spikePixel(other,(i*13)%256);
            columns.push_back(source|((i%4)<<21));
        }
    }
    rows.back()=static_cast<unsigned>(columns.size());require(columns.size()<=SpikeLongCapacity,"CSR fixture overflow");
    const auto sv=vabpnnSpikeModelView(f.spike);
    check(cudaMemcpyAsync(const_cast<unsigned*>(sv.longRows),rows.data(),rows.size()*sizeof(unsigned),cudaMemcpyHostToDevice,f.stream));
    check(cudaMemcpyAsync(const_cast<unsigned*>(sv.longColumns),columns.data(),columns.size()*sizeof(unsigned),cudaMemcpyHostToDevice,f.stream));
    auto counts=f.read(sv.counters);counts.longEdges=static_cast<unsigned>(columns.size());f.put(const_cast<SpikeCounters*>(sv.counters),counts);
    auto frame=f.frame();frame.evaluationMode=true;
    for(int mode=0;mode<3;++mode) {
        BlockConfig config;config.communication=mode;vabpnnConfigureBlockModel(vabpnnSpikeBlockModel(f.spike),config,f.stream);
        ++frame.frameId;frame.captureNanos+=20'000'000;vabpnnPrepareMemes(frame,false);check(cudaStreamSynchronize(f.stream));
        auto old=vabpnnMemeModelView(memes);packetFixture<<<(MemeBlocks+255)/256,256,0,f.stream>>>(old,frame.frameId);
        check(cudaStreamSynchronize(f.stream));const auto before=f.read(&old.persistent->counters).longSelections;
        ++frame.frameId;frame.captureNanos+=20'000'000;vabpnnPrepareMemes(frame,true);check(cudaStreamSynchronize(f.stream));
        const auto current=vabpnnMemeModelView(memes);const auto blocks=vabpnnBlockModelView(vabpnnSpikeBlockModel(f.spike));
        check(cudaMemcpyAsync(refPersistent,current.persistent,sizeof(MemePersistent),cudaMemcpyDeviceToDevice,f.stream));
        check(cudaMemsetAsync(&refPersistent->counters.longSelections,0,sizeof(std::uint64_t),f.stream));
        ReferenceRuntime c{};c.frame=frame.frameId;c.fresh=1;f.put(clock,c);
        referenceSelectRoutes<<<MemeBlocks,256,0,f.stream>>>(refPersistent,clock,blocks.config,old.packets,current.contexts,sv,routes);
        check(cudaStreamSynchronize(f.stream));f.same(routes,current.routes,MemeBlocks*2*sizeof(unsigned),"CSR top-2 routes differ from 1.0.1");
        require(f.read(&refPersistent->counters).longSelections==f.read(&current.persistent->counters).longSelections-before,"route counts differ from 1.0.1");
    }
    std::puts("PASS: exact 1.0.1 distinct CSR top-2 routes/counts across all rows, duplicate sources, invalid IDs, ties, expiry/hops and 3 communication modes.");
}
}
int main() {
    try {Fixture f;voteEquivalence(f);contributionEquivalence(f);routeEquivalence(f);return 0;}
    catch(const std::exception& e){std::fprintf(stderr,"FAIL: performance equivalence: %s\n",e.what());return 1;}
}
