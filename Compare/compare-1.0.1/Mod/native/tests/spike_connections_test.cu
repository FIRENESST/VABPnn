#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "SpikeConnections.h"
#include "FrameOperator.h"
#include "model_checkpoint.h"
#include <vector>
#include <array>
#include <map>
#include <set>
#include <cmath>
#include <cstdio>
#include <limits>
using namespace vabpnn;
void require(bool ok,const char* msg) { if(!ok) throw std::runtime_error(msg); }
void check(cudaError_t c) { if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c)); }
template<class T> std::vector<T> download(const T* p,std::size_t n) {
    std::vector<T> v(n); check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v;
}
template<class T> T scalar(const T* p) { return download(p,1)[0]; }
template<class T> void put(T* p,const T& v,cudaStream_t s) {
    check(cudaMemcpyAsync(p,&v,sizeof(v),cudaMemcpyHostToDevice,s)); check(cudaStreamSynchronize(s));
}
void verifyLocal(VabpnnSpikeModel* m,bool initial) {
    auto view=vabpnnSpikeModelView(m);
    auto out=download(view.localOutgoing,NeuronCount*4),rows=download(view.localRows,SpikeTiles*257);
    auto cols=download(view.localColumns,SpikeTiles*1024);
    std::uint64_t total=0; bool lower=false;
    for(int tile=0;tile<SpikeTiles;++tile) {
        unsigned counts[256]={},degree[256]={};
        for(int src=0;src<256;++src) {
            std::set<int> unique;
            for(int slot=0;slot<4;++slot) {
                const auto e=out[(tile*256+src)*4+slot]; if(!(e&0x400)) continue;
                require((e&~0x7ff)==0 && (e&255)!=src,"local edge has a valid kind and no self loop");
                require(unique.insert(e&255).second,"no repeated outgoing targets");
                ++counts[e&255]; ++degree[src];
            }
            require(degree[src]<=4 && (!initial || degree[src]==4),"bounded outgoing fanout");
            lower|=degree[src]<4;
        }
        require(rows[tile*257]==0 && rows[tile*257+256]<=1024,"CSR bounds");
        for(int target=0;target<256;++target) {
            const int begin=rows[tile*257+target],end=rows[tile*257+target+1];
            require(end-begin==counts[target],"destination CSR row is the outgoing transpose");
            std::set<int> unique;
            for(int e=begin;e<end;++e) {
                const auto encoded=cols[tile*1024+e]; const int src=encoded&255;
                require(unique.insert(src).second,"incoming CSR no duplicate source");
                bool found=false;
                for(int slot=0;slot<4;++slot)
                    found|=out[(tile*256+src)*4+slot]==(target|(encoded&0x300)|0x400);
                require(found,"CSR kind/source exactly matches the source adjacency");
            }
        }
        total+=rows[tile*257+256];
    }
    require(total==scalar(view.counters).localEdges,"CSR counter matches complete graph");
    require(initial || lower,"plasticity can remove connections");
}

unsigned referenceHash(unsigned x) {x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void seedTopologyReference(SpikePixelStatistics* stats) {
    const unsigned pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=NeuronCount)return;
    const unsigned tile=pixel/NeuronWidth/16*120+pixel%NeuronWidth/16;
    const unsigned local=pixel/NeuronWidth%16*16+pixel%NeuronWidth%16;
    stats[pixel]={1,0,0,static_cast<float>(tile*10+local%9),0}; // Exact half-integer importance.
}
void optimizedTopologyTest(VabpnnSpikeModel* m,VabpnnGpuFrame& frame) {
    SpikeConfig config;put(vabpnnMutableSpikeConfig(m,frame.stream),config,frame.stream);
    const auto previous=scalar(vabpnnSpikeModelView(m).counters).topologyUpdates;
    vabpnnResetSpikeHistory(m,frame.stream);
    seedTopologyReference<<<(NeuronCount+255)/256,256,0,frame.stream>>>(vabpnnMutableSpikeStatistics(m,frame.stream));
    for(int i=0;i<6;++i) {frame.captureNanos+=20'000'000;vabpnnLaunchSpikeForward(frame);}
    check(cudaStreamSynchronize(frame.stream));
    const auto view=vabpnnSpikeModelView(m);
    auto actual=download(view.localOutgoing,2*256*4);
    for(unsigned tile=0;tile<2;++tile)for(unsigned local=0;local<256;++local) {
        unsigned seed=referenceHash(config.seed ^ spikePixel(tile,local) ^ referenceHash(2));
        const unsigned degree=seed%5;int targets[4]={-1,-1,-1,-1};
        for(unsigned slot=0;slot<4;++slot) {
            unsigned short expected=0;
            if(slot<degree) {
                seed=referenceHash(seed+slot+37);int target=seed&255,alternate=(seed>>8)&255;
                if(alternate%9>target%9)target=alternate;
                for(int attempt=0;attempt<6;++attempt) {
                    bool duplicate=target==static_cast<int>(local);
                    for(unsigned k=0;k<slot;++k)duplicate|=targets[k]==target;
                    if(!duplicate)break;target=(target+1)&255;
                }
                targets[slot]=target;expected=static_cast<unsigned short>(target|((seed>>16)&3)<<8|0x400u);
            }
            require(actual[(tile*256+local)*4+slot]==expected,"Fused score/outgoing kernel exactly matches the prior generator");
        }
    }
    // Independent original quadratic region/dedup rule, with distinct tile keys.
    std::vector<unsigned> expected;std::set<unsigned> seen;
    auto add=[&](unsigned id,bool balanced) {
        if(seen.count(id))return;
        unsigned count=0,region=id/NeuronWidth/64*30+id%NeuronWidth/64;
        for(auto old:expected)count+=old/NeuronWidth/64*30+old%NeuronWidth/64==region;
        if(balanced && count>=4)return;
        seen.insert(id);expected.push_back(id);
    };
    for(int k=0;k<SpikeAnchorLimit/8;++k) {
        const unsigned index=referenceHash(config.seed ^ referenceHash(static_cast<unsigned>(previous)+1) ^ referenceHash(k+1))%SpikeTiles;
        add(spikePixel(SpikeTiles-1-index,8),false);
    }
    for(int k=0;k<SpikeTiles && expected.size()<SpikeAnchorLimit;++k)add(spikePixel(SpikeTiles-1-k,8),true);
    require(download(view.anchorIds,expected.size())==expected && scalar(view.counters).anchors==expected.size(),
        "Bounded hash/region tables preserve the exact prior ordered high-contribution selection");
    std::puts("PASS: fused local scores/outgoing and bounded anchor lookup match independent prior arithmetic and quadratic selection exactly.");
}

void longTest(VabpnnSpikeModel* m,VabpnnGpuFrame& frame) {
    vabpnnResetSpikeHistory(m,frame.stream);
    auto view=vabpnnSpikeModelView(m); auto* stats=vabpnnMutableSpikeStatistics(m,frame.stream);
    check(cudaMemsetAsync(stats,0,NeuronCount*sizeof(SpikePixelStatistics),frame.stream));
    SpikeConfig config; config.maxAnchors=3; config.refractorySeconds=0;
    put(vabpnnMutableSpikeConfig(m,frame.stream),config,frame.stream);
    const std::array<unsigned,3> ids={15*1920+15,15*1920+16,500*1920+500};
    for(auto id:ids) { SpikePixelStatistics stat{.3f,.3f,.4f,10,0}; put(stats+id,stat,frame.stream); }
    for(int i=0;i<6;++i) {
        frame.captureNanos+=20'000'000; frame.frameId++;
        vabpnnLaunchSpikeForward(frame);
    }
    check(cudaStreamSynchronize(frame.stream)); view=vabpnnSpikeModelView(m);
    const auto c=scalar(view.counters);
    require(c.anchors==3 && c.longEdges<=SpikeLongCapacity,"bounded high-contribution anchor selection");
    auto anchors=download(view.anchorIds,3); std::set<unsigned> chosen(anchors.begin(),anchors.end());
    require(chosen==std::set<unsigned>(ids.begin(),ids.end()),"top contributors from different blocks selected");
    std::map<unsigned,std::set<unsigned>> expected;
    for(auto src:chosen) {
        for(auto dst:chosen) if(src!=dst) expected[dst].insert(src);
        for(int dy=-6;dy<=6;++dy) for(int dx=-6;dx<=6;++dx) {
            if(std::abs(dx)+std::abs(dy)==0 || std::abs(dx)+std::abs(dy)>6) continue;
            const int x=src%1920+dx,y=src/1920+dy;
            if(x>=0 && x<1920 && y>=0 && y<1024) expected[y*1920+x].insert(src);
        }
    }
    auto rows=download(view.longRows,NeuronCount+1),cols=download(view.longColumns,c.longEdges);
    unsigned total=0;
    for(unsigned dst=0;dst<NeuronCount;++dst) {
        require(rows[dst]<=rows[dst+1],"long CSR rows monotonic");
        if(rows[dst]==rows[dst+1]) continue;
        std::set<unsigned> actual;
        for(unsigned e=rows[dst];e<rows[dst+1];++e) {
            require((cols[e]>>21)<4,"only four long-edge kinds");
            require(actual.insert(cols[e]&SpikeSourceMask).second,"long CSR deduplicates clique and nearby edges");
        }
        require(actual==expected[dst],"all anchor pairs and Manhattan <= 6 neighbors, including cross-grid neighbors");
        total+=actual.size(); expected.erase(dst);
    }
    require(expected.empty() && total==c.longEdges,"complete dense long topology");
    // Persist graph plastic state, reconstruct transient CSR on resume.
    ModelPart parts[5]; require(vabpnnSpikeCheckpointParts(m,parts)==5,"five persistent SNN parts");
    auto path=(std::filesystem::current_path()/L"spike-test-checkpoint.bin").wstring();
    std::vector<ModelPart> layout(parts,parts+5); vabpnn::saveModelCheckpoint(path,layout);
    check(cudaMemset(vabpnnMutableSpikeOutgoing(m,frame.stream),0,parts[0].bytes));
    require(vabpnn::loadModelCheckpoint(path,layout),"SNN snapshot restored");
    vabpnnResumeSpikeModel(m,frame.stream); check(cudaStreamSynchronize(frame.stream));
    require(scalar(view.counters).longEdges==c.longEdges,"long CSR rebuilt from saved anchors");
    for(auto s:download(vabpnnSpikeModelView(m).spikes,NeuronCount)) require(s==0,"resume never replays prior spikes");
    verifyLocal(m,false); std::filesystem::remove(path);
}
void contributionTest(VabpnnSpikeModel* m,VabpnnGpuFrame& frame) {
    auto* stats=vabpnnMutableSpikeStatistics(m,frame.stream);
    check(cudaMemsetAsync(stats,0,NeuronCount*sizeof(SpikePixelStatistics),frame.stream));
    const auto pixel=spikePixel(0,0); unsigned packed=0;
    for(int k=0;k<VoteCount;++k) packed|=(k==0 ? 2u:1u)<<(k*2);
    check(cudaMemsetAsync(vabpnnMutableSpikeVoteCache(m,frame.stream),0x55,NeuronCount*sizeof(unsigned),frame.stream));
    put(vabpnnMutableSpikeVoteCache(m,frame.stream)+pixel,packed,frame.stream);
    SpikeConfig config; put(vabpnnMutableSpikeConfig(m,frame.stream),config,frame.stream);
    vabpnnRecordSpikeVotes(m,frame.stream);
    DecisionOutput out{}; out.valid=1; out.selectedMask=1; out.posterior[0]=.75f;
    DecisionParameters param{}; param.rawVoteWeight[0]=2;
    DecisionOutput* gpuOut=nullptr; DecisionParameters* gpuParam=nullptr; int* valid=nullptr; unsigned* label=nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&gpuOut),sizeof(out)));
    check(cudaMalloc(reinterpret_cast<void**>(&gpuParam),sizeof(param)));
    check(cudaMalloc(reinterpret_cast<void**>(&valid),sizeof(int)));
    check(cudaMalloc(reinterpret_cast<void**>(&label),sizeof(unsigned)));
    put(gpuOut,out,frame.stream); put(gpuParam,param,frame.stream);
    put(valid,1,frame.stream); put(label,1u,frame.stream);
    const auto steps=scalar(vabpnnSpikeModelView(m).counters).contributionUpdates;
    vabpnnObserveSpikeContribution(m,gpuOut,gpuParam,valid,label,frame.stream);
    check(cudaStreamSynchronize(frame.stream));
    auto actual=scalar(stats+pixel);
    const double delta=2.0/NeuronCount;
    const double exact=(std::log1p(.75*std::expm1(-delta))+delta)*NeuronCount;
    require(std::fabs(actual.crossEntropy-exact*.05/VoteCount)<1e-7,"stable leave-one-neuron-out cross entropy gain");
    require(std::fabs(actual.hit-exact*.05)<1e-6,"true-positive largest action-hit contribution");
    require(std::fabs(actual.positive-.05/VoteCount)<1e-7 && std::fabs(actual.zero-.05*(VoteCount-1)/VoteCount)<1e-7,
            "EMA signed vote histogram for Shannon and Simpson");
    // Next topology tick reduces the per-action maxima to a global winner.
    vabpnnResetSpikeHistory(m,frame.stream);
    for(int i=0;i<6;++i) { frame.captureNanos+=20'000'000; frame.frameId++; vabpnnLaunchSpikeForward(frame); }
    check(cudaStreamSynchronize(frame.stream));
    require(scalar(vabpnnSpikeModelView(m).championIds)==pixel,"action champion pixel selected globally");
    put(valid,0,frame.stream);
    vabpnnObserveSpikeContribution(m,gpuOut,gpuParam,valid,label,frame.stream); check(cudaStreamSynchronize(frame.stream));
    require(scalar(vabpnnSpikeModelView(m).counters).contributionUpdates==steps+1,"no supervised plastic signal in model mode");
    cudaFree(gpuOut); cudaFree(gpuParam); cudaFree(valid); cudaFree(label);
}
void lifTest(VabpnnSpikeModel* m,VabpnnGpuFrame& frame,VabpnnNeuronModel* neuron) {
    vabpnnResetSpikeHistory(m,frame.stream);
    auto* outgoing=vabpnnMutableSpikeOutgoing(m,frame.stream);
    check(cudaMemsetAsync(outgoing,0,NeuronCount*4*sizeof(unsigned short),frame.stream));
    for(unsigned short kind=0;kind<4;++kind) {
        const unsigned short edge=(kind+1)|(kind<<8)|0x400;
        put(outgoing+kind,edge,frame.stream);
    }
    vabpnnRebuildSpikeLocalCsr(m,frame.stream);
    // Clear long topology from earlier tests by loading empty anchors and counters.
    ModelPart parts[5]; vabpnnSpikeCheckpointParts(m,parts);
    check(cudaMemsetAsync(parts[4].device,0xff,parts[4].bytes,frame.stream));
    SpikeCounters counters{}; put(static_cast<SpikeCounters*>(parts[3].device),counters,frame.stream);
    vabpnnResumeSpikeModel(m,frame.stream);
    SpikeConfig config; config.membraneTau=.001f; config.threshold=.6f; config.minimumThreshold=.1f;
    config.maximumThreshold=2; config.stimulationGain=.2f; config.polarityGain=.2f; config.negativeMargin=.04f;
    config.refractorySeconds=.04f; config.maxAnchors=0;
    put(vabpnnMutableSpikeConfig(m,frame.stream),config,frame.stream);
    vabpnnLaunchSpikeForward(frame); check(cudaStreamSynchronize(frame.stream));
    const std::int8_t one=1; put(vabpnnMutableSpikeSignals(m,frame.stream),one,frame.stream);
    frame.captureNanos+=20'000'000; frame.frameId++;
    vabpnnLaunchSpikeForward(frame); check(cudaStreamSynchronize(frame.stream));
    auto signals=download(vabpnnSpikeModelView(m).spikes,20);
    require(signals[1]==1 && signals[2]==0 && signals[3]==1 && signals[4]==-1 && signals[5]==0,
            "four typed CSR effects control stimulation and positive/negative release");
    // Use independent weights to ensure signed pulses actually gate the 13 raw votes.
    check(cudaMemsetAsync(vabpnnMutableNeuronWeights(neuron,frame.stream),0,ParameterCount,frame.stream));
    for(int k=0;k<VoteCount;++k) for(unsigned pixel:{1u,2u,3u,4u})
        put(vabpnnMutableNeuronWeights(neuron,frame.stream)+weightIndex(pixel,0,k),TernaryWeight::Positive,frame.stream);
    vabpnnLaunchTernaryVotes(frame,neuron); check(cudaStreamSynchronize(frame.stream));
    for(auto total:download(vabpnnNeuronModelView(neuron).voteSums,13)) require(total==1,"raw totals = positive + positive - negative; quiet neurons abstain");
    require((scalar(vabpnnSpikeModelView(m).packedVotes+4)&3)==0,"negative pulse encoded as negative vote");
    frame.captureNanos+=20'000'000; frame.frameId++; vabpnnLaunchSpikeForward(frame);
    check(cudaStreamSynchronize(frame.stream));
    signals=download(vabpnnSpikeModelView(m).spikes,5);
    require(signals[1]==0 && signals[3]==0 && signals[4]==0,"refractory interval suppresses repeated spikes");
    config.membraneTau=std::numeric_limits<float>::quiet_NaN();
    put(vabpnnMutableSpikeConfig(m,frame.stream),config,frame.stream);
    frame.captureNanos+=20'000'000; vabpnnLaunchSpikeForward(frame); check(cudaStreamSynchronize(frame.stream));
    require(scalar(vabpnnSpikeModelView(m).counters).invalidConfig==1,"invalid time constants fail closed");
    for(auto s:download(vabpnnSpikeModelView(m).spikes,NeuronCount)) require(s==0,"invalid config emits no spikes");
    config=SpikeConfig{}; put(vabpnnMutableSpikeConfig(m,frame.stream),config,frame.stream);
    frame.captureNanos+=500'000'000; vabpnnLaunchSpikeForward(frame); check(cudaStreamSynchronize(frame.stream));
    for(auto s:download(vabpnnSpikeModelView(m).spikes,NeuronCount)) require(s==0,"long gaps reset rather than replay history");
}

void placementTest(VabpnnSpikeModel* m,VabpnnGpuFrame& frame,VabpnnNeuronModel* neuron) {
    // A dim, silent pixel supports the teacher's action; its bright neighbor opposes
    // it. Probe observations must discover the dim pixel without any LIF firing.
    auto saved=frame;VabpnnDecisionModel* decision=vabpnnCreateDecisionModel(frame.stream);
    unsigned char* image=nullptr;std::size_t pitch=0;
    check(cudaMallocPitch(reinterpret_cast<void**>(&image),&pitch,NeuronWidth*4,NeuronHeight));
    check(cudaMemset2DAsync(image,pitch,0,NeuronWidth*4,NeuronHeight,frame.stream));
    frame.rgba=image;frame.rgbaPitch=pitch;frame.width=NeuronWidth;frame.height=NeuronHeight;frame.decisionModel=decision;
    const unsigned dim=32*NeuronWidth+32,bright=dim+1,tile=2*120+2;
    const std::array<unsigned char,4> darkColor{32,32,32,255},brightColor{240,240,240,255};
    check(cudaMemcpyAsync(image+(NeuronHeight-1-32)*pitch+32*4,darkColor.data(),4,cudaMemcpyHostToDevice,frame.stream));
    check(cudaMemcpyAsync(image+(NeuronHeight-1-32)*pitch+33*4,brightColor.data(),4,cudaMemcpyHostToDevice,frame.stream));
    check(cudaMemsetAsync(vabpnnMutableNeuronWeights(neuron,frame.stream),0,ParameterCount,frame.stream));
    put(vabpnnMutableNeuronWeights(neuron,frame.stream)+weightIndex(dim,0,0),TernaryWeight::Positive,frame.stream);
    put(vabpnnMutableNeuronWeights(neuron,frame.stream)+weightIndex(bright,0,0),TernaryWeight::Negative,frame.stream);
    auto view=vabpnnSpikeModelView(m);
    check(cudaMemsetAsync(vabpnnMutableSpikeStatistics(m,frame.stream),0,NeuronCount*sizeof(SpikePixelStatistics),frame.stream));
    auto placementPart=vabpnnSpikePlacementCheckpointPart(m);
    check(cudaMemsetAsync(placementPart.device,0,placementPart.bytes,frame.stream));
    SpikePlacementTile state{};
    state.arms[0]={bright,0,0,0};state.arms[1]={dim,0,0,0};
    for(int a=2;a<SpikePlacementScouts;++a) state.arms[a]={spikePixel(tile,a+20),-4,16,100};
    put(static_cast<SpikePlacementTile*>(placementPart.device)+tile,state,frame.stream);
    SpikeConfig config;config.maxAnchors=1;put(vabpnnMutableSpikeConfig(m,frame.stream),config,frame.stream);
    DecisionParameters params{};params.rawVoteWeight[0]=2;
    DecisionOutput output{};output.valid=1;for(float& p:output.posterior)p=.75f;
    put(vabpnnMutableDecisionParameters(decision,frame.stream),params,frame.stream);
    put(const_cast<DecisionOutput*>(vabpnnDecisionModelView(decision).output),output,frame.stream);
    int* valid=nullptr;unsigned* label=nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&valid),sizeof(int)));check(cudaMalloc(reinterpret_cast<void**>(&label),sizeof(unsigned)));
    put(valid,1,frame.stream);put(label,1u,frame.stream);
    vabpnnResetSpikeHistory(m,frame.stream);
    for(int i=0;i<5;++i) {frame.captureNanos+=20'000'000;vabpnnLaunchSpikeForward(frame);vabpnnLaunchTernaryVotes(frame,neuron);}
    check(cudaStreamSynchronize(frame.stream));
    require(scalar(vabpnnSpikeModelView(m).spikes+dim)==0,"Dim candidate is genuinely silent, not a bright firing neuron");
    vabpnnObserveSpikePrediction(frame);
    vabpnnObserveSpikeContribution(m,vabpnnDecisionModelView(decision).output,vabpnnDecisionModelView(decision).parameters,valid,label,frame.stream);
    check(cudaStreamSynchronize(frame.stream));state=scalar(static_cast<SpikePlacementTile*>(placementPart.device)+tile);
    require(state.arms[1].utility>state.arms[0].utility && state.arms[1].visits==1 && state.supervisedRounds==1,
        "Causal feedback rewards a dim positive contributor above a bright wrong voter");
    frame.captureNanos+=20'000'000;vabpnnLaunchSpikeForward(frame);check(cudaStreamSynchronize(frame.stream));
    require(scalar(vabpnnSpikeModelView(m).anchorIds)==dim,"Placement rank, not brightness, determines the actual CSR anchor");
    put(label,0u,frame.stream);
    for(int i=0;i<4;++i) {frame.captureNanos+=20'000'000;vabpnnLaunchSpikeForward(frame);vabpnnLaunchTernaryVotes(frame,neuron);}
    vabpnnObserveSpikePrediction(frame);
    vabpnnObserveSpikeContribution(m,vabpnnDecisionModelView(decision).output,vabpnnDecisionModelView(decision).parameters,valid,label,frame.stream);
    check(cudaStreamSynchronize(frame.stream));state=scalar(static_cast<SpikePlacementTile*>(placementPart.device)+tile);
    require(state.arms[0].utility>state.arms[1].utility && state.supervisedRounds==2,
        "The same scout pool learns changed feedback instead of permanently fixing its location");
    frame.captureNanos+=20'000'000;vabpnnLaunchSpikeForward(frame);check(cudaStreamSynchronize(frame.stream));
    require(scalar(vabpnnSpikeModelView(m).anchorIds)==bright,"Changed learned utility relocates the real high-contribution anchor");
    std::vector<ModelPart> parts{placementPart};
    const auto path=(std::filesystem::current_path()/L"placement-v7-test.bin").wstring();
    saveModelCheckpoint(path,parts,7);
    check(cudaMemset(placementPart.device,0,placementPart.bytes));
    require(loadModelCheckpoint(path,parts,false,7),"Placement values are independently persisted in version seven");
    auto restored=scalar(static_cast<SpikePlacementTile*>(placementPart.device)+tile);
    require(std::memcmp(&state,&restored,sizeof(state))==0,"Placement visits, feedback counts and values restored exactly");
    std::filesystem::remove(path);
    const auto before=restored.rounds;frame.evaluationMode=true;
    for(int i=0;i<5;++i) {frame.captureNanos+=20'000'000;vabpnnLaunchSpikeForward(frame);vabpnnLaunchTernaryVotes(frame,neuron);vabpnnObserveSpikePrediction(frame);}
    check(cudaStreamSynchronize(frame.stream));
    require(scalar(static_cast<SpikePlacementTile*>(placementPart.device)+tile).rounds>before,
        "Placement learning keeps probing and adapting in weight-frozen prediction");
    // A restored pixel at ID zero must not compare the new scene against a
    // fictitious black history just because a zeroed transient ID also equals zero.
    check(cudaMemset2DAsync(image,pitch,64,NeuronWidth*4,NeuronHeight,frame.stream));
    state=SpikePlacementTile{};state.arms[0]={0,0,0,10};
    for(int a=1;a<SpikePlacementScouts;++a)state.arms[a]={spikePixel(tile,a+20),0,0,10};
    put(static_cast<SpikePlacementTile*>(placementPart.device)+tile,state,frame.stream);
    vabpnnResetSpikeHistory(m,frame.stream);
    for(int i=0;i<5;++i) {frame.captureNanos+=20'000'000;vabpnnLaunchSpikeForward(frame);vabpnnLaunchTernaryVotes(frame,neuron);vabpnnObserveSpikePrediction(frame);}
    check(cudaStreamSynchronize(frame.stream));
    require(scalar(static_cast<SpikePlacementTile*>(placementPart.device)+tile).arms[0].utility==0,
        "New-scene pixel zero has no fictitious RGB novelty after transient reset");
    std::puts("PASS: ungated dim scout wins by contribution, changed feedback relocates anchors, v7 restores learned placement, frozen prediction keeps scouting; scene reset never invents novelty.");
    cudaFree(valid);cudaFree(label);cudaFree(image);vabpnnDestroyDecisionModel(decision);frame=saved;
}

void legacyHeaderTest(cudaStream_t stream) {
    float* pointer=nullptr; check(cudaMalloc(reinterpret_cast<void**>(&pointer),sizeof(float)));
    put(pointer,3.5f,stream); std::vector<ModelPart> parts{{pointer,sizeof(float)}};
    auto path=(std::filesystem::current_path()/L"legacy-header-test.bin").wstring();
    vabpnn::saveModelCheckpoint(path,parts);
    { std::fstream f(std::filesystem::path(path),std::ios::binary|std::ios::in|std::ios::out);
      CheckpointHeader h; f.read(reinterpret_cast<char*>(&h),sizeof(h)); h.magic[6]='6'; h.version=1;
      f.seekp(0); f.write(reinterpret_cast<char*>(&h),sizeof(h)); }
    put(pointer,0.0f,stream);
    require(vabpnn::loadModelCheckpoint(path,parts,true) && scalar(pointer)==3.5f,"explicit legacy v1 layout remains readable");
    bool rejected=false; try { vabpnn::loadModelCheckpoint(path,parts); } catch(...) { rejected=true; }
    require(rejected,"legacy layout cannot be mistaken for new layout");
    std::filesystem::remove(path); cudaFree(pointer);
}
int main() {
    cudaStream_t stream=nullptr; VabpnnSpikeModel* model=nullptr; VabpnnNeuronModel* neuron=nullptr;
    unsigned char* image=nullptr; int result=0;
    try {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); model=vabpnnCreateSpikeModel(stream);
        neuron=vabpnnCreateNeuronModel(stream); check(cudaStreamSynchronize(stream));
        verifyLocal(model,true);
        check(cudaMalloc(reinterpret_cast<void**>(&image),16)); check(cudaMemset(image,128,16));
        VabpnnGpuFrame frame{image,8,nullptr,2,2,1,1'000'000'000ULL,stream,neuron,nullptr,nullptr,nullptr,model};
        for(int i=0;i<5;++i) { frame.captureNanos+=20'000'000; vabpnnLaunchSpikeForward(frame); }
        check(cudaStreamSynchronize(stream));
        require(scalar(vabpnnSpikeModelView(model).counters).topologyUpdates==0,"exactly five predictions use initial adjacency");
        frame.captureNanos+=20'000'000; vabpnnLaunchSpikeForward(frame); check(cudaStreamSynchronize(stream));
        require(scalar(vabpnnSpikeModelView(model).counters).topologyUpdates==1,"rewire before sixth prediction");
        verifyLocal(model,false); optimizedTopologyTest(model,frame);longTest(model,frame); contributionTest(model,frame); lifTest(model,frame,neuron);placementTest(model,frame,neuron);
        legacyHeaderTest(stream);
        std::printf("PASS: bounded local CSR, five-prediction plasticity, dense anchor clique/radius6, supervised CE/diversity/hit, signed LIF/SpMV, refractory/reset and checkpoint migration; spikeBytes=%llu\n",
            static_cast<unsigned long long>(vabpnnSpikeModelBytes(model)));
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); result=1; }
    if(stream) cudaStreamSynchronize(stream);
    vabpnnDestroySpikeModel(model); vabpnnDestroyNeuronModel(neuron); if(image) cudaFree(image);
    if(stream) cudaStreamDestroy(stream); return result;
}

