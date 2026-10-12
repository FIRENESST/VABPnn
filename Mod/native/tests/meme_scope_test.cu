#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "FrameOperator.h"
#include "MemeState.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "NetworkScope.h"
#include "model_checkpoint.h"
#include "action_checkpoint_migration.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
using namespace vabpnn;
void check(cudaError_t c) { if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c)); }
void require(bool ok,const char* msg) {if(!ok) throw std::runtime_error(msg);}
template<class T> T read(const T* p) {T v;check(cudaMemcpy(&v,p,sizeof(v),cudaMemcpyDeviceToHost));return v;}
template<class T> void put(T* p,const T& v,cudaStream_t s) {
    check(cudaMemcpyAsync(p,&v,sizeof(v),cudaMemcpyHostToDevice,s));check(cudaStreamSynchronize(s));
}
struct Fixture {
    cudaStream_t stream{}; VabpnnNeuronModel* neuron{};VabpnnDecisionModel* decision{};
    VabpnnOnlineModel* online{};VabpnnSpikeModel* spike{};VabpnnJepaModel* jepa{};
    unsigned char* image{};float* rgb{};int* valid{};unsigned* label{};
    VabpnnBlockModel* blocks{};VabpnnMemeModel* memes{};MemePersistent* p{};BlockPersistent* bp{};
    VabpnnGpuFrame frame{};DemoSample demo{};int width=7,height=5;
    std::vector<unsigned char> pixels;
    Fixture() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        neuron=vabpnnCreateNeuronModel(stream);decision=vabpnnCreateDecisionModel(stream);
        online=vabpnnCreateOnlineModel(stream);spike=vabpnnCreateSpikeModel(stream);jepa=vabpnnCreateJepaModel(stream);
        blocks=vabpnnSpikeBlockModel(spike);memes=vabpnnBlockMemeModel(blocks);
        p=vabpnnMutableMemePersistent(memes,stream);bp=vabpnnMutableBlockPersistent(blocks,stream);
        check(cudaMalloc(reinterpret_cast<void**>(&image),width*height*4));
        check(cudaMalloc(reinterpret_cast<void**>(&rgb),width*height*3*sizeof(float)));
        check(cudaMalloc(reinterpret_cast<void**>(&valid),sizeof(int)));check(cudaMalloc(reinterpret_cast<void**>(&label),sizeof(unsigned)));
        put(valid,1,stream);put(label,1u,stream);
        pixels.resize(width*height*4);
        for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
            int at=(y*width+x)*4;pixels[at]=33+x*27;pixels[at+1]=45+y*37;pixels[at+2]=31+x*11+y*13;pixels[at+3]=255;
        }
        check(cudaMemcpyAsync(image,pixels.data(),pixels.size(),cudaMemcpyHostToDevice,stream));
        // RGB tensor deliberately contains stale unrelated values: context must use the current RGBA.
        check(cudaMemsetAsync(rgb,0xff,width*height*3*sizeof(float),stream));
        frame={image,static_cast<std::size_t>(width*4),rgb,width,height,1,1'000'000'000ULL,stream,neuron,decision,jepa,online,spike};
        demo.active=1;demo.epoch=1;demo.pointerMode=1;vabpnnSetDemoSample(online,&demo,stream);
        auto bc=BlockConfig{};bc.warmup=0;bc.trainEvery=1;bc.rankEvery=65536;bc.communication=1;
        vabpnnConfigureBlockModel(blocks,bc,stream);sync();
    }
    ~Fixture() {
        if(stream) cudaStreamSynchronize(stream);
        vabpnnDestroySpikeModel(spike);vabpnnDestroyOnlineModel(online);vabpnnDestroyJepaModel(jepa);
        vabpnnDestroyDecisionModel(decision);vabpnnDestroyNeuronModel(neuron);
        if(image) cudaFree(image);if(rgb) cudaFree(rgb);if(valid) cudaFree(valid);if(label) cudaFree(label);
        if(stream) cudaStreamDestroy(stream);
    }
    void sync() {check(cudaStreamSynchronize(stream));}
    DeviceMemeView view() {return vabpnnMemeModelView(memes);}
    void next(double dt=.02) {++frame.frameId;frame.captureNanos+=static_cast<unsigned long long>(dt*1e9);}
    void prepare() {vabpnnPrepareMemes(frame,true);sync();}
    void clear(int mode=1) {
        auto bc=read(&bp->config);bc.communication=mode;vabpnnConfigureBlockModel(blocks,bc,stream);
        check(cudaMemsetAsync(p->prototypes,0,MemeBlocks*sizeof(MemePacket),stream));
        check(cudaMemsetAsync(p->policy,0,MemeBlocks*sizeof(MemePolicy),stream));
        auto sv=vabpnnSpikeModelView(spike);auto c=read(sv.counters);c.longEdges=0;
        put(const_cast<SpikeCounters*>(sv.counters),c,stream);next();prepare();
    }
    MemePacket learned(int tile,float confidence=.9f,float utility=.7f) {
        auto v=view();MemePacket packet;check(cudaMemcpy(packet.context,v.contexts+tile*MemeContext,sizeof(packet.context),cudaMemcpyDeviceToHost));
        for(int k=0;k<MemeChannels;++k) packet.evidence[k]=k==0 ? .8f:-.6f;
        packet.support=64;packet.revision=7;packet.origin=tile;packet.confidence=confidence;
        packet.utility=utility;packet.agreement=1;packet.coverage=1;return packet;
    }
    void inject(int tile,float confidence=.9f,float utility=.7f) {
        auto packet=learned(tile,confidence,utility);packet.sequence=frame.frameId;packet.age=0;
        put(view().packets+tile*2,packet,stream);
    }
    void csr(const std::vector<int>& sources) {
        std::vector<unsigned> rows(NeuronCount+1);std::vector<unsigned> columns;
        for(unsigned pixel=0;pixel<NeuronCount;++pixel) {
            rows[pixel]=static_cast<unsigned>(columns.size());
            if(pixel/1920<16 && pixel%1920<16)
                for(int source:sources) columns.push_back(spikePixel(source,73));
        }
        rows[NeuronCount]=static_cast<unsigned>(columns.size());
        auto v=vabpnnSpikeModelView(spike);
        check(cudaMemcpyAsync(const_cast<unsigned*>(v.longRows),rows.data(),rows.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(const_cast<unsigned*>(v.longColumns),columns.data(),columns.size()*4,cudaMemcpyHostToDevice,stream));
        auto c=read(v.counters);c.longEdges=static_cast<unsigned>(columns.size());put(const_cast<SpikeCounters*>(v.counters),c,stream);
    }
};

std::unique_ptr<MemeScopeTelemetry> capture(Fixture& f,MemeScopeTelemetry* device,int tile) {
    vabpnnLaunchMemeScope(f.frame,tile,device);f.sync();
    auto host=std::make_unique<MemeScopeTelemetry>();
    check(cudaMemcpy(host.get(),device,sizeof(*host),cudaMemcpyDeviceToHost));return host;
}
int main() {
    MemeScopeTelemetry* device=nullptr;MemeRelayTrace* trace=nullptr;
    try {
        Fixture f;check(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(MemeScopeTelemetry)));
        check(cudaMalloc(reinterpret_cast<void**>(&trace),MemeBlocks*sizeof(MemeRelayTrace)));
        check(cudaMemsetAsync(trace,0,MemeBlocks*sizeof(MemeRelayTrace),f.stream));f.frame.memeTrace=trace;
        f.clear();f.inject(1,.95f);f.next();f.prepare();
        auto s=capture(f,device,0);
        require(s->meta[0]==f.frame.frameId && s->meta[1]==f.frame.captureNanos && s->meta[3]==0 && s->meta[39]==1,"coherent schema/frame/time");
        require(s->meta[18]==1 && s->meta[19]==1 && s->meta[20]==0 && s->meta[34]==1 && s->meta[35]==1,
            "actual neighbor source, channel, originating packet and hop");
        require(s->meta[17]==f.frame.frameId && s->meta[37]==f.frame.captureNanos,"receipt timestamp");
        const auto acceptedFrame=s->meta[17],acceptedSequence=s->meta[33];
        // No new dispatch: held packets age, and duplicate replay must not invent a new hop.
        f.next(.2);f.prepare();s=capture(f,device,0);
        require(s->meta[17]==acceptedFrame && s->meta[33]==acceptedSequence,"retained packet keeps real last accepted hop");
        std::vector<MemePacket> packets(MemeBlocks*2);
        std::vector<float> keys(MemeBlocks*MemeContext);
        std::vector<MemeRelayTrace> traces(MemeBlocks);
        check(cudaMemcpy(packets.data(),f.view().packets,packets.size()*sizeof(MemePacket),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(keys.data(),f.view().contexts,keys.size()*sizeof(float),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(traces.data(),trace,traces.size()*sizeof(MemeRelayTrace),cudaMemcpyDeviceToHost));
        const auto cfg=read(&f.p->config);
        for(int tile=0;tile<MemeBlocks;++tile) {
            const auto& own=packets[tile*2];const auto& imported=packets[tile*2+1];
            const bool known=memeTraceMatches(traces[tile],imported);
            require(std::abs(s->values[MemeScopeOwn+tile]-memePacketWeight(own,keys.data()+tile*8,cfg))<1e-6
                && std::abs(s->values[MemeScopeImported+tile]-memePacketWeight(imported,keys.data()+tile*8,cfg))<1e-6,"every map tile equals actual GPU packet weight");
            require(s->discrete[MemeScopeOrigin+tile]==(imported.support ? static_cast<int>(imported.origin):-1)
                && s->discrete[MemeScopeSender+tile]==(known ? static_cast<int>(traces[tile].sourceTile):-1)
                && s->identities[tile]==imported.sequence,"all source identities agree with the actual packet bank");
        }
        // Independently verify detailed evidence and separate actual cached actor feature.
        MemeAction action;action.active=1;action.feature[0]=-.31f*MemeInputScale;
        put(f.view().actions,action,f.stream);s=capture(f,device,0);
        require(s->values[MemeScopeDetail+2*MemeScopePacketFloats]==packets[1].evidence[0]
            && std::abs(s->values[MemeScopeAction]+.31f)<1e-6
            && std::abs(s->values[MemeScopeFeatures]-read(f.view().features)/MemeInputScale)<1e-6,
            "packet evidence, candidate mix and actually used cached actor input stay separate");
        f.clear();f.inject(240,.9f);f.csr({240});f.next();f.prepare();s=capture(f,device,0);
        require(s->meta[18]==240 && s->meta[19]==2 && s->meta[21]==240 && s->meta[34]==240,
            "actual long CSR relay is distinguished from candidate routes");
        auto before=std::make_unique<MemePersistent>(),after=std::make_unique<MemePersistent>();
        check(cudaMemcpy(before.get(),f.p,sizeof(*before),cudaMemcpyDeviceToHost));
        f.frame.evaluationMode=true;
        for(int i=0;i<7;++i)s=capture(f,device,(i*137)%MemeBlocks);
        check(cudaMemcpy(after.get(),f.p,sizeof(*after),cudaMemcpyDeviceToHost));
        require(std::memcmp(before.get(),after.get(),sizeof(*before))==0 && s->meta[2]==1,
            "viewer is read-only, including learned knowledge, AdamW, counters and evaluation state");
        // A viewer opened later must not infer a sender from the packet's origin.
        check(cudaMemsetAsync(trace,0,MemeBlocks*sizeof(MemeRelayTrace),f.stream));
        s=capture(f,device,0);require(s->meta[18]==~0ULL && s->discrete[MemeScopeSender]==-1,"unknown historical hop remains unknown");
        f.clear(0);f.inject(240);f.csr({240});f.next();f.prepare();s=capture(f,device,0);
        require(s->meta[6]==0 && s->values[MemeScopeImported]==0 && s->discrete[MemeScopeSender]==-1,
            "independent mode clears receiving and recorded relay");
        f.clear();f.inject(1);f.next();f.prepare();
        for(int i=0;i<6;++i) {f.next(.6);f.prepare();}
        s=capture(f,device,0);
        require(s->values[MemeScopeImported]==0 && s->discrete[MemeScopeSender]==-1,"expired packet cannot leave a false active arrow");
        bool rejected=false;try{capture(f,device,MemeBlocks);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"invalid tile rejected before kernel launch");
        std::printf("PASS: detailed meme snapshot %zu bytes; all 7680 GPU packet weights/identities, real grid/CSR last hops, retention/TTL, actual input and read-only frozen state.\n",sizeof(MemeScopeTelemetry));
        f.frame.memeTrace=nullptr;cudaFree(device);cudaFree(trace);return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what());if(device)cudaFree(device);if(trace)cudaFree(trace);return 1;
    }
}
