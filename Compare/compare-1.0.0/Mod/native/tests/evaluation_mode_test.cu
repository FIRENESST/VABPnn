#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "FrameOperator.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "MemeState.h"
#include "model_checkpoint.h"
#include <cstdio>
#include <vector>
#include <array>
#include <stdexcept>
using namespace vabpnn;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void check(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
template<class T>T get(const T* device){T value;check(cudaMemcpy(&value,device,sizeof(T),cudaMemcpyDeviceToHost));return value;}
struct Fixture {
    cudaStream_t stream=nullptr;VabpnnNeuronModel* neuron=nullptr;VabpnnDecisionModel* decision=nullptr;
    VabpnnJepaModel* jepa=nullptr;VabpnnOnlineModel* online=nullptr;VabpnnSpikeModel* spike=nullptr;
    VabpnnGpuFrame frame{};DemoSample demo{};MemeRelayTrace* trace=nullptr;
    Fixture() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        neuron=vabpnnCreateNeuronModel(stream);decision=vabpnnCreateDecisionModel(stream);
        jepa=vabpnnCreateJepaModel(stream);online=vabpnnCreateOnlineModel(stream);spike=vabpnnCreateSpikeModel(stream);
        frame.width=37;frame.height=23;frame.frameId=0;frame.captureNanos=1'000'000'000;
        frame.stream=stream;frame.neuronModel=neuron;frame.decisionModel=decision;frame.jepaModel=jepa;frame.onlineModel=online;frame.spikeModel=spike;
        unsigned char* image=nullptr;float* rgb=nullptr;
        check(cudaMallocPitch(reinterpret_cast<void**>(&image),&frame.rgbaPitch,37*4,23));frame.rgba=image;
        check(cudaMemset2DAsync(image,frame.rgbaPitch,204,37*4,23,stream));
        check(cudaMalloc(reinterpret_cast<void**>(&rgb),37*23*3*sizeof(float)));frame.rgbNchw=rgb;
        demo.active=1;demo.epoch=1;demo.pointerMode=1;
        check(cudaMalloc(reinterpret_cast<void**>(&trace),MemeBlocks*sizeof(MemeRelayTrace)));
        check(cudaMemsetAsync(trace,0,MemeBlocks*sizeof(MemeRelayTrace),stream));frame.memeTrace=trace;
        check(cudaStreamSynchronize(stream));
    }
    ~Fixture() {
        cudaStreamSynchronize(stream);cudaFree(const_cast<unsigned char*>(frame.rgba));cudaFree(frame.rgbNchw);cudaFree(trace);
        vabpnnDestroySpikeModel(spike);vabpnnDestroyOnlineModel(online);vabpnnDestroyJepaModel(jepa);
        vabpnnDestroyDecisionModel(decision);vabpnnDestroyNeuronModel(neuron);cudaStreamDestroy(stream);
    }
    void next() {frame.evaluationMode=demo.modelControl!=0;frame.frameId++;frame.captureNanos+=20'000'000;vabpnnSetDemoSample(online,&demo,stream);vabpnnLaunchOperator(frame);check(cudaStreamSynchronize(stream));}
    std::vector<ModelPart> parts() {
        std::vector<ModelPart> result{{vabpnnMutableNeuronWeights(neuron,stream),ParameterCount},
            {vabpnnMutableDecisionParameters(decision,stream),sizeof(DecisionParameters)}};
        ModelPart extra[8];auto add=[&](std::size_t n){result.insert(result.end(),extra,extra+n);};
        add(vabpnnJepaCheckpointParts(jepa,extra));add(vabpnnOnlineCheckpointParts(online,extra));
        add(vabpnnSpikeCheckpointParts(spike,extra));add(vabpnnDecisionPositionCheckpointParts(decision,extra));
        auto* block=vabpnnSpikeBlockModel(spike);result.push_back(vabpnnBlockCheckpointPart(block));
        result.push_back(vabpnnMemeCheckpointPart(vabpnnBlockMemeModel(block)));check(cudaStreamSynchronize(stream));return result;
    }
};
using Digest=std::array<unsigned char,32>;
Digest deviceDigest(ModelPart part){
    Sha256 hash;std::vector<unsigned char> bytes(1024*1024);
    for(std::size_t offset=0;offset<part.bytes;){
        const auto length=std::min(bytes.size(),part.bytes-offset);
        check(cudaMemcpy(bytes.data(),static_cast<unsigned char*>(part.device)+offset,length,cudaMemcpyDeviceToHost));
        hash.add(bytes.data(),length);offset+=length;
    }
    Digest result;hash.finish(result.data());return result;
}
struct Signature{
    std::vector<Digest> values;std::vector<std::string> names;
    void add(const char* name,ModelPart part){names.emplace_back(name);values.push_back(deviceDigest(part));}
};
Signature learnedState(Fixture& f){
    Signature result;auto parts=f.parts();
    const char* names[]={"ternary","decision","JEPA","EMA","JEPA first","JEPA second","Adam groups",
        "pixel master","wheel/online Adam","local outgoing","SNN statistics","SNN configuration"};
    for(int i=0;i<9;++i)result.add(names[i],parts[i]);
    result.add("minutes/attention/AdamW",vabpnnDecisionMinuteCheckpointPart(f.decision));
    result.add("SNN configuration",parts[11]);result.add("coordinates/Adam",parts[14]);
    auto block=vabpnnBlockModelView(vabpnnSpikeBlockModel(f.spike));
    result.add("block weights/Adam/best/scores",{const_cast<BlockState*>(block.state),BlockCount*sizeof(BlockState)});
    auto meme=vabpnnMemeModelView(vabpnnBlockMemeModel(vabpnnSpikeBlockModel(f.spike)));
    result.add("meme projection/Adam/best",{meme.persistent->policy,MemeBlocks*sizeof(MemePolicy)});
    std::vector<MemePacket> prototypes(MemeBlocks);
    check(cudaMemcpy(prototypes.data(),meme.persistent->prototypes,prototypes.size()*sizeof(MemePacket),cudaMemcpyDeviceToHost));
    for(auto& p:prototypes)p.age=0; // Monotonic inference age is not learned knowledge.
    Sha256 hash;hash.add(reinterpret_cast<unsigned char*>(prototypes.data()),prototypes.size()*sizeof(MemePacket));
    Digest digest;hash.finish(digest.data());result.names.emplace_back("meme knowledge/support/confidence");result.values.push_back(digest);
    return result;
}
void equal(const Signature& a,const Signature& b){
    require(a.names==b.names,"Signature layout");
    for(std::size_t i=0;i<a.values.size();++i)if(a.values[i]!=b.values[i])
        throw std::runtime_error("Evaluation modified "+a.names[i]);
}
void reset(Fixture& f){
    vabpnnResetDecisionState(f.decision,f.stream);vabpnnResetJepaHistory(f.jepa,f.stream);
    vabpnnResetOnlineHistory(f.online,f.stream);vabpnnResetSpikeHistory(f.spike,f.stream);
}
int main(){
    try{
        Fixture f;JepaConfig cfg;cfg.trainEveryFrames=1;
        check(cudaMemcpyAsync(vabpnnMutableJepaConfig(f.jepa,f.stream),&cfg,sizeof(cfg),cudaMemcpyHostToDevice,f.stream));
        auto* blocks=vabpnnSpikeBlockModel(f.spike);BlockConfig bcfg;bcfg.warmup=0;bcfg.rankEvery=5;
        vabpnnConfigureBlockModel(blocks,bcfg,f.stream);
        auto* memes=vabpnnBlockMemeModel(blocks);
        f.demo.heldKeys=1u|(1u<<13)|(1u<<14);
        auto advance=[&](){
            for(int k:{0,6,7})f.demo.keyHeldNanos[k]+=20'000'000;
            f.demo.mouseX+=.4;f.demo.mouseY+=.2;f.demo.wheel+=.05;
            f.frame.position={f.frame.captureNanos+20'000'000,f.demo.epoch,42.25,64,-7.125};f.next();
        };
        for(int i=0;i<64;++i)advance();
        const auto trained=get(vabpnnOnlineOutput(f.online));const auto j=get(vabpnnJepaModelView(f.jepa).output);
        const auto bb=get(vabpnnBlockModelView(blocks).counters);const auto mb=get(&vabpnnMemeModelView(memes).persistent->counters);
        const auto pb=get(vabpnnDecisionModelView(f.decision).position);
        require(trained.examples>40 && trained.pixelSteps && trained.decisionSteps && j.selfSupervisedSteps>40
            && j.motionSteps && pb.steps && bb.actorUpdates && mb.supervisedFrames
            && get(vabpnnDecisionModelView(f.decision).minutes.persistent).steps>40
            && get(vabpnnDecisionModelView(f.decision).minutes.persistent).motion.steps>40,"Every learner exercised before evaluation");
        f.demo.modelControl=1;f.demo.epoch++;reset(f);f.next();
        const auto placementBefore=deviceDigest(vabpnnSpikePlacementCheckpointPart(f.spike));
        const auto outgoingBefore=deviceDigest(f.parts()[9]);
        const auto frozen=learnedState(f);const auto ob=get(vabpnnOnlineOutput(f.online));
        const auto minuteBefore=get(vabpnnDecisionModelView(f.decision).minutes.persistent).steps;
        const auto jb=get(vabpnnJepaModelView(f.jepa).output);const auto sb=get(vabpnnSpikeModelView(f.spike).counters);
        const auto lastFrame=f.frame.frameId;
        for(int i=0;i<32;++i){if(i==16)reset(f);advance();}
        equal(frozen,learnedState(f));
        const auto oa=get(vabpnnOnlineOutput(f.online));const auto ja=get(vabpnnJepaModelView(f.jepa).output);
        const auto sa=get(vabpnnSpikeModelView(f.spike).counters);const auto decision=get(vabpnnDecisionModelView(f.decision).output);
        require(std::memcmp(&ob,&oa,sizeof(ob))==0,"All online persistent state frozen");
        require(ja.selfSupervisedSteps==jb.selfSupervisedSteps && ja.motionSteps==jb.motionSteps,"JEPA AdamW/head steps frozen at trainEveryFrames=1");
        require(sa.predictions>=sb.predictions+32 && sa.topologyUpdates>sb.topologyUpdates && sa.contributionUpdates>=sb.contributionUpdates+32,
            "Frozen neural inference keeps salience, scouting and CSR rewiring dynamic");
        require(decision.valid && decision.frameId==f.frame.frameId && decision.selectedMask && ja.valid && ja.pairReady
            && ja.frameId>lastFrame && get(vabpnnJepaModelView(f.jepa).config).motionEnabled,"Fresh model actions and motion inference continue");
        require(placementBefore!=deviceDigest(vabpnnSpikePlacementCheckpointPart(f.spike))
            && outgoingBefore!=deviceDigest(f.parts()[9]),"Actual placement learner and local connectivity change with neural weights frozen");
        f.frame.onlineModel=nullptr;f.next();f.next();equal(frozen,learnedState(f));f.frame.onlineModel=f.online;
        std::puts("PASS: exact GPU hashes of neural/attention weights, AdamW, EMA, master weights and knowledge stay frozen; salience, placement and CSR remain dynamic through resets.");
        f.demo.modelControl=0;f.demo.epoch++;reset(f);for(int i=0;i<32;++i)advance();
        const auto ro=get(vabpnnOnlineOutput(f.online));const auto rj=get(vabpnnJepaModelView(f.jepa).output);
        require(ro.examples>ob.examples && ro.pixelSteps>ob.pixelSteps && ro.decisionSteps>ob.decisionSteps
            && rj.selfSupervisedSteps>jb.selfSupervisedSteps && rj.motionSteps>jb.motionSteps
            && get(vabpnnDecisionModelView(f.decision).position).steps>pb.steps
            && get(vabpnnBlockModelView(blocks).counters).actorUpdates>bb.actorUpdates
            && get(&vabpnnMemeModelView(memes).persistent->counters).supervisedFrames>mb.supervisedFrames
            && get(vabpnnSpikeModelView(f.spike).counters).topologyUpdates>sb.topologyUpdates
            && get(vabpnnDecisionModelView(f.decision).minutes.persistent).steps>minuteBefore,
            "Every learner and plasticity resumes in demonstration");
        require(frozen.values!=learnedState(f).values,"Resumed learning changes actual parameters");
        std::puts("PASS: demonstration resumes every learner/optimizer and topology while retaining learned parameters.");
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
