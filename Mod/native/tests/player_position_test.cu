#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "TemporalDecision.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "model_checkpoint.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
using namespace vabpnn;
void check(cudaError_t c) { if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c)); }
void require(bool v,const char* m) { if(!v) throw std::runtime_error(m); }
void positionNear(double a,double b,const char* m,double tolerance=2e-6) {
    if(!std::isfinite(a) || std::abs(a-b)>tolerance) {
        std::fprintf(stderr,"%s got %.10g expected %.10g\n",m,a,b); throw std::runtime_error(m);
    }
}
template<class T> T read(const T* p) { T v; check(cudaMemcpy(&v,p,sizeof(v),cudaMemcpyDeviceToHost)); return v; }
int main(int argc,char** argv) {
    cudaStream_t stream=nullptr; std::int32_t* votes=nullptr; VabpnnDecisionModel* model=nullptr;
    VabpnnNeuronModel* neuron=nullptr; VabpnnJepaModel* jepa=nullptr;
    VabpnnOnlineModel* online=nullptr; VabpnnSpikeModel* spike=nullptr;
    int result=0;
    try {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        model=vabpnnCreateDecisionModel(stream);
        check(cudaMalloc(reinterpret_cast<void**>(&votes),VoteCount*sizeof(*votes)));
        check(cudaMemsetAsync(votes,0,VoteCount*sizeof(*votes),stream));
        VabpnnGpuFrame frame{}; frame.stream=stream; frame.decisionModel=model;
        frame.frameId=1; frame.captureNanos=1'000'000'000;
        auto run=[&]() {
            vabpnnLaunchTemporalDecision(frame,votes,model); check(cudaStreamSynchronize(stream));
            return read(vabpnnDecisionModelView(model).output);
        };
        auto reset=[&]() { vabpnnResetDecisionState(model,stream); ++frame.frameId; };
        auto missing=run();
        frame.position={frame.captureNanos,1,123.125,64.0,-81.25};
        reset(); auto first=run();
        require(first.positionValid && first.status.positionNanos==frame.captureNanos,"timestamp paired to captured frame");
        for(int k=0;k<VoteCount;++k) positionNear(first.posterior[k],missing.posterior[k],"neutral coordinate branch preserves old Bayes");
        for(int k=0;k<DecisionKChoices;++k) positionNear(first.kProbability[k],missing.kProbability[k],"neutral coordinate branch preserves old k");
        for(int d=0;d<6;++d) positionNear(first.positionFeatures[d],0,"first observation has zero relative XYZ / speed");
        positionNear(first.positionFeatures[6],std::sin(123.125/64),"absolute X phase");
        positionNear(first.positionFeatures[9],std::cos(-81.25/64),"absolute Z phase");
        positionNear(first.positionFeatures[10],std::tanh(64.0/128),"absolute height");
        positionNear(first.positionFeatures[11],1,"presence");
        frame.captureNanos+=33'333'333; frame.position.sampleNanos=frame.captureNanos; frame.position.x+=0.1; ++frame.frameId;
        auto second=run();
        positionNear(second.positionFeatures[0],std::tanh(0.1/64),"independent relative XYZ");
        const double dt=33'333'333e-9;
        positionNear(second.positionFeatures[3],std::tanh((1-std::exp(-12*dt))*0.1/dt/10),"timestamp-derived filtered velocity");
        frame.captureNanos+=16'666'667; ++frame.frameId; auto reused=run();
        require(std::memcmp(second.positionFeatures,reused.positionFeatures,sizeof(second.positionFeatures))==0,
                "60 Hz predictions reuse the exact 30 Hz coordinate feature, without zeroing speed");
        frame.captureNanos+=33'333'333; frame.position.sampleNanos=frame.captureNanos; frame.position.x+=1000; ++frame.frameId;
        auto teleport=run();
        for(int d=0;d<6;++d) positionNear(teleport.positionFeatures[d],0,"teleport resets local origin and velocity");
        ++frame.position.epoch; frame.position.x=-25; ++frame.frameId;
        auto world=run();
        for(int d=0;d<6;++d) positionNear(world.positionFeatures[d],0,"world/respawn epoch resets position history");
        frame.position.x=29'999'000.125; frame.position.y=75; frame.position.z=-29'998'000.25;
        ++frame.position.epoch; ++frame.frameId; auto edge=run();
        frame.captureNanos+=33'333'333; frame.position.sampleNanos=frame.captureNanos; frame.position.x+=0.25; ++frame.frameId;
        edge=run();
        positionNear(edge.positionFeatures[0],std::tanh(0.25/64),"double coordinate delta retains sub-block precision at world edge");
        DecisionPositionAdapter adapter{};
        adapter.evidence[0][10]=2.0f; adapter.k[1][11]=4.0f;
        auto* gpuAdapter=vabpnnMutablePositionAdapter(model,stream);
        check(cudaMemcpyAsync(gpuAdapter,&adapter,sizeof(adapter),cudaMemcpyHostToDevice,stream));
        frame.position={frame.captureNanos,10,0,64,0}; reset(); auto fused=run();
        positionNear(fused.posterior[0],1/(1+std::exp(-2*std::tanh(0.5))),"coordinate contribution enters conditional evidence before Bayes");
        require(fused.requestedK==1 && fused.selectedMask==1,"coordinate k branch participates in actual action selection");
        for(int problem=0;problem<5;++problem) {
            reset(); frame.position={frame.captureNanos,11,0,64,0};
            if(problem==0) frame.position.sampleNanos=0;
            if(problem==1) frame.position.sampleNanos=frame.captureNanos+1;
            if(problem==2) frame.position.sampleNanos=frame.captureNanos-PositionMaximumAgeNanos-1;
            if(problem==3) frame.position.x=std::numeric_limits<double>::quiet_NaN();
            if(problem==4) frame.position.z=30'000'001;
            const auto invalid=run();
            require(!invalid.positionValid && invalid.valid && invalid.selectedMask==0,"missing/future/stale/invalid position falls back to visual policy");
            positionNear(invalid.posterior[0],0.5,"unavailable position contributes zero evidence");
        }
        for(int k=0;k<VoteCount;++k) for(int f=0;f<PositionFeatures;++f) {
            adapter.evidenceFirst[k][f]=(k+f+1)*.0125f;
            adapter.evidenceSecond[k][f]=(k*PositionFeatures+f+1)*.0025f;
        }
        adapter.steps=123;
        check(cudaMemcpyAsync(gpuAdapter,&adapter,sizeof(adapter),cudaMemcpyHostToDevice,stream));
        check(cudaStreamSynchronize(stream));
        ModelPart part[1]; require(vabpnnDecisionPositionCheckpointParts(model,part)==1,"separate appended checkpoint part");
        auto path=(std::filesystem::current_path()/L"position-checkpoint-test.bin").wstring();
        saveModelCheckpoint(path,{part[0]},3);
        check(cudaMemset(gpuAdapter,0,sizeof(adapter)));
        require(loadModelCheckpoint(path,{part[0]},false,3),"V3 coordinate checkpoint restores");
        const auto restored=read(gpuAdapter);
        require(std::memcmp(&adapter,&restored,sizeof(adapter))==0,"coordinate weights / AdamW / independent steps resume exactly");
        bool wrongVersion=false;
        try { loadModelCheckpoint(path,{part[0]}); } catch(const std::exception&) { wrongVersion=true; }
        require(wrongVersion,"V3 cannot be confused with a V2 layout");
        std::filesystem::remove(path);
        std::puts("PASS: frame-aligned coordinates, neutral legacy predictions, XYZ / velocity / world-edge precision, cached 30 Hz features, teleport / epoch reset, Bayes / learned-k fusion, stale / future / missing fallback and V3 AdamW checkpoint.");
        if(argc==2) {
            // Read-only acceptance against the user's existing V2 model. No save operation.
            neuron=vabpnnCreateNeuronModel(stream); jepa=vabpnnCreateJepaModel(stream);
            online=vabpnnCreateOnlineModel(stream); spike=vabpnnCreateSpikeModel(stream);
            check(cudaStreamSynchronize(stream)); check(cudaMemset(gpuAdapter,0,sizeof(adapter)));
            std::vector<ModelPart> parts{{vabpnnMutableNeuronWeights(neuron,stream),ParameterCount},
                {vabpnnMutableDecisionParameters(model,stream),sizeof(DecisionParameters)}};
            ModelPart extra[8];
            auto append=[&](std::size_t n) { parts.insert(parts.end(),extra,extra+n); };
            append(vabpnnJepaCheckpointParts(jepa,extra));
            append(vabpnnOnlineCheckpointParts(online,extra));
            append(vabpnnSpikeCheckpointParts(spike,extra));
            check(cudaStreamSynchronize(stream));
            require(loadModelCheckpoint(std::filesystem::path(argv[1]).wstring(),parts),"read-only actual V2 migration source");
            const auto old=read(vabpnnOnlineOutput(online)); const auto fresh=read(gpuAdapter);
            DecisionPositionAdapter zero{};
            require(std::memcmp(&fresh,&zero,sizeof(zero))==0,"legacy resume leaves new coordinate branch neutral");
            frame.position={}; reset(); const auto legacyVisual=run();
            frame.position={frame.captureNanos,25,123.25,64,-81.125}; reset(); const auto legacyPosition=run();
            for(int k=0;k<VoteCount;++k)
                positionNear(legacyPosition.posterior[k],legacyVisual.posterior[k],"actual trained V2 neutral coordinate posterior");
            for(int k=0;k<DecisionKChoices;++k)
                positionNear(legacyPosition.kProbability[k],legacyVisual.kProbability[k],"actual trained V2 neutral coordinate k");
            require(legacyPosition.selectedMask==legacyVisual.selectedMask && legacyPosition.requestedK==legacyVisual.requestedK,
                    "actual trained V2 action selection remains identical before position teaching");
            std::puts("PASS: actual trained V2 probabilities, learned k and selected actions are unchanged with a neutral position branch.");
            std::printf("PASS: actual V2 checkpoint restored read-only: parts=%zu examples=%llu decisionSteps=%llu; new coordinate adapter remains neutral.\n",
                parts.size(),static_cast<unsigned long long>(old.examples),static_cast<unsigned long long>(old.decisionSteps));
        }
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); result=1; }
    if(stream) cudaStreamSynchronize(stream);
    vabpnnDestroySpikeModel(spike); vabpnnDestroyOnlineModel(online); vabpnnDestroyJepaModel(jepa);
    vabpnnDestroyNeuronModel(neuron); vabpnnDestroyDecisionModel(model);
    if(votes) cudaFree(votes); if(stream) cudaStreamDestroy(stream);
    return result;
}
