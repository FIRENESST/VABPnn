#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "OnlineLearning.h"
#include "FrameOperator.h"
#include "model_checkpoint.h"
#include <cstdio>
#include <memory>
#include <cmath>
#include <algorithm>
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
void cudaCheck(cudaError_t code) { if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
template<class T> T read(const T* pointer) { T v; cudaCheck(cudaMemcpy(&v,pointer,sizeof(T),cudaMemcpyDeviceToHost)); return v; }
// Independent CPU reference for the first AdamW step across every policy head.
// This also verifies that the parallel lanes own all parameters exactly once.
void verifyFirstPolicyStep(const vabpnn::DecisionParameters& before,
                          const vabpnn::DecisionOutput& observation,
                          const vabpnn::DecisionParameters& after,std::uint32_t label) {
    auto expected=before;
    auto update=[](float& value,double gradient,bool decay) {
        gradient=std::max(-1.0,std::min(1.0,gradient));
        const double first=double(0.1f)*gradient/(1-double(0.9f));
        const double second=double(0.001f)*gradient*gradient/(1-double(0.999f));
        value=static_cast<float>(double(value)*(decay ? double(0.999997f):1.0)
            -double(0.0003f)*first/(std::sqrt(second)+double(1e-8f)));
    };
    for(int i=0;i<vabpnn::VoteCount;++i) {
        const double y=(label>>i)&1u;
        const double prob=std::max(double(1e-6f),std::min(double(1-1e-6f),double(observation.posterior[i])));
        const double gradient=(prob-y)/vabpnn::VoteCount;
        const double threshold=(observation.threshold[i]-(y ? 0.25:0.75))/vabpnn::VoteCount;
        update(expected.logLikelihoodBias[i],gradient,false);
        update(expected.rawVoteWeight[i],gradient*observation.normalizedVotes[i],true);
        update(expected.thresholdBias[i],threshold,false);
        for(int d=0;d<vabpnn::DecisionWidth;++d) {
            update(expected.logLikelihoodWeight[i][d],gradient*observation.hidden[i][d],true);
            update(expected.thresholdWeight[i][d],threshold*observation.hidden[i][d],true);
        }
    }
    int count=0;
    for(int i=0;i<vabpnn::VoteCount;++i) count+=(label>>i)&1u;
    for(int k=0;k<vabpnn::DecisionKChoices;++k) {
        const double gradient=double(observation.kProbability[k])-(k==count);
        update(expected.kBias[k],gradient,false);
        for(int d=0;d<vabpnn::DecisionKFeatures;++d)
            update(expected.kWeight[k][d],gradient*observation.kFeatures[d],true);
    }
    const auto* wanted=reinterpret_cast<const float*>(&expected);
    const auto* actual=reinterpret_cast<const float*>(&after);
    for(std::size_t i=0;i<sizeof(expected)/sizeof(float);++i)
        require(std::isfinite(actual[i]) && std::fabs(double(actual[i])-wanted[i])<2e-6,
                "all policy parameters agree with independent CPU AdamW; frozen backbone retained");
}
void verifyFirstPositionStep(const vabpnn::DecisionOutput& observation,
                             const vabpnn::DecisionPositionAdapter& after,std::uint32_t label) {
    require(observation.positionValid && after.steps==1,"position AdamW has its own step counter");
    int actions=0; for(int k=0;k<vabpnn::VoteCount;++k) actions+=(label>>k)&1u;
    for(int node=0;node<vabpnn::VoteCount+vabpnn::DecisionKChoices;++node)
        for(int f=0;f<vabpnn::PositionFeatures;++f) {
            const bool action=node<vabpnn::VoteCount;
            const int k=action ? node:node-vabpnn::VoteCount;
            double gradient=action ? (double(observation.posterior[k])-((label>>k)&1u))/vabpnn::VoteCount
                : double(observation.kProbability[k])-(k==actions);
            gradient=std::max(-1.0,std::min(1.0,gradient*observation.positionFeatures[f]));
            const double m=double(.1f)*gradient,v=double(.001f)*gradient*gradient;
            const double expected=-double(.0003f)*(m/(1-double(.9f)))/(std::sqrt(v/(1-double(.999f)))+double(1e-8f));
            const float value=action ? after.evidence[k][f]:after.k[k][f];
            const float first=action ? after.evidenceFirst[k][f]:after.kFirst[k][f];
            const float second=action ? after.evidenceSecond[k][f]:after.kSecond[k][f];
            require(std::fabs(value-expected)<2e-6 && std::fabs(first-m)<2e-6 && std::fabs(second-v)<2e-6,
                "all 324 coordinate parameters and moments follow CPU AdamW using the previous observation");
        }
}
void checkpointTest() {
    float* gpu=nullptr; cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu),4*sizeof(float)));
    const float initial[4]={1,2,-3,4}, changed[4]={7,7,7,7};
    auto path=(std::filesystem::current_path()/L"online-checkpoint-test.bin").wstring();
    std::vector<vabpnn::ModelPart> parts{{gpu,4*sizeof(float)}};
    cudaCheck(cudaMemcpy(gpu,initial,sizeof(initial),cudaMemcpyHostToDevice));
    vabpnn::saveModelCheckpoint(path,parts);
    cudaCheck(cudaMemcpy(gpu,changed,sizeof(changed),cudaMemcpyHostToDevice));
    require(vabpnn::loadModelCheckpoint(path,parts),"checkpoint restore");
    float restored[4]; cudaCheck(cudaMemcpy(restored,gpu,sizeof(restored),cudaMemcpyDeviceToHost));
    require(std::memcmp(initial,restored,sizeof(initial))==0,"checkpoint parameters exact");
    { std::fstream file(std::filesystem::path(path),std::ios::binary|std::ios::in|std::ios::out);
      file.seekp(sizeof(vabpnn::CheckpointHeader)+2); char c=0x7f; file.write(&c,1); }
    cudaCheck(cudaMemcpy(gpu,changed,sizeof(changed),cudaMemcpyHostToDevice));
    bool rejected=false;
    try { vabpnn::loadModelCheckpoint(path,parts); } catch(const std::exception&) { rejected=true; }
    require(rejected,"SHA256 corrupted checkpoint rejected");
    cudaCheck(cudaMemcpy(restored,gpu,sizeof(restored),cudaMemcpyDeviceToHost));
    require(std::memcmp(changed,restored,sizeof(changed))==0,"corruption rejected before any GPU mutation");
    std::filesystem::remove(path); cudaFree(gpu);
}
int main() {
    cudaStream_t stream=nullptr; unsigned char* rgba=nullptr; float* rgb=nullptr;
    VabpnnNeuronModel* neuron=nullptr; VabpnnDecisionModel* decision=nullptr;
    VabpnnJepaModel* jepa=nullptr; VabpnnOnlineModel* online=nullptr;
    int result=0;
    try {
        cudaCheck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&rgba),16));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&rgb),12*sizeof(float)));
        const unsigned char image[16]={200,90,30,255,200,90,30,255,200,90,30,255,200,90,30,255};
        cudaCheck(cudaMemcpy(rgba,image,16,cudaMemcpyHostToDevice));
        neuron=vabpnnCreateNeuronModel(stream); decision=vabpnnCreateDecisionModel(stream);
        jepa=vabpnnCreateJepaModel(stream); online=vabpnnCreateOnlineModel(stream);
        vabpnn::DemoSample demo; demo.active=1; demo.epoch=1; demo.heldKeys=1; demo.pointerMode=1;
        VabpnnGpuFrame frame{rgba,8,rgb,2,2,1,1'000'000'000,stream,neuron,decision,jepa,online};
        auto run=[&]() {
            frame.evaluationMode=demo.modelControl!=0;
            vabpnnSetDemoSample(online,&demo,stream); vabpnnLaunchOperator(frame);
            cudaCheck(cudaStreamSynchronize(stream));
        };
        frame.position={frame.captureNanos,1,123.25,64,-81.125};
        run();
        require(read(vabpnnOnlineOutput(online)).examples==0,"no label for the first observation");
        const auto referenceParameters=read(vabpnnDecisionModelView(decision).parameters);
        const auto referenceObservation=read(vabpnnDecisionModelView(decision).output);
        frame.frameId++; frame.captureNanos+=20'000'000; demo.keyHeldNanos[0]+=20'000'000; demo.mouseX+=0.4;
        frame.position={frame.captureNanos,1,-3200,128,1800}; // Future relative to the cached training observation.
        run();
        auto status=read(vabpnnOnlineOutput(online));
        require(status.examples==1 && status.lastLabel==(1u|(1u<<5)),"causal held W / mouse X target");
        auto weights=std::make_unique<vabpnn::DecisionParameters>();
        cudaCheck(cudaMemcpy(weights.get(),vabpnnDecisionModelView(decision).parameters,sizeof(*weights),cudaMemcpyDeviceToHost));
        require(fabs(weights->logLikelihoodBias[0]-0.0003f)<2e-7,"first AdamW evidence update");
        verifyFirstPolicyStep(referenceParameters,referenceObservation,*weights,status.lastLabel);
        verifyFirstPositionStep(referenceObservation,read(vabpnnDecisionModelView(decision).position),status.lastLabel);
        std::puts("PASS: coordinate evidence/k weights and all AdamW moments agree with independent CPU reference; previous-frame XYZ only.");
        require(fabs(status.decisionLoss-(logf(2.0f)+logf(static_cast<float>(vabpnn::DecisionKChoices))))<1e-4,"BCE and k cross entropy");
        for(int i=0;i<46;++i) {
            frame.frameId++; frame.captureNanos+=20'000'000;
            demo.keyHeldNanos[0]+=20'000'000; demo.mouseX+=0.4;
            run();
        }
        status=read(vabpnnOnlineOutput(online));
        require(status.examples==47 && status.pixelSteps==11,"bounded four-frame latent pixel updates");
        const auto j=read(vabpnnJepaModelView(jepa).output);
        require(j.motionSteps>0 && j.selfSupervisedSteps>0,"real motion labels and JEPA AdamW both run");
        const auto neuronView=vabpnnNeuronModelView(neuron);
        require(read(neuronView.weights+vabpnn::weightIndex(0,0,0))==vabpnn::TernaryWeight::Positive
                && read(neuronView.weights+vabpnn::weightIndex(0,0,1))==vabpnn::TernaryWeight::Negative,
                "learned RGB weights quantize to exact ternary values");
        // A nonzero wheel target exercises the separately updated wheel parameters.
        demo.wheel+=0.1; demo.mouseX+=0.4; demo.keyHeldNanos[0]+=20'000'000;
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        status=read(vabpnnOnlineOutput(online));
        require((status.lastLabel&(1u<<11))!=0 && std::fabs(status.wheelLoss-0.03125f)<2e-6,
                "nonzero wheel demonstration trains the wheel head with the original MSE");
        auto before=read(vabpnnOnlineOutput(online));
        // Artificially drop 5 capture IDs. Cumulative deltas still label the complete 100 ms interval.
        frame.frameId+=5; frame.captureNanos+=100'000'000; demo.mouseX+=2; demo.keyHeldNanos[0]+=100'000'000;
        run(); require(read(vabpnnOnlineOutput(online)).examples==before.examples+1,"dropped frames preserve input interval");
        // Mutually exclusive real button edges are not converted to inconsistent policy targets.
        before=read(vabpnnOnlineOutput(online));
        frame.frameId++; frame.captureNanos+=20'000'000; demo.edge[0]++; demo.edge[1]++;
        run(); status=read(vabpnnOnlineOutput(online));
        require(status.examples==before.examples && status.rejected>before.rejected,"only specified conflicts rejected");
        // Switching modes never uses model controls as imitation targets.
        demo.modelControl=1; demo.epoch++;
        vabpnnResetDecisionState(decision,stream);
        vabpnnResetJepaHistory(jepa,stream);
        vabpnnResetOnlineHistory(online,stream);
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        auto selected=read(vabpnnDecisionModelView(decision).output);
        std::printf("MODE_SWITCH: k=%d eligible=%x selected=%x W_p=%.5f W_threshold=%.5f\n",
            selected.requestedK,selected.eligibleMask,selected.selectedMask,selected.posterior[0],selected.threshold[0]);
        require(selected.valid && (selected.selectedMask&1u),"trained W still selected after F8-style history reset");
        require(selected.status.frameId==frame.frameId && selected.status.requestedK==selected.requestedK,
                "decision counters match the actual control output");
        before=read(vabpnnOnlineOutput(online));
        const auto mouseBefore=read(vabpnnJepaModelView(jepa).output).motionSteps;
        frame.frameId++; frame.captureNanos+=20'000'000; demo.mouseX+=1; run();
        require(read(vabpnnOnlineOutput(online)).examples==before.examples,"model control is not human BC");
        require(read(vabpnnJepaModelView(jepa).output).motionSteps==mouseBefore,"model output not a motion target");
        require(read(vabpnnJepaModelView(jepa).config).motionEnabled==1,"motion inference only in model mode");
        demo.modelControl=0; demo.epoch++; frame.frameId++; frame.captureNanos+=20'000'000; run();
        // An ungrabbed inventory pointer still supplies real left/right presses, holds and releases.
        demo.pointerMode=0; demo.heldKeys=0; demo.epoch++;
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        demo.heldKeys=1u<<7; demo.edge[0]++;
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        require((read(vabpnnOnlineOutput(online)).lastLabel&(1u<<7))!=0,"inventory left press enters GPU training");
        before=read(vabpnnOnlineOutput(online));
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        status=read(vabpnnOnlineOutput(online));
        require(status.examples==before.examples+1 && (status.lastLabel&(1u<<7))!=0,
                "held inventory button is not falsely trained as inactive");
        demo.heldKeys=0; demo.edge[2]++;
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        status=read(vabpnnOnlineOutput(online));
        require((status.lastLabel&(1u<<9))!=0 && !(status.lastLabel&(1u<<7)),"inventory left release enters GPU training");
        demo.heldKeys=1u<<8; demo.edge[1]++;
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        require((read(vabpnnOnlineOutput(online)).lastLabel&(1u<<8))!=0,"inventory right press enters GPU training");
        demo.heldKeys=0; demo.edge[3]++;
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        require((read(vabpnnOnlineOutput(online)).lastLabel&(1u<<10))!=0,"inventory right release enters GPU training");
        before=read(vabpnnOnlineOutput(online));
        demo.active=0; frame.frameId++; frame.captureNanos+=20'000'000; run();
        require(read(vabpnnOnlineOutput(online)).examples==before.examples,"inactive window does not train");
        demo.active=1; demo.epoch++; frame.frameId++; frame.captureNanos+=20'000'000; run();
        before=read(vabpnnOnlineOutput(online));
        frame.frameId++; frame.captureNanos+=500'000'000; run();
        require(read(vabpnnOnlineOutput(online)).examples==before.examples,"long gaps do not make false targets");
        vabpnnResetOnlineHistory(online,stream);
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        require(read(vabpnnOnlineOutput(online)).examples==before.examples,"scene reset discards demonstration history");
        vabpnn::ModelPart parts[8];
        require(vabpnnOnlineCheckpointParts(online,parts)==2 && parts[0].bytes==vabpnn::ParameterCount*sizeof(float),
                "checkpoint contains latent parameters / optimizer, excludes observations");
        const auto oldSteps=read(vabpnnJepaModelView(jepa).output).selfSupervisedSteps;
        vabpnnResumeJepaOptimizer(jepa,stream);
        frame.frameId++; frame.captureNanos+=20'000'000; run();
        require(read(vabpnnJepaModelView(jepa).output).selfSupervisedSteps==oldSteps,"resume preserves AdamW step count");
        checkpointTest();
        std::puts("PASS: causal GPU demonstration, ternary learning, Bayesian heads/k, motion AdamW, modes, drops, gaps and atomic SHA256 checkpoints.");
    } catch(const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); result=1; }
    if(stream) cudaStreamSynchronize(stream);
    vabpnnDestroyOnlineModel(online); vabpnnDestroyJepaModel(jepa);
    vabpnnDestroyDecisionModel(decision); vabpnnDestroyNeuronModel(neuron);
    if(rgb) cudaFree(rgb); if(rgba) cudaFree(rgba);
    if(stream) cudaStreamDestroy(stream);
    return result;
}
