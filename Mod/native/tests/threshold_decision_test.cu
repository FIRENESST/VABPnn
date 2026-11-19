#include "TemporalDecision.h"
#include "JepaMotion.h"
#include "OnlineLearning.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace vabpnn;
void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main()try {
    auto* neurons=vabpnnCreateNeuronModel(nullptr);auto* model=vabpnnCreateDecisionModel(nullptr);
    auto* jepa=vabpnnCreateJepaModel(nullptr);auto* online=vabpnnCreateOnlineModel(nullptr);
    auto jp=std::make_unique<JepaParameters>();check(cudaMemcpy(jp.get(),vabpnnJepaModelView(jepa).parameters,sizeof(*jp),cudaMemcpyDeviceToHost));
    jp->motion.outputBias[0]=.2f;jp->motion.outputBias[1]=-.1f;
    check(cudaMemcpy(vabpnnMutableJepaParameters(jepa,nullptr),jp.get(),sizeof(*jp),cudaMemcpyHostToDevice));
    float* rgb=nullptr;check(cudaMalloc(&rgb,37*23*3*sizeof(float)));check(cudaMemset(rgb,0,37*23*3*sizeof(float)));
    DemoSample demo{};demo.epoch=1;demo.active=0;demo.modelControl=1;vabpnnSetDemoSample(online,&demo,nullptr);
    const auto view=vabpnnDecisionModelView(model);auto parameter=std::make_unique<DecisionParameters>();
    check(cudaMemcpy(parameter.get(),view.parameters,sizeof(*parameter),cudaMemcpyDeviceToHost));
    parameter->logLikelihoodBias[0]=2;parameter->thresholdBias[0]=3; // Positive posterior still below threshold.
    parameter->logLikelihoodBias[1]=4;parameter->thresholdBias[1]=2;
    parameter->logLikelihoodBias[5]=parameter->logLikelihoodBias[6]=3;
    parameter->thresholdBias[5]=parameter->thresholdBias[6]=30; // Both axes remain blocked across repeated Bayes updates.
    parameter->logLikelihoodBias[12]=4;parameter->thresholdBias[12]=1;
    parameter->kBias[5]=4;
    check(cudaMemcpy(vabpnnMutableDecisionParameters(model,nullptr),parameter.get(),sizeof(*parameter),cudaMemcpyHostToDevice));
    MotionProposal motion{42,-37,2};check(cudaMemcpy(vabpnnMutableMotionProposal(model,nullptr),&motion,sizeof(motion),cudaMemcpyHostToDevice));
    auto frame=VabpnnGpuFrame{};frame.neuronModel=neurons;frame.frameId=1;frame.captureNanos=1'000'000'000;frame.evaluationMode=true;
    frame.jepaModel=jepa;frame.onlineModel=online;frame.decisionModel=model;frame.rgbNchw=rgb;frame.width=37;frame.height=23;
    vabpnnLearnPreviousDemo(frame);
    vabpnnLaunchJepaForward(frame,vabpnnNeuronModelView(neurons).voteSums,jepa,vabpnnMutableMotionProposal(model,nullptr));
    vabpnnLaunchTemporalDecision(frame,vabpnnNeuronModelView(neurons).voteSums,model);check(cudaDeviceSynchronize());
    auto out=std::make_unique<DecisionOutput>();check(cudaMemcpy(out.get(),view.output,sizeof(*out),cudaMemcpyDeviceToHost));
    require(out->valid && out->requestedK==5,"frozen prediction still computes positive learned k");
    require(out->eligibleMask==((1u<<1)|(1u<<12)) && out->selectedMask==out->eligibleMask && out->selectedCount==2,
        "frozen evaluation filters candidate actions by their learned thresholds before top-k");
    require(out->command.mouseDx==0 && out->command.mouseDy==0 && out->command.wheelSteps==0,
        "blocked mouse axes/wheel cannot leak motion");
    for(int i=2;i<=4;++i) {
        frame.frameId=i;frame.captureNanos+=20'000'000;
        vabpnnLearnPreviousDemo(frame);vabpnnLaunchJepaForward(frame,vabpnnNeuronModelView(neurons).voteSums,jepa,vabpnnMutableMotionProposal(model,nullptr));
        vabpnnLaunchTemporalDecision(frame,vabpnnNeuronModelView(neurons).voteSums,model);check(cudaDeviceSynchronize());
        check(cudaMemcpy(out.get(),view.output,sizeof(*out),cudaMemcpyDeviceToHost));unsigned expected=0;
        for(int k=0;k<VoteCount;++k)if(out->confidence[k]>out->threshold[k])expected|=1u<<k;
        require(out->valid && out->frameId==i && out->eligibleMask==expected
            && !(out->selectedMask&~expected),"every fresh frozen frame keeps the learned threshold rule");
        JepaOutput jo{};check(cudaMemcpy(&jo,vabpnnJepaModelView(jepa).output,sizeof(jo),cudaMemcpyDeviceToHost));
        require(jo.pairReady && jo.motionActive && jo.suggestedDx>0 && jo.suggestedDy<0,"inactive input still computes frozen JEPA/Bayes advice");
        require(!(out->selectedMask&96) && out->command.mouseDx==0 && out->command.mouseDy==0,"pre-gate advice cannot bypass learned thresholds");
        require(out->status.mousePosteriorX==out->posterior[5] && out->status.mouseThresholdX==out->threshold[5],"diagnostic uses exact axis probabilities/thresholds");
    }
    auto after=std::make_unique<DecisionParameters>();check(cudaMemcpy(after.get(),view.parameters,sizeof(*after),cudaMemcpyDeviceToHost));
    require(!std::memcmp(parameter.get(),after.get(),sizeof(*parameter)),"threshold-filtered inference keeps neural weights frozen");
    // Even a positive k must return a valid empty packet when all confidence values stay below their thresholds.
    for(int k=0;k<VoteCount;++k){parameter->logLikelihoodBias[k]=0;parameter->thresholdBias[k]=10;}
    check(cudaMemcpy(vabpnnMutableDecisionParameters(model,nullptr),parameter.get(),sizeof(*parameter),cudaMemcpyHostToDevice));
    vabpnnResetDecisionState(model,nullptr);frame.frameId=5;frame.captureNanos+=20'000'000;
    vabpnnLaunchTemporalDecision(frame,vabpnnNeuronModelView(neurons).voteSums,model);check(cudaDeviceSynchronize());
    check(cudaMemcpy(out.get(),view.output,sizeof(*out),cudaMemcpyDeviceToHost));
    require(out->valid && out->requestedK==5 && !out->eligibleMask && !out->selectedMask && !out->selectedCount,
        "positive k cannot bypass an empty confidence gate");
    parameter->logLikelihoodBias[5]=parameter->logLikelihoodBias[6]=3;
    parameter->thresholdBias[5]=parameter->thresholdBias[6]=-10;
    check(cudaMemcpy(vabpnnMutableDecisionParameters(model,nullptr),parameter.get(),sizeof(*parameter),cudaMemcpyHostToDevice));
    vabpnnResetDecisionState(model,nullptr);frame.frameId=6;frame.captureNanos+=20'000'000;
    vabpnnLearnPreviousDemo(frame);vabpnnLaunchJepaForward(frame,vabpnnNeuronModelView(neurons).voteSums,jepa,vabpnnMutableMotionProposal(model,nullptr));
    vabpnnLaunchTemporalDecision(frame,vabpnnNeuronModelView(neurons).voteSums,model);check(cudaDeviceSynchronize());
    check(cudaMemcpy(out.get(),view.output,sizeof(*out),cudaMemcpyDeviceToHost));JepaOutput allowed{};
    check(cudaMemcpy(&allowed,vabpnnJepaModelView(jepa).output,sizeof(allowed),cudaMemcpyDeviceToHost));
    require(out->selectedMask==96 && out->command.mouseDx==allowed.suggestedDx
        && out->command.mouseDy==allowed.suggestedDy && out->command.mouseDx>0 && out->command.mouseDy<0,"allowed axes deliver exact Bayesian mouse advice");
    require(!allowed.selfSupervisedSteps && !allowed.motionSteps,"frozen mouse diagnostics update no JEPA optimizer");
    auto jeAfter=std::make_unique<JepaParameters>();check(cudaMemcpy(jeAfter.get(),vabpnnJepaModelView(jepa).parameters,sizeof(*jeAfter),cudaMemcpyDeviceToHost));
    require(!std::memcmp(jp.get(),jeAfter.get(),sizeof(*jp)),"all JEPA weights remain exact during inactive/frozen inference");
    vabpnnDestroyOnlineModel(online);vabpnnDestroyJepaModel(jepa);cudaFree(rgb);
    vabpnnDestroyDecisionModel(model);vabpnnDestroyNeuronModel(neurons);
    std::puts("PASS: inactive frozen JEPA inference, nonzero pre-gate advice, exact diagnostics, blocked-zero/allowed-exact delivery, unchanged neural weights.");
    std::printf("PASS: continuous frozen control/prediction enforce learned thresholds on every frame, cap top-k to eligible actions and publish valid zero-action packets.\n");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
