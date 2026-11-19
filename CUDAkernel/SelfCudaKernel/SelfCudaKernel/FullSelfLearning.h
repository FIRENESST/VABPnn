#pragma once
#include "RewardPolicy.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "LongTermMemory.h"
#include "MemeState.h"
struct VabpnnFullLearning;
namespace vabpnn {
constexpr int FullDecisionParameters=sizeof(DecisionParameters)/sizeof(float);
constexpr int FullPositionParameters=(VoteCount+DecisionKChoices)*PositionFeatures;
constexpr int FullMotionParameters=sizeof(JepaMotionHead)/sizeof(float);
constexpr int FullWheelParameters=DecisionWidth+VoteCount+1;
constexpr int FullParameters=FullDecisionParameters+MinuteParameterCount+FullPositionParameters
    +2*BayesianMotionFeatures+LongMemoryParameterCount+FullMotionParameters+FullWheelParameters;
constexpr int FullDecisionOffset=0,FullMinuteOffset=FullDecisionParameters;
constexpr int FullPositionOffset=FullMinuteOffset+MinuteParameterCount;
constexpr int FullBayesOffset=FullPositionOffset+FullPositionParameters;
constexpr int FullMemoryOffset=FullBayesOffset+2*BayesianMotionFeatures;
constexpr int FullMotionOffset=FullMemoryOffset+LongMemoryParameterCount;
constexpr int FullWheelOffset=FullMotionOffset+FullMotionParameters;
constexpr int FullVisualEvery=4,FullVisualSlots=RewardCacheSlots/FullVisualEvery;
struct FullLearningPersistent {
    float first[FullParameters],second[FullParameters];
    std::uint64_t updates=0,secondsSteps=0,pixelSteps=0,minuteSteps=0,positionSteps=0;
    std::uint64_t bayesSteps=0,motionSteps=0,memorySteps=0,blockSteps=0,memeSteps=0,wheelSteps=0;
};
struct FullLearningStatus {
    std::uint64_t updates,secondsSteps,pixelSteps,minuteSteps,positionSteps,bayesSteps,motionSteps;
    std::uint64_t memorySteps,blockSteps,memeSteps,wheelSteps,cached,dropped,bytes;
};
// Pure observations of the causal seconds graph, never persisted as weights.
struct SecondsLearningState {
    std::uint64_t lastCaptureNanos,originNanos,historyNanos[DecisionHistory];
    float previousPosterior[VoteCount];
    std::int32_t historyVotes[DecisionHistory][VoteCount];
    float historyDeltaSeconds[DecisionHistory];
    int started,count,head;
    float priorSource[VoteCount];int first;
    int latentDecoder;
    float historyLatents[DecisionHistory][VoteCount][DecisionWidth];
};
struct BlockRewardObservation {
    float feature[BlockFeatures],noise[BlockOutputs],memeFeature[MemeChannels];
    MemeObservation observation;
    std::uint64_t mutation;
    std::uint32_t exploring,memeActive;
};
struct FullRewardRoots {
    float evidence[VoteCount],k[DecisionKChoices],hidden[VoteCount][DecisionWidth];
    float threshold[VoteCount],temperature,prior[VoteCount],vote[VoteCount];
};
struct FullRewardObservation {
    RewardProposal proposal;
    FullRewardRoots roots;
    DecisionOutput decision;
    float seconds[FullDecisionParameters],minutes[MinuteParameterCount],voteScore[VoteCount];
    float motionInput[JepaMotionInput],motionHidden[JepaHidden],motionHiddenScore[JepaHidden],rawMotionScore[2],headRate[2],meanDerivative[2],motionScore[2];
    float bayesFeatures[BayesianMotionFeatures],wheelFeatures[FullWheelParameters],wheelScore;
    LongMemoryTrace memory;
    std::uint32_t visual,motionValid,pnnVisualSupport;
};
struct DeviceFullLearningView {
    FullLearningPersistent* persistent;
    FullRewardObservation* cache;
    FullRewardRoots* roots;
    const float* gradient;
    const int* active;
};
}
VabpnnFullLearning* vabpnnCreateFullLearning(cudaStream_t stream);
void vabpnnDestroyFullLearning(VabpnnFullLearning* model) noexcept;
void vabpnnEnableFullLearning(VabpnnFullLearning* model, cudaStream_t stream, bool pnnJepa = false);
void vabpnnResetFullLearning(VabpnnFullLearning* model,cudaStream_t stream);
std::uint64_t vabpnnFullLearningBytes(const VabpnnFullLearning* model) noexcept;
vabpnn::ModelPart vabpnnFullLearningCheckpointPart(VabpnnFullLearning* model);
vabpnn::DeviceFullLearningView vabpnnFullLearningView(VabpnnFullLearning* model);
// Executed on the GPU after the actual policy sample and before the next parameter update.
void vabpnnCacheFullReward(const VabpnnGpuFrame& frame,VabpnnFullLearning* model,
    vabpnn::RewardCache* reward);
// The exact same validated batch as the policy heads. No prediction is recast as a human label.
void vabpnnPrepareFullReward(VabpnnFullLearning* model,const VabpnnGpuFrame& models,
    const vabpnn::RewardBatch* batch,int* accepted,cudaStream_t stream);
void vabpnnCommitFullReward(VabpnnFullLearning* model,const VabpnnGpuFrame& models,const int* accepted,cudaStream_t stream);
void vabpnnReadFullLearningStatus(VabpnnFullLearning* model,vabpnn::FullLearningStatus* host,cudaStream_t stream);
void vabpnnSecondsRewardScore(const VabpnnDecisionModel* model,const vabpnn::FullRewardRoots* roots,
    float* gradient, float* votes, void* scratch, cudaStream_t stream,
    float* spatialGradient = nullptr, const int* active = nullptr);
std::size_t vabpnnSecondsScratchBytes();
void vabpnnMinuteRewardScore(VabpnnDecisionModel* model,const vabpnn::FullRewardRoots* roots,
    float* gradient,float* votes,cudaStream_t stream);
void vabpnnCopyBlockRewardObservation(VabpnnBlockModel* model,vabpnn::BlockRewardObservation* target,cudaStream_t stream);
void vabpnnApplyBlockReward(VabpnnBlockModel* model,const vabpnn::BlockRewardObservation* const* traces,
    const int* count,const float* credit,cudaStream_t stream);
void vabpnnFullJepaMotionStep(VabpnnJepaModel* model,const int* active,cudaStream_t stream);
float* vabpnnOnlineMasterWeights(VabpnnOnlineModel* model);
vabpnn::OnlinePersistent* vabpnnOnlinePersistent(VabpnnOnlineModel* model);

const float* vabpnnOnlineWheelFeatures(VabpnnOnlineModel* model);
