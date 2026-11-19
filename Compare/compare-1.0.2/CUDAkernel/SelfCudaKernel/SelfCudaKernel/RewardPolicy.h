#pragma once
#include "TemporalDecision.h"
#include "JepaMotion.h"
#include <cstddef>
struct VabpnnRewardPolicy;
namespace vabpnn {
constexpr int RewardCacheSlots=512;
constexpr int RewardEvidenceStride=DecisionWidth+2;
constexpr int RewardEvidenceParameters=VoteCount*RewardEvidenceStride;
constexpr int RewardKStride=DecisionKFeatures+1;
constexpr int RewardPolicyParameters=RewardEvidenceParameters+DecisionKChoices*RewardKStride;
static_assert(RewardPolicyParameters==1022,"Existing evidence/raw-vote and k-head weights only.");
struct RewardPersistent {
    float first[RewardPolicyParameters],second[RewardPolicyParameters];
    std::uint64_t steps,policyVersion;
    float lastCredit;std::uint32_t lastCount;
};
static_assert(sizeof(RewardPersistent)==8200,"Independent reward AdamW; no new learned parameters.");
struct RewardProposal {
    std::uint64_t frameId,captureNanos,epoch,policyVersion;
    std::uint32_t mask,eligibleMask,k,count,valid;
    float logProbability;
};
struct RewardCache {
    RewardProposal proposal;
    int order[VoteCount];
    float score[RewardPolicyParameters]; // d log actual sampled joint policy / d existing parameter.
};
struct RewardBatch {
    std::uint64_t epoch,rewardId,frames[RewardCacheSlots];
    std::uint32_t masks[RewardCacheSlots],count;
    float credit; // Caller normalizes by Scorer's reward_clip; bounded -1..1.
};
enum class RewardReason:std::uint32_t {Ready,Disabled,Epoch,Duplicate,CacheMissing,OldPolicy,
    MaskMismatch,InvalidBatch,NoSupport,ZeroCredit,Updated};
struct RewardStatus {
    std::uint64_t epoch,policyVersion,updates,appliedRewards,droppedRewards,lastRewardId,proposals;
    std::uint32_t enabled,reason,lastCount;
    float lastCredit;
};
struct DeviceRewardView {
    RewardPersistent* persistent;RewardCache* cache;
    const RewardProposal* proposal;const RewardStatus* status;const float* gradient;
};
}
VabpnnRewardPolicy* vabpnnCreateRewardPolicy(cudaStream_t stream);
void vabpnnDestroyRewardPolicy(VabpnnRewardPolicy* model) noexcept;
std::uint64_t vabpnnRewardPolicyBytes(const VabpnnRewardPolicy* model) noexcept;
vabpnn::DeviceRewardView vabpnnRewardPolicyView(VabpnnRewardPolicy* model);
vabpnn::ModelPart vabpnnRewardCheckpointPart(VabpnnRewardPolicy* model);
void vabpnnRefreshRewardStatus(VabpnnRewardPolicy* model,cudaStream_t stream);
void vabpnnConfigureRewardPolicy(VabpnnRewardPolicy* model,std::uint64_t epoch,bool enabled,cudaStream_t stream);
// Stochastic k and without-replacement posterior sampling ONLY in explicit selftrain mode.
// All sampled actions belong to the existing strict threshold candidate set and obey conflicts.
void vabpnnProposeRewardAction(VabpnnRewardPolicy* model,const vabpnn::DecisionParameters* parameters,
    const vabpnn::DecisionConstraints* constraints,vabpnn::DecisionOutput* output,
    const vabpnn::JepaOutput* mouse,const vabpnn::MotionProposal* motion,cudaStream_t stream);
// Caller must already have verified actual execution and reward/server-tick identity windows.
// CUDA independently checks exact frame/mask/epoch/policy version; batch updates once after aggregation.
void vabpnnApplyRewardBatch(VabpnnRewardPolicy* model,vabpnn::DecisionParameters* parameters,
    const vabpnn::RewardBatch& batch,cudaStream_t stream);
