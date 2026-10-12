#pragma once
#include "GpuFrame.h"
#include "NeuronModel.h"
#include "ModelSnapshot.h"
#include <cstddef>

struct VabpnnMinuteModel;
namespace vabpnn {
constexpr int MinuteHistory = 8; // Current observed minute and up to seven previous minutes.
constexpr int MinuteWidth = 8, MinuteHeads = 2, MinuteHeadWidth = MinuteWidth / MinuteHeads;
constexpr int MinuteTimeFeatures = 4, MinuteKChoices = VoteCount + 1;
constexpr std::uint64_t MinuteNanos = 60'000'000'000ull;

// Independent minute encoder, graph attention, decay/frequency and fusion heads.
// Zero fusion heads preserve an existing seconds model exactly on migration.
struct DecisionMinuteParameters {
    float nodeEmbedding[VoteCount][MinuteWidth], voteEmbedding[MinuteWidth];
    float timeEmbedding[MinuteTimeFeatures][MinuteWidth];
    float query[MinuteWidth][MinuteWidth], key[MinuteWidth][MinuteWidth];
    float value[MinuteWidth][MinuteWidth], attentionOutput[MinuteWidth][MinuteWidth];
    float graphBias[MinuteHeads][VoteCount][VoteCount];
    float logAttentionDecay[MinuteHeads], logClockFrequency; // Rates measured per MINUTE.
    float evidence[VoteCount][MinuteWidth], k[MinuteKChoices][MinuteWidth];
};
constexpr int MinuteParameterCount = sizeof(DecisionMinuteParameters) / sizeof(float);
static_assert(MinuteParameterCount == 1117, "Compact independently learnable minute axis.");
static_assert(offsetof(DecisionMinuteParameters,k)+sizeof(DecisionMinuteParameters::k)
    == sizeof(DecisionMinuteParameters), "Contiguous float parameter layout.");
constexpr int BayesianMotionFeatures=VoteCount*2+3; // Posterior, prior, JEPA X/Y and bias.
struct BayesianMotionPersistent {
    float weights[2][BayesianMotionFeatures],first[2][BayesianMotionFeatures],second[2][BayesianMotionFeatures];
    std::uint64_t steps;
};
static_assert(sizeof(BayesianMotionPersistent)==800,"Neutral learnable Bayesian JEPA residual.");
struct DecisionMinutePersistent {
    DecisionMinuteParameters parameters, first, second;
    std::uint64_t steps;
    BayesianMotionPersistent motion;
};
static_assert(sizeof(DecisionMinutePersistent)==14216, "Append-only minute weights and AdamW part.");

// GPU diagnostics and raw bounded summaries, not images; production copies only status counters.
struct DecisionMinuteOutput {
    std::uint64_t frameId, captureNanos;
    float elapsedMinutes;
    int buckets;
    float bucketMinutes[MinuteHistory], ageMinutes[MinuteHistory], coverageMinutes[MinuteHistory];
    float votes[MinuteHistory][VoteCount], currentVotes[VoteCount];
    float hidden[VoteCount][MinuteWidth], mean[MinuteWidth];
    float evidence[VoteCount], kLogits[MinuteKChoices];
    std::uint32_t valid;
};
struct DeviceMinuteView {
    const DecisionMinutePersistent* persistent;
    const DecisionMinuteOutput* output;
};
}
VabpnnMinuteModel* vabpnnCreateMinuteModel(cudaStream_t stream);
void vabpnnDestroyMinuteModel(VabpnnMinuteModel* model) noexcept;
std::uint64_t vabpnnMinuteModelBytes(const VabpnnMinuteModel* model) noexcept;
vabpnn::DeviceMinuteView vabpnnMinuteModelView(const VabpnnMinuteModel* model);
vabpnn::DecisionMinutePersistent* vabpnnMutableMinuteModel(VabpnnMinuteModel* model,cudaStream_t stream);
vabpnn::ModelPart vabpnnMinuteModelCheckpointPart(VabpnnMinuteModel* model);
void vabpnnLaunchMinuteTimeline(const VabpnnGpuFrame& frame,const std::int32_t* votes,VabpnnMinuteModel* model);
void vabpnnResetMinuteHistory(VabpnnMinuteModel* model,cudaStream_t stream);
