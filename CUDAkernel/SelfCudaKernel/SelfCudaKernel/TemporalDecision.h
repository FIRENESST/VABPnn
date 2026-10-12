#pragma once
#include "GpuFrame.h"
#include "ActionContracts.h"
#include "NeuronModel.h"
#include "ModelSnapshot.h"
#include "MinuteTimeline.h"

struct VabpnnDecisionModel;

namespace vabpnn {
static_assert(static_cast<int>(Action::Count) == VoteCount, "Exactly 15 atomic actions.");

// Small execution diagnostics; sampled as host counters, never image/tensor data.
struct DecisionStatus {
    std::uint64_t frameId;
    std::uint32_t eligibleMask;
    std::int32_t requestedK, selectedCount, valid;
    std::uint64_t positionNanos, positionEpoch, positionSteps;
    double positionX, positionY, positionZ;
    std::uint32_t positionValid;
    double elapsedMinutes;
    std::uint64_t minuteSteps;
    std::uint32_t minuteBuckets;
    std::uint64_t bayesianMotionSteps;
};
static_assert(sizeof(DecisionStatus)==112,"Bounded decision, position and dual-clock counters.");
constexpr int DecisionWidth = 16;
constexpr int DecisionHeads = 2;
constexpr int DecisionHeadWidth = DecisionWidth / DecisionHeads;
constexpr int DecisionHiddenWidth = 32;
constexpr int DecisionHistory = 8;
constexpr int DecisionTimeFeatures = 4;
constexpr int PositionFeatures = 12;
constexpr std::uint64_t PositionMaximumAgeNanos = 200'000'000;
constexpr int DecisionKChoices = VoteCount + 1; // Includes doing nothing.
constexpr int DecisionKFeatures = DecisionWidth + VoteCount * 2;
static_assert(DecisionWidth % DecisionHeads == 0, "Whole attention heads.");

// Mutable GPU parameters. All heads are trainable; no optimizer or training labels are implied.
// The evidence head must be trained/calibrated as a CONDITIONAL log likelihood ratio,
// not as a posterior that already includes the same prior.
// Empty base preserves the exact 13-action ABI; extended models persist an active k prefix.
template<int Actions,bool Extended=(Actions>13)> struct DecisionExpansionState {};
template<int Actions> struct DecisionExpansionState<Actions,true> {
    float enabledKChoices=Actions+1; // Metadata, activated by real label counts; not an AdamW weight.
};
template<int Actions> struct DecisionParametersT : DecisionExpansionState<Actions> {
    float nodeEmbedding[Actions][DecisionWidth];
    float voteEmbedding[DecisionWidth];
    float timeEmbedding[DecisionTimeFeatures][DecisionWidth];
    float query[DecisionWidth][DecisionWidth];
    float key[DecisionWidth][DecisionWidth];
    float value[DecisionWidth][DecisionWidth];
    float attentionOutput[DecisionWidth][DecisionWidth];
    float graphBias[DecisionHeads][Actions][Actions];
    float logAttentionDecay[DecisionHeads]; // softplus -> decay per second
    float logClockFrequency;                // softplus -> radians per second
    float attentionGain[DecisionWidth], attentionBias[DecisionWidth];
    float feedForwardUp[DecisionWidth][DecisionHiddenWidth];
    float feedForwardUpBias[DecisionHiddenWidth];
    float feedForwardDown[DecisionHiddenWidth][DecisionWidth];
    float feedForwardDownBias[DecisionWidth];
    float outputGain[DecisionWidth], outputBias[DecisionWidth];
    float logLikelihoodWeight[Actions][DecisionWidth], logLikelihoodBias[Actions];
    float rawVoteWeight[Actions]; // Online visual evidence from this action's raw total.
    float thresholdWeight[Actions][DecisionWidth], thresholdBias[Actions];
    float kWeight[(Actions+1)][(DecisionWidth+Actions*2)], kBias[(Actions+1)];
    float initialPriorLogit[Actions];
    float logOnRate[Actions], logOffRate[Actions]; // softplus -> transitions/second
    float logGateTemperature; // Training extension: relaxed gate, inference remains strict.
};
using DecisionParameters=DecisionParametersT<VoteCount>;

// Separate checkpoint part: old pixel, graph, JEPA and optimizer layouts stay unchanged.
// Zero initialization makes fresh position observations neutral until human teaching.
template<int Actions> struct DecisionPositionAdapterT {
    float evidence[Actions][PositionFeatures], k[(Actions+1)][PositionFeatures];
    float evidenceFirst[Actions][PositionFeatures], evidenceSecond[Actions][PositionFeatures];
    float kFirst[(Actions+1)][PositionFeatures], kSecond[(Actions+1)][PositionFeatures];
    std::uint64_t steps;
};
using DecisionPositionAdapter=DecisionPositionAdapterT<VoteCount>;
static_assert(sizeof(DecisionParametersT<13>)==15544 && sizeof(DecisionParameters)==17028,"Explicit legacy and 15-action decision layouts.");
static_assert(sizeof(DecisionPositionAdapterT<13>)==3896 && sizeof(DecisionPositionAdapter)==4472,"372 trainable position weights plus AdamW state.");

// Fixed action constraints, separate from learned weights. Defaults: (8,9), (10,11), (8,12), (9,12), in one-based action numbering.
// Either direction of a pair is sufficient to prevent selecting the two together.
struct DecisionConstraints {
    std::uint32_t conflictMask[VoteCount];
};

// GPU result; one reusable buffer for the whole model. Consume on the same stream.
// confidence is the posterior probability that the object SHOULD be triggered.
struct DecisionOutput {
    float hidden[VoteCount][DecisionWidth], kFeatures[DecisionKFeatures], normalizedVotes[VoteCount];
    float prior[VoteCount];
    float logLikelihoodRatio[VoteCount];
    float posterior[VoteCount];
    float confidence[VoteCount];
    float threshold[VoteCount];
    float softEligibility[VoteCount]; // For a later differentiable training objective.
    float kProbability[DecisionKChoices];
    int selectedIndices[VoteCount];  // Descending posterior; unused entries are -1.
    std::uint32_t eligibleMask, selectedMask;
    int requestedK, selectedCount, valid;
    float deltaSeconds;
    float positionFeatures[PositionFeatures];
    std::uint32_t positionValid;
    std::uint64_t frameId, captureNanos;
    ControlCommand command;
    DecisionStatus status;
};

struct DeviceDecisionView {
    const DecisionParameters* parameters;
    const DecisionConstraints* constraints;
    const DecisionOutput* output;
    const ControlCommand* command;
    const DecisionPositionAdapter* position;
    DeviceMinuteView minutes;
};
}

// Worker-owned allocations; survive frame-pool resize. Creation initializes neutral heads:
// prior=posterior=threshold=0.5, strict eligibility false, k=0. No automatic game input.
VabpnnDecisionModel* vabpnnCreateDecisionModel(cudaStream_t initializationStream);
void vabpnnDestroyDecisionModel(VabpnnDecisionModel* model) noexcept;
std::uint64_t vabpnnDecisionModelBytes(const VabpnnDecisionModel* model) noexcept;
vabpnn::DeviceDecisionView vabpnnDecisionModelView(const VabpnnDecisionModel* model);
vabpnn::DecisionParameters* vabpnnMutableDecisionParameters(VabpnnDecisionModel* model, cudaStream_t stream);
vabpnn::DecisionConstraints* vabpnnMutableDecisionConstraints(VabpnnDecisionModel* model, cudaStream_t stream);
vabpnn::DecisionPositionAdapter* vabpnnMutablePositionAdapter(VabpnnDecisionModel* model, cudaStream_t stream);
std::size_t vabpnnDecisionPositionCheckpointParts(VabpnnDecisionModel* model, vabpnn::ModelPart* parts);
vabpnn::DecisionMinutePersistent* vabpnnMutableMinuteTimeline(VabpnnDecisionModel* model,cudaStream_t stream);
vabpnn::ModelPart vabpnnDecisionMinuteCheckpointPart(VabpnnDecisionModel* model);
// Called only on a valid cached human demonstration, before changing the seconds heads.
void vabpnnTrainMinuteDecision(VabpnnDecisionModel* model,const int* active,const std::uint32_t* label,cudaStream_t stream);
// A cached Bayesian/JEPA residual learns only from the previous real mouse target.
void vabpnnTrainBayesianMotion(VabpnnDecisionModel* model,const int* active,const float* rateX,const float* rateY,cudaStream_t stream);
void vabpnnTrainMinuteTimeline(VabpnnMinuteModel* model,const vabpnn::DecisionParameters* base,
    const vabpnn::DecisionOutput* output,const int* active,const std::uint32_t* label,cudaStream_t stream);

// Queue causal graph/time attention -> CTMC prior -> Bayes -> learned k.
// Demonstrations retain the calibrated threshold gate. Frozen prediction/control use all
// valid actions as candidates, continuously producing top-k packets (k=0 remains a learned rest).
// votes is 15 raw int32 totals in device memory. No per-frame allocation, wait, or CPU copy.
// One active consumer stream per model; join other streams before the next call.
void vabpnnLaunchTemporalDecision(const VabpnnGpuFrame& frame,
                                  const std::int32_t* votes, VabpnnDecisionModel* model);

// Next motion operator queues device writes here before the decision kernel.
// Clear each frame so a missing proposal cannot repeat old displacement.
vabpnn::MotionProposal* vabpnnMutableMotionProposal(VabpnnDecisionModel* model, cudaStream_t stream);
void vabpnnClearMotionProposal(VabpnnDecisionModel* model, cudaStream_t stream);

// Episode/world reset: clear history and beliefs, retain learned parameters/constraints.
// Queue on the model's consumer stream after any previous consumers.
void vabpnnResetDecisionState(VabpnnDecisionModel* model, cudaStream_t stream);


