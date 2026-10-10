#pragma once
#include "GpuFrame.h"
#include "NeuronModel.h"
#include "ModelSnapshot.h"

struct VabpnnDecisionModel;

namespace vabpnn {
// One-based user numbering is Action index + 1.
enum class Action : int {
    W = 0, A, S, D, E, MouseX, MouseY,
    LeftPress, RightPress, LeftRelease, RightRelease, Wheel, Space, Count
};
static_assert(static_cast<int>(Action::Count) == VoteCount, "Exactly 13 atomic actions.");
__host__ __device__ constexpr std::uint32_t actionBit(Action action) {
    return 1u << static_cast<int>(action);
}

// Filled by the later GPU motion operator. Relative mouse units and vertical wheel steps.
struct MotionProposal { float mouseDx, mouseDy, wheelSteps; };

// Final control only. JNI transfers this small command, never frames/votes/probabilities.
struct ControlCommand {
    std::uint64_t frameId, captureNanos;
    std::uint32_t selectedMask, valid;
    float mouseDx, mouseDy, wheelSteps;
};
static_assert(sizeof(ControlCommand) == 40, "One 40-byte final control command.");
// Small execution diagnostics; sampled as host counters, never image/tensor data.
struct DecisionStatus {
    std::uint64_t frameId;
    std::uint32_t eligibleMask;
    std::int32_t requestedK, selectedCount, valid;
    std::uint64_t positionNanos, positionEpoch, positionSteps;
    double positionX, positionY, positionZ;
    std::uint32_t positionValid;
};
static_assert(sizeof(DecisionStatus)==80,"Bounded decision and position counters.");
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
struct DecisionParameters {
    float nodeEmbedding[VoteCount][DecisionWidth];
    float voteEmbedding[DecisionWidth];
    float timeEmbedding[DecisionTimeFeatures][DecisionWidth];
    float query[DecisionWidth][DecisionWidth];
    float key[DecisionWidth][DecisionWidth];
    float value[DecisionWidth][DecisionWidth];
    float attentionOutput[DecisionWidth][DecisionWidth];
    float graphBias[DecisionHeads][VoteCount][VoteCount];
    float logAttentionDecay[DecisionHeads]; // softplus -> decay per second
    float logClockFrequency;                // softplus -> radians per second
    float attentionGain[DecisionWidth], attentionBias[DecisionWidth];
    float feedForwardUp[DecisionWidth][DecisionHiddenWidth];
    float feedForwardUpBias[DecisionHiddenWidth];
    float feedForwardDown[DecisionHiddenWidth][DecisionWidth];
    float feedForwardDownBias[DecisionWidth];
    float outputGain[DecisionWidth], outputBias[DecisionWidth];
    float logLikelihoodWeight[VoteCount][DecisionWidth], logLikelihoodBias[VoteCount];
    float rawVoteWeight[VoteCount]; // Online visual evidence from this action's raw total.
    float thresholdWeight[VoteCount][DecisionWidth], thresholdBias[VoteCount];
    float kWeight[DecisionKChoices][DecisionKFeatures], kBias[DecisionKChoices];
    float initialPriorLogit[VoteCount];
    float logOnRate[VoteCount], logOffRate[VoteCount]; // softplus -> transitions/second
    float logGateTemperature; // Training extension: relaxed gate, inference remains strict.
};

// Separate checkpoint part: old pixel, graph, JEPA and optimizer layouts stay unchanged.
// Zero initialization makes fresh position observations neutral until human teaching.
struct DecisionPositionAdapter {
    float evidence[VoteCount][PositionFeatures], k[DecisionKChoices][PositionFeatures];
    float evidenceFirst[VoteCount][PositionFeatures], evidenceSecond[VoteCount][PositionFeatures];
    float kFirst[DecisionKChoices][PositionFeatures], kSecond[DecisionKChoices][PositionFeatures];
    std::uint64_t steps;
};
static_assert(sizeof(DecisionParameters)==15544,"Keep the legacy decision checkpoint layout.");
static_assert(sizeof(DecisionPositionAdapter)==3896,"324 trainable position weights plus AdamW state.");

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

// Queue causal graph/time attention -> CTMC prior -> Bayes -> strict threshold -> learned k.
// votes is 13 raw int32 totals in device memory. No per-frame allocation, wait, or CPU copy.
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

