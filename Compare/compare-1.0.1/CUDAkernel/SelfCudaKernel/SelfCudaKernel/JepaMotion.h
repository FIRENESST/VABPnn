#pragma once
#include "GpuFrame.h"
#include "NeuronModel.h"
#include "ActionContracts.h"
#include <type_traits>

struct VabpnnJepaModel;
namespace vabpnn {
// Small temporal JEPA, implemented in CUDA; this is not a pretrained Meta checkpoint.
constexpr int JepaGridX = 30, JepaGridY = 16, JepaTokens = JepaGridX * JepaGridY;
constexpr int JepaCells = 4, JepaInput = JepaCells * JepaCells * 3;
constexpr int JepaWidth = 128, JepaHidden = 256;
constexpr int JepaMoments = 3 * JepaWidth; // mean, horizontal moment, vertical moment
constexpr int JepaMaskX = 10, JepaMaskY = 8, JepaTargets = JepaMaskX * JepaMaskY;
constexpr int JepaPredictInput = JepaWidth + JepaMoments + 4; // local, moments, xy, time
constexpr int JepaMotionInput = JepaMoments * 3 + VoteCount + 2; // current, past, difference

template<int Input, int Hidden, int Output> struct JepaMlp {
    // Input-major matrices. tanh hidden activation, linear output.
    float inputWeight[Input][Hidden], hiddenBias[Hidden];
    float outputWeight[Hidden][Output], outputBias[Output];
};
using JepaEncoder = JepaMlp<JepaInput, JepaHidden, JepaWidth>;
using JepaPredictor = JepaMlp<JepaPredictInput, JepaHidden, JepaWidth>;
template<int Actions> using JepaMotionHeadT=JepaMlp<JepaMoments*3+Actions+2,JepaHidden,2>;
using JepaMotionHead=JepaMotionHeadT<VoteCount>;
template<int Actions> struct JepaParametersT {
    JepaEncoder encoder;
    JepaPredictor predictor;
    JepaMotionHeadT<Actions> motion;
};
using JepaParameters=JepaParametersT<VoteCount>;
constexpr std::size_t JepaEncoderParameters = sizeof(JepaEncoder) / sizeof(float);
constexpr std::size_t JepaSelfSupervisedParameters =
    (sizeof(JepaEncoder) + sizeof(JepaPredictor)) / sizeof(float);
constexpr std::size_t JepaParameterCount = sizeof(JepaParameters) / sizeof(float);
constexpr std::size_t JepaTotalParameterCount = JepaParameterCount + JepaEncoderParameters;
static_assert(JepaParameterCount == 510722 && JepaTotalParameterCount == 556162, "Parameter budget.");
static_assert(JepaTotalParameterCount < 50000000, "JEPA including EMA must be below 0.05B.");
static_assert(std::is_standard_layout<JepaParameters>::value, "Contiguous float parameters.");
// Parameters are large: allocate host copies on the heap, not the Windows default stack.

struct JepaConfig {
    float learningRate = 0.0003f, beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
    float weightDecay = 0.01f, gradientClip = 1.0f, emaDecay = 0.996f;
    float varianceTarget = 0.2f, varianceWeight = 0.1f;
    float maxPairGapSeconds = 0.25f, maxMotionDtSeconds = 0.1f;
    float maxMouseUnitsPerSecond = 1200.0f, maxMouseDelta = 32.0f;
    int selfSupervisedEnabled = 1, trainEveryFrames = 4;
    int motionEnabled = 0; // Enable only after supplying trained head parameters.
};

// A later trainer writes this AFTER forward, then calls vabpnnTrainJepaMotion.
// Targets are desired raw GLFW-relative mouse units / second, before vanilla sensitivity.
struct JepaMotionTarget {
    std::uint64_t frameId;
    float rateX, rateY;
    std::uint32_t valid;
};
struct JepaOutput {
    std::uint64_t frameId, captureNanos, selfSupervisedSteps, motionSteps;
    float deltaSeconds, rateX, rateY, mouseDx, mouseDy;
    float headRateX,headRateY; // Unconditioned JEPA rates, also cached in demonstration mode.
    float predictionLoss, variancePenalty, jepaLoss, gradientNorm, motionLoss;
    int valid, pairReady, motionActive, lastLearningValid;
};
struct DeviceJepaView {
    const JepaParameters* parameters;
    const JepaEncoder* targetEncoder; // EMA, stop-gradient
    const JepaConfig* config;
    const JepaOutput* output;
    const float* currentPatchFeatures;
    const float* previousPatchFeatures;
    const float* currentLatents;
    const float* previousLatents;
    const float* targetLatents;
    const float* predictions;
    const float* gradients; // Last computed parameter gradients, before norm clipping.
    const float* firstMoment;
    const float* secondMoment;
};
}

VabpnnJepaModel* vabpnnCreateJepaModel(cudaStream_t initializationStream);
void vabpnnDestroyJepaModel(VabpnnJepaModel* model) noexcept;
std::uint64_t vabpnnJepaModelBytes(const VabpnnJepaModel* model) noexcept;
vabpnn::DeviceJepaView vabpnnJepaModelView(const VabpnnJepaModel* model);
vabpnn::JepaParameters* vabpnnMutableJepaParameters(VabpnnJepaModel* model, cudaStream_t stream);
vabpnn::JepaConfig* vabpnnMutableJepaConfig(VabpnnJepaModel* model, cudaStream_t stream);
vabpnn::JepaMotionTarget* vabpnnMutableJepaMotionTarget(VabpnnJepaModel* model, cudaStream_t stream);

// Same stream only; forward caches small GPU patch features, never retains frame-pool pointers.
// votes contains the 15 raw int32 totals. proposal may be null for representation-only tests.
// Writes X/Y only; wheel remains available to the later wheel operator.
void vabpnnLaunchJepaForward(const VabpnnGpuFrame& frame, const std::int32_t* votes,
                            VabpnnJepaModel* model, vabpnn::MotionProposal* proposal);
// Actual backprop + AdamW + EMA, normally after decision; gated by GPU config/pair validity.
// Only encoder/predictor are updated. No labels or reward are invented for the motion head.
void vabpnnTrainJepaSelfSupervised(VabpnnJepaModel* model, cudaStream_t stream);
// Optional supervised head-only MSE training on the cached forward features.
// The caller supplies a matching-frame valid device target; there is no automatic invocation.
void vabpnnTrainJepaMotion(VabpnnJepaModel* model, cudaStream_t stream);
// New world/resize/clock reversal: discard history, retain parameters and optimizer moments.
void vabpnnResetJepaHistory(VabpnnJepaModel* model, cudaStream_t stream);


#include "ModelSnapshot.h"
std::size_t vabpnnJepaCheckpointParts(VabpnnJepaModel* model, vabpnn::ModelPart* parts);
void vabpnnResumeJepaOptimizer(VabpnnJepaModel* model, cudaStream_t stream);

