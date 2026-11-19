#pragma once
#include "NeuronModel.h"
#include "TemporalDecision.h"
#include "ModelSnapshot.h"
#include <type_traits>

struct VabpnnDenseBlockModel;
namespace vabpnn {
constexpr int DenseNodes=120*64, DenseWidth=16, DenseHeads=2, DenseHeadWidth=8, DenseHidden=32;
constexpr int DenseVoteHistory=8, DenseTimeFeatures=8;
constexpr int DenseFeatures=VoteCount*DenseVoteHistory+DenseTimeFeatures+PositionFeatures+2;
constexpr std::size_t DensePairParameters=static_cast<std::size_t>(DenseHeads)*DenseNodes*DenseNodes;
// Every ordered pair, including the diagonal, owns a separate FP32 parameter per head.
struct DenseBlockParameters {
    float nodeEmbedding[DenseNodes][DenseWidth];
    float input[DenseFeatures][DenseWidth];
    float query[DenseWidth][DenseWidth],key[DenseWidth][DenseWidth],value[DenseWidth][DenseWidth];
    float attentionOutput[DenseWidth][DenseWidth];
    float attentionGain[DenseWidth],attentionBias[DenseWidth];
    float up[DenseWidth][DenseHidden],upBias[DenseHidden];
    float down[DenseHidden][DenseWidth],downBias[DenseWidth];
    float outputGain[DenseWidth],outputBias[DenseWidth];
    float readout[VoteCount][DenseWidth];
    float logSecondsFrequency,logMinutesFrequency;
};
constexpr std::size_t DenseSmallParameters=sizeof(DenseBlockParameters)/sizeof(float);
constexpr std::size_t DenseParameterCount=DensePairParameters+DenseSmallParameters;
static_assert(DenseNodes==7680 && DenseHeads==2 && DenseFeatures==142,"Fixed comparison topology.");
static_assert(DensePairParameters==117964800 && DenseSmallParameters==127554
    && DenseParameterCount==118092354,"Exact count of the comparison Transformer, excluding PNN/JEPA.");
static_assert(std::is_trivially_copyable<DenseBlockParameters>::value,"Contiguous FP32 parameters.");
static_assert(offsetof(DenseBlockParameters,logMinutesFrequency)+sizeof(float)==sizeof(DenseBlockParameters),
    "No hidden padding in trainable parameters.");
struct DenseBlockPersistent {
    DenseBlockParameters parameters,first,second;
    std::uint64_t steps;
};
static_assert(sizeof(DenseBlockPersistent)==1530656,"Independent comparison parameters and AdamW.");
struct DenseBlockOutput {
    float hidden[VoteCount][DenseWidth];
    std::uint64_t frameId,captureNanos;
    std::uint32_t valid;
};
struct DenseBlockStatus {
    std::uint64_t frameId,steps,parameters,pairs,allocatedBytes;
    std::uint32_t nodes,heads,valid;
};
struct DeviceDenseBlockView {
    DenseBlockPersistent* persistent;
    float* graphBias;
    float* first;
    float* second;
    float* graphGradient;
    DenseBlockParameters* gradient;
    const float* attention; // Exact [head][query node][key node] softmax, no masks/top-k/factorization.
    const float* features;
    const DenseBlockOutput* output;
    const DenseBlockStatus* status;
    int nodes;
};
}
VabpnnDenseBlockModel* vabpnnCreateDenseBlockModel(cudaStream_t stream,int nodes=vabpnn::DenseNodes);
void vabpnnDestroyDenseBlockModel(VabpnnDenseBlockModel* model) noexcept;
std::uint64_t vabpnnDenseBlockModelBytes(const VabpnnDenseBlockModel* model) noexcept;
vabpnn::DeviceDenseBlockView vabpnnDenseBlockModelView(VabpnnDenseBlockModel* model);
std::size_t vabpnnDenseBlockCheckpointParts(VabpnnDenseBlockModel* model,vabpnn::ModelPart* parts);
// blockVotes is the existing [15][7680] signed GPU reduction. Each node keeps its last 8 observations.
void vabpnnLaunchDenseBlocks(VabpnnDenseBlockModel* model,const VabpnnGpuFrame& frame,
    const std::int32_t* blockVotes,const float* positionFeatures);
void vabpnnTrainDenseBlocks(VabpnnDenseBlockModel* model,const vabpnn::DecisionParameters* base,
    const vabpnn::DecisionOutput* previous,const int* active,const std::uint32_t* label,cudaStream_t stream);
void vabpnnResetDenseBlocks(VabpnnDenseBlockModel* model,cudaStream_t stream);
// Numerical verification uses these exact production kernels with a smaller complete graph.
void vabpnnDenseForwardFeatures(VabpnnDenseBlockModel* model,const float* deviceFeatures,
    std::uint64_t frameId,std::uint64_t captureNanos,cudaStream_t stream);
void vabpnnDenseBackwardHidden(VabpnnDenseBlockModel* model,const float* hiddenGradient,
    const int* active,cudaStream_t stream);
void vabpnnDenseApplyAdamW(VabpnnDenseBlockModel* model,cudaStream_t stream);
