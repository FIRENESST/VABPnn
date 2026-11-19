#pragma once
#include "GpuFrame.h"
#include "ModelSnapshot.h"
struct VabpnnBlockModel;
struct VabpnnMemeModel;
namespace vabpnn {
constexpr int BlockCount=120*64, BlockFeatures=16, BlockOutputs=4, BlockWeights=BlockFeatures*BlockOutputs;
constexpr int BlockControlEvery=5;
enum class BlockCommunication : int { Independent=0, Cooperate=1, Compete=2 };
struct BlockConfig {
    int enabled=1, communication=1, radius=1, warmup=32, trainEvery=5, rankEvery=128;
    int minimumExamples=64, minimumWindow=32, poorWindows=3, cooldown=256;
    float exploration=0.05f, learningRate=0.0003f, minimumCoverage=0.05f, rankMargin=0.002f, mutation=0.01f;
};
struct BlockControl {
    float thresholdScale=1, polarityBias=0, memoryScale=1, learningScale=1;
};
struct BlockState {
    float weights[BlockWeights]{}, first[BlockWeights]{}, second[BlockWeights]{}, bestWeights[BlockWeights]{};
    float score=0, baseline=0, variance=0, contribution=0, agreement=0, coverage=0, progress=0;
    float windowSum=0, windowSquares=0, windowCoverage=0, rankScore=0, previousRankScore=0, bestScore=-1e20f;
    std::uint64_t examples=0, optimizerSteps=0, mutations=0, cooldownUntil=0;
    std::uint32_t windowExamples=0, poorStreak=0, protectedAction=0, reserved=0;
};
struct BlockCounters {
    std::uint64_t scoredFrames=0, actorUpdates=0, rankings=0, mutations=0, controlUpdates=0, broadcasts=0;
    std::uint32_t enabled=1, communication=1, radius=1, eligible=0, candidates=0, punished=0, invalidConfig=0, reserved=0;
};
static_assert(sizeof(BlockCounters)==80,"Bounded block-controller status.");
struct BlockPersistent {
    BlockConfig config{};
    BlockCounters counters{};
    BlockState state[BlockCount]{};
};
struct DeviceBlockView {
    const BlockConfig* config;
    const BlockCounters* counters;
    const BlockState* state;
    const BlockControl* controls;
};
}
VabpnnBlockModel* vabpnnCreateBlockModel(cudaStream_t stream);
void vabpnnDestroyBlockModel(VabpnnBlockModel* model) noexcept;
VabpnnMemeModel* vabpnnBlockMemeModel(const VabpnnBlockModel* model);
std::uint64_t vabpnnBlockModelBytes(const VabpnnBlockModel* model) noexcept;
vabpnn::DeviceBlockView vabpnnBlockModelView(const VabpnnBlockModel* model);
vabpnn::BlockPersistent* vabpnnMutableBlockPersistent(VabpnnBlockModel* model,cudaStream_t stream);
void vabpnnConfigureBlockModel(VabpnnBlockModel* model,const vabpnn::BlockConfig& config,cudaStream_t stream);
void vabpnnConfigureBlockMode(VabpnnBlockModel* model,int enabled,int communication,cudaStream_t stream);
// Called after OLD feedback and before both trainPixels and the CURRENT LIF forward.
void vabpnnPrepareBlockControllers(const VabpnnGpuFrame& frame);
void vabpnnLearnBlockControllers(const VabpnnGpuFrame& frame,const int* valid,const std::uint32_t* label);
void vabpnnRankBlockControllers(VabpnnBlockModel* model,cudaStream_t stream);
void vabpnnResetBlockHistory(VabpnnBlockModel* model,cudaStream_t stream);
void vabpnnResumeBlockModel(VabpnnBlockModel* model,cudaStream_t stream);
vabpnn::ModelPart vabpnnBlockCheckpointPart(VabpnnBlockModel* model);
