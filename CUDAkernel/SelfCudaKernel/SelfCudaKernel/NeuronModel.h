#pragma once
#include "GpuFrame.h"

namespace vabpnn {
constexpr int NeuronWidth = 1920;
constexpr int NeuronHeight = 1024;
constexpr int InputChannels = 3;
constexpr int VoteCount = 13;
constexpr std::size_t NeuronCount = static_cast<std::size_t>(NeuronWidth) * NeuronHeight;
constexpr std::size_t ParameterCount = NeuronCount * InputChannels * VoteCount;
static_assert(NeuronCount == 1966080 && ParameterCount == 76677120,
              "Each pixel has 3 x 13 independent ternary weights.");

enum class TernaryWeight : std::int8_t { Negative = -1, Neutral = 0, Positive = 1 };
static_assert(sizeof(TernaryWeight) == 1, "Use one byte per ternary weight.");

// Logical per-neuron weights: 3 RGB channels x 13 vote items.
// Physical layout [channel][vote][pixel] keeps adjacent pixel loads contiguous.
__host__ __device__ constexpr std::size_t weightIndex(std::size_t pixel, int channel, int vote) {
    return (static_cast<std::size_t>(channel) * VoteCount + vote) * NeuronCount + pixel;
}
static_assert(weightIndex(NeuronCount - 1, 2, 12) == ParameterCount - 1,
              "Last weight must be inside the parameter buffer.");

struct DeviceNeuronView {
    const TernaryWeight* weights;
    const std::int32_t* voteSums;   // 13 global sums; no CPU download.
    const std::int32_t* output;    // Canonical output: 13 raw sums, alias of voteSums.
    std::size_t parameterCount;
    const std::int32_t* blockVotes; // Existing [13][7680] reductions, reused for block credit.
};
}

// Worker-thread lifecycle, on the same CUDA device as the incoming frames.
// Allocate once; keep weights across frame-pool resizes. Initial weights are all zero.
VabpnnNeuronModel* vabpnnCreateNeuronModel(cudaStream_t initializationStream);
void vabpnnDestroyNeuronModel(VabpnnNeuronModel* model) noexcept;
std::uint64_t vabpnnNeuronModelBytes(const VabpnnNeuronModel* model) noexcept;
vabpnn::DeviceNeuronView vabpnnNeuronModelView(const VabpnnNeuronModel* model);

// Enqueue sign(R*wR+G*wG+B*wB) per neuron, then sum all votes.
// Keep signed integer totals unchanged for the following temporal model.
// Non-1920x1024 source frames are nearest-neighbor sampled on the GPU.
void vabpnnLaunchTernaryVotes(const VabpnnGpuFrame& frame, VabpnnNeuronModel* model);

// Training extension: initialization dependency is queued on the supplied stream.
// Modify weights on that stream, keeping every value exactly -1, 0 or +1.
// No labels, loss, optimizer or update rule are implied by this interface.
vabpnn::TernaryWeight* vabpnnMutableNeuronWeights(VabpnnNeuronModel* model, cudaStream_t stream);
