#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

struct VabpnnNeuronModel;
struct VabpnnDecisionModel;
struct VabpnnJepaModel;
struct VabpnnOnlineModel;
struct VabpnnSpikeModel;

namespace vabpnn {
// Same monotonic clock as captureNanos; zero timestamp means no current local player.
struct PlayerPositionSample {
    std::uint64_t sampleNanos = 0, epoch = 0;
    double x = 0, y = 0, z = 0;
};
static_assert(sizeof(PlayerPositionSample) == 40, "Bounded host player metadata.");
}

// Device pointers only. Lifetime ends after all work in frame.stream has completed.
struct VabpnnGpuFrame {
    const unsigned char* rgba; // OpenGL bottom-left origin, RGBA8
    std::size_t rgbaPitch;
    float* rgbNchw;            // Preallocated 3 x H x W, top-left origin
    int width;
    int height;
    std::uint64_t frameId;
    std::uint64_t captureNanos;
    cudaStream_t stream;
    VabpnnNeuronModel* neuronModel = nullptr; // Persistent model, owned by the CUDA worker.
    VabpnnDecisionModel* decisionModel = nullptr; // Time graph / Markov / Bayes / selection.
    VabpnnJepaModel* jepaModel = nullptr; // Visual JEPA / mouse-motion head.
    VabpnnOnlineModel* onlineModel = nullptr; // Cached human demonstrations and online training.
    VabpnnSpikeModel* spikeModel = nullptr; // CSR connections, membrane and signed spikes.
    vabpnn::PlayerPositionSample position{}; // Causal 30 Hz sidecar-style observation.
};

// Replace this consumer with your operator. Queue work on frame.stream without host waits.
// If you use additional streams, join them back to frame.stream before returning.
void vabpnnLaunchOperator(const VabpnnGpuFrame& frame);

