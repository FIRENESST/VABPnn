#pragma once
#include "GpuFrame.h"
#include "NeuronModel.h"
#include "TemporalDecision.h"
#include "JepaMotion.h"

namespace vabpnn {
constexpr int TargetWidth = NeuronWidth;
constexpr int TargetHeight = NeuronHeight;
constexpr int BlockWidth = 16;
constexpr int BlockHeight = 16;
constexpr int ThreadsPerBlock = BlockWidth * BlockHeight;
constexpr int TargetGridWidth = TargetWidth / BlockWidth;
constexpr int TargetGridHeight = TargetHeight / BlockHeight;
static_assert(TargetWidth % BlockWidth == 0 && TargetHeight % BlockHeight == 0,
              "The target frame must contain whole tiles.");
static_assert(ThreadsPerBlock == 256 && TargetGridWidth == 120 && TargetGridHeight == 64,
              "1920x1024 uses a 120x64 grid of 16x16 thread blocks.");

inline dim3 blockShape() { return dim3(BlockWidth, BlockHeight, 1); }
// Positive frame dimensions only. Round up for resized windows and guard edges in the kernel.
inline dim3 gridShape(int width, int height) {
    return dim3(1 + (width - 1) / BlockWidth, 1 + (height - 1) / BlockHeight, 1);
}

struct Pixel {
    int x, y;                   // Output coordinates, top-left origin.
    int blockX, blockY;         // Tile coordinates.
    int localX, localY;         // Coordinates inside the 16x16 tile.
    std::size_t index, plane;   // NCHW pixel offset and one channel's element count.
    float red, green, blue;     // Normalized RGB input for future calculations.
};
}

// Host-side extension point. Enqueue subsequent kernels on frame.stream.
// The input and output pointers are borrowed from the mod's GPU pool; never free them.
void vabpnnLaunchNextStage(const VabpnnGpuFrame& frame);

// JEPA self-supervised learning runs after forward/decision. Other trainers remain extension points.
void vabpnnLaunchLearningStep(const VabpnnGpuFrame& frame);
