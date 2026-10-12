#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace vabpnn {
// One-based user numbering is Action index + 1.
enum class Action : int {
    W = 0, A, S, D, E, MouseX, MouseY,
    LeftPress, RightPress, LeftRelease, RightRelease, Wheel, Space, Ctrl, Shift, Count
};
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
}
