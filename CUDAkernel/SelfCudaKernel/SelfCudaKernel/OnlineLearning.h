#pragma once
#include "GpuFrame.h"
#include "JepaMotion.h"
#include "ModelSnapshot.h"
struct VabpnnOnlineModel;
namespace vabpnn {
// Cumulative real inputs survive dropped frames. Java never fills these from the policy.
struct DemoSample {
    std::uint64_t epoch = 0, edge[4]{}, keyHeldNanos[6]{};
    double mouseX = 0, mouseY = 0, wheel = 0;
    // Action mask: keyboard holds plus LeftPress/RightPress while physically held.
    std::uint32_t heldKeys = 0, active = 0, modelControl = 0, pointerMode = 0;
};
struct OnlineOutput {
    std::uint64_t examples, rejected, pixelSteps, decisionSteps;
    float decisionLoss, wheelLoss;
    std::uint32_t lastLabel, valid;
};
}
VabpnnOnlineModel* vabpnnCreateOnlineModel(cudaStream_t stream);
void vabpnnDestroyOnlineModel(VabpnnOnlineModel* model) noexcept;
std::uint64_t vabpnnOnlineModelBytes(const VabpnnOnlineModel* model) noexcept;
const vabpnn::OnlineOutput* vabpnnOnlineOutput(const VabpnnOnlineModel* model);
const vabpnn::DemoSample* vabpnnOnlineDemoSample(const VabpnnOnlineModel* model);
void vabpnnSetDemoSample(VabpnnOnlineModel* model, const vabpnn::DemoSample* sample, cudaStream_t stream);
// Before the next forward: label the CACHED previous observation with intervening real input.
void vabpnnLearnPreviousDemo(const VabpnnGpuFrame& frame);
void vabpnnCacheDemoObservation(const VabpnnGpuFrame& frame);
void vabpnnOnlineWheel(const VabpnnGpuFrame& frame, vabpnn::MotionProposal* proposal);
void vabpnnResetOnlineHistory(VabpnnOnlineModel* model, cudaStream_t stream);
std::size_t vabpnnOnlineCheckpointParts(VabpnnOnlineModel* model, vabpnn::ModelPart* parts);
