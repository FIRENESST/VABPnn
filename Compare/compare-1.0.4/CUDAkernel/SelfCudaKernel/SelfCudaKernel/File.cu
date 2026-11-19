#include "FrameOperator.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "MemeState.h"
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

// DEVICE EXTENSION POINT: each valid pixel calls this once.
// Add per-pixel calculations here. No allocation, CPU copy, or block-wide barrier here;
// partial edge tiles may have threads that return before entering this function.
__device__ __forceinline__
void vabpnnComputePixel(const VabpnnGpuFrame& frame, const vabpnn::Pixel& pixel) {
    // Initial stage: keep the normalized RGB tensor ready in device memory.
    frame.rgbNchw[pixel.index] = pixel.red;
    frame.rgbNchw[pixel.plane + pixel.index] = pixel.green;
    frame.rgbNchw[pixel.plane * 2 + pixel.index] = pixel.blue;
    // TODO: per-pixel computation, using pixel coordinates / tile coordinates / RGB.
}

__global__ __launch_bounds__(vabpnn::ThreadsPerBlock)
void vabpnnFrameKernel(VabpnnGpuFrame frame) {
    int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= frame.width || y >= frame.height) return;

    // Input is bottom-left-origin RGBA8; output is top-left-origin RGB NCHW.
    const unsigned char* rgba = frame.rgba
        + static_cast<std::size_t>(frame.height - 1 - y) * frame.rgbaPitch
        + static_cast<std::size_t>(x) * 4;
    vabpnn::Pixel pixel{
        x, y,
        static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
        static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y),
        static_cast<std::size_t>(y) * frame.width + x,
        static_cast<std::size_t>(frame.width) * frame.height,
        rgba[0] / 255.0f, rgba[1] / 255.0f, rgba[2] / 255.0f
    };
    vabpnnComputePixel(frame, pixel);
}
}

// HOST EXTENSION POINT: the first stage has been queued on frame.stream.
// TODO: launch later kernels here with <<<grid, block, sharedBytes, frame.stream>>>.
// rgbNchw already belongs to the same stream's sequence; no CPU wait is needed.
// Work on other streams must be joined back before returning. Do not retain pool pointers.
void vabpnnLaunchNextStage(const VabpnnGpuFrame& frame) {
    if (!frame.neuronModel) return; // Numeric preprocessing tests can omit the model.
    if(frame.spikeModel) vabpnnLaunchSpikeForward(frame);
    vabpnnLaunchTernaryVotes(frame, frame.neuronModel);
    if(frame.spikeModel) vabpnnRecordSpikeVotes(frame.spikeModel,frame.stream);
    vabpnnObserveMemes(frame);
    const auto votes = vabpnnNeuronModelView(frame.neuronModel);
    if (!frame.decisionModel) return;
    vabpnnClearMotionProposal(frame.decisionModel, frame.stream);
    auto* motion = vabpnnMutableMotionProposal(frame.decisionModel, frame.stream);
    if (frame.jepaModel) vabpnnLaunchJepaForward(frame, votes.output, frame.jepaModel, motion);
    if (frame.onlineModel) vabpnnOnlineWheel(frame, motion);
    // WHEEL EXTENSION POINT: queue a later GPU wheel operator here. JEPA writes X/Y only.
    // Mouse movement is enabled only after head training, then gated by actions 6 and 7.
    vabpnnLaunchTemporalDecision(frame, votes.output, frame.decisionModel);
    vabpnnObserveSpikePrediction(frame); // Structural learning also runs with frozen neural weights.
    // GPU probabilities/history remain here. The mod receives only the final 40-byte command.
}

// LEARNING EXTENSION POINT: weights are mutable, but an update rule needs labels/loss.
// Implement GPU updates on frame.stream using vabpnnMutableNeuronWeights(...).
// Keep pixel weights exactly -1, 0 or +1; updates affect subsequent frames.
// TemporalDecision parameters are float32; update via vabpnnMutableDecisionParameters(...).
// Pixel/decision labels and rules remain external. JEPA already has GPU AdamW self-supervision.
void vabpnnLaunchLearningStep(const VabpnnGpuFrame& frame) {
    if (frame.evaluationMode) return; // Includes JEPA AdamW/EMA and observation training caches.
    if (frame.jepaModel && frame.neuronModel)
        vabpnnTrainJepaSelfSupervised(frame.jepaModel, frame.stream);
    if (frame.onlineModel) vabpnnCacheDemoObservation(frame);
}

// The mod invokes this once per frame, from its CUDA worker thread.
void vabpnnLaunchOperator(const VabpnnGpuFrame& frame) {
    if (!frame.rgba || !frame.rgbNchw || frame.width <= 0 || frame.height <= 0
            || frame.rgbaPitch < static_cast<std::size_t>(frame.width) * 4)
        throw std::invalid_argument("Invalid GPU frame pointers, dimensions or row pitch");

    if (frame.onlineModel) vabpnnLearnPreviousDemo(frame);
    else vabpnnPrepareBlockControllers(frame);
    const dim3 block = vabpnn::blockShape();
    const dim3 grid = vabpnn::gridShape(frame.width, frame.height);
    // 1920x1024: block=(16,16,1), grid=(120,64,1), one thread per pixel.
    vabpnnFrameKernel<<<grid, block, 0, frame.stream>>>(frame);
    checkCuda(cudaGetLastError(), "launch 16x16 frame kernel");
    vabpnnLaunchNextStage(frame);
    vabpnnLaunchLearningStep(frame);
}

#ifndef VABPNN_OPERATOR_LIBRARY
// Standalone VS validation uses synthetic GPU buffers only.
// Real game frames enter through vabpnnLaunchOperator in the mod DLL.
int main() {
    unsigned char* rgba = nullptr;
    float* rgb = nullptr;
    cudaStream_t stream = nullptr;
    VabpnnNeuronModel* model = nullptr;
    VabpnnDecisionModel* decision = nullptr;
    VabpnnJepaModel* jepa = nullptr;
    VabpnnSpikeModel* spike = nullptr;
    int exitCode = 0;
    try {
        const dim3 block = vabpnn::blockShape();
        const dim3 grid = vabpnn::gridShape(vabpnn::TargetWidth, vabpnn::TargetHeight);
        std::printf("Frame %dx%d; block %ux%u; grid %ux%u; %u blocks.\n",
                    vabpnn::TargetWidth, vabpnn::TargetHeight,
                    block.x, block.y, grid.x, grid.y, grid.x * grid.y);
        std::size_t pitch = 0;
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        checkCuda(cudaMallocPitch(reinterpret_cast<void**>(&rgba), &pitch,
                                 vabpnn::TargetWidth * 4, vabpnn::TargetHeight), "allocate RGBA");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&rgb),
                  static_cast<std::size_t>(vabpnn::TargetWidth) * vabpnn::TargetHeight
                  * 3 * sizeof(float)), "allocate RGB tensor");
        checkCuda(cudaMemset2DAsync(rgba, pitch, 0, vabpnn::TargetWidth * 4,
                                   vabpnn::TargetHeight, stream), "initialize synthetic input");
        model = vabpnnCreateNeuronModel(stream);
        decision = vabpnnCreateDecisionModel(stream);
        spike = vabpnnCreateSpikeModel(stream);
        jepa = vabpnnCreateJepaModel(stream);
        std::printf("Neurons: %zu; trainable ternary weights: %zu.\n",
                    vabpnn::NeuronCount, vabpnn::ParameterCount);
        VabpnnGpuFrame frame{rgba, pitch, rgb, vabpnn::TargetWidth,
                            vabpnn::TargetHeight, 0, 0, stream, model, decision, jepa,nullptr,spike};
        vabpnnLaunchOperator(frame);
        checkCuda(cudaStreamSynchronize(stream), "verify standalone kernel completion");
        std::puts("PASS: GPU JEPA/AdamW + raw votes -> graph/Markov/Bayes -> threshold/top-k; no image readback.");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        exitCode = 1;
    }
    // Standalone cleanup only; the DLL entry above never synchronizes or frees pool memory.
    if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    vabpnnDestroyJepaModel(jepa);
    vabpnnDestroyDecisionModel(decision);
    vabpnnDestroySpikeModel(spike);
    vabpnnDestroyNeuronModel(model);
    if (rgb) cudaFree(rgb);
    if (rgba) cudaFree(rgba);
    return exitCode;
}
#endif
