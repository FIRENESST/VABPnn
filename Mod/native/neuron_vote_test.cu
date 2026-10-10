#include "FrameOperator.h"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

using Trit = vabpnn::TernaryWeight;
static void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

// Independent fixture, not a training rule. All three ternary states are exercised.
__host__ __device__ int fixtureWeight(std::size_t pixel, int channel, int k) {
    switch (k) {
    case 0: return channel == 0 ? 1 : 0;
    case 1: return channel == 1 ? -1 : 0;
    case 2: return channel == 0 ? -1 : channel == 1 ? 1 : 0;
    case 3: return channel < 2 ? 1 : -1;
    case 4: return channel == 0 ? (pixel % 2 == 0 ? 1 : -1) : 0;
    case 5: return channel == 0 ? (pixel % 3 == 0 ? -1 : 1) : 0;
    case 6: return channel == 0 && pixel < 5 ? -1 : 0;
    case 7: return channel == 0 ? (pixel % 3 == 0 ? 1 : -1) : 0;
    case 8: return channel == 0 ? -1 : channel == 2 ? 1 : 0;
    case 9: return channel == 0 ? 1 : channel == 2 ? -1 : 0;
    case 10: return 1;
    case 11: return -1;
    case 12: return channel == 0 && pixel < 3 ? 1 : 0;
    default: return 0;
    }
}
__global__ void fixtureWeights(Trit* weights, int direction) {
    const std::size_t pixel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= vabpnn::NeuronCount) return;
    for (int channel = 0; channel < 3; ++channel) for (int k = 0; k < 13; ++k) {
        // Explicit [channel][vote][pixel] layout; exercise the production layout helper separately.
        const std::size_t offset = (channel * 13 + k) * 1966080ull + pixel;
        weights[offset] = static_cast<Trit>(direction * fixtureWeight(pixel, channel, k));
    }
}

static void verifyFrame(VabpnnNeuronModel* model, cudaStream_t stream,
                        int width, int height, int direction, const char* label) {
    unsigned char* rgba = nullptr;
    float* rgb = nullptr;
    try {
        // Artificial bottom-left-origin RGBA. R+G-B includes exact zero, +1 and -1 scores.
        std::vector<unsigned char> input(static_cast<std::size_t>(width) * height * 4);
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 4;
            const int r = (x * 3 + y * 7) % 100 + 1;
            const int g = (x * 5 + y * 2) % 100 + 1;
            input[i] = r; input[i + 1] = g;
            input[i + 2] = r + g + ((x + y) % 3 - 1);
            input[i + 3] = 255;
        }
        std::size_t pitch = 0;
        check(cudaMallocPitch(reinterpret_cast<void**>(&rgba), &pitch, width * 4, height));
        check(cudaMalloc(reinterpret_cast<void**>(&rgb),
                         static_cast<std::size_t>(width) * height * 3 * sizeof(float)));
        check(cudaMemcpy2DAsync(rgba, pitch, input.data(), width * 4, width * 4, height,
                               cudaMemcpyHostToDevice, stream));
        VabpnnGpuFrame frame{rgba, pitch, rgb, width, height, 123, 0, stream, model};
        vabpnnLaunchOperator(frame);
        const auto view = vabpnnNeuronModelView(model);
        if (view.parameterCount != 76677120) throw std::runtime_error("Wrong parameter count");
        std::array<std::int32_t, 13> actualSums{};
        std::array<std::int32_t, 13> actualOutput{};
        if (view.output != view.voteSums) throw std::runtime_error("Output must expose the original vote sums");
        // Test fixture only. Production never downloads game images or voting results.
        check(cudaMemcpyAsync(actualSums.data(), view.voteSums, sizeof(actualSums),
                              cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(actualOutput.data(), view.output, sizeof(actualOutput),
                              cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));

        std::array<std::int32_t, 13> expected{};
        if (direction != 0) {
            for (int y = 0; y < 1024; ++y) for (int x = 0; x < 1920; ++x) {
                const int sourceX = (2 * x + 1) * width / (2 * 1920);
                const int sourceY = (2 * y + 1) * height / (2 * 1024);
                const std::size_t i = (static_cast<std::size_t>(height - 1 - sourceY) * width + sourceX) * 4;
                const std::size_t pixel = static_cast<std::size_t>(y) * 1920 + x;
                for (int k = 0; k < 13; ++k) {
                    int score = 0;
                    for (int c = 0; c < 3; ++c)
                        score += input[i + c] * direction * fixtureWeight(pixel, c, k);
                    expected[k] += (score > 0) - (score < 0);
                }
            }
            // These votes have analytically known global outcomes, including ties.
            if (expected[0] != direction * 1966080 || expected[1] != -direction * 1966080
                    || expected[4] != 0 || expected[5] != direction * 655360
                    || expected[6] != -direction * 5 || expected[7] != -direction * 655360
                    || expected[12] != direction * 3)
                throw std::runtime_error("Fixture analytical totals are inconsistent");
        }
        for (int k = 0; k < 13; ++k) {
            if (actualSums[k] != expected[k] || actualOutput[k] != expected[k]) {
                std::fprintf(stderr, "Vote %d: got sum=%d output=%d, expected sum=%d output=%d\n",
                    k, actualSums[k], actualOutput[k], expected[k], expected[k]);
                throw std::runtime_error("Raw vote total mismatch");
            }
        }
        std::printf("PASS: %s, source %dx%d -> 1920x1024 neurons, all 13 signed integer totals.\n",
                    label, width, height);
    } catch (...) {
        cudaStreamSynchronize(stream);
        if (rgba) cudaFree(rgba);
        if (rgb) cudaFree(rgb);
        throw;
    }
    cudaFree(rgba); cudaFree(rgb);
}

int main() {
    cudaStream_t stream = nullptr;
    VabpnnNeuronModel* model = nullptr;
    int result = 0;
    try {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        model = vabpnnCreateNeuronModel(stream);
        const auto originalView = vabpnnNeuronModelView(model);
        const std::uint64_t bytes = vabpnnNeuronModelBytes(model);
        if (bytes != 76677120ull + 13ull * 7680 * sizeof(std::int32_t) + 13ull * sizeof(std::int32_t))
            throw std::runtime_error("Model allocation estimate is incorrect");
        verifyFrame(model, stream, 37, 23, 0, "zero-initialized standby");

        auto* weights = vabpnnMutableNeuronWeights(model, stream);
        fixtureWeights<<<(vabpnn::NeuronCount + 255) / 256, 256, 0, stream>>>(weights, 1);
        check(cudaGetLastError());
        verifyFrame(model, stream, 1920, 1024, 1, "small +/- counts, hundreds-of-thousands totals and ties");

        // GPU weight updates affect the next frame, including differently sized input.
        fixtureWeights<<<(vabpnn::NeuronCount + 255) / 256, 256, 0, stream>>>(weights, -1);
        check(cudaGetLastError());
        verifyFrame(model, stream, 37, 23, -1, "weight update plus GPU resampling");
        if (vabpnnNeuronModelView(model).weights != originalView.weights
                || vabpnnNeuronModelBytes(model) != bytes)
            throw std::runtime_error("Weights reallocated during frame/size changes");
        std::puts("PASS: persistent 76,677,120 ternary parameters and 13-item raw int32 device output.");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }
    if (stream) cudaStreamSynchronize(stream);
    vabpnnDestroyNeuronModel(model);
    if (stream) cudaStreamDestroy(stream);
    return result;
}
