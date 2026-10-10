#include "frame_operator.h"
#include "FrameOperator.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

static void verifyFrame(int width, int height) {
    unsigned char* deviceRgba = nullptr;
    float* allocation = nullptr;
    cudaStream_t stream = nullptr;
    auto cleanup = [&]() {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        if (deviceRgba) cudaFree(deviceRgba);
        if (allocation) cudaFree(allocation);
    };
    try {
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        std::vector<unsigned char> input(pixels * 4);
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 4;
            input[i] = (x * 17 + y * 13) & 255;
            input[i + 1] = (x * 3 + y * 29) & 255;
            input[i + 2] = (x * 11 + y * 5) & 255;
            input[i + 3] = 111;
        }
        constexpr std::size_t guard = 32;
        constexpr float sentinel = -42.0f;
        std::vector<float> output(pixels * 3 + guard * 2, sentinel);
        std::size_t pitch = 0;
        check(cudaMallocPitch(reinterpret_cast<void**>(&deviceRgba), &pitch, width * 4, height));
        check(cudaMalloc(reinterpret_cast<void**>(&allocation), output.size() * sizeof(float)));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaMemcpy2DAsync(deviceRgba, pitch, input.data(), width * 4, width * 4, height,
                               cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(allocation, output.data(), output.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
        VabpnnGpuFrame frame{deviceRgba, pitch, allocation + guard, width, height, 123, 0, stream};
        vabpnnLaunchOperator(frame);
        check(cudaGetLastError());
        // Artificial fixture only. The game path never downloads frames.
        check(cudaMemcpyAsync(output.data(), allocation, output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < guard; ++i) {
            if (output[i] != sentinel || output[output.size() - guard + i] != sentinel)
                throw std::runtime_error("Operator overwrote output guard memory");
        }
        for (int c = 0; c < 3; ++c) for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            const std::size_t source = (static_cast<std::size_t>(height - 1 - y) * width + x) * 4 + c;
            const float expected = input[source] / 255.0f;
            const float actual = output[guard + static_cast<std::size_t>(c) * pixels
                                        + static_cast<std::size_t>(y) * width + x];
            if (std::abs(expected - actual) > 0.000001f)
                throw std::runtime_error("RGB normalization/channel order/vertical flip mismatch");
        }
        const dim3 grid = vabpnn::gridShape(width, height);
        std::cout << "PASS: " << width << "x" << height << ", block 16x16, grid "
                  << grid.x << "x" << grid.y
                  << ", all pixels / pitched input / RGB NCHW / guard memory." << std::endl;
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

int main() {
    try {
        const dim3 block = vabpnn::blockShape();
        const dim3 grid = vabpnn::gridShape(1920, 1024);
        if (block.x != 16 || block.y != 16 || block.z != 1
                || grid.x != 120 || grid.y != 64 || grid.z != 1
                || grid.x * grid.y * block.x * block.y != 1920u * 1024u)
            throw std::runtime_error("Target frame launch layout is incorrect");
        verifyFrame(1920, 1024);
        verifyFrame(37, 23); // Partial tiles and row padding must remain safe.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
