#include "PnnJepa.h"
#include "SpikeConnections.h"
#include <cmath>
#include <stdexcept>

namespace {
using namespace vabpnn;
constexpr float LatentGain = 4.f;
constexpr float PredictionMix = .25f;
constexpr unsigned WarpMask = 0xffffffffu;

void checkLaunch() {
    if (const auto status = cudaGetLastError(); status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

__device__ float predictionMix(int patch, const JepaOutput* frame) {
    const int maskX = frame->frameId % (JepaGridX - JepaMaskX + 1);
    const int maskY = (frame->frameId / 3) % (JepaGridY - JepaMaskY + 1);
    const int x = patch % JepaGridX, y = patch / JepaGridX;
    return frame->pairReady && x >= maskX && x < maskX + JepaMaskX && y >= maskY && y < maskY + JepaMaskY
        ? PredictionMix : 0.f;
}

__global__ void poolSpikes(const std::int32_t* votes, const float* membrane, float* features) {
    const int patch = blockIdx.x, lane = threadIdx.x & 31, band = threadIdx.x / 32;
    const int tileX = patch % JepaGridX * JepaCells;
    const int tileY = patch / JepaGridX * JepaCells + pnnJepaBandStart(band);
    const int rows = pnnJepaBandRows(band);
    const int pixels = rows * JepaCells * 16 * 16;
    float* output = features + patch * JepaInput + band * PnnJepaBandFeatures;
    if (lane < VoteCount) {
        int sum = 0;
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < JepaCells; ++x)
                sum += votes[lane * SpikeTiles + (tileY + y) * PnnJepaTilesAcross + tileX + x];
        output[lane] = static_cast<float>(sum) / pixels;
    }
    // Membrane activity retains information before a neuron emits a signed spike.
    float sum = 0;
    constexpr int patchWidth = JepaCells * 16;
    for (int at = lane; at < pixels; at += 32)
        sum += membrane[(tileY * 16 + at / patchWidth) * NeuronWidth + tileX * 16 + at % patchWidth];
    for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(WarpMask, sum, offset);
    if (lane == 0) output[VoteCount] = tanhf(sum / pixels);
}

__global__ void poolLatents(const float* observed, const float* predicted,
    const JepaOutput* output, JepaDecoderInput* decoder) {
    const int region = blockIdx.x, d = threadIdx.x;
    const int x0 = jepaRegionStart(region % JepaDecoderColumns, JepaGridX, JepaDecoderColumns);
    const int x1 = jepaRegionStart(region % JepaDecoderColumns + 1, JepaGridX, JepaDecoderColumns);
    const int y0 = jepaRegionStart(region / JepaDecoderColumns, JepaGridY, JepaDecoderRows);
    const int y1 = jepaRegionStart(region / JepaDecoderColumns + 1, JepaGridY, JepaDecoderRows);
    float sum = 0;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const int patch = y * JepaGridX + x;
            const float mix = predictionMix(patch, output);
            for (int k = d; k < JepaWidth; k += DecisionWidth) {
                const int index = patch * JepaWidth + k;
                sum += mix > 0 ? observed[index] * (1 - mix) + predicted[index] * mix : observed[index];
            }
        }
    }
    decoder->spatial[region][d] = output->valid
        ? tanhf(LatentGain * sum / ((x1 - x0) * (y1 - y0) * JepaDecoderChannels)) : 0;
}

__global__ void summarizeLatents(JepaDecoderInput* decoder) {
    const int action = threadIdx.x;
    float sum = 0;
    for (int region = 0; region < VoteCount; ++region) sum += decoder->spatial[region][action];
    decoder->votes[action] = __float2int_rn(sum * (static_cast<float>(NeuronCount) / VoteCount));
}

__global__ void backpropEncoderInput(const JepaEncoder* encoder, const float* hidden,
    const JepaOutput* output, const JepaDecoderInput* decoder, const float* spatialScore,
    const float* scalarScore, float* patchScore) {
    const int patch = blockIdx.x, h = threadIdx.x;
    const int region = jepaDecoderRegion(patch);
    const int width = jepaRegionStart(region % JepaDecoderColumns + 1, JepaGridX, JepaDecoderColumns)
                    - jepaRegionStart(region % JepaDecoderColumns, JepaGridX, JepaDecoderColumns);
    const int height = jepaRegionStart(region / JepaDecoderColumns + 1, JepaGridY, JepaDecoderRows)
                     - jepaRegionStart(region / JepaDecoderColumns, JepaGridY, JepaDecoderRows);
    // Forecasts and past observations remain fixed during action credit assignment.
    // The predictor and latent encoder weights learn from JEPA's own temporal loss.
    const float scale = LatentGain * (1 - predictionMix(patch, output))
        / (width * height * JepaDecoderChannels);
    __shared__ float latentScore[DecisionWidth], hiddenScore[JepaHidden];
    if (h < DecisionWidth) {
        const float value = decoder->spatial[region][h];
        const float summary = h < VoteCount ? scalarScore[h] / VoteCount : 0.f;
        latentScore[h] = (spatialScore[region * DecisionWidth + h] + summary)
            * scale * (1 - value * value);
    }
    __syncthreads();
    float sum = 0;
    for (int k = 0; k < JepaWidth; ++k)
        sum += encoder->outputWeight[h][k] * latentScore[k % DecisionWidth];
    const float value = hidden[patch * JepaHidden + h];
    hiddenScore[h] = sum * (1 - value * value);
    __syncthreads();
    if (h < JepaInput) {
        sum = 0;
        for (int k = 0; k < JepaHidden; ++k) sum += hiddenScore[k] * encoder->inputWeight[h][k];
        patchScore[patch * JepaInput + h] = sum;
    }
}
}

void vabpnnPoolPnnForJepa(const VabpnnGpuFrame& frame, float* patchFeatures) {
    if (!frame.neuronModel || !frame.spikeModel)
        throw std::invalid_argument("PNN JEPA requires ternary votes and the spiking model");
    const auto pnn = vabpnnNeuronModelView(frame.neuronModel);
    const auto spikes = vabpnnSpikeModelView(frame.spikeModel);
    poolSpikes<<<vabpnn::JepaTokens, 32 * vabpnn::PnnJepaBands, 0, frame.stream>>>(
        pnn.blockVotes, spikes.membrane, patchFeatures);
    checkLaunch();
}

void vabpnnPoolJepaDecoderInput(const float* observed, const float* predicted,
    const vabpnn::JepaOutput* output, vabpnn::JepaDecoderInput* decoder, cudaStream_t stream) {
    poolLatents<<<vabpnn::VoteCount, vabpnn::DecisionWidth, 0, stream>>>(observed, predicted, output, decoder);
    summarizeLatents<<<1, vabpnn::VoteCount, 0, stream>>>(decoder);
    checkLaunch();
}

void vabpnnBackpropPnnJepa(const vabpnn::JepaEncoder* encoder, const float* hidden,
    const vabpnn::JepaOutput* output, const vabpnn::JepaDecoderInput* decoder,
    const float* spatialScore, const float* scalarScore, float* patchScore, cudaStream_t stream) {
    backpropEncoderInput<<<vabpnn::JepaTokens, vabpnn::JepaHidden, 0, stream>>>(
        encoder, hidden, output, decoder, spatialScore, scalarScore, patchScore);
    checkLaunch();
}
