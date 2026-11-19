#pragma once
#include "JepaMotion.h"
#include "TemporalDecision.h"

namespace vabpnn {
constexpr int PnnJepaTilesAcross = JepaGridX * JepaCells;
constexpr int PnnJepaTilesDown = JepaGridY * JepaCells;
constexpr int PnnJepaBands = 3;
constexpr int PnnJepaBandFeatures = VoteCount + 1;
constexpr int JepaDecoderColumns = 5, JepaDecoderRows = 3;
constexpr int JepaDecoderChannels = JepaWidth / DecisionWidth;
static_assert(PnnJepaBandFeatures * PnnJepaBands == JepaInput);
static_assert(PnnJepaTilesAcross * 16 == NeuronWidth && PnnJepaTilesDown * 16 == NeuronHeight);
static_assert(JepaDecoderColumns * JepaDecoderRows == VoteCount && JepaWidth % DecisionWidth == 0);

struct JepaDecoderInput {
    float spatial[VoteCount][DecisionWidth];
    // The auxiliary scalar inputs to the action heads and minute timeline are
    // reductions of the same latent map, stored on the original vote scale.
    std::int32_t votes[VoteCount];
};

__host__ __device__ constexpr int pnnJepaBand(int rowInPatch) {
    return (rowInPatch + 1) / 2;
}
__host__ __device__ constexpr int pnnJepaBandStart(int band) {
    return band == 0 ? 0 : 2 * band - 1;
}
__host__ __device__ constexpr int pnnJepaBandRows(int band) {
    return band == 1 ? 2 : 1;
}
__host__ __device__ constexpr int pnnJepaPatch(int x, int y) {
    return (y / 64) * JepaGridX + x / 64;
}
__host__ __device__ constexpr int jepaDecoderRegion(int patch) {
    return (patch / JepaGridX * JepaDecoderRows / JepaGridY) * JepaDecoderColumns
        + patch % JepaGridX * JepaDecoderColumns / JepaGridX;
}
__host__ __device__ constexpr int jepaRegionStart(int region, int extent, int divisions) {
    return (region * extent + divisions - 1) / divisions;
}
}

void vabpnnPoolPnnForJepa(const VabpnnGpuFrame& frame, float* patchFeatures);
void vabpnnPoolJepaDecoderInput(const float* observed, const float* predicted,
    const vabpnn::JepaOutput* output, vabpnn::JepaDecoderInput* decoder, cudaStream_t stream);
// Returns the current decoder input, produced on the JEPA frame stream.
const vabpnn::JepaDecoderInput* vabpnnJepaDecoderInput(const VabpnnJepaModel* model);
void vabpnnBackpropPnnJepa(const vabpnn::JepaEncoder* encoder, const float* hidden,
    const vabpnn::JepaOutput* output, const vabpnn::JepaDecoderInput* decoder,
    const float* spatialScore, const float* scalarScore, float* patchScore, cudaStream_t stream);
void vabpnnScoreJepaPnnFeatures(const VabpnnJepaModel* model, const float* spatialScore,
    const float* scalarScore, float* patchScore, cudaStream_t stream);
