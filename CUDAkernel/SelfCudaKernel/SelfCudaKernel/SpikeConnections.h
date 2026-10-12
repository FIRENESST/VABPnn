#pragma once
#include "GpuFrame.h"
#include "NeuronModel.h"
#include "TemporalDecision.h"
#include "ModelSnapshot.h"
#include "BlockControllers.h"
struct VabpnnSpikeModel;
namespace vabpnn {
constexpr int SpikeTile=16, SpikeTileSize=256, SpikeTiles=120*64;
constexpr int LocalFanout=4, LocalTileEdges=SpikeTileSize*LocalFanout;
constexpr int SpikeAnchorLimit=256, SpikeRadius=6, SpikeNearCount=2*SpikeRadius*(SpikeRadius+1);
constexpr int SpikeLongCapacity=SpikeAnchorLimit*(SpikeAnchorLimit-1+SpikeNearCount);
static_assert(VoteCount<=16,"Two-bit votes must fit in a 32-bit cache.");
constexpr std::uint32_t NoNeuron=0xffffffffu, SpikeSourceMask=(1u<<21)-1;
enum class ConnectionKind : unsigned char { EasierStimulate, HarderStimulate, EasierPositive, EasierNegative };
struct SpikeConfig {
    int enabled=1, maxAnchors=SpikeAnchorLimit;
    unsigned int seed=0x7a6b9d31u;
    float membraneTau=0.035f, threshold=0.45f, minimumThreshold=0.1f, maximumThreshold=1.5f;
    float stimulationGain=0.22f, polarityGain=0.18f, negativeMargin=0.04f;
    float refractorySeconds=0.016f, maxGapSeconds=0.25f, statisticsDecay=0.95f, minimumScore=1e-5f;
};
// Eight adaptive scouts per tile estimate ungated action utility, so low membrane
// activity never excludes a pixel from placement learning. These values are structural
// bandit statistics, independent of frozen neural weights and AdamW state.
constexpr int SpikePlacementScouts=8;
struct SpikePlacementArm {
    std::uint32_t pixel;
    float utility, secondMoment, visits;
};
struct SpikePlacementTile {
    SpikePlacementArm arms[SpikePlacementScouts];
    std::uint64_t rounds, supervisedRounds;
};
static_assert(sizeof(SpikePlacementTile)==144,"Bounded persistent placement statistics.");
// Plastic statistics; no cached RGB, instantaneous membrane, or spikes in checkpoints.
struct SpikePixelStatistics { float positive, negative, zero, crossEntropy, hit; };
struct SpikeCounters {
    std::uint64_t predictions, topologyUpdates, localEdges, contributionUpdates;
    std::uint32_t anchors, longEdges, invalidConfig, reserved;
};
struct DeviceSpikeView {
    const SpikeConfig* config;
    const SpikeCounters* counters;
    const std::int8_t* spikes;
    const float* membrane;
    const float* refractory;
    const SpikePixelStatistics* statistics;
    const std::uint32_t* packedVotes;
    const std::uint16_t* localOutgoing; // [tile][source local index][4], bit 10 = enabled
    const std::uint16_t* localRows;     // [tile][257], compact CSR offsets <= 1024
    const std::uint16_t* localColumns;  // [tile][1024], source local index + kind << 8
    const std::uint32_t* longRows;      // [NeuronCount+1], destination rows
    const std::uint32_t* longColumns;   // source global index + kind << 21
    const std::uint32_t* anchorIds;
    const std::uint16_t* anchorIndex;   // 0xffff means not selected
    const std::uint32_t* championIds;
    const BlockControl* blockControls;
    const SpikePlacementTile* placement;
};
__host__ __device__ constexpr std::uint32_t spikePixel(int tile,int local) {
    return static_cast<std::uint32_t>((tile/120*16+local/16)*NeuronWidth+tile%120*16+local%16);
}
}
VabpnnSpikeModel* vabpnnCreateSpikeModel(cudaStream_t stream);
void vabpnnDestroySpikeModel(VabpnnSpikeModel* model) noexcept;
VabpnnBlockModel* vabpnnSpikeBlockModel(const VabpnnSpikeModel* model);
std::uint64_t vabpnnSpikeModelBytes(const VabpnnSpikeModel* model) noexcept;
vabpnn::DeviceSpikeView vabpnnSpikeModelView(const VabpnnSpikeModel* model);
vabpnn::SpikeConfig* vabpnnMutableSpikeConfig(VabpnnSpikeModel* model,cudaStream_t stream);
vabpnn::SpikePixelStatistics* vabpnnMutableSpikeStatistics(VabpnnSpikeModel* model,cudaStream_t stream);
std::int8_t* vabpnnMutableSpikeSignals(VabpnnSpikeModel* model,cudaStream_t stream);
std::uint16_t* vabpnnMutableSpikeOutgoing(VabpnnSpikeModel* model,cudaStream_t stream);
// Rebuild incoming CSR after an external outgoing edit; no self edges / duplicates / fanout > 4.
void vabpnnRebuildSpikeLocalCsr(VabpnnSpikeModel* model,cudaStream_t stream);
// Dense synchronous tick; CSR four typed SpMVs use |previousSpike|, never event queues.
void vabpnnLaunchSpikeForward(const VabpnnGpuFrame& frame);
void vabpnnRecordSpikeVotes(VabpnnSpikeModel* model,cudaStream_t stream);
std::uint32_t* vabpnnMutableSpikeVoteCache(VabpnnSpikeModel* model,cudaStream_t stream);
void vabpnnObserveSpikeContribution(VabpnnSpikeModel* model,const vabpnn::DecisionOutput* decision,
    const vabpnn::DecisionParameters* parameters,const int* valid,const std::uint32_t* label,cudaStream_t stream);
// Refresh prediction salience and probe placement every fifth frame on the consumer stream.
// Probes use model sensitivity without treating a model prediction as a human label.
void vabpnnObserveSpikePrediction(const VabpnnGpuFrame& frame);
vabpnn::ModelPart vabpnnSpikePlacementCheckpointPart(VabpnnSpikeModel* model);
void vabpnnResetSpikeHistory(VabpnnSpikeModel* model,cudaStream_t stream);
std::size_t vabpnnSpikeCheckpointParts(VabpnnSpikeModel* model,vabpnn::ModelPart* parts);
void vabpnnResumeSpikeModel(VabpnnSpikeModel* model,cudaStream_t stream);

