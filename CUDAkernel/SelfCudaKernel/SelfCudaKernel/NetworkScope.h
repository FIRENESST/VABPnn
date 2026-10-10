#pragma once
#include "GpuFrame.h"
#include "SpikeConnections.h"
namespace vabpnn {
// Explicit flat layout shared with ScopeSnapshot.java. Observation telemetry only, never RGB.
constexpr int ScopeEdges=512, ScopeMetrics=8;
constexpr int ScopeMapMem=ScopeMetrics, ScopeMapPositive=ScopeMapMem+SpikeTiles;
constexpr int ScopeMapNegative=ScopeMapPositive+SpikeTiles, ScopeLocalMem=ScopeMapNegative+SpikeTiles;
constexpr int ScopePosterior=ScopeLocalMem+SpikeTileSize, ScopeThreshold=ScopePosterior+VoteCount;
constexpr int ScopePrior=ScopeThreshold+VoteCount, ScopeBlockScore=ScopePrior+VoteCount;
constexpr int ScopeBlockSelected=ScopeBlockScore+BlockCount, ScopeFloats=ScopeBlockSelected+8;
constexpr int ScopeLocalSpike=8, ScopeOutgoing=ScopeLocalSpike+SpikeTileSize;
constexpr int ScopeAnchors=ScopeOutgoing+LocalTileEdges, ScopeEdgePairs=ScopeAnchors+SpikeAnchorLimit;
constexpr int ScopeVotes=ScopeEdgePairs+ScopeEdges*2, ScopeBlockData=ScopeVotes+VoteCount, ScopeInts=ScopeBlockData+8;
struct ScopeTelemetry {
    std::uint64_t frameId,captureNanos;
    float values[ScopeFloats];
    std::int32_t discrete[ScopeInts];
};
static_assert(sizeof(ScopeTelemetry)==134496,"Bounded 132 KiB telemetry with block-controller scores.");
}
// Device/pinned allocations belong to the native worker, not model checkpoints.
// Queue on the consumer stream AFTER the complete model forward and training; no host waits.
void vabpnnLaunchNetworkScope(const VabpnnGpuFrame& frame,int inspectedTile,vabpnn::ScopeTelemetry* destination);
