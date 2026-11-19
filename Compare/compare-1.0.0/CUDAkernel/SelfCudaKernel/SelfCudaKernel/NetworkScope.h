#pragma once
#include "GpuFrame.h"
#include "SpikeConnections.h"
#include "MemeState.h"
namespace vabpnn {
// Explicit flat layout shared with ScopeSnapshot.java. Observation telemetry only, never RGB.
constexpr int ScopeEdges=512, ScopeMetrics=8;
constexpr int ScopeMapMem=ScopeMetrics, ScopeMapPositive=ScopeMapMem+SpikeTiles;
constexpr int ScopeMapNegative=ScopeMapPositive+SpikeTiles, ScopeLocalMem=ScopeMapNegative+SpikeTiles;
constexpr int ScopePosterior=ScopeLocalMem+SpikeTileSize, ScopeThreshold=ScopePosterior+VoteCount;
constexpr int ScopePrior=ScopeThreshold+VoteCount, ScopeBlockScore=ScopePrior+VoteCount;
constexpr int ScopeBlockSelected=ScopeBlockScore+BlockCount, ScopeMemeMap=ScopeBlockSelected+8;
constexpr int ScopeMemeEvidence=ScopeMemeMap+BlockCount, ScopeMemeSelected=ScopeMemeEvidence+VoteCount, ScopeFloats=ScopeMemeSelected+8;
constexpr int ScopeLocalSpike=8, ScopeOutgoing=ScopeLocalSpike+SpikeTileSize;
constexpr int ScopeAnchors=ScopeOutgoing+LocalTileEdges, ScopeEdgePairs=ScopeAnchors+SpikeAnchorLimit;
constexpr int ScopeVotes=ScopeEdgePairs+ScopeEdges*2, ScopeBlockData=ScopeVotes+VoteCount, ScopeMemeData=ScopeBlockData+8, ScopeInts=ScopeMemeData+8;
struct ScopeTelemetry {
    std::uint64_t frameId,captureNanos;
    float values[ScopeFloats];
    std::int32_t discrete[ScopeInts];
};
static_assert(sizeof(ScopeTelemetry)==165376,"Bounded 162 KiB telemetry with controller scores and meme activity.");
}
// Device/pinned allocations belong to the native worker, not model checkpoints.
// Queue on the consumer stream AFTER the complete model forward and training; no host waits.
void vabpnnLaunchNetworkScope(const VabpnnGpuFrame& frame,int inspectedTile,vabpnn::ScopeTelemetry* destination);



// Independent 5 Hz viewer. SoA overview for every block and three full selected
// packets; no pixels, weights, optimizer buffers or checkpoint parts are read back.
namespace vabpnn {
constexpr int MemeScopeMeta=40;
constexpr int MemeScopeOwn=0,MemeScopeImported=MemeBlocks,MemeScopeAge=2*MemeBlocks;
constexpr int MemeScopeConfidence=3*MemeBlocks,MemeScopeKnowledge=4*MemeBlocks;
constexpr int MemeScopeCompatibility=5*MemeBlocks,MemeScopeDetail=6*MemeBlocks;
constexpr int MemeScopePacketFloats=MemeChannels+MemeContext+5;
constexpr int MemeScopeVisual=MemeScopeDetail+3*MemeScopePacketFloats;
constexpr int MemeScopeFeatures=MemeScopeVisual+MemeContext;
constexpr int MemeScopeAction=MemeScopeFeatures+MemeChannels;
constexpr int MemeScopeConfig=MemeScopeAction+MemeChannels,MemeScopeFloats=MemeScopeConfig+10;
constexpr int MemeScopeOrigin=0,MemeScopeHops=MemeBlocks,MemeScopeSender=2*MemeBlocks;
constexpr int MemeScopeChannel=3*MemeBlocks,MemeScopeInts=4*MemeBlocks,MemeScopeLongs=2*MemeBlocks;
struct MemeScopeTelemetry {
    std::uint64_t meta[MemeScopeMeta];
    float values[MemeScopeFloats];
    std::int32_t discrete[MemeScopeInts];
    std::uint64_t identities[MemeScopeLongs];
};
static_assert(sizeof(MemeScopeTelemetry)==430928,"Bounded coherent 5 Hz meme snapshot.");
}
void vabpnnLaunchMemeScope(const VabpnnGpuFrame& frame,int inspectedTile,vabpnn::MemeScopeTelemetry* destination);
