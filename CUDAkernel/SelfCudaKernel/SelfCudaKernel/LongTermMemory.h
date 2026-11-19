#pragma once
#include "GpuFrame.h"
#include "NeuronModel.h"
#include "ModelSnapshot.h"
#include "TemporalDecision.h"
#include <cstddef>

struct VabpnnLongMemory;
namespace vabpnn {
struct DecisionOutput;
constexpr int LongMemoryCapacity=8192,LongMemoryDimension=64,LongMemoryTopK=8;
constexpr int LongMemoryGateFeatures=8,LongMemoryKChoices=VoteCount+1;
struct LongMemoryParameters {
    float read[8],write[8],retain[8];
    float evidenceGain[VoteCount],kGain[LongMemoryKChoices];
};
constexpr int LongMemoryParameterCount=sizeof(LongMemoryParameters)/sizeof(float);
static_assert(LongMemoryParameterCount==55,"Small external read/write/retention gates and neutral fusion gains.");
struct LongMemoryPersistent {
    LongMemoryParameters parameters,first,second;
    std::uint64_t steps;
};
static_assert(sizeof(LongMemoryPersistent)==672,"Append-only memory gate weights and AdamW.");
// This catalog/vector index is independent of the neural checkpoint.
struct LongMemoryCatalog {
    std::uint64_t revision,queries,writes,hits,supervised,ordinal;
    double observedSeconds;
    std::uint32_t used,evictionCursor;
};
struct LongMemoryEntry {
    float key[LongMemoryDimension],action[VoteCount],kMass[LongMemoryKChoices];
    std::uint64_t worldHi,worldLo,sessionHi,sessionLo,ordinal,wallMillis,lastSeenMillis;
    double observedSeconds;
    float seconds,minutes,quality;
    std::uint32_t labels,visits,valid;
};
static_assert(sizeof(LongMemoryEntry)==472,"Version-one independent vector record.");
struct LongMemoryStatus {
    std::uint64_t frame,queries,writes,hits,supervised,steps,revision;
    std::uint32_t used,capacity,retrieved,enabled,valid;
    float seconds,minutes,ageSeconds,ageMinutes,similarity,readGate,writeGate,retainGate,strength;
};
struct LongMemoryOutput {
    float evidence[VoteCount],kLogits[LongMemoryKChoices];
    std::uint32_t active,valid;
    LongMemoryStatus status;
};
// All cached values are bounded state vectors, never images or frame-pool pointers.
struct LongMemoryTrace {
    std::uint64_t frame,captureNanos,ordinal;
    LongMemoryContext context;
    double observedSeconds;
    float key[LongMemoryDimension],seconds,minutes;
    float readFeatures[8],writeFeatures[8],retainFeatures[8];
    float memoryEvidence[VoteCount],memoryK[LongMemoryKChoices];
    float basePosterior[VoteCount],posterior[VoteCount],kProbability[LongMemoryKChoices];
    float softDerivative[VoteCount];
    float readGate,writeGate,retainGate,novelty,support,maximumSimilarity;
    int selected[LongMemoryTopK],count,writtenSlot,nearestSlot,ready,frozen,consumed;
};
struct DeviceLongMemoryView {
    LongMemoryPersistent* persistent;
    LongMemoryCatalog* catalog;
    LongMemoryEntry* entries;
    const LongMemoryOutput* output;
    LongMemoryTrace* trace;
    const float* gradients;
};
}
VabpnnLongMemory* vabpnnCreateLongMemory(cudaStream_t stream);
void vabpnnDestroyLongMemory(VabpnnLongMemory* memory) noexcept;
std::uint64_t vabpnnLongMemoryBytes(const VabpnnLongMemory* memory) noexcept;
bool vabpnnLongMemoryEnabled(const VabpnnLongMemory* memory) noexcept;
vabpnn::DeviceLongMemoryView vabpnnLongMemoryView(VabpnnLongMemory* memory);
vabpnn::ModelPart vabpnnLongMemoryCheckpointPart(VabpnnLongMemory* memory);
std::size_t vabpnnLongMemoryStoreParts(VabpnnLongMemory* memory,vabpnn::ModelPart* parts);
void vabpnnConfigureLongMemory(VabpnnLongMemory* memory,bool enabled,cudaStream_t stream);
void vabpnnResetLongMemoryTransient(VabpnnLongMemory* memory,cudaStream_t stream);
void vabpnnRecallLongMemory(const VabpnnGpuFrame& frame,const vabpnn::DecisionOutput* decision,
    const float* targetLatents,VabpnnLongMemory* memory);
void vabpnnRecordLongMemory(const VabpnnGpuFrame& frame,const vabpnn::DecisionOutput* decision,VabpnnLongMemory* memory);
// The parameters argument is a device DecisionParameters pointer; no ownership transfer.
void vabpnnTrainLongMemory(VabpnnLongMemory* memory,const vabpnn::DecisionParameters* parameters,
    const std::uint64_t* previousFrame,const int* active,const std::uint32_t* label,
    const vabpnn::LongMemoryContext& currentContext,cudaStream_t stream);
