#pragma once
#include "GpuFrame.h"
#include "ModelSnapshot.h"
#include "NeuronModel.h"
#include <cmath>
struct VabpnnMemeModel;
namespace vabpnn {
constexpr int MemeBlocks=120*64, MemeChannels=VoteCount, MemeContext=8, MemeOutputs=4;
constexpr int MemeWeights=MemeChannels*MemeOutputs, MemePacketSlots=2;
// Preserve the learned pre-v2 projection scale when extending its action channels.
constexpr float MemeInputScale=0.15f/13.0f;
constexpr float MemeMaximumMeanShift=1.5f*MemeInputScale*MemeChannels;
struct MemeConfig {
    int enabled=1, minimumSupport=16, maximumHops=8, reserved=0;
    float contextSigma=0.15f, minimumCompatibility=0.35f, minimumConfidence=0.1f;
    float foreignMix=0.2f, packetLifetimeSeconds=2.0f, packetDecaySeconds=0.75f;
};
// A prototype is learned knowledge. A dispatched packet has sequence>0 and finite age.
// Only an origin with a new matching visual observation may dispatch a fresh sequence.
// Forwarding preserves origin/revision/sequence/support and never increases confidence.
template<int Channels> struct MemePacketT {
    float evidence[Channels]{},context[MemeContext]{};
    float utility=0,agreement=0,coverage=0,confidence=0,age=0;
    std::uint64_t support=0,revision=0,sequence=0;
    std::uint32_t origin=0,hops=0;
};
using MemePacket=MemePacketT<MemeChannels>;
static_assert(sizeof(MemePacketT<13>)==136 && sizeof(MemePacket)==144,"Bounded context/evidence packet with legacy migration.");
template<int Channels> struct MemePolicyT {
    float weights[(Channels*MemeOutputs)]{},first[(Channels*MemeOutputs)]{},second[(Channels*MemeOutputs)]{},bestWeights[(Channels*MemeOutputs)]{};
    std::uint64_t optimizerSteps=0;
};
using MemePolicy=MemePolicyT<MemeChannels>;
struct MemeCounters {
    std::uint64_t observedFrames=0,supervisedFrames=0,projectionUpdates=0,propagations=0;
    std::uint64_t receipts=0,longSelections=0,dispatches=0;
    std::uint32_t enabled=1,localActive=0,importActive=0,invalidConfig=0;
};
static_assert(sizeof(MemeCounters)==72,"Fixed completed-worker status.");
template<int Channels> struct MemePersistentT {
    MemeConfig config{};
    MemeCounters counters{};
    MemePacketT<Channels> prototypes[MemeBlocks]{};
    MemePolicyT<Channels> policy[MemeBlocks]{};
};
using MemePersistent=MemePersistentT<MemeChannels>;
static_assert(sizeof(MemePersistentT<13>)==7495792 && sizeof(MemePersistent)==8540272,"Versioned persistent meme layout.");
// Last accepted hop for the imported packet. Viewer memory is separate from
// persistent knowledge. Matching packet identities prevent stale arrows.
struct MemeRelayTrace {
    std::uint64_t frame=0,captureNanos=0,revision=0,sequence=0;
    std::uint32_t origin=0,sourceTile=0,sourceSlot=0,channel=0; // 0 unknown, 1 grid, 2 CSR
    float sourceWeight=0;
    std::uint32_t reserved=0;
};
static_assert(sizeof(MemeRelayTrace)==56,"Observation-only last-hop record.");
__host__ __device__ inline bool memeTraceMatches(const MemeRelayTrace& trace,const MemePacket& packet) {
    return trace.channel!=0 && packet.support!=0 && trace.origin==packet.origin
        && trace.revision==packet.revision && trace.sequence==packet.sequence;
}
struct MemeAction {
    float feature[MemeChannels]{},gradient[MemeWeights]{};
    std::uint32_t batch=0,active=0;
};
struct MemeActorCredit {
    std::uint64_t frame=0;
    float advantage=0,contribution=0,agreement=0,coverage=0;
    float noise[MemeOutputs]{},exploration=0.05f;
    std::uint32_t valid=0,exploring=0;
};
struct MemeObservation { float evidence[MemeChannels]{},context[MemeContext]{}; };
struct DeviceMemeView {
    MemePersistent* persistent=nullptr;
    MemePacket* packets=nullptr; // [tile][own dispatch, imported dispatch], current bank
    float* features=nullptr; // [tile][15], already multiplied by MemeInputScale
    MemeAction* actions=nullptr; // Actual feature used when a block action was generated
    MemeActorCredit* credit=nullptr;
    const MemeObservation* observations=nullptr;
    const float* contexts=nullptr;
    std::uint32_t* routes=nullptr; // Two distinct actual long-CSR source tiles, or NoNeuron
};
__host__ __device__ inline float memeCompatibility(const float* a,const float* b,const MemeConfig& c) {
    float distance=0;
    for(int i=0;i<MemeContext;++i) {
        if(!isfinite(a[i]) || !isfinite(b[i])) return 0;
        const float difference=a[i]-b[i]; distance+=difference*difference/MemeContext;
    }
    return expf(-distance/(c.contextSigma*c.contextSigma));
}
__host__ __device__ inline float memePacketWeight(const MemePacket& packet,const float* context,
                                                  const MemeConfig& config) {
    if(packet.support<static_cast<unsigned>(config.minimumSupport) || packet.sequence==0
        || packet.origin>=MemeBlocks || packet.hops>static_cast<unsigned>(config.maximumHops)
        || !isfinite(packet.age) || packet.age<0 || packet.age>config.packetLifetimeSeconds
        || !isfinite(packet.confidence) || packet.confidence<config.minimumConfidence) return 0;
    if(!isfinite(packet.utility)) return 0;
    for(int k=0;k<MemeChannels;++k) if(!isfinite(packet.evidence[k])) return 0;
    const float match=memeCompatibility(packet.context,context,config);
    if(match<config.minimumCompatibility) return 0;
    return fminf(1.0f,packet.confidence)*match*expf(-packet.age/config.packetDecaySeconds)
        *powf(0.9f,static_cast<float>(packet.hops));
}
}
VabpnnMemeModel* vabpnnCreateMemeModel(cudaStream_t stream);
void vabpnnDestroyMemeModel(VabpnnMemeModel* model) noexcept;
std::uint64_t vabpnnMemeModelBytes(const VabpnnMemeModel* model) noexcept;
vabpnn::DeviceMemeView vabpnnMemeModelView(const VabpnnMemeModel* model);
vabpnn::MemePersistent* vabpnnMutableMemePersistent(VabpnnMemeModel* model,cudaStream_t stream);
void vabpnnConfigureMemeModel(VabpnnMemeModel* model,const vabpnn::MemeConfig& config,cudaStream_t stream);
void vabpnnConfigureMemeEnabled(VabpnnMemeModel* model,int enabled,cudaStream_t stream);
void vabpnnPrepareMemes(const VabpnnGpuFrame& frame,bool generateControl);
void vabpnnObserveMemes(const VabpnnGpuFrame& frame);
void vabpnnLearnMemes(const VabpnnGpuFrame& frame,const int* valid,const std::uint32_t* label);
void vabpnnResetMemeHistory(VabpnnMemeModel* model,cudaStream_t stream);
void vabpnnResumeMemeModel(VabpnnMemeModel* model,cudaStream_t stream);
vabpnn::ModelPart vabpnnMemeCheckpointPart(VabpnnMemeModel* model);


