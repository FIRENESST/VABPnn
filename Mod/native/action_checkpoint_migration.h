#pragma once
#include "model_checkpoint.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "MemeState.h"
#include <memory>

namespace vabpnn {
// The released layout has 15 actions. These types remain the exact 13-action disk ABI.
constexpr int LegacyActions=13;
using LegacyDecision=DecisionParametersT<LegacyActions>;
using LegacyJepa=JepaParametersT<LegacyActions>;
using LegacyOnline=OnlinePersistentT<LegacyActions>;
using LegacyPosition=DecisionPositionAdapterT<LegacyActions>;
using LegacyMeme=MemePersistentT<LegacyActions>;
static_assert(sizeof(LegacyOnline)==31496 && sizeof(LegacyJepa)==2040840,"Frozen 13-action optimizer layouts.");

inline void copyCommon(float& to,const float& from) { to=from; }
template<std::size_t To,std::size_t From> inline void copyCommon(float (&to)[To],const float (&from)[From]) {
    std::copy_n(from,std::min(To,From),to);
}
template<std::size_t RowsTo,std::size_t ColsTo,std::size_t RowsFrom,std::size_t ColsFrom>
inline void copyCommon(float (&to)[RowsTo][ColsTo],const float (&from)[RowsFrom][ColsFrom]) {
    for(std::size_t i=0;i<std::min(RowsTo,RowsFrom);++i) copyCommon(to[i],from[i]);
}
inline void expandDecision(DecisionParameters& to,const LegacyDecision& from) {
#define VABPNN_COPY(field) copyCommon(to.field,from.field)
    VABPNN_COPY(nodeEmbedding);VABPNN_COPY(voteEmbedding);VABPNN_COPY(timeEmbedding);
    VABPNN_COPY(query);VABPNN_COPY(key);VABPNN_COPY(value);VABPNN_COPY(attentionOutput);
    for(int h=0;h<DecisionHeads;++h) copyCommon(to.graphBias[h],from.graphBias[h]);
    VABPNN_COPY(logAttentionDecay);VABPNN_COPY(logClockFrequency);
    VABPNN_COPY(attentionGain);VABPNN_COPY(attentionBias);
    VABPNN_COPY(feedForwardUp);VABPNN_COPY(feedForwardUpBias);
    VABPNN_COPY(feedForwardDown);VABPNN_COPY(feedForwardDownBias);
    VABPNN_COPY(outputGain);VABPNN_COPY(outputBias);
    VABPNN_COPY(logLikelihoodWeight);VABPNN_COPY(logLikelihoodBias);VABPNN_COPY(rawVoteWeight);
    VABPNN_COPY(thresholdWeight);VABPNN_COPY(thresholdBias);
    VABPNN_COPY(kBias);VABPNN_COPY(initialPriorLogit);VABPNN_COPY(logOnRate);VABPNN_COPY(logOffRate);
    VABPNN_COPY(logGateTemperature);
#undef VABPNN_COPY
    for(int k=0;k<=LegacyActions;++k) {
        std::copy_n(from.kWeight[k],DecisionWidth+LegacyActions,to.kWeight[k]);
        std::copy_n(from.kWeight[k]+DecisionWidth+LegacyActions,LegacyActions,to.kWeight[k]+DecisionWidth+VoteCount);
    }
}
inline void expandJepa(JepaParameters& to,const LegacyJepa& from) {
    std::memset(&to,0,sizeof(to));to.encoder=from.encoder;to.predictor=from.predictor;
    auto& motion=to.motion;const auto& old=from.motion;
    for(int i=0;i<JepaMoments*3+LegacyActions;++i) copyCommon(motion.inputWeight[i],old.inputWeight[i]);
    for(int i=0;i<2;++i) copyCommon(motion.inputWeight[JepaMoments*3+VoteCount+i],old.inputWeight[JepaMoments*3+LegacyActions+i]);
    copyCommon(motion.hiddenBias,old.hiddenBias);copyCommon(motion.outputWeight,old.outputWeight);
    copyCommon(motion.outputBias,old.outputBias);
}
inline void expandOnline(OnlinePersistent& to,const LegacyOnline& from) {
    std::memset(&to,0,sizeof(to));expandDecision(to.first,from.first);expandDecision(to.second,from.second);
    auto wheel=[](auto& target,const auto& source) {
        std::copy_n(source,DecisionWidth+LegacyActions,target);
        target[DecisionWidth+VoteCount]=source[DecisionWidth+LegacyActions];
    };
    wheel(to.wheel,from.wheel);wheel(to.wheelFirst,from.wheelFirst);wheel(to.wheelSecond,from.wheelSecond);
    to.output=from.output;
}
inline void expandPosition(DecisionPositionAdapter& to,const LegacyPosition& from) {
    std::memset(&to,0,sizeof(to));
    copyCommon(to.evidence,from.evidence);copyCommon(to.evidenceFirst,from.evidenceFirst);copyCommon(to.evidenceSecond,from.evidenceSecond);
    copyCommon(to.k,from.k);copyCommon(to.kFirst,from.kFirst);copyCommon(to.kSecond,from.kSecond);
    to.steps=from.steps;
}
inline void expandMemes(MemePersistent& to,const LegacyMeme& from) {
    std::memset(&to,0,sizeof(to));to.config=from.config;to.counters=from.counters;
    for(int tile=0;tile<MemeBlocks;++tile) {
        auto& p=to.prototypes[tile];const auto& old=from.prototypes[tile];
        copyCommon(p.evidence,old.evidence);copyCommon(p.context,old.context);
        p.utility=old.utility;p.agreement=old.agreement;p.coverage=old.coverage;
        p.confidence=old.confidence;p.age=old.age;p.support=old.support;
        p.revision=old.revision;p.sequence=old.sequence;p.origin=old.origin;p.hops=old.hops;
        auto& policy=to.policy[tile];const auto& source=from.policy[tile];
        for(int output=0;output<MemeOutputs;++output) {
            std::copy_n(source.weights+output*LegacyActions,LegacyActions,policy.weights+output*MemeChannels);
            std::copy_n(source.first+output*LegacyActions,LegacyActions,policy.first+output*MemeChannels);
            std::copy_n(source.second+output*LegacyActions,LegacyActions,policy.second+output*MemeChannels);
            std::copy_n(source.bestWeights+output*LegacyActions,LegacyActions,policy.bestWeights+output*MemeChannels);
        }
        policy.optimizerSteps=source.optimizerSteps;
    }
}
inline void checkpointCuda(cudaError_t result) {
    if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
// Validate the full original SHA before writing any GPU part. Convert at startup only:
// at most ~16 MiB of host parameter staging, never image readback or per-frame work.
class LegacyCheckpointReader {
    std::ifstream file;
    std::vector<std::uint64_t> offsets;
    std::vector<unsigned char> staging;
public:
    LegacyCheckpointReader(const std::wstring& path,const std::vector<ModelPart>& current,int version)
        :file(std::filesystem::path(path),std::ios::binary),staging(1024*1024) {
        if(current.size()!=17 || version<1 || version>5) throw std::runtime_error("Invalid legacy migration request");
        if(current[0].bytes!=ParameterCount || current[1].bytes!=sizeof(DecisionParameters)
            || current[2].bytes!=sizeof(JepaParameters) || current[7].bytes!=ParameterCount*sizeof(float)
            || current[8].bytes!=sizeof(OnlinePersistent) || current[14].bytes!=sizeof(DecisionPositionAdapter)
            || current[16].bytes!=sizeof(MemePersistent)) throw std::runtime_error("Invalid released model layout");
        auto layout=current;
        layout[0].bytes=NeuronCount*InputChannels*LegacyActions;
        layout[1].bytes=sizeof(LegacyDecision);
        for(int i:{2,4,5}) layout[i].bytes=sizeof(LegacyJepa);
        layout[7].bytes=NeuronCount*InputChannels*LegacyActions*sizeof(float);
        layout[8].bytes=sizeof(LegacyOnline);
        layout[14].bytes=sizeof(LegacyPosition);layout[16].bytes=sizeof(LegacyMeme);
        layout.resize(version==1 ? 9:version==2 ? 14:version==3 ? 15:version==4 ? 16:17);
        std::uint64_t offset=version==5 ? sizeof(CheckpointHeaderV5):sizeof(CheckpointHeader);
        if(version==5) validateModelCheckpointFormat<CheckpointHeaderV5>(file,path,layout,false,5);
        else validateModelCheckpointFormat<CheckpointHeader>(file,path,layout,version==1,version);
        for(const auto& part:layout) { offsets.push_back(offset);offset+=part.bytes; }
    }
    std::size_t count() const { return offsets.size(); }
    void read(int part,std::uint64_t offset,void* destination,std::size_t bytes) {
        file.clear();file.seekg(offsets.at(part)+offset);
        if(!file.read(static_cast<char*>(destination),bytes)) throw std::runtime_error("Legacy checkpoint read failed");
    }
    template<class T> std::unique_ptr<T> readObject(int part) {
        auto value=std::make_unique<T>();read(part,0,value.get(),sizeof(T));return value;
    }
    void copy(int part,std::uint64_t source,void* destination,std::size_t bytes) {
        for(std::size_t at=0;at<bytes;) {
            const auto size=std::min(bytes-at,staging.size());
            read(part,source+at,staging.data(),size);
            checkpointCuda(cudaMemcpy(static_cast<unsigned char*>(destination)+at,staging.data(),size,cudaMemcpyHostToDevice));
            at+=size;
        }
    }
    void planes(int part,void* destination,std::size_t element) {
        const auto plane=NeuronCount*element;
        for(int channel=0;channel<InputChannels;++channel)
            checkpointCuda(cudaMemset(static_cast<unsigned char*>(destination)+(channel*VoteCount+LegacyActions)*plane,0,(VoteCount-LegacyActions)*plane));
        for(int channel=0;channel<InputChannels;++channel) for(int k=0;k<LegacyActions;++k)
            copy(part,(channel*LegacyActions+k)*plane,static_cast<unsigned char*>(destination)+(channel*VoteCount+k)*plane,plane);
    }
};
template<class Current,class Legacy,class Convert>
inline void migrateObject(LegacyCheckpointReader& reader,int index,void* destination,Convert convert,bool keepInitial=false) {
    auto source=reader.readObject<Legacy>(index);
    auto target=std::make_unique<Current>();
    if(keepInitial) checkpointCuda(cudaMemcpy(target.get(),destination,sizeof(Current),cudaMemcpyDeviceToHost));
    convert(*target,*source);
    checkpointCuda(cudaMemcpy(destination,target.get(),sizeof(Current),cudaMemcpyHostToDevice));
}
inline bool loadLegacyActionCheckpoint(const std::wstring& path,const std::vector<ModelPart>& current,int version) {
    if(!std::filesystem::exists(path)) return false;
    LegacyCheckpointReader reader(path,current,version);
    reader.planes(0,current[0].device,1);
    migrateObject<DecisionParameters,LegacyDecision>(reader,1,current[1].device,[](auto& to,const auto& from){expandDecision(to,from);to.enabledKChoices=LegacyActions+1;},true);
    for(int i:{2,4,5}) migrateObject<JepaParameters,LegacyJepa>(reader,i,current[i].device,expandJepa);
    reader.planes(7,current[7].device,sizeof(float));
    migrateObject<OnlinePersistent,LegacyOnline>(reader,8,current[8].device,expandOnline);
    for(int i=0;i<static_cast<int>(reader.count());++i) {
        if(i==0 || i==1 || i==2 || i==4 || i==5 || i==7 || i==8) continue;
        if(i==14) migrateObject<DecisionPositionAdapter,LegacyPosition>(reader,i,current[i].device,expandPosition);
        else if(i==16) migrateObject<MemePersistent,LegacyMeme>(reader,i,current[i].device,expandMemes);
        else reader.copy(i,0,current[i].device,current[i].bytes);
    }
    return true;
}
} // namespace vabpnn
