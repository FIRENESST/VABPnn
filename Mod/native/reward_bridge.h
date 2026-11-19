#pragma once
#include "RewardPolicy.h"
#include "FullSelfLearning.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <atomic>
#include <deque>
#include <mutex>
#include <vector>
// Worker owns CUDA objects. JNI/main thread only uses these bounded CPU receipts.
class RewardBridge {
    struct Receipt {vabpnn::RewardProposal proposal{};std::uint64_t tickBefore=0,tickAfter=0;bool executed=false,full=false;};
    std::mutex mutex;
    std::array<Receipt,vabpnn::RewardCacheSlots> receipts{};
    std::deque<vabpnn::RewardBatch> requests;
    std::atomic<std::uint64_t> epoch{0};
    std::atomic<bool> enabled{false},changed{false},fullScope{false};
    std::uint64_t lastQueued=0,hostDropped=0;
    float exploration=0;bool allowSubThreshold=false;
    VabpnnRewardPolicy* model=nullptr;
    VabpnnFullLearning* fullModel=nullptr;
    vabpnn::FullLearningStatus *pendingFull=nullptr,latestFull{};
    vabpnn::RewardProposal* pendingProposal=nullptr;
    vabpnn::RewardStatus* pendingStatus=nullptr;
    vabpnn::RewardStatus latest{};
    static void check(cudaError_t code) {
        if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
    }
public:
    void initialize() {
        model=vabpnnCreateRewardPolicy(nullptr);fullModel=vabpnnCreateFullLearning(nullptr);
        check(cudaMallocHost(reinterpret_cast<void**>(&pendingFull),sizeof(*pendingFull)));*pendingFull={};
        check(cudaMallocHost(reinterpret_cast<void**>(&pendingProposal),sizeof(*pendingProposal)));
        check(cudaMallocHost(reinterpret_cast<void**>(&pendingStatus),sizeof(*pendingStatus)));
        *pendingProposal={};*pendingStatus={};
    }
    void close() noexcept {
        vabpnnDestroyRewardPolicy(model);model=nullptr;vabpnnDestroyFullLearning(fullModel);fullModel=nullptr;
        if(pendingFull)cudaFreeHost(pendingFull);pendingFull=nullptr;
        if(pendingProposal)cudaFreeHost(pendingProposal);pendingProposal=nullptr;
        if(pendingStatus)cudaFreeHost(pendingStatus);pendingStatus=nullptr;
    }
    std::uint64_t bytes() const noexcept{return vabpnnRewardPolicyBytes(model)+vabpnnFullLearningBytes(fullModel);}
    vabpnn::ModelPart checkpoint() {return vabpnnRewardCheckpointPart(model);}
    vabpnn::ModelPart fullCheckpoint(){return vabpnnFullLearningCheckpointPart(fullModel);}
    VabpnnFullLearning* fullLearningModel(){return fullModel;}
    bool allNetworks() const noexcept{return enabled.load()&&fullScope.load();}
    vabpnn::FullLearningStatus fullStatus(){std::lock_guard<std::mutex> lock(mutex);return latestFull;}
    void configure(std::uint64_t nextEpoch,bool active,float probability=0,bool bootstrap=false,bool all=false) {
        if(!std::isfinite(probability) || probability<0 || probability>1)
            throw std::invalid_argument("Exploration probability must be finite and in 0..1");
        if(!active){probability=0;bootstrap=false;all=false;}
        if(!nextEpoch)throw std::invalid_argument("Positive self-training epoch required");
        std::lock_guard<std::mutex> lock(mutex);
        if(nextEpoch<epoch.load())throw std::invalid_argument("Self-training epoch moved backwards");
        if(nextEpoch==epoch.load()) {
            if(active==enabled.load() && probability==exploration && bootstrap==allowSubThreshold && all==fullScope.load())return;
            throw std::invalid_argument("Exploration changes require a new mode epoch");
        }
        exploration=probability;allowSubThreshold=bootstrap;
        epoch.store(nextEpoch);enabled.store(active);fullScope.store(all);changed.store(true);
        receipts={};requests.clear();lastQueued=0;latest.enabled=active;latest.epoch=nextEpoch;
    }
    bool active() const noexcept{return enabled.load();}
    void resume(cudaStream_t stream) {
        // One-time restore publication; disabled frames otherwise have zero added GPU work.
        vabpnnRefreshRewardStatus(model,stream);
        const auto view=vabpnnRewardPolicyView(model);
        check(cudaMemcpyAsync(pendingStatus,view.status,sizeof(*pendingStatus),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        vabpnnReadFullLearningStatus(fullModel,pendingFull,stream);check(cudaStreamSynchronize(stream));
        std::lock_guard<std::mutex> lock(mutex);latest=*pendingStatus;latestFull=*pendingFull;
    }
    void service(const VabpnnGpuFrame& owners,cudaStream_t stream) {
        auto* decision=owners.decisionModel;
        if(!model)return;
        bool configureNow;std::uint64_t modeEpoch;bool activeNow,bootstrapNow,allNow;float probabilityNow;
        std::deque<vabpnn::RewardBatch> pending;
        {std::lock_guard<std::mutex> lock(mutex);
            configureNow=changed.exchange(false);modeEpoch=epoch.load();activeNow=enabled.load();
            probabilityNow=exploration;bootstrapNow=allowSubThreshold;allNow=fullScope.load();
            pending.swap(requests);}
        if(configureNow){vabpnnConfigureRewardPolicy(model,modeEpoch,activeNow,stream,probabilityNow,bootstrapNow);if(allNow&&activeNow)vabpnnEnableFullLearning(fullModel,stream,owners.pnnJepa);vabpnnResetFullLearning(fullModel,stream);}
        for(const auto& request:pending) {
            if(!enabled.load() || request.epoch!=epoch.load())continue;
            vabpnnApplyRewardBatch(model,vabpnnMutableDecisionParameters(decision,stream),request,stream,allNow?&owners:nullptr);
        }
        if(configureNow || !pending.empty()) {
            const auto view=vabpnnRewardPolicyView(model);
            check(cudaMemcpyAsync(pendingStatus,view.status,sizeof(*pendingStatus),cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream)); // Worker only; no image/large tensor readback.
            std::lock_guard<std::mutex> lock(mutex);
            const auto oldVersion=latest.policyVersion;latest=*pendingStatus;
            if(latest.policyVersion!=oldVersion)receipts={}; // Old-policy executions never train a new policy.
        }
    }
    void enqueue(const VabpnnGpuFrame& frame,bool inputActive,cudaStream_t stream) {
        auto* decision=frame.decisionModel;auto* jepa=frame.jepaModel;
        // Disabled default/F8/predict paths add no GPU work or extra readback per frame.
        if(!enabled.load()) {
            std::lock_guard<std::mutex> lock(mutex);*pendingProposal={};*pendingStatus=latest;return;
        }
        const auto view=vabpnnRewardPolicyView(model);
        vabpnnRefreshRewardStatus(model,stream);
        if(enabled.load() && inputActive) {
            const auto policy=vabpnnDecisionModelView(decision);
            vabpnnProposeRewardAction(model,policy.parameters,policy.constraints,
                const_cast<vabpnn::DecisionOutput*>(policy.output),vabpnnJepaModelView(jepa).output,
                vabpnnMutableMotionProposal(decision,stream),stream);
            if(frame.selfLearning){
                auto* slot=view.cache+frame.frameId%vabpnn::RewardCacheSlots;
                vabpnnCacheFullReward(frame,fullModel,slot);
                check(cudaMemcpyAsync(const_cast<vabpnn::RewardProposal*>(view.proposal),&slot->proposal,sizeof(vabpnn::RewardProposal),cudaMemcpyDeviceToDevice,stream));
            }
        } else check(cudaMemsetAsync(const_cast<vabpnn::RewardProposal*>(view.proposal),0,sizeof(vabpnn::RewardProposal),stream));
        vabpnnReadFullLearningStatus(fullModel,pendingFull,stream);
        check(cudaMemcpyAsync(pendingProposal,view.proposal,sizeof(*pendingProposal),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(pendingStatus,view.status,sizeof(*pendingStatus),cudaMemcpyDeviceToHost,stream));
    }
    void complete(std::uint64_t frame,bool currentInputEpoch) {
        std::lock_guard<std::mutex> lock(mutex);latest=*pendingStatus;latestFull=*pendingFull;
        if(!currentInputEpoch || !enabled.load())return;
        const auto& proposal=*pendingProposal;
        if(proposal.valid && proposal.frameId==frame && proposal.epoch==epoch.load()
                && proposal.policyVersion==latest.policyVersion)
            receipts[frame%vabpnn::RewardCacheSlots]={proposal,0,0,false,fullScope.load()};
    }
    vabpnn::RewardProposal proposal(std::uint64_t frame) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto value=receipts[frame%vabpnn::RewardCacheSlots].proposal;
        return enabled.load() && value.valid && value.frameId==frame && value.epoch==epoch.load() ? value:vabpnn::RewardProposal{};
    }
    bool acknowledge(std::uint64_t modeEpoch,std::uint64_t frame,std::uint64_t captured,unsigned mask,
                      std::uint64_t before,std::uint64_t after,std::uint64_t now,std::uint64_t version,bool actualMotion=false,float dx=0,float dy=0,float wheel=0) {
        std::lock_guard<std::mutex> lock(mutex);
        auto& receipt=receipts[frame%vabpnn::RewardCacheSlots];const auto& p=receipt.proposal;
        bool valid=enabled.load() && modeEpoch==epoch.load() && p.valid && p.epoch==modeEpoch
            && p.frameId==frame && p.captureNanos==captured && p.mask==mask && !receipt.executed
            && version==p.policyVersion && version==latest.policyVersion
            && before<=after && after-before<=1 && now>=captured && now-captured<=500'000'000ull;
        if(receipt.full){auto same=[](float a,float b){return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=std::max(1e-4f,1e-5f*std::abs(a));};
            valid=valid&&actualMotion&&same(p.expectedDx,dx)&&same(p.expectedDy,dy)&&same(p.expectedWheel,wheel);}
        if(!valid){++hostDropped;return false;}
        receipt.tickBefore=before;receipt.tickAfter=after;receipt.executed=true;return true;
    }
    bool submit(std::uint64_t modeEpoch,std::uint64_t rewardId,std::uint64_t start,std::uint64_t end,
                float credit,const std::vector<std::uint64_t>& frames) {
        std::lock_guard<std::mutex> lock(mutex);
        if(!enabled.load() || modeEpoch!=epoch.load() || !rewardId || rewardId<=lastQueued || start>=end
                || !std::isfinite(credit) || std::abs(credit)>1 || frames.empty()
                || frames.size()>vabpnn::RewardCacheSlots || requests.size()>=8){++hostDropped;return false;}
        vabpnn::RewardBatch batch{};batch.epoch=modeEpoch;batch.rewardId=rewardId;batch.credit=credit;batch.count=static_cast<unsigned>(frames.size());
        for(unsigned i=0;i<batch.count;++i) {
            const auto frame=frames[i];const auto& receipt=receipts[frame%vabpnn::RewardCacheSlots];const auto& p=receipt.proposal;
            if(!frame || (i && frame<=frames[i-1]) || !p.valid || p.frameId!=frame || p.epoch!=modeEpoch || !receipt.executed
                    || receipt.tickBefore<=start || receipt.tickAfter>end || p.policyVersion!=latest.policyVersion){
                ++hostDropped;return false;
            }
            batch.frames[i]=frame;batch.masks[i]=p.mask;
        }
        for(const auto frame:frames)receipts[frame%vabpnn::RewardCacheSlots].executed=false;
        lastQueued=rewardId;requests.push_back(batch);return true;
    }
    vabpnn::RewardStatus status() {
        std::lock_guard<std::mutex> lock(mutex);auto out=latest;
        out.epoch=epoch.load();out.enabled=enabled.load();out.droppedRewards+=hostDropped;return out;
    }
};
