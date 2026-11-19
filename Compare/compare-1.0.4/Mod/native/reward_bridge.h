#pragma once
#include "RewardPolicy.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <atomic>
#include <deque>
#include <mutex>
#include <vector>
// Worker owns CUDA objects. JNI/main thread only uses these bounded CPU receipts.
class RewardBridge {
    struct Receipt {vabpnn::RewardProposal proposal{};std::uint64_t tickBefore=0,tickAfter=0;bool executed=false;};
    std::mutex mutex;
    std::array<Receipt,vabpnn::RewardCacheSlots> receipts{};
    std::deque<vabpnn::RewardBatch> requests;
    std::atomic<std::uint64_t> epoch{0};
    std::atomic<bool> enabled{false},changed{false};
    std::uint64_t lastQueued=0,hostDropped=0;
    float exploration=0;bool allowSubThreshold=false;
    VabpnnRewardPolicy* model=nullptr;
    vabpnn::RewardProposal* pendingProposal=nullptr;
    vabpnn::RewardStatus* pendingStatus=nullptr;
    vabpnn::RewardStatus latest{};
    static void check(cudaError_t code) {
        if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
    }
public:
    void initialize() {
        model=vabpnnCreateRewardPolicy(nullptr);
        check(cudaMallocHost(reinterpret_cast<void**>(&pendingProposal),sizeof(*pendingProposal)));
        check(cudaMallocHost(reinterpret_cast<void**>(&pendingStatus),sizeof(*pendingStatus)));
        *pendingProposal={};*pendingStatus={};
    }
    void close() noexcept {
        vabpnnDestroyRewardPolicy(model);model=nullptr;
        if(pendingProposal)cudaFreeHost(pendingProposal);pendingProposal=nullptr;
        if(pendingStatus)cudaFreeHost(pendingStatus);pendingStatus=nullptr;
    }
    std::uint64_t bytes() const noexcept{return vabpnnRewardPolicyBytes(model);}
    vabpnn::ModelPart checkpoint() {return vabpnnRewardCheckpointPart(model);}
    void configure(std::uint64_t nextEpoch,bool active,float probability=0,bool bootstrap=false) {
        if(!std::isfinite(probability) || probability<0 || probability>1)
            throw std::invalid_argument("Exploration probability must be finite and in 0..1");
        if(!active){probability=0;bootstrap=false;}
        if(!nextEpoch)throw std::invalid_argument("Positive self-training epoch required");
        std::lock_guard<std::mutex> lock(mutex);
        if(nextEpoch<epoch.load())throw std::invalid_argument("Self-training epoch moved backwards");
        if(nextEpoch==epoch.load()) {
            if(active==enabled.load() && probability==exploration && bootstrap==allowSubThreshold)return;
            throw std::invalid_argument("Exploration changes require a new mode epoch");
        }
        exploration=probability;allowSubThreshold=bootstrap;
        epoch.store(nextEpoch);enabled.store(active);changed.store(true);
        receipts={};requests.clear();lastQueued=0;latest.enabled=active;latest.epoch=nextEpoch;
    }
    bool active() const noexcept{return enabled.load();}
    void service(VabpnnDecisionModel* decision,cudaStream_t stream) {
        if(!model)return;
        bool configureNow;std::uint64_t modeEpoch;bool activeNow,bootstrapNow;float probabilityNow;
        std::deque<vabpnn::RewardBatch> pending;
        {std::lock_guard<std::mutex> lock(mutex);
            configureNow=changed.exchange(false);modeEpoch=epoch.load();activeNow=enabled.load();
            probabilityNow=exploration;bootstrapNow=allowSubThreshold;
            pending.swap(requests);}
        if(configureNow)vabpnnConfigureRewardPolicy(model,modeEpoch,activeNow,stream,probabilityNow,bootstrapNow);
        for(const auto& request:pending) {
            if(!enabled.load() || request.epoch!=epoch.load())continue;
            vabpnnApplyRewardBatch(model,vabpnnMutableDecisionParameters(decision,stream),request,stream);
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
    void enqueue(VabpnnDecisionModel* decision,VabpnnJepaModel* jepa,bool inputActive,cudaStream_t stream) {
        const auto view=vabpnnRewardPolicyView(model);
        vabpnnRefreshRewardStatus(model,stream);
        if(enabled.load() && inputActive) {
            const auto policy=vabpnnDecisionModelView(decision);
            vabpnnProposeRewardAction(model,policy.parameters,policy.constraints,
                const_cast<vabpnn::DecisionOutput*>(policy.output),vabpnnJepaModelView(jepa).output,
                vabpnnMutableMotionProposal(decision,stream),stream);
        } else check(cudaMemsetAsync(const_cast<vabpnn::RewardProposal*>(view.proposal),0,sizeof(vabpnn::RewardProposal),stream));
        check(cudaMemcpyAsync(pendingProposal,view.proposal,sizeof(*pendingProposal),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(pendingStatus,view.status,sizeof(*pendingStatus),cudaMemcpyDeviceToHost,stream));
    }
    void complete(std::uint64_t frame,bool currentInputEpoch) {
        std::lock_guard<std::mutex> lock(mutex);latest=*pendingStatus;
        if(!currentInputEpoch || !enabled.load())return;
        const auto& proposal=*pendingProposal;
        if(proposal.valid && proposal.frameId==frame && proposal.epoch==epoch.load()
                && proposal.policyVersion==latest.policyVersion)
            receipts[frame%vabpnn::RewardCacheSlots]={proposal,0,0,false};
    }
    vabpnn::RewardProposal proposal(std::uint64_t frame) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto value=receipts[frame%vabpnn::RewardCacheSlots].proposal;
        return enabled.load() && value.valid && value.frameId==frame && value.epoch==epoch.load() ? value:vabpnn::RewardProposal{};
    }
    bool acknowledge(std::uint64_t modeEpoch,std::uint64_t frame,std::uint64_t captured,unsigned mask,
                      std::uint64_t before,std::uint64_t after,std::uint64_t now,std::uint64_t version) {
        std::lock_guard<std::mutex> lock(mutex);
        auto& receipt=receipts[frame%vabpnn::RewardCacheSlots];const auto& p=receipt.proposal;
        bool valid=enabled.load() && modeEpoch==epoch.load() && p.valid && p.epoch==modeEpoch
            && p.frameId==frame && p.captureNanos==captured && p.mask==mask && !receipt.executed
            && version==p.policyVersion && version==latest.policyVersion
            && before<=after && after-before<=1 && now>=captured && now-captured<=500'000'000ull;
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
