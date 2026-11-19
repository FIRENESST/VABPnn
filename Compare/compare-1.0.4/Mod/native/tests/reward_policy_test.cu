#include "RewardPolicy.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace vabpnn;
void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class T>struct Device {
    T* p=nullptr;explicit Device(std::size_t n=1){check(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){cudaFree(p);}void set(const T& value){check(cudaMemcpy(p,&value,sizeof(value),cudaMemcpyHostToDevice));}
    T get(){T out{};check(cudaMemcpy(&out,p,sizeof(out),cudaMemcpyDeviceToHost));return out;}
};
double sigmoid(double x){return 1/(1+std::exp(-x));}
DecisionOutput evaluate(const DecisionParameters& p,unsigned long long frame=1) {
    DecisionOutput o{};o.valid=1;o.frameId=frame;o.captureNanos=frame*20'000'000+1'000'000'000;
    o.command={o.frameId,o.captureNanos,0,1,0,0,0};o.status.frameId=frame;o.status.valid=1;
    const double temperature=std::log1p(std::exp(p.logGateTemperature));
    for(int a=0;a<VoteCount;++a) {
        o.prior[a]=.4f+.03f*std::sin(a+.7f);o.normalizedVotes[a]=.2f*std::cos(.31f*(a+1));
        double x=p.logLikelihoodBias[a]+p.rawVoteWeight[a]*o.normalizedVotes[a],t=p.thresholdBias[a];
        for(int d=0;d<DecisionWidth;++d) {
            o.hidden[a][d]=.3f*std::sin(.21f*(a*DecisionWidth+d+1));
            x+=p.logLikelihoodWeight[a][d]*o.hidden[a][d];t+=p.thresholdWeight[a][d]*o.hidden[a][d];
            o.kFeatures[d]+=o.hidden[a][d]/VoteCount;
        }
        o.posterior[a]=o.confidence[a]=static_cast<float>(sigmoid(std::log(o.prior[a]/(1-o.prior[a]))+x));
        o.threshold[a]=static_cast<float>(sigmoid(t));
        o.softEligibility[a]=static_cast<float>(sigmoid((o.posterior[a]-o.threshold[a])/temperature));
        if(o.posterior[a]>o.threshold[a])o.eligibleMask|=1u<<a;
        o.kFeatures[DecisionWidth+a]=o.posterior[a];o.kFeatures[DecisionWidth+VoteCount+a]=o.softEligibility[a];
    }
    double logits[DecisionKChoices],maximum=-INFINITY,total=0;
    for(int k=0;k<DecisionKChoices;++k) {
        logits[k]=p.kBias[k];
        for(int f=0;f<DecisionKFeatures;++f)logits[k]+=p.kWeight[k][f]*o.kFeatures[f];
        maximum=std::max(maximum,logits[k]);
    }
    for(int k=0;k<DecisionKChoices;++k){logits[k]=std::exp(logits[k]-maximum);total+=logits[k];}
    for(int k=0;k<DecisionKChoices;++k)o.kProbability[k]=static_cast<float>(logits[k]/total);
    return o;
}
double logProbability(const DecisionParameters& p,const RewardCache& trace) {
    const auto o=evaluate(p);const int maximum=static_cast<int>(trace.choices);
    double logits[DecisionKChoices]{},highest=-INFINITY,total=0;
    for(int k=0;k<maximum;++k) {
        logits[k]=p.kBias[k];
        for(int f=0;f<DecisionKFeatures;++f)logits[k]+=p.kWeight[k][f]*o.kFeatures[f];
        highest=std::max(highest,logits[k]);
    }
    for(int k=0;k<maximum;++k){logits[k]=std::exp(logits[k]-highest);total+=logits[k];}
    const double mix=trace.mixture;
    double log=std::log(trace.branchProbability)+std::log(mix*logits[trace.proposal.k]/total+(1-mix)/maximum);
    unsigned remaining=trace.proposal.eligibleMask;
    const unsigned conflicts[VoteCount]={0,0,0,0,0,0,0,(1u<<8)|(1u<<11),(1u<<7)|(1u<<11),1u<<10,1u<<9,(1u<<7)|(1u<<8),0,0,0};
    for(unsigned i=0;i<trace.proposal.count;++i) {
        const int chosen=trace.order[i];double denominator=0;
        for(int a=0;a<VoteCount;++a)if(remaining&(1u<<a))denominator+=mix*o.posterior[a]+1-mix;
        log+=std::log((mix*o.posterior[chosen]+1-mix)/denominator);remaining&=~(1u<<chosen);
        for(int a=0;a<VoteCount;++a)if((conflicts[chosen]&(1u<<a)) || (conflicts[a]&(1u<<chosen)))remaining&=~(1u<<a);
    }
    return log;
}
float& parameter(DecisionParameters& p,int index) {
    if(index<RewardEvidenceParameters) {
        const int a=index/RewardEvidenceStride,d=index%RewardEvidenceStride;
        return d<DecisionWidth ? p.logLikelihoodWeight[a][d]:d==DecisionWidth ? p.logLikelihoodBias[a]:p.rawVoteWeight[a];
    }
    const int at=index-RewardEvidenceParameters,k=at/RewardKStride,f=at%RewardKStride;
    return f<DecisionKFeatures ? p.kWeight[k][f]:p.kBias[k];
}
int main()try {
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto* model=vabpnnCreateRewardPolicy(stream);const auto view=vabpnnRewardPolicyView(model);
    Device<DecisionParameters> deviceParameters;Device<DecisionConstraints> constraints;Device<DecisionOutput> decision;
    auto base=std::make_unique<DecisionParameters>();base->enabledKChoices=DecisionKChoices;
    for(int a=0;a<VoteCount;++a){base->thresholdBias[a]=a<3 ? -2:10;base->logLikelihoodBias[a]=1.2f+.03f*a;}
    for(int i=0;i<RewardPolicyParameters;++i)if(i>=RewardEvidenceParameters || i%RewardEvidenceStride<DecisionWidth)
        parameter(*base,i)=.04f*std::sin(.37f*(i+1));
    DecisionConstraints c{};c.conflictMask[7]=(1u<<8)|(1u<<11);c.conflictMask[8]=(1u<<7)|(1u<<11);
    c.conflictMask[9]=1u<<10;c.conflictMask[10]=1u<<9;c.conflictMask[11]=(1u<<7)|(1u<<8);constraints.set(c);
    deviceParameters.set(*base);vabpnnConfigureRewardPolicy(model,1,true,stream);check(cudaStreamSynchronize(stream));
    auto snapshot=evaluate(*base);decision.set(snapshot);
    vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
    RewardCache trace{};check(cudaMemcpy(&trace,view.cache+1,sizeof(trace),cudaMemcpyDeviceToHost));
    require(trace.proposal.valid && !(trace.proposal.mask&~snapshot.eligibleMask),"proposal uses strict threshold candidates");
    require(std::abs(trace.proposal.logProbability-logProbability(*base,trace))<1e-6,"actual sampled joint log probability");
    constexpr float epsilon=.005f;double maxError=0;unsigned nonzero=0;
    for(int i=0;i<RewardPolicyParameters;++i) {
        float& weight=parameter(*base,i);const float old=weight;
        weight=old+epsilon;const double plus=logProbability(*base,trace);
        weight=old-epsilon;const double minus=logProbability(*base,trace);weight=old;
        const double expected=(plus-minus)/(2*epsilon),error=std::abs(expected-trace.score[i]);
        maxError=std::max(maxError,error);if(std::abs(trace.score[i])>1e-7)++nonzero;
        require(error<2e-5+std::abs(expected)*.002,"REINFORCE score finite difference including k-to-evidence chain");
    }
    require(nonzero>150,"reward gradients actually active");
    RewardBatch batch{};batch.epoch=1;batch.rewardId=1;batch.count=1;batch.frames[0]=1;batch.masks[0]=trace.proposal.mask;batch.credit=1;
    const double before=logProbability(*base,trace);vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);check(cudaStreamSynchronize(stream));
    auto positive=std::make_unique<DecisionParameters>(deviceParameters.get());
    require(logProbability(*positive,trace)>before,"positive reward increases sampled behavior likelihood");
    RewardStatus status{};check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
    require(status.updates==1 && status.appliedRewards==1 && status.lastRewardId==1,"one aggregated reward update");
    auto allowed=std::make_unique<DecisionParameters>(*base);
    for(int i=0;i<RewardPolicyParameters;++i)parameter(*allowed,i)=parameter(*positive,i);
    require(!std::memcmp(allowed.get(),positive.get(),sizeof(*positive)),"reward updates only the approved 1022 policy parameters");
    const auto first=*positive;
    vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);check(cudaStreamSynchronize(stream));
    const auto duplicate=deviceParameters.get();require(!std::memcmp(&first,&duplicate,sizeof(first)),"duplicate reward leaves all weights unchanged");
    // Isolated negative case starts from identical weights and optimizer, never a user checkpoint.
    check(cudaMemset(view.persistent,0,sizeof(RewardPersistent)));deviceParameters.set(*base);
    vabpnnConfigureRewardPolicy(model,2,true,stream);check(cudaStreamSynchronize(stream));decision.set(snapshot);
    vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&trace,view.cache+1,sizeof(trace),cudaMemcpyDeviceToHost));
    batch.epoch=2;batch.masks[0]=trace.proposal.mask;batch.credit=-1;
    const double negativeBefore=logProbability(*base,trace);
    vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);check(cudaStreamSynchronize(stream));
    const auto negative=deviceParameters.get();require(logProbability(negative,trace)<negativeBefore,"negative reward decreases sampled behavior likelihood");
    // Freeze explicitly invalidates eligibility and rejects future rewards.
    vabpnnConfigureRewardPolicy(model,3,false,stream);batch.epoch=3;batch.rewardId=2;
    vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);check(cudaStreamSynchronize(stream));
    const auto frozen=deviceParameters.get();require(!std::memcmp(&negative,&frozen,sizeof(frozen)),"frozen/disabled reward mode updates no neural weight");
    // Zero rewards consume the valid receipt but do not move weights, moments or counters.
    check(cudaMemset(view.persistent,0,sizeof(RewardPersistent)));deviceParameters.set(*base);
    vabpnnConfigureRewardPolicy(model,4,true,stream);check(cudaStreamSynchronize(stream));decision.set(snapshot);
    vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
    auto meta=Device<RewardProposal>();RewardProposal proposal{};check(cudaMemcpy(&proposal,view.proposal,sizeof(proposal),cudaMemcpyDeviceToHost));
    batch.epoch=4;batch.rewardId=1;batch.credit=0;batch.masks[0]=proposal.mask;
    vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);check(cudaStreamSynchronize(stream));
    const auto zero=deviceParameters.get();require(!std::memcmp(base.get(),&zero,sizeof(zero)),"zero reward does not change weights");
    RewardPersistent optimizer{};check(cudaMemcpy(&optimizer,view.persistent,sizeof(optimizer),cudaMemcpyDeviceToHost));
    RewardPersistent neutral{};require(!std::memcmp(&optimizer,&neutral,sizeof(neutral)),"zero reward does not change moments/version/steps");
    // Cache actual-mask proof is mandatory, even for valid reward IDs.
    vabpnnConfigureRewardPolicy(model,5,true,stream);check(cudaStreamSynchronize(stream));decision.set(snapshot);
    vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&proposal,view.proposal,sizeof(proposal),cudaMemcpyDeviceToHost));
    batch.epoch=5;batch.rewardId=1;batch.credit=1;batch.masks[0]=proposal.mask^1u;
    vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
    require(!status.updates && status.reason==static_cast<unsigned>(RewardReason::MaskMismatch),"mismatched actual mask rejected");
    // Consecutive non-blocking batches cannot overwrite the first receipt; the old policy is rejected.
    vabpnnConfigureRewardPolicy(model,6,true,stream);check(cudaStreamSynchronize(stream));
    auto one=evaluate(*base,1),two=evaluate(*base,2);
    decision.set(one);vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
    RewardProposal p1{};check(cudaMemcpy(&p1,view.proposal,sizeof(p1),cudaMemcpyDeviceToHost));
    decision.set(two);vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
    RewardProposal p2{};check(cudaMemcpy(&p2,view.proposal,sizeof(p2),cudaMemcpyDeviceToHost));
    batch.epoch=6;batch.rewardId=1;batch.frames[0]=1;batch.masks[0]=p1.mask;batch.credit=1;
    vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);
    batch.rewardId=2;batch.frames[0]=2;batch.masks[0]=p2.mask;batch.credit=-1;
    vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
    require(status.updates==1 && status.lastRewardId==1 && status.appliedRewards>=1
        && status.reason==static_cast<unsigned>(RewardReason::OldPolicy),"same-stream consecutive receipts and stale-policy rejection");
    // Bootstrap is explicit, bounded, and can move even from a neutral k=0-only checkpoint.
    auto boot=std::make_unique<DecisionParameters>(*base);
    boot->enabledKChoices=1;
    for(int a=0;a<VoteCount;++a)boot->thresholdBias[a]=10;
    check(cudaMemset(view.persistent,0,sizeof(RewardPersistent)));
    deviceParameters.set(*boot);const auto bootBefore=*boot;
    vabpnnConfigureRewardPolicy(model,7,true,stream,1,true);check(cudaStreamSynchronize(stream));
    unsigned observed=0;bool gotTrace=false;RewardCache exploratory{};
    for(unsigned frame=10;frame<330;++frame) {
        const auto input=evaluate(*boot,frame);require(!input.eligibleMask,"fixture has no threshold candidates");decision.set(input);
        vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
        RewardCache sample{};check(cudaMemcpy(&sample,view.cache+frame%RewardCacheSlots,sizeof(sample),cudaMemcpyDeviceToHost));
        require(sample.proposal.valid && sample.exploration && sample.proposal.k<=2
            && sample.proposal.count<=2 && sample.proposal.eligibleMask==((1u<<VoteCount)-1),"bounded bootstrap opens existing 15 atomic actions");
        for(int a=0;a<VoteCount;++a)if(sample.proposal.mask&(1u<<a))
            require(!(sample.proposal.mask&c.conflictMask[a]),"exploration always obeys atomic-action conflicts");
        require(std::abs(sample.proposal.logProbability-logProbability(*boot,sample))<2e-6,"exploration joint density includes actual branch/k/order");
        observed|=sample.proposal.mask;
        if(!gotTrace && sample.proposal.count==2){exploratory=sample;gotTrace=true;}
    }
    require(observed==((1u<<VoteCount)-1) && gotTrace,"all 15 atomic actions can be explored from cold start");
    const auto samplingOnly=deviceParameters.get();require(!std::memcmp(&samplingOnly,&bootBefore,sizeof(samplingOnly)),"sampling edits no neural weight/support/threshold");
    double explorationError=0;unsigned explorationNonzero=0;
    for(int i=0;i<RewardPolicyParameters;++i) {
        float& weight=parameter(*boot,i);const float old=weight;
        weight=old+epsilon;const double plus=logProbability(*boot,exploratory);
        weight=old-epsilon;const double minus=logProbability(*boot,exploratory);weight=old;
        const double expected=(plus-minus)/(2*epsilon),error=std::abs(expected-exploratory.score[i]);
        explorationError=std::max(explorationError,error);if(std::abs(exploratory.score[i])>1e-7)++explorationNonzero;
        require(error<2e-5+std::abs(expected)*.002,"exploration mixture score including k-to-evidence chain finite difference");
    }
    require(explorationNonzero>100,"bootstrap random decisions retain useful reward gradients");
    // Actual below-threshold actions can learn: reward increases exactly the sampled branch likelihood.
    deviceParameters.set(*boot);
    batch={};batch.epoch=7;batch.rewardId=1;batch.count=1;batch.frames[0]=exploratory.proposal.frameId;
    batch.masks[0]=exploratory.proposal.mask;batch.credit=1;
    const double explorationBefore=logProbability(*boot,exploratory);
    vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);check(cudaStreamSynchronize(stream));
    require(logProbability(deviceParameters.get(),exploratory)>explorationBefore,"reward teaches the actual exploratory action");
    // A strength change invalidates old traces; strict exploration and disabled mode never bypass thresholds.
    vabpnnConfigureRewardPolicy(model,8,true,stream,1,false);check(cudaStreamSynchronize(stream));
    for(unsigned frame=340;frame<345;++frame) {
        decision.set(evaluate(*boot,frame));
        vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
        RewardProposal value{};check(cudaMemcpy(&value,view.proposal,sizeof(value),cudaMemcpyDeviceToHost));
        require(value.valid && !value.mask && !value.eligibleMask,"strict exploration preserves empty threshold set");
    }
    batch.epoch=8;batch.rewardId=2;vabpnnApplyRewardBatch(model,deviceParameters.p,batch,stream);check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
    require(status.reason==static_cast<unsigned>(RewardReason::CacheMissing),"changed exploration epoch invalidates old policy traces");
    vabpnnConfigureRewardPolicy(model,9,false,stream,1,true);check(cudaStreamSynchronize(stream));
    decision.set(evaluate(*boot,346));vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
    RewardProposal disabled{};check(cudaMemcpy(&disabled,view.proposal,sizeof(disabled),cudaMemcpyDeviceToHost));
    require(!disabled.valid,"frozen reward mode emits no exploratory proposal");
    // Reproducible mixed-branch frequency: epsilon is the chance of exploring, not a threshold rewrite.
    deviceParameters.set(*base);vabpnnConfigureRewardPolicy(model,10,true,stream,.2f,true);check(cudaStreamSynchronize(stream));
    unsigned explorationFrames=0,ordinaryFrames=0;
    for(unsigned frame=400;frame<900;++frame) {
        const auto input=evaluate(*base,frame);decision.set(input);
        vabpnnProposeRewardAction(model,deviceParameters.p,constraints.p,decision.p,nullptr,nullptr,stream);check(cudaStreamSynchronize(stream));
        RewardCache sample{};check(cudaMemcpy(&sample,view.cache+frame%RewardCacheSlots,sizeof(sample),cudaMemcpyDeviceToHost));
        require(sample.proposal.valid,"mixed policy has valid proposals");
        if(sample.exploration){++explorationFrames;require(sample.proposal.k<=2,"exploration k bound");}
        else {++ordinaryFrames;require(!(sample.proposal.mask&~input.eligibleMask),"ordinary branch preserves thresholds");}
        require(std::abs(sample.proposal.logProbability-logProbability(*base,sample))<3e-6,"mixed-branch joint density");
    }
    require(explorationFrames>55 && explorationFrames<150 && ordinaryFrames>350,"20-percent configured exploration produces bounded mixed behavior");
    bool invalid=false;try{vabpnnConfigureRewardPolicy(model,11,true,stream,NAN,true);}catch(const std::invalid_argument&){invalid=true;}
    require(invalid,"nonfinite exploration configuration rejected");
    std::printf("PASS: exploration 1022 finite differences (%u nonzero, max %.3g); cold-start action diversity/conflicts/k<=2, 20%% branch %u/500, exact exploratory reward, epoch/freeze/threshold protections.\n",explorationNonzero,explorationError,explorationFrames);
    std::printf("PASS: %d actual policy score gradients (%u nonzero), max error %.3g; positive/negative likelihood direction.\n",RewardPolicyParameters,nonzero,maxError);
    std::printf("PASS: exact frame/mask/epoch/version, duplicate/zero/frozen protection; aggregate once and consecutive non-blocking batch lifetime.\n");
    std::printf("REWARD_LAYOUT size=%zu evidence=%zu evidenceBias=%zu rawVote=%zu threshold=%zu thresholdBias=%zu k=%zu kBias=%zu prior=%zu steps=%zu version=%zu\n",sizeof(DecisionParameters),offsetof(DecisionParameters,logLikelihoodWeight),offsetof(DecisionParameters,logLikelihoodBias),offsetof(DecisionParameters,rawVoteWeight),offsetof(DecisionParameters,thresholdWeight),offsetof(DecisionParameters,thresholdBias),offsetof(DecisionParameters,kWeight),offsetof(DecisionParameters,kBias),offsetof(DecisionParameters,initialPriorLogit),offsetof(RewardPersistent,steps),offsetof(RewardPersistent,policyVersion));
    vabpnnDestroyRewardPolicy(model);cudaStreamDestroy(stream);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
