#include "RewardPolicy.h"
#include "FullSelfLearning.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using namespace vabpnn;
struct Runtime {RewardStatus status;std::uint64_t lastFrame;int trainActive;
    float exploration;bool allowSubThreshold;};
struct Buffers {
    RewardPersistent* persistent;RewardCache* cache;RewardProposal* proposal;Runtime* runtime;
    RewardBatch* batch;float* gradient;
};
void check(cudaError_t code,const char* operation) {
    if(code!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(code));
}
__device__ float uniform(std::uint64_t& state) {
    state^=state>>12;state^=state<<25;state^=state>>27;
    return (static_cast<float>((state*2685821657736338717ull)>>40)+.5f)/16777216.f;
}
__global__ void refresh(Buffers b) {
    b.runtime->status.updates=b.persistent->steps;b.runtime->status.policyVersion=b.persistent->policyVersion;
}
__global__ void configure(Buffers b,std::uint64_t epoch,bool enabled,float exploration,bool allowSubThreshold) {
    b.runtime->status.epoch=epoch;b.runtime->status.enabled=enabled;
    b.runtime->status.reason=static_cast<unsigned>(enabled ? RewardReason::Ready:RewardReason::Disabled);
    b.runtime->status.policyVersion=b.persistent->policyVersion;b.runtime->status.updates=b.persistent->steps;
    b.runtime->lastFrame=0;b.runtime->trainActive=0;b.runtime->status.lastRewardId=0;
    b.runtime->exploration=enabled ? exploration:0;
    b.runtime->allowSubThreshold=enabled && allowSubThreshold;
    *b.proposal={};
}
__global__ void propose(Buffers b,const DecisionParameters* p,const DecisionConstraints* constraints,
                        DecisionOutput* out,const JepaOutput* mouse,const MotionProposal* motion) {
    auto& rt=*b.runtime;*b.proposal={};
    if(!rt.status.enabled || !out->valid || !out->frameId || rt.lastFrame==out->frameId)return;
    rt.lastFrame=out->frameId;auto& cache=b.cache[out->frameId%RewardCacheSlots];cache.proposal={};
    for(int a=0;a<VoteCount;++a)cache.order[a]=-1;
    for(int j=0;j<RewardPolicyParameters;++j)cache.score[j]=0;
    std::uint64_t rng=(out->frameId*0x9e3779b97f4a7c15ull)^out->captureNanos
        ^(rt.status.epoch*0xd1b54a32d192ed03ull)^(b.persistent->policyVersion+1);
    if(!rng)rng=1;
    const bool exploring=rt.exploration>0 && uniform(rng)<rt.exploration;
    const unsigned eligible=exploring && rt.allowSubThreshold ? ((1u<<VoteCount)-1):out->eligibleMask;
    const int available=__popc(eligible);
    const int active=isfinite(p->enabledKChoices) ? static_cast<int>(fminf(DecisionKChoices,fmaxf(1,p->enabledKChoices))):1;
    // Bootstrap exposes existing k=0,1,2 logits without editing learned support metadata.
    const int choices=min(exploring ? (rt.allowSubThreshold ? 3:min(active,3)):active,available+1);
    const float mixture=exploring ? .5f:1.f;
    const float branchProbability=rt.exploration>0 ? (exploring ? rt.exploration:1-rt.exploration):1;
    float s[DecisionKChoices]{},q[DecisionKChoices]{},maximum=-INFINITY,total=0;
    for(int k=0;k<choices;++k) {
        float x=p->kBias[k]+out->kContextLogits[k];
        for(int f=0;f<DecisionKFeatures;++f)x+=p->kWeight[k][f]*out->kFeatures[f];
        if(!isfinite(x))return;s[k]=x;maximum=fmaxf(maximum,x);
    }
    for(int k=0;k<choices;++k){s[k]=expf(s[k]-maximum);total+=s[k];}
    if(!isfinite(total) || total<=0 || branchProbability<=0)return;
    for(int k=0;k<choices;++k){s[k]/=total;q[k]=mixture*s[k]+(1-mixture)/choices;}
    const float draw=uniform(rng);float cumulative=0;int chosenK=choices-1;
    for(int k=0;k<choices;++k){cumulative+=q[k];if(draw<cumulative){chosenK=k;break;}}
    float scoreEvidence[VoteCount]{},dk[DecisionKChoices]{};
    const float kDerivative=mixture*s[chosenK]/q[chosenK];
    for(int k=0;k<choices;++k)dk[k]=kDerivative*((k==chosenK ? 1.f:0.f)-s[k]);
    cache.exploration=exploring;cache.choices=choices;cache.mixture=mixture;
    cache.branchProbability=branchProbability;
    const float temperature=p->logGateTemperature>20 ? p->logGateTemperature:log1pf(expf(p->logGateTemperature));
    if(!isfinite(temperature) || temperature<=0)return;
    for(int a=0;a<VoteCount;++a) {
        const float probability=out->posterior[a],gate=out->softEligibility[a];
        float derivative=0;
        for(int k=0;k<choices;++k)
            derivative+=dk[k]*(p->kWeight[k][DecisionWidth+a]
                +p->kWeight[k][DecisionWidth+VoteCount+a]*gate*(1-gate)/temperature);
        scoreEvidence[a]=derivative*probability*(1-probability);
    }
    unsigned remaining=eligible,selected=0;int count=0;
    float logProbability=logf(branchProbability)+logf(fmaxf(q[chosenK],1e-30f));
    float weights[VoteCount]{};
    for(int a=0;a<VoteCount;++a)weights[a]=mixture*out->posterior[a]+(1-mixture);
    while(count<chosenK && remaining) {
        float sum=0;for(int a=0;a<VoteCount;++a)if(remaining&(1u<<a))sum+=weights[a];
        if(!isfinite(sum) || sum<=0)return;
        const float value=uniform(rng)*sum;float acc=0;int chosen=-1;
        for(int a=0;a<VoteCount;++a)if(remaining&(1u<<a)) {
            chosen=a;acc+=weights[a];if(value<acc)break;
        }
        if(chosen<0)return;
        for(int a=0;a<VoteCount;++a)if(remaining&(1u<<a)) {
            const float probability=out->posterior[a];
            const float derivative=mixture*probability*(1-probability)/weights[a];
            scoreEvidence[a]+=((a==chosen ? 1.f:0.f)-weights[a]/sum)*derivative;
        }
        logProbability+=logf(fmaxf(weights[chosen]/sum,1e-30f));
        selected|=1u<<chosen;cache.order[count++]=chosen;remaining&=~(1u<<chosen);
        for(int a=0;a<VoteCount;++a)
            if((constraints->conflictMask[chosen]&(1u<<a)) || (constraints->conflictMask[a]&(1u<<chosen)))
                remaining&=~(1u<<a);
    }
    for(int a=0;a<VoteCount;++a) {
        const int at=a*RewardEvidenceStride;
        for(int d=0;d<DecisionWidth;++d)cache.score[at+d]=scoreEvidence[a]*out->hidden[a][d];
        cache.score[at+DecisionWidth]=scoreEvidence[a];
        cache.score[at+DecisionWidth+1]=scoreEvidence[a]*out->normalizedVotes[a];
    }
    for(int k=0;k<DecisionKChoices;++k) {
        const int at=RewardEvidenceParameters+k*RewardKStride;
        for(int f=0;f<DecisionKFeatures;++f)cache.score[at+f]=dk[k]*out->kFeatures[f];
        cache.score[at+DecisionKFeatures]=dk[k];
    }
    for(int j=0;j<RewardPolicyParameters;++j)if(!isfinite(cache.score[j]))return;
    const float dx=mouse && mouse->valid && mouse->motionActive ? mouse->mouseDx:0;
    const float dy=mouse && mouse->valid && mouse->motionActive ? mouse->mouseDy:0;
    const float wheel=motion ? motion->wheelSteps:0;
    if(!isfinite(dx) || !isfinite(dy) || !isfinite(wheel))return;
    cache.proposal=RewardProposal{out->frameId,out->captureNanos,rt.status.epoch,b.persistent->policyVersion,
        selected,eligible,static_cast<unsigned>(chosenK),static_cast<unsigned>(count),1,logProbability};
    cache.proposal.expectedDx=(selected&actionBit(Action::MouseX))?dx:0;
    cache.proposal.expectedDy=(selected&actionBit(Action::MouseY))?dy:0;
    cache.proposal.expectedWheel=(selected&actionBit(Action::Wheel))?wheel:0;
    *b.proposal=cache.proposal;++rt.status.proposals;
    if(exploring)++rt.status.explorationProposals;
    if(selected&~out->eligibleMask)++rt.status.subThresholdProposals;
    out->selectedMask=selected;out->requestedK=chosenK;out->selectedCount=count;
    for(int a=0;a<VoteCount;++a)out->selectedIndices[a]=cache.order[a];
    out->command.selectedMask=selected;
    out->command.mouseDx=(selected&actionBit(Action::MouseX)) ? dx:0;
    out->command.mouseDy=(selected&actionBit(Action::MouseY)) ? dy:0;
    out->command.wheelSteps=(selected&actionBit(Action::Wheel)) ? wheel:0;
    out->status.requestedK=chosenK;out->status.selectedCount=count;
}
__global__ void validate(Buffers b) {
    auto& rt=*b.runtime;rt.trainActive=0;
    auto reject=[&](RewardReason reason){rt.status.reason=static_cast<unsigned>(reason);++rt.status.droppedRewards;};
    const auto& batch=*b.batch;
    if(!rt.status.enabled){reject(RewardReason::Disabled);return;}
    if(batch.epoch!=rt.status.epoch){reject(RewardReason::Epoch);return;}
    if(!batch.rewardId || batch.rewardId<=rt.status.lastRewardId){reject(RewardReason::Duplicate);return;}
    if(!batch.count || batch.count>RewardCacheSlots || !isfinite(batch.credit) || fabsf(batch.credit)>1){
        reject(RewardReason::InvalidBatch);return;
    }
    for(unsigned i=0;i<batch.count;++i) {
        if(!batch.frames[i] || (i && batch.frames[i]<=batch.frames[i-1])){reject(RewardReason::InvalidBatch);return;}
        const auto& proposal=b.cache[batch.frames[i]%RewardCacheSlots].proposal;
        if(!proposal.valid || proposal.frameId!=batch.frames[i] || proposal.epoch!=batch.epoch){
            reject(RewardReason::CacheMissing);return;
        }
        if(proposal.policyVersion!=b.persistent->policyVersion){reject(RewardReason::OldPolicy);return;}
        if(proposal.mask!=batch.masks[i]){reject(RewardReason::MaskMismatch);return;}
    }
    rt.status.lastRewardId=batch.rewardId;rt.status.lastCredit=batch.credit;rt.status.lastCount=batch.count;
    ++rt.status.appliedRewards;rt.status.reason=static_cast<unsigned>(RewardReason::Ready);rt.trainActive=1;
}
__global__ void aggregate(Buffers b) {
    const int parameter=blockIdx.x*blockDim.x+threadIdx.x;if(parameter>=RewardPolicyParameters)return;
    if(!b.runtime->trainActive){b.gradient[parameter]=0;return;}
    float score=0;
    for(unsigned i=0;i<b.batch->count;++i)score+=b.cache[b.batch->frames[i]%RewardCacheSlots].score[parameter];
    // One normalized reward for the window, distributed across its acknowledged decisions.
    b.gradient[parameter]=-b.batch->credit*score/b.batch->count;
}
__global__ void prepareAdam(Buffers b,const int* fullSupport=nullptr) {
    if(!b.runtime->trainActive)return;
    for(unsigned i=0;i<b.batch->count;++i)b.cache[b.batch->frames[i]%RewardCacheSlots].proposal.valid=0;
    if(b.batch->credit==0){b.runtime->trainActive=0;b.runtime->status.reason=static_cast<unsigned>(RewardReason::ZeroCredit);return;}
    bool support=false;
    for(int i=0;i<RewardPolicyParameters;++i) {
        if(!isfinite(b.gradient[i])){b.runtime->trainActive=0;b.runtime->status.reason=static_cast<unsigned>(RewardReason::InvalidBatch);return;}
        support=support || b.gradient[i]!=0;
    }
    if(fullSupport && *fullSupport)support=true;
    if(!support){b.runtime->trainActive=0;b.runtime->status.reason=static_cast<unsigned>(RewardReason::NoSupport);return;}
    ++b.persistent->steps;++b.persistent->policyVersion;
    b.persistent->lastCredit=b.batch->credit;b.persistent->lastCount=b.batch->count;
    b.runtime->status.updates=b.persistent->steps;b.runtime->status.policyVersion=b.persistent->policyVersion;
    b.runtime->status.reason=static_cast<unsigned>(RewardReason::Updated);
}
__global__ void adam(Buffers b,DecisionParameters* p) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=RewardPolicyParameters || !b.runtime->trainActive)return;
    float* parameter;
    if(index<RewardEvidenceParameters) {
        const int action=index/RewardEvidenceStride,d=index%RewardEvidenceStride;
        parameter=d<DecisionWidth ? &p->logLikelihoodWeight[action][d]
            :d==DecisionWidth ? &p->logLikelihoodBias[action]:&p->rawVoteWeight[action];
    } else {
        const int at=index-RewardEvidenceParameters,k=at/RewardKStride,f=at%RewardKStride;
        parameter=f<DecisionKFeatures ? &p->kWeight[k][f]:&p->kBias[k];
    }
    const float g=fminf(1,fmaxf(-1,b.gradient[index]));
    const float m=b.persistent->first[index]=.9f*b.persistent->first[index]+.1f*g;
    const float v=b.persistent->second[index]=.999f*b.persistent->second[index]+.001f*g*g;
    const float step=static_cast<float>(b.persistent->steps);
    *parameter-=.0001f*(m/(1-powf(.9f,step)))/(sqrtf(v/(1-powf(.999f,step)))+1e-8f);
}
}
struct VabpnnRewardPolicy {Buffers b{};std::vector<void*> allocations;std::uint64_t bytes=0;
    cudaEvent_t initialized=nullptr,uploaded=nullptr;vabpnn::RewardBatch* staging=nullptr;};
namespace {
template<class T>void allocate(VabpnnRewardPolicy* m,T*& p,std::size_t count,cudaStream_t stream) {
    check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)),"allocate reward policy buffer");
    m->allocations.push_back(p);m->bytes+=count*sizeof(T);check(cudaMemsetAsync(p,0,count*sizeof(T),stream),"zero reward policy buffer");
}
void wait(VabpnnRewardPolicy* m,cudaStream_t stream) {
    if(!m)throw std::invalid_argument("Reward policy not initialized");
    check(cudaStreamWaitEvent(stream,m->initialized,0),"wait for reward policy");
}
}
VabpnnRewardPolicy* vabpnnCreateRewardPolicy(cudaStream_t stream) {
    auto* m=new VabpnnRewardPolicy();
    try {
        allocate(m,m->b.persistent,1,stream);allocate(m,m->b.cache,vabpnn::RewardCacheSlots,stream);
        allocate(m,m->b.proposal,1,stream);allocate(m,m->b.runtime,1,stream);
        allocate(m,m->b.batch,1,stream);allocate(m,m->b.gradient,vabpnn::RewardPolicyParameters,stream);
        check(cudaMallocHost(reinterpret_cast<void**>(&m->staging),sizeof(vabpnn::RewardBatch)),"allocate bounded pinned reward receipt");
        check(cudaEventCreateWithFlags(&m->uploaded,cudaEventDisableTiming),"reward receipt upload event");
        check(cudaEventRecord(m->uploaded,stream),"initialize receipt upload lifetime");
        check(cudaEventCreateWithFlags(&m->initialized,cudaEventDisableTiming),"reward policy ready event");
        check(cudaEventRecord(m->initialized,stream),"record reward policy initialization");
        return m;
    }catch(...){vabpnnDestroyRewardPolicy(m);throw;}
}
void vabpnnDestroyRewardPolicy(VabpnnRewardPolicy* m) noexcept {
    if(!m)return;if(m->initialized)cudaEventSynchronize(m->initialized);
    if(m->uploaded)cudaEventSynchronize(m->uploaded);
    for(auto p:m->allocations)cudaFree(p);if(m->staging)cudaFreeHost(m->staging);
    if(m->uploaded)cudaEventDestroy(m->uploaded);if(m->initialized)cudaEventDestroy(m->initialized);delete m;
}
std::uint64_t vabpnnRewardPolicyBytes(const VabpnnRewardPolicy* m) noexcept{return m ? m->bytes:0;}
vabpnn::DeviceRewardView vabpnnRewardPolicyView(VabpnnRewardPolicy* m) {
    if(!m)throw std::invalid_argument("Reward policy not initialized");
    return {m->b.persistent,m->b.cache,m->b.proposal,&m->b.runtime->status,m->b.gradient,&m->b.runtime->trainActive};
}
vabpnn::ModelPart vabpnnRewardCheckpointPart(VabpnnRewardPolicy* m) {return {m->b.persistent,sizeof(vabpnn::RewardPersistent)};}
void vabpnnRefreshRewardStatus(VabpnnRewardPolicy* m,cudaStream_t stream) {
    wait(m,stream);refresh<<<1,1,0,stream>>>(m->b);check(cudaGetLastError(),"refresh reward optimizer counters");
}
void vabpnnConfigureRewardPolicy(VabpnnRewardPolicy* m,std::uint64_t epoch,bool enabled,cudaStream_t stream,
    float exploration,bool allowSubThreshold) {
    wait(m,stream);if(!epoch)throw std::invalid_argument("Positive reward-mode epoch required");
    if(!std::isfinite(exploration) || exploration<0 || exploration>1)
        throw std::invalid_argument("Exploration probability must be finite and in 0..1");
    check(cudaMemsetAsync(m->b.cache,0,vabpnn::RewardCacheSlots*sizeof(vabpnn::RewardCache),stream),"discard old reward eligibility");
    configure<<<1,1,0,stream>>>(m->b,epoch,enabled,exploration,allowSubThreshold);check(cudaGetLastError(),"configure reward policy");
}
void vabpnnProposeRewardAction(VabpnnRewardPolicy* m,const vabpnn::DecisionParameters* p,const vabpnn::DecisionConstraints* constraints,
    vabpnn::DecisionOutput* output,const vabpnn::JepaOutput* mouse,const vabpnn::MotionProposal* motion,cudaStream_t stream) {
    wait(m,stream);propose<<<1,1,0,stream>>>(m->b,p,constraints,output,mouse,motion);check(cudaGetLastError(),"sample threshold-filtered reward policy");
}
void vabpnnApplyRewardBatch(VabpnnRewardPolicy* m,vabpnn::DecisionParameters* parameters,const vabpnn::RewardBatch& batch,cudaStream_t stream,const VabpnnGpuFrame* fullModels) {
    wait(m,stream);
    if(!batch.count || batch.count>vabpnn::RewardCacheSlots || !std::isfinite(batch.credit) || std::abs(batch.credit)>1)
        throw std::invalid_argument("Invalid bounded reward batch");
    // Only this bounded CPU staging lifetime waits; GPU receipt writes and reads are
    // ordered on the SAME operator stream, including consecutive submissions.
    check(cudaEventSynchronize(m->uploaded),"reuse pinned reward receipt after upload");
    *m->staging=batch;
    check(cudaMemcpyAsync(m->b.batch,m->staging,sizeof(batch),cudaMemcpyHostToDevice,stream),"upload bounded reward receipt on operator stream");
    check(cudaEventRecord(m->uploaded,stream),"record safe reward receipt staging lifetime");
    validate<<<1,1,0,stream>>>(m->b);aggregate<<<(vabpnn::RewardPolicyParameters+255)/256,256,0,stream>>>(m->b);
    if(fullModels && fullModels->fullLearning)vabpnnPrepareFullReward(fullModels->fullLearning,*fullModels,m->b.batch,&m->b.runtime->trainActive,stream);
    prepareAdam<<<1,1,0,stream>>>(m->b,fullModels && fullModels->fullLearning?vabpnnFullLearningView(fullModels->fullLearning).active:nullptr);
    if(fullModels && fullModels->fullLearning)vabpnnCommitFullReward(fullModels->fullLearning,*fullModels,&m->b.runtime->trainActive,stream);adam<<<(vabpnn::RewardPolicyParameters+255)/256,256,0,stream>>>(m->b,parameters);
    check(cudaGetLastError(),"apply acknowledged on-policy reward gradient");
}
