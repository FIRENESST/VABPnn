#include "TemporalDecision.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
using namespace vabpnn;
constexpr int MinuteTokens=MinuteHistory*VoteCount;
struct MinuteState {
    std::uint64_t origin, lastNanos, bucket[MinuteHistory], sampleNanos[MinuteHistory];
    float coverageSeconds[MinuteHistory], average[MinuteHistory][VoteCount];
    int started, count, head;
};
struct MinuteCache {
    DecisionMinuteOutput output;
    float input[MinuteHistory][VoteCount][MinuteWidth], current[VoteCount][MinuteWidth];
    float query[VoteCount][MinuteWidth], key[MinuteHistory][VoteCount][MinuteWidth];
    float value[MinuteHistory][VoteCount][MinuteWidth];
    float attention[VoteCount][MinuteHeads][MinuteTokens], context[VoteCount][MinuteWidth];
    float time[MinuteHistory][MinuteTimeFeatures], currentTime[MinuteTimeFeatures];
    int learningFrozen;
};
struct MinuteBackward {
    float gradient[MinuteParameterCount];
    float current[VoteCount][MinuteWidth], query[VoteCount][MinuteWidth];
    float key[MinuteHistory][VoteCount][MinuteWidth], value[MinuteHistory][VoteCount][MinuteWidth];
    float input[MinuteHistory][VoteCount][MinuteWidth];
    float hidden[VoteCount][MinuteWidth], context[VoteCount][MinuteWidth];
    float evidence[VoteCount], k[MinuteKChoices];
    int active;
};
static_assert(std::is_trivially_copyable<DecisionMinutePersistent>::value,"GPU minute checkpoint.");
void check(cudaError_t code,const char* operation) {
    if(code!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(code));
}
__device__ float sigmoid(float x) {
    const float e=expf(-fabsf(x));return x>=0 ? 1/(1+e):e/(1+e);
}
__device__ float softplus(float x) {return x>20 ? x:log1pf(expf(x));}
__device__ void initialize(float* values,int count,unsigned seed,float scale) {
    for(int i=0;i<count;++i) {
        unsigned h=static_cast<unsigned>(i)+seed;
        h^=h>>16;h*=0x7feb352du;h^=h>>15;h*=0x846ca68bu;h^=h>>16;
        values[i]=(static_cast<float>(h&0xffffu)/32767.5f-1)*scale;
    }
}
__global__ void initializeMinute(DecisionMinutePersistent* state) {
    auto& p=state->parameters;
    initialize(&p.nodeEmbedding[0][0],VoteCount*MinuteWidth,307,.08f);
    initialize(p.voteEmbedding,MinuteWidth,311,.25f);
    initialize(&p.timeEmbedding[0][0],MinuteTimeFeatures*MinuteWidth,313,.05f);
    initialize(&p.query[0][0],MinuteWidth*MinuteWidth,317,.25f);
    initialize(&p.key[0][0],MinuteWidth*MinuteWidth,331,.25f);
    initialize(&p.value[0][0],MinuteWidth*MinuteWidth,337,.25f);
    initialize(&p.attentionOutput[0][0],MinuteWidth*MinuteWidth,347,.25f);
    for(int h=0;h<MinuteHeads;++h)p.logAttentionDecay[h]=-2.016290584f; // softplus=.125/minute.
    // logClockFrequency=0 -> softplus=.693 radians/minute; evidence/k/moments remain zero.
}
__global__ void minuteForward(const std::int32_t* votes,const DecisionMinutePersistent* persistent,
                              MinuteState* state,MinuteCache* cache,std::uint64_t id,std::uint64_t nanos,bool frozen) {
    const int i=threadIdx.x;
    __shared__ int slot,fresh,count;
    __shared__ float alpha;
    if(i==0) {
        if(state->started && nanos<state->lastNanos) {state->started=0;state->count=0;}
        if(!state->started) {state->origin=nanos;state->count=0;state->head=0;}
        const auto bucket=(nanos-state->origin)/MinuteNanos;
        // Expire missing/gapped minutes; never manufacture observations during a pause.
        while(state->count) {
            const int oldest=(state->head-state->count+1+MinuteHistory)%MinuteHistory;
            if(bucket-state->bucket[oldest]<MinuteHistory)break;
            --state->count;
        }
        fresh=!state->count || state->bucket[state->head]!=bucket;
        if(fresh) {
            slot=state->count ? (state->head+1)%MinuteHistory:0;
            state->head=slot;state->count=state->count<MinuteHistory ? state->count+1:MinuteHistory;
            state->bucket[slot]=bucket;state->coverageSeconds[slot]=0;
        } else slot=state->head;
        const float dt=state->started ? static_cast<float>((nanos-state->lastNanos)*1e-9):0;
        // Duration weighting caps unseen time at .25 s; identical timestamps add no samples.
        const float weight=fresh ? 1.0f/30.0f:fminf(.25f,dt);
        const float total=state->coverageSeconds[slot]+weight;
        alpha=total>0 ? weight/total:0;
        state->coverageSeconds[slot]=total;state->sampleNanos[slot]=nanos;
        state->started=1;state->lastNanos=nanos;count=state->count;
        cache->learningFrozen=frozen;
        cache->output.frameId=id;cache->output.captureNanos=nanos;
        cache->output.elapsedMinutes=static_cast<float>((nanos-state->origin)*(1e-9/60.0));
        cache->output.buckets=count;cache->output.valid=1;
        for(int t=0;t<count;++t) {
            const int position=(slot-t+MinuteHistory)%MinuteHistory;
            cache->output.bucketMinutes[t]=static_cast<float>((state->sampleNanos[position]-state->origin)*(1e-9/60.0));
            cache->output.ageMinutes[t]=static_cast<float>((nanos-state->sampleNanos[position])*(1e-9/60.0));
            cache->output.coverageMinutes[t]=state->coverageSeconds[position]/60.0f;
            const float phase=cache->output.bucketMinutes[t]*softplus(persistent->parameters.logClockFrequency);
            cache->time[t][0]=log1pf(cache->output.coverageMinutes[t]);
            cache->time[t][1]=sinf(phase);cache->time[t][2]=cosf(phase);
            cache->time[t][3]=log1pf(cache->output.bucketMinutes[t]);
        }
        for(int f=0;f<MinuteTimeFeatures;++f)cache->currentTime[f]=cache->time[0][f];
    }
    __syncthreads();
    const auto& p=persistent->parameters;
    if(i<VoteCount) {
        const float vote=static_cast<float>(votes[i])/NeuronCount;
        state->average[slot][i]=fresh ? vote:state->average[slot][i]+alpha*(vote-state->average[slot][i]);
        cache->output.currentVotes[i]=vote;
        for(int d=0;d<MinuteWidth;++d) {
            float x=p.nodeEmbedding[i][d]+vote*p.voteEmbedding[d];
            for(int f=0;f<MinuteTimeFeatures;++f)x+=cache->currentTime[f]*p.timeEmbedding[f][d];
            cache->current[i][d]=x;
        }
        for(int t=0;t<count;++t) {
            const int position=(slot-t+MinuteHistory)%MinuteHistory;
            const float summary=state->average[position][i];cache->output.votes[t][i]=summary;
            for(int d=0;d<MinuteWidth;++d) {
                float x=p.nodeEmbedding[i][d]+summary*p.voteEmbedding[d];
                for(int f=0;f<MinuteTimeFeatures;++f)x+=cache->time[t][f]*p.timeEmbedding[f][d];
                cache->input[t][i][d]=x;
            }
            for(int d=0;d<MinuteWidth;++d) {
                float q=0,k=0,v=0;
                for(int e=0;e<MinuteWidth;++e) {
                    if(t==0)q+=cache->current[i][e]*p.query[e][d];
                    k+=cache->input[t][i][e]*p.key[e][d];v+=cache->input[t][i][e]*p.value[e][d];
                }
                if(t==0)cache->query[i][d]=q;
                cache->key[t][i][d]=k;cache->value[t][i][d]=v;
            }
        }
    }
    __syncthreads();
    if(i<VoteCount) {
        for(int h=0;h<MinuteHeads;++h) {
            float maximum=-INFINITY;
            for(int t=0;t<count;++t)for(int j=0;j<VoteCount;++j) {
                float score=0;
                for(int d=0;d<MinuteHeadWidth;++d) {
                    const int e=h*MinuteHeadWidth+d;score+=cache->query[i][e]*cache->key[t][j][e];
                }
                score=score*.5f+p.graphBias[h][i][j]-softplus(p.logAttentionDecay[h])*cache->output.ageMinutes[t];
                cache->attention[i][h][t*VoteCount+j]=score;maximum=fmaxf(maximum,score);
            }
            float denominator=0;
            for(int t=0;t<count*VoteCount;++t) {
                const float weight=expf(cache->attention[i][h][t]-maximum);
                cache->attention[i][h][t]=weight;denominator+=weight;
            }
            for(int d=0;d<MinuteHeadWidth;++d)cache->context[i][h*MinuteHeadWidth+d]=0;
            for(int t=0;t<count;++t)for(int j=0;j<VoteCount;++j) {
                const float weight=cache->attention[i][h][t*VoteCount+j]/denominator;
                cache->attention[i][h][t*VoteCount+j]=weight;
                for(int d=0;d<MinuteHeadWidth;++d) {
                    const int e=h*MinuteHeadWidth+d;cache->context[i][e]+=weight*cache->value[t][j][e];
                }
            }
        }
        float evidence=0;
        for(int d=0;d<MinuteWidth;++d) {
            float x=cache->current[i][d];
            for(int e=0;e<MinuteWidth;++e)x+=cache->context[i][e]*p.attentionOutput[e][d];
            const float hidden=tanhf(x);cache->output.hidden[i][d]=hidden;evidence+=hidden*p.evidence[i][d];
        }
        cache->output.evidence[i]=evidence;
    }
    __syncthreads();
    if(i==0) {
        for(int d=0;d<MinuteWidth;++d) {
            float mean=0;for(int j=0;j<VoteCount;++j)mean+=cache->output.hidden[j][d];
            cache->output.mean[d]=mean/VoteCount;
        }
        for(int k=0;k<MinuteKChoices;++k) {
            float x=0;for(int d=0;d<MinuteWidth;++d)x+=cache->output.mean[d]*p.k[k][d];
            cache->output.kLogits[k]=x;cache->output.valid&=isfinite(x);
        }
        for(int j=0;j<VoteCount;++j)cache->output.valid&=isfinite(cache->output.evidence[j]);
    }
}
// Local, one-step temporal supervision: histories are causal raw observations, not
// stale learned projections. The CTMC prior is detached; k's posterior/soft-gate path
// is differentiated, as well as both minute heads and the entire minute attention.
__global__ void minuteBackward(const DecisionMinutePersistent* persistent,const MinuteCache* c,
        const DecisionParameters* base,const DecisionOutput* out,const int* active,const std::uint32_t* label,
        MinuteBackward* b) {
    const int i=threadIdx.x;
    if(i==0)b->active=*active && !c->learningFrozen && out->valid && c->output.valid
        && out->frameId==c->output.frameId && __popc(*label)<base->enabledKChoices;
    __syncthreads();
    if(!b->active)return;
    for(int at=i;at<MinuteParameterCount;at+=blockDim.x)b->gradient[at]=0;
    for(int at=i;at<MinuteHistory*VoteCount*MinuteWidth;at+=blockDim.x) {
        reinterpret_cast<float*>(b->key)[at]=0;reinterpret_cast<float*>(b->value)[at]=0;
    }
    for(int at=i;at<VoteCount*MinuteWidth;at+=blockDim.x)reinterpret_cast<float*>(b->query)[at]=0;
    const auto& p=persistent->parameters;const int count=c->output.buckets;
    const int target=__popc(*label);
    if(i<MinuteKChoices)b->k[i]=out->kProbability[i]-(i==target);
    __syncthreads();
    if(i<VoteCount) {
        const float posterior=out->posterior[i];
        float g=(posterior-static_cast<float>((*label>>i)&1u))/VoteCount;
        const float soft=out->softEligibility[i],temperature=softplus(base->logGateTemperature);
        float chain=0;
        for(int k=0;k<MinuteKChoices;++k)
            chain+=b->k[k]*(base->kWeight[k][DecisionWidth+i]
                +base->kWeight[k][DecisionWidth+VoteCount+i]*soft*(1-soft)/temperature);
        g+=chain*posterior*(1-posterior);b->evidence[i]=g;
        for(int d=0;d<MinuteWidth;++d) {
            float hiddenGradient=g*p.evidence[i][d];
            for(int k=0;k<MinuteKChoices;++k)hiddenGradient+=b->k[k]*p.k[k][d]/VoteCount;
            const float hidden=c->output.hidden[i][d];
            b->hidden[i][d]=b->current[i][d]=hiddenGradient*(1-hidden*hidden);
        }
        for(int e=0;e<MinuteWidth;++e) {
            float gContext=0;for(int d=0;d<MinuteWidth;++d)gContext+=b->hidden[i][d]*p.attentionOutput[e][d];
            b->context[i][e]=gContext;
        }
    }
    __syncthreads();
    constexpr int EvidenceOffset=offsetof(DecisionMinuteParameters,evidence)/sizeof(float);
    constexpr int KOffset=offsetof(DecisionMinuteParameters,k)/sizeof(float);
    constexpr int OutputOffset=offsetof(DecisionMinuteParameters,attentionOutput)/sizeof(float);
    for(int at=i;at<VoteCount*MinuteWidth;at+=blockDim.x) {
        const int node=at/MinuteWidth,d=at%MinuteWidth;
        b->gradient[EvidenceOffset+at]=b->evidence[node]*c->output.hidden[node][d];
    }
    for(int at=i;at<MinuteKChoices*MinuteWidth;at+=blockDim.x)
        b->gradient[KOffset+at]=b->k[at/MinuteWidth]*c->output.mean[at%MinuteWidth];
    for(int at=i;at<MinuteWidth*MinuteWidth;at+=blockDim.x) {
        const int e=at/MinuteWidth,d=at%MinuteWidth;float g=0;
        for(int j=0;j<VoteCount;++j)g+=c->context[j][e]*b->hidden[j][d];
        b->gradient[OutputOffset+at]=g;
    }
    __syncthreads();
    constexpr int GraphOffset=offsetof(DecisionMinuteParameters,graphBias)/sizeof(float);
    constexpr int DecayOffset=offsetof(DecisionMinuteParameters,logAttentionDecay)/sizeof(float);
    if(i<VoteCount)for(int h=0;h<MinuteHeads;++h) {
        float baseline=0,decayGradient=0;
        for(int d=0;d<MinuteHeadWidth;++d) {
            const int e=h*MinuteHeadWidth+d;baseline+=b->context[i][e]*c->context[i][e];
        }
        for(int t=0;t<count;++t)for(int j=0;j<VoteCount;++j) {
            float dot=0;
            for(int d=0;d<MinuteHeadWidth;++d) {
                const int e=h*MinuteHeadWidth+d;dot+=b->context[i][e]*c->value[t][j][e];
            }
            const float a=c->attention[i][h][t*VoteCount+j],g=a*(dot-baseline);
            b->gradient[GraphOffset+(h*VoteCount+i)*VoteCount+j]+=g;
            decayGradient-=g*c->output.ageMinutes[t]*sigmoid(p.logAttentionDecay[h]);
            for(int d=0;d<MinuteHeadWidth;++d) {
                const int e=h*MinuteHeadWidth+d;
                b->query[i][e]+=g*.5f*c->key[t][j][e];
                atomicAdd(&b->key[t][j][e],g*.5f*c->query[i][e]);
                atomicAdd(&b->value[t][j][e],a*b->context[i][e]);
            }
        }
        atomicAdd(&b->gradient[DecayOffset+h],decayGradient);
    }
    __syncthreads();
    if(i<VoteCount) {
        for(int e=0;e<MinuteWidth;++e) {
            float g=0;for(int d=0;d<MinuteWidth;++d)g+=b->query[i][d]*p.query[e][d];
            b->current[i][e]+=g;
        }
        for(int t=0;t<count;++t)for(int e=0;e<MinuteWidth;++e) {
            float g=0;
            for(int d=0;d<MinuteWidth;++d)g+=b->key[t][i][d]*p.key[e][d]+b->value[t][i][d]*p.value[e][d];
            b->input[t][i][e]=g;
        }
    }
    __syncthreads();
    constexpr int QueryOffset=offsetof(DecisionMinuteParameters,query)/sizeof(float);
    constexpr int KeyOffset=offsetof(DecisionMinuteParameters,key)/sizeof(float);
    constexpr int ValueOffset=offsetof(DecisionMinuteParameters,value)/sizeof(float);
    for(int at=i;at<MinuteWidth*MinuteWidth;at+=blockDim.x) {
        const int e=at/MinuteWidth,d=at%MinuteWidth;float q=0,k=0,v=0;
        for(int j=0;j<VoteCount;++j) {
            q+=c->current[j][e]*b->query[j][d];
            for(int t=0;t<count;++t) {
                k+=c->input[t][j][e]*b->key[t][j][d];
                v+=c->input[t][j][e]*b->value[t][j][d];
            }
        }
        b->gradient[QueryOffset+at]=q;b->gradient[KeyOffset+at]=k;b->gradient[ValueOffset+at]=v;
    }
    constexpr int NodeOffset=offsetof(DecisionMinuteParameters,nodeEmbedding)/sizeof(float);
    constexpr int VoteOffset=offsetof(DecisionMinuteParameters,voteEmbedding)/sizeof(float);
    constexpr int TimeOffset=offsetof(DecisionMinuteParameters,timeEmbedding)/sizeof(float);
    constexpr int FrequencyOffset=offsetof(DecisionMinuteParameters,logClockFrequency)/sizeof(float);
    for(int at=i;at<VoteCount*MinuteWidth;at+=blockDim.x) {
        const int j=at/MinuteWidth,e=at%MinuteWidth;float g=b->current[j][e];
        for(int t=0;t<count;++t)g+=b->input[t][j][e];
        b->gradient[NodeOffset+at]=g;
    }
    if(i<MinuteWidth) {
        float g=0;
        for(int j=0;j<VoteCount;++j) {
            g+=b->current[j][i]*c->output.currentVotes[j];
            for(int t=0;t<count;++t)g+=b->input[t][j][i]*c->output.votes[t][j];
        }
        b->gradient[VoteOffset+i]=g;
    }
    for(int at=i;at<MinuteTimeFeatures*MinuteWidth;at+=blockDim.x) {
        const int f=at/MinuteWidth,e=at%MinuteWidth;float g=0;
        for(int j=0;j<VoteCount;++j) {
            g+=b->current[j][e]*c->currentTime[f];
            for(int t=0;t<count;++t)g+=b->input[t][j][e]*c->time[t][f];
        }
        b->gradient[TimeOffset+at]=g;
    }
    if(i==0) {
        float g=0;
        for(int j=0;j<VoteCount;++j)for(int e=0;e<MinuteWidth;++e) {
            g+=b->current[j][e]*c->output.elapsedMinutes
                *(p.timeEmbedding[1][e]*c->currentTime[2]-p.timeEmbedding[2][e]*c->currentTime[1]);
            for(int t=0;t<count;++t)g+=b->input[t][j][e]*c->output.bucketMinutes[t]
                *(p.timeEmbedding[1][e]*c->time[t][2]-p.timeEmbedding[2][e]*c->time[t][1]);
        }
        b->gradient[FrequencyOffset]=g*sigmoid(p.logClockFrequency);
    }
    __syncthreads();
    // Reject the entire update on a nonfinite gradient; no partial weight/moment writes.
    for(int at=i;at<MinuteParameterCount;at+=blockDim.x)if(!isfinite(b->gradient[at]))atomicExch(&b->active,0);
}
__global__ void applyMinuteAdam(DecisionMinutePersistent* persistent,const MinuteBackward* b) {
    if(!b->active)return;
    __shared__ float bias1,bias2;
    if(threadIdx.x==0) {
        const float step=static_cast<float>(++persistent->steps);
        bias1=1-powf(.9f,step);bias2=1-powf(.999f,step);
    }
    __syncthreads();
    auto* p=reinterpret_cast<float*>(&persistent->parameters);
    auto* first=reinterpret_cast<float*>(&persistent->first);auto* second=reinterpret_cast<float*>(&persistent->second);
    for(int at=threadIdx.x;at<MinuteParameterCount;at+=blockDim.x) {
        const float g=fminf(1,fmaxf(-1,b->gradient[at]));
        const float m=first[at]=.9f*first[at]+.1f*g;
        const float v=second[at]=.999f*second[at]+.001f*g*g;
        p[at]=p[at]*.999997f-.0003f*(m/bias1)/(sqrtf(v/bias2)+1e-8f);
    }
}
}
struct VabpnnMinuteModel {
    vabpnn::DecisionMinutePersistent* persistent=nullptr;
    MinuteState* state=nullptr;MinuteCache* cache=nullptr;MinuteBackward* backward=nullptr;
    cudaEvent_t initialized=nullptr;
};
VabpnnMinuteModel* vabpnnCreateMinuteModel(cudaStream_t stream) {
    auto* m=new VabpnnMinuteModel();
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&m->persistent),sizeof(DecisionMinutePersistent)),"allocate minute weights/AdamW");
        check(cudaMalloc(reinterpret_cast<void**>(&m->state),sizeof(MinuteState)),"allocate observed minute summaries");
        check(cudaMalloc(reinterpret_cast<void**>(&m->cache),sizeof(MinuteCache)),"allocate causal minute attention cache");
        check(cudaMalloc(reinterpret_cast<void**>(&m->backward),sizeof(MinuteBackward)),"allocate reusable minute gradients");
        check(cudaMemsetAsync(m->persistent,0,sizeof(DecisionMinutePersistent),stream),"zero neutral minute fusion");
        check(cudaMemsetAsync(m->state,0,sizeof(MinuteState),stream),"zero minute history");
        check(cudaMemsetAsync(m->cache,0,sizeof(MinuteCache),stream),"zero minute output");
        check(cudaMemsetAsync(m->backward,0,sizeof(MinuteBackward),stream),"zero minute gradients");
        check(cudaEventCreateWithFlags(&m->initialized,cudaEventDisableTiming),"create minute ready event");
        initializeMinute<<<1,1,0,stream>>>(m->persistent);check(cudaGetLastError(),"initialize minute axis");
        check(cudaEventRecord(m->initialized,stream),"record minute initialization");return m;
    } catch(...) {vabpnnDestroyMinuteModel(m);throw;}
}
void vabpnnDestroyMinuteModel(VabpnnMinuteModel* m) noexcept {
    if(!m)return;if(m->initialized)cudaEventSynchronize(m->initialized);
    if(m->persistent)cudaFree(m->persistent);if(m->state)cudaFree(m->state);
    if(m->cache)cudaFree(m->cache);if(m->backward)cudaFree(m->backward);
    if(m->initialized)cudaEventDestroy(m->initialized);delete m;
}
std::uint64_t vabpnnMinuteModelBytes(const VabpnnMinuteModel* m) noexcept {
    return m ? sizeof(DecisionMinutePersistent)+sizeof(MinuteState)+sizeof(MinuteCache)+sizeof(MinuteBackward):0;
}
vabpnn::DeviceMinuteView vabpnnMinuteModelView(const VabpnnMinuteModel* m) {
    if(!m)throw std::invalid_argument("Minute model is not initialized");
    return {m->persistent,&m->cache->output};
}
vabpnn::DecisionMinutePersistent* vabpnnMutableMinuteModel(VabpnnMinuteModel* m,cudaStream_t stream) {
    if(!m)throw std::invalid_argument("Minute model is not initialized");
    check(cudaStreamWaitEvent(stream,m->initialized,0),"wait before minute update");return m->persistent;
}
vabpnn::ModelPart vabpnnMinuteModelCheckpointPart(VabpnnMinuteModel* m) {
    return {m->persistent,sizeof(vabpnn::DecisionMinutePersistent)};
}
void vabpnnLaunchMinuteTimeline(const VabpnnGpuFrame& frame,const std::int32_t* votes,VabpnnMinuteModel* m) {
    check(cudaStreamWaitEvent(frame.stream,m->initialized,0),"wait before minute inference");
    minuteForward<<<1,32,0,frame.stream>>>(votes,m->persistent,m->state,m->cache,frame.frameId,frame.captureNanos,frame.evaluationMode);
    check(cudaGetLastError(),"infer causal minute graph attention");
}
void vabpnnTrainMinuteTimeline(VabpnnMinuteModel* m,const vabpnn::DecisionParameters* base,
        const vabpnn::DecisionOutput* out,const int* active,const std::uint32_t* label,cudaStream_t stream) {
    check(cudaStreamWaitEvent(stream,m->initialized,0),"wait before minute teaching");
    minuteBackward<<<1,32,0,stream>>>(m->persistent,m->cache,base,out,active,label,m->backward);
    applyMinuteAdam<<<1,128,0,stream>>>(m->persistent,m->backward);
    check(cudaGetLastError(),"learn previous minute attention with AdamW");
}
void vabpnnResetMinuteHistory(VabpnnMinuteModel* m,cudaStream_t stream) {
    check(cudaStreamWaitEvent(stream,m->initialized,0),"wait before minute reset");
    check(cudaMemsetAsync(m->state,0,sizeof(MinuteState),stream),"reset minute observations");
    check(cudaMemsetAsync(m->cache,0,sizeof(MinuteCache),stream),"reset minute inference cache");
}
