#include "TemporalDecision.h"
#include "JepaMotion.h"
#include "DenseBlockTransformer.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
using namespace vabpnn;
struct DecisionState {
    std::uint64_t lastCaptureNanos, originNanos;
    std::uint64_t historyNanos[DecisionHistory];
    float previousPosterior[VoteCount];
    std::int32_t historyVotes[DecisionHistory][VoteCount];
    float historyDeltaSeconds[DecisionHistory];
    int started, count, head;
};
struct PositionState {
    double origin[3], previous[3];
    float velocity[3], features[PositionFeatures];
    std::uint64_t sampleNanos, epoch;
    int ready;
};
struct BayesianMotionCache {
    std::uint64_t frameId;
    float features[BayesianMotionFeatures],predicted[2],scale;
    int ready,learningFrozen;
};
constexpr std::uint64_t DecisionBytes = sizeof(DecisionParameters) + sizeof(DecisionConstraints)
    + sizeof(DecisionState) + sizeof(DecisionOutput) + sizeof(MotionProposal)
    + sizeof(PositionState) + sizeof(DecisionPositionAdapter)+sizeof(BayesianMotionCache);
static_assert(std::is_trivially_copyable<DecisionParameters>::value, "GPU parameter object.");
static_assert(std::is_trivially_copyable<DecisionOutput>::value, "GPU result object.");

void check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
__device__ float sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}
__device__ float softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}
__device__ float probability(float p) {
    return fminf(1.0f - 1.0e-6f, fmaxf(1.0e-6f, p));
}
__device__ float logit(float p) {
    p = probability(p);
    return logf(p) - log1pf(-p);
}
__device__ void layerNorm(float* x, const float* gain, const float* bias) {
    float mean = 0, variance = 0;
    for (int d = 0; d < DecisionWidth; ++d) mean += x[d];
    mean /= DecisionWidth;
    for (int d = 0; d < DecisionWidth; ++d) {
        const float difference = x[d] - mean;
        variance += difference * difference;
    }
    const float scale = rsqrtf(variance / DecisionWidth + 1.0e-5f);
    for (int d = 0; d < DecisionWidth; ++d) x[d] = (x[d] - mean) * scale * gain[d] + bias[d];
}
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
__device__ void initializeMatrix(float* values, int count, unsigned int seed, float scale) {
    for (int i = 0; i < count; ++i) {
        unsigned int hash = static_cast<unsigned int>(i) + seed;
        hash ^= hash >> 16; hash *= 0x7feb352du;
        hash ^= hash >> 15; hash *= 0x846ca68bu; hash ^= hash >> 16;
        values[i] = (static_cast<float>(hash & 0xffffu) / 32767.5f - 1.0f) * scale;
    }
}
__global__ void initializeParameters(DecisionParameters* p, DecisionConstraints* constraints) {
    // Buffer was zeroed first. Nonzero representations; neutral evidence/decision heads.
    p->enabledKChoices=DecisionKChoices;
    initializeMatrix(&p->nodeEmbedding[0][0], VoteCount * DecisionWidth, 11, 0.1f);
    initializeMatrix(p->voteEmbedding, DecisionWidth, 23, 0.25f);
    initializeMatrix(&p->timeEmbedding[0][0], DecisionTimeFeatures * DecisionWidth, 37, 0.02f);
    initializeMatrix(&p->query[0][0], DecisionWidth * DecisionWidth, 41, 0.25f);
    initializeMatrix(&p->key[0][0], DecisionWidth * DecisionWidth, 53, 0.25f);
    initializeMatrix(&p->value[0][0], DecisionWidth * DecisionWidth, 67, 0.25f);
    initializeMatrix(&p->attentionOutput[0][0], DecisionWidth * DecisionWidth, 79, 0.25f);
    initializeMatrix(&p->feedForwardUp[0][0], DecisionWidth * DecisionHiddenWidth, 83, 0.2f);
    initializeMatrix(&p->feedForwardDown[0][0], DecisionHiddenWidth * DecisionWidth, 97, 0.2f);
    for (int d = 0; d < DecisionWidth; ++d) p->attentionGain[d] = p->outputGain[d] = 1.0f;
    p->logGateTemperature = -2.252168461f; // softplus = 0.1
    constraints->conflictMask[7] = (1u << 8) | (1u << 11);
    constraints->conflictMask[8] = (1u << 7) | (1u << 11);
    constraints->conflictMask[9] = 1u << 10;
    constraints->conflictMask[10] = 1u << 9;
    constraints->conflictMask[11] = (1u << 7) | (1u << 8);
}

__device__ void encodePosition(const PlayerPositionSample& sample, std::uint64_t nanos,
                               PositionState* state, DecisionOutput* result) {
    for(int f=0;f<PositionFeatures;++f) result->positionFeatures[f]=0;
    result->positionValid=0;
    const double xyz[3]={sample.x,sample.y,sample.z};
    bool valid=sample.sampleNanos && sample.epoch && sample.sampleNanos<=nanos
        && nanos-sample.sampleNanos<=PositionMaximumAgeNanos;
    for(int d=0;d<3;++d) valid=valid && isfinite(xyz[d]) && fabs(xyz[d])<=30'000'000;
    if(!valid) { state->ready=0; return; }
    if(state->ready && state->epoch==sample.epoch && sample.sampleNanos<state->sampleNanos) {
        state->ready=0; return;
    }
    bool fresh=!state->ready || state->epoch!=sample.epoch;
    if(!fresh && sample.sampleNanos!=state->sampleNanos) {
        const double dt=(sample.sampleNanos-state->sampleNanos)*1e-9;
        double squared=0;
        for(int d=0;d<3;++d) squared+=(xyz[d]-state->previous[d])*(xyz[d]-state->previous[d]);
        // Do not turn a teleport, resume or long sampling gap into extreme motion.
        fresh=dt>0.25 || squared>128.0*128.0*dt*dt;
        if(!fresh) {
            const float alpha=static_cast<float>(1-exp(-12.0*dt));
            for(int d=0;d<3;++d)
                state->velocity[d]+=alpha*(static_cast<float>((xyz[d]-state->previous[d])/dt)-state->velocity[d]);
        }
    }
    if(fresh) for(int d=0;d<3;++d) { state->origin[d]=xyz[d]; state->velocity[d]=0; }
    if(fresh || sample.sampleNanos!=state->sampleNanos) {
        for(int d=0;d<3;++d) state->previous[d]=xyz[d];
        state->sampleNanos=sample.sampleNanos; state->epoch=sample.epoch; state->ready=1;
        // Relative XYZ, smoothed blocks/second, world X/Z phase, absolute height and presence.
        // Differences and phase are formed from doubles before conversion, even at world edges.
        for(int d=0;d<3;++d) {
            state->features[d]=static_cast<float>(tanh((state->previous[d]-state->origin[d])/64.0));
            state->features[3+d]=tanhf(state->velocity[d]/10.0f);
        }
        state->features[6]=static_cast<float>(sin(state->previous[0]/64.0));
        state->features[7]=static_cast<float>(cos(state->previous[0]/64.0));
        state->features[8]=static_cast<float>(sin(state->previous[2]/64.0));
        state->features[9]=static_cast<float>(cos(state->previous[2]/64.0));
        state->features[10]=static_cast<float>(tanh(state->previous[1]/128.0));
        state->features[11]=1;
    }
    for(int f=0;f<PositionFeatures;++f) result->positionFeatures[f]=state->features[f];
    result->positionValid=1;
}

__global__ void densePosition(PlayerPositionSample position,std::uint64_t nanos,
    PositionState* state,DecisionOutput* output) {encodePosition(position,nanos,state,output);}

__global__ void decide(const std::int32_t* votes, const DecisionParameters* p,
                       const DecisionConstraints* constraints, const MotionProposal* motion, DecisionState* state,
                       DecisionOutput* result, std::uint64_t frameId, std::uint64_t nanos,
                       PlayerPositionSample position, PositionState* positionState,
                       const DecisionPositionAdapter* adapter, const DecisionMinuteOutput* minutes,
                       const DecisionMinutePersistent* minuteParameters, const DenseBlockOutput* dense) {
    // Action heads read the learned complete 7680-node block graph.
    __shared__ float hidden[VoteCount][DecisionWidth];
    __shared__ int nodeValid[VoteCount], first, slot;
    __shared__ float dt;
    const int i = threadIdx.x;
    if (i == 0) {
        // A reversed clock denotes a new episode; discard temporal data, retain parameters.
        if (state->started && nanos < state->lastCaptureNanos) {
            state->started = 0; state->count = 0; positionState->ready=0;
        }
        encodePosition(position,nanos,positionState,result);
        first = !state->started;
        if (first) state->originNanos = nanos;
        dt = first ? 0.0f : static_cast<float>((nanos - state->lastCaptureNanos) * 1.0e-9);
        slot = state->count == 0 ? 0 : (state->head + 1) % DecisionHistory;
        state->head = slot;
        state->count = state->count < DecisionHistory ? state->count + 1 : DecisionHistory;
        state->historyNanos[slot] = nanos;
        state->historyDeltaSeconds[slot] = dt;
        result->frameId = frameId; result->captureNanos = nanos; result->deltaSeconds = dt;
    }
    __syncthreads();
    if (i < VoteCount) {
        float prior;
        if (first) {
            prior = sigmoid(p->initialPriorLogit[i]);
        } else {
            const float on = softplus(p->logOnRate[i]), off = softplus(p->logOffRate[i]);
            const float rate = on + off;
            // Closed form of a two-state continuous-time Markov transition.
            const float stationary = rate > 0 ? on / rate : state->previousPosterior[i];
            prior = stationary + (state->previousPosterior[i] - stationary) * expf(-rate * dt);
        }
        nodeValid[i] = isfinite(prior) && isfinite(p->logOnRate[i]) && isfinite(p->logOffRate[i]);
        result->prior[i] = nodeValid[i] ? probability(prior) : 0.5f;

        state->historyVotes[slot][i]=votes[i];
        for(int d=0;d<DecisionWidth;++d)hidden[i][d]=dense->hidden[i][d];
        nodeValid[i]=nodeValid[i] && dense->valid && dense->frameId==frameId;
        result->normalizedVotes[i] = static_cast<float>(votes[i]) / static_cast<float>(NeuronCount);
        float evidence = p->logLikelihoodBias[i] + p->rawVoteWeight[i] * result->normalizedVotes[i];
        float thresholdLogit = p->thresholdBias[i];
        for (int d = 0; d < DecisionWidth; ++d) result->hidden[i][d] = hidden[i][d];
        for (int d = 0; d < DecisionWidth; ++d) {
            evidence += hidden[i][d] * p->logLikelihoodWeight[i][d];
            thresholdLogit += hidden[i][d] * p->thresholdWeight[i][d];
        }
        if(result->positionValid) for(int f=0;f<PositionFeatures;++f)
            evidence+=adapter->evidence[i][f]*result->positionFeatures[f];
        evidence+=minutes->evidence[i];
        const float temperature = softplus(p->logGateTemperature);
        nodeValid[i] = nodeValid[i] && minutes->valid && minutes->frameId==frameId
            && isfinite(evidence) && isfinite(thresholdLogit)
            && isfinite(temperature) && temperature > 0;
        const float posterior = nodeValid[i] ? sigmoid(logit(result->prior[i]) + evidence) : result->prior[i];
        result->logLikelihoodRatio[i] = nodeValid[i] ? evidence : 0;
        result->posterior[i] = result->confidence[i] = posterior;
        result->threshold[i] = nodeValid[i] ? sigmoid(thresholdLogit) : 1.0f;
        result->softEligibility[i] = nodeValid[i]
            ? sigmoid((posterior - result->threshold[i]) / temperature) : 0;
        state->previousPosterior[i] = posterior;
    }
    __syncthreads();
    if (i == 0) {
        result->valid = 1; result->eligibleMask = result->selectedMask = 0;
        result->requestedK = result->selectedCount = 0;
        for (int j = 0; j < VoteCount; ++j) {
            result->valid &= nodeValid[j];
            result->selectedIndices[j] = -1;
            if (nodeValid[j] && result->confidence[j] > result->threshold[j]) result->eligibleMask |= 1u << j;
        }
        float features[DecisionKFeatures]{};
        for (int d = 0; d < DecisionWidth; ++d) {
            for (int j = 0; j < VoteCount; ++j) features[d] += hidden[j][d];
            features[d] /= VoteCount;
        }
        for (int j = 0; j < VoteCount; ++j) {
            features[DecisionWidth + j] = result->posterior[j];
            features[DecisionWidth + VoteCount + j] = result->softEligibility[j];
        }
        for (int d = 0; d < DecisionKFeatures; ++d) result->kFeatures[d] = features[d];
        const int activeK=isfinite(p->enabledKChoices) ? static_cast<int>(fminf(DecisionKChoices,fmaxf(1,p->enabledKChoices))):1;
        result->valid &= isfinite(p->enabledKChoices);
        float logits[DecisionKChoices], maximum = -INFINITY;
        for (int k = 0; k < DecisionKChoices; ++k) {
            float x = p->kBias[k];
            for (int d = 0; d < DecisionKFeatures; ++d) x += features[d] * p->kWeight[k][d];
            if(result->positionValid) for(int f=0;f<PositionFeatures;++f)
                x+=adapter->k[k][f]*result->positionFeatures[f];
            x+=minutes->kLogits[k];
            logits[k] = k<activeK ? x:-INFINITY;
            if (!isfinite(x)) result->valid = 0;
            if (k<activeK && (k == 0 || x > maximum)) { maximum = x; result->requestedK = k; }
        }
        float denominator = 0;
        for (int k = 0; k < DecisionKChoices; ++k) {
            result->kProbability[k] = expf(logits[k] - maximum); denominator += result->kProbability[k];
        }
        for (int k = 0; k < DecisionKChoices; ++k)
            result->kProbability[k] = result->valid ? result->kProbability[k] / denominator : 0;
        if (!result->valid) { result->eligibleMask = 0; result->requestedK = 0; }
        std::uint32_t remaining = result->eligibleMask;
        while (result->selectedCount < result->requestedK && remaining) {
            int best = -1;
            for (int j = 0; j < VoteCount; ++j)
                if ((remaining & (1u << j)) && (best < 0 || result->posterior[j] > result->posterior[best])) best = j;
            result->selectedMask |= 1u << best;
            result->selectedIndices[result->selectedCount++] = best;
            remaining &= ~(1u << best);
            for (int j = 0; j < VoteCount; ++j)
                if ((constraints->conflictMask[best] & (1u << j))
                        || (constraints->conflictMask[j] & (1u << best))) remaining &= ~(1u << j);
        }
        float dx = (result->selectedMask & actionBit(Action::MouseX)) ? motion->mouseDx : 0;
        float dy = (result->selectedMask & actionBit(Action::MouseY)) ? motion->mouseDy : 0;
        float wheel = (result->selectedMask & actionBit(Action::Wheel)) ? motion->wheelSteps : 0;
        if (!isfinite(dx) || !isfinite(dy) || !isfinite(wheel)) {
            result->valid = 0; result->selectedMask = 0; result->selectedCount = result->requestedK = 0;
            for (int j = 0; j < VoteCount; ++j) result->selectedIndices[j] = -1;
            dx = dy = wheel = 0;
        }
        result->command = ControlCommand{frameId, nanos, result->selectedMask,
            static_cast<std::uint32_t>(result->valid), dx, dy, wheel};
        result->status=DecisionStatus{frameId,result->eligibleMask,result->requestedK,result->selectedCount,result->valid,
            result->positionValid ? positionState->sampleNanos:0, result->positionValid ? positionState->epoch:0,
            adapter->steps, result->positionValid ? positionState->previous[0]:0,
            result->positionValid ? positionState->previous[1]:0, result->positionValid ? positionState->previous[2]:0,
            result->positionValid,minutes->elapsedMinutes,minuteParameters->steps,
            static_cast<std::uint32_t>(minutes->buckets),minuteParameters->motion.steps,
            result->posterior[5],result->posterior[6],result->threshold[5],result->threshold[6]};
        state->started = 1; state->lastCaptureNanos = nanos;
    }
}
// Bayesian probabilities condition the displacement value as well as action selection.
// A learned residual preserves old JEPA rates at zero initialization; there is no
// unconditional multiplication by confidence that would shrink every mouse movement.
__global__ void fuseBayesianMotion(DecisionOutput* decision,JepaOutput* jepa,const JepaConfig* cfg,
        const BayesianMotionPersistent* p,BayesianMotionCache* cache,bool frozen) {
    cache->ready=0;cache->frameId=decision->frameId;cache->learningFrozen=frozen;
    if(!decision->valid || !jepa->valid || !jepa->pairReady || decision->frameId!=jepa->frameId
            || !isfinite(cfg->maxMouseUnitsPerSecond) || cfg->maxMouseUnitsPerSecond<=0)return;
    cache->scale=cfg->maxMouseUnitsPerSecond;
    for(int k=0;k<VoteCount;++k) {
        cache->features[k]=2*decision->posterior[k]-1;
        cache->features[VoteCount+k]=2*decision->prior[k]-1;
    }
    cache->features[VoteCount*2]=jepa->headRateX/cache->scale;
    cache->features[VoteCount*2+1]=jepa->headRateY/cache->scale;
    cache->features[BayesianMotionFeatures-1]=1;
    float rates[2]={jepa->headRateX,jepa->headRateY};
    for(int axis=0;axis<2;++axis) {
        float residual=0;
        for(int f=0;f<BayesianMotionFeatures;++f)residual+=p->weights[axis][f]*cache->features[f];
        cache->predicted[axis]=rates[axis]/cache->scale+residual;
        rates[axis]=fminf(cache->scale,fmaxf(-cache->scale,rates[axis]+cache->scale*residual));
        if(!isfinite(residual) || !isfinite(cache->predicted[axis])) {
            decision->command.mouseDx=decision->command.mouseDy=0;
            jepa->mouseDx=jepa->mouseDy=0;return;
        }
    }
    cache->ready=1;
    const float dt=fminf(jepa->deltaSeconds,cfg->maxMotionDtSeconds);
    const float dx=fminf(cfg->maxMouseDelta,fmaxf(-cfg->maxMouseDelta,rates[0]*dt));
    const float dy=fminf(cfg->maxMouseDelta,fmaxf(-cfg->maxMouseDelta,rates[1]*dt));
    jepa->suggestedDx=dx;jepa->suggestedDy=dy;
    if(!jepa->motionActive)return; // Recommendations remain observable without injecting or learning.
    jepa->rateX=rates[0];jepa->rateY=rates[1];jepa->mouseDx=dx;jepa->mouseDy=dy;
    decision->command.mouseDx=(decision->selectedMask&actionBit(Action::MouseX)) ? dx:0;
    decision->command.mouseDy=(decision->selectedMask&actionBit(Action::MouseY)) ? dy:0;
}
__global__ void trainBayesianMotion(BayesianMotionPersistent* p,const BayesianMotionCache* cache,
        const DecisionOutput* decision,const int* active,const float* rateX,const float* rateY) {
    if(!*active || !cache->ready || cache->learningFrozen || !decision->valid
            || cache->frameId!=decision->frameId || !isfinite(*rateX) || !isfinite(*rateY)
            || fabsf(*rateX)>cache->scale || fabsf(*rateY)>cache->scale)return;
    __shared__ float gradients[2],bias1,bias2;
    if(threadIdx.x==0) {
        const float step=static_cast<float>(++p->steps);
        bias1=1-powf(.9f,step);bias2=1-powf(.999f,step);
        gradients[0]=.5f*(cache->predicted[0]-*rateX/cache->scale);
        gradients[1]=.5f*(cache->predicted[1]-*rateY/cache->scale);
    }
    __syncthreads();
    for(int at=threadIdx.x;at<2*BayesianMotionFeatures;at+=blockDim.x) {
        const int axis=at/BayesianMotionFeatures,f=at%BayesianMotionFeatures;
        const float g=fminf(1,fmaxf(-1,gradients[axis]*cache->features[f]));
        const float m=p->first[axis][f]=.9f*p->first[axis][f]+.1f*g;
        const float v=p->second[axis][f]=.999f*p->second[axis][f]+.001f*g*g;
        p->weights[axis][f]=p->weights[axis][f]*(f==BayesianMotionFeatures-1 ? 1:.999997f)
            -.0003f*(m/bias1)/(sqrtf(v/bias2)+1e-8f);
    }
}
} // namespace

struct VabpnnDecisionModel {
    vabpnn::DecisionParameters* parameters = nullptr;
    vabpnn::DecisionConstraints* constraints = nullptr;
    DecisionState* state = nullptr;
    PositionState* positionState = nullptr;
    vabpnn::DecisionPositionAdapter* position = nullptr;
    vabpnn::DecisionOutput* output = nullptr;
    vabpnn::MotionProposal* motion = nullptr;
    VabpnnMinuteModel* minutes = nullptr;
    VabpnnDenseBlockModel* dense = nullptr;
    BayesianMotionCache* bayesianMotion = nullptr;
    cudaEvent_t initialized = nullptr;
};

VabpnnDecisionModel* vabpnnCreateDecisionModel(cudaStream_t stream) {
    auto* model = new VabpnnDecisionModel();
    try {
        model->minutes=vabpnnCreateMinuteModel(stream);
        model->dense=vabpnnCreateDenseBlockModel(stream);
        check(cudaMalloc(reinterpret_cast<void**>(&model->bayesianMotion),sizeof(BayesianMotionCache)),"allocate Bayesian mouse cache");
        check(cudaMemsetAsync(model->bayesianMotion,0,sizeof(BayesianMotionCache),stream),"zero Bayesian mouse cache");
        check(cudaMalloc(reinterpret_cast<void**>(&model->parameters), sizeof(vabpnn::DecisionParameters)), "allocate decision parameters");
        check(cudaMalloc(reinterpret_cast<void**>(&model->constraints), sizeof(vabpnn::DecisionConstraints)), "allocate action constraints");
        check(cudaMalloc(reinterpret_cast<void**>(&model->state), sizeof(DecisionState)), "allocate temporal state");
        check(cudaMalloc(reinterpret_cast<void**>(&model->output), sizeof(vabpnn::DecisionOutput)), "allocate decision output");
        check(cudaMalloc(reinterpret_cast<void**>(&model->motion), sizeof(vabpnn::MotionProposal)), "allocate motion proposal");
        check(cudaMalloc(reinterpret_cast<void**>(&model->positionState), sizeof(PositionState)), "allocate position history");
        check(cudaMalloc(reinterpret_cast<void**>(&model->position), sizeof(vabpnn::DecisionPositionAdapter)), "allocate position adapter");
        check(cudaMemsetAsync(model->positionState, 0, sizeof(PositionState), stream), "zero position history");
        check(cudaMemsetAsync(model->position, 0, sizeof(vabpnn::DecisionPositionAdapter), stream), "zero neutral position adapter");
        check(cudaEventCreateWithFlags(&model->initialized, cudaEventDisableTiming), "create decision ready event");
        check(cudaMemsetAsync(model->parameters, 0, sizeof(vabpnn::DecisionParameters), stream), "zero decision heads");
        check(cudaMemsetAsync(model->constraints, 0, sizeof(vabpnn::DecisionConstraints), stream), "zero action constraints");
        check(cudaMemsetAsync(model->state, 0, sizeof(DecisionState), stream), "zero temporal state");
        check(cudaMemsetAsync(model->output, 0, sizeof(vabpnn::DecisionOutput), stream), "zero decision output");
        check(cudaMemsetAsync(model->motion, 0, sizeof(vabpnn::MotionProposal), stream), "zero motion proposal");
        initializeParameters<<<1, 1, 0, stream>>>(model->parameters, model->constraints);
        check(cudaGetLastError(), "initialize decision parameters");
        check(cudaEventRecord(model->initialized, stream), "record decision initialization");
        return model;
    } catch (...) { vabpnnDestroyDecisionModel(model); throw; }
}
void vabpnnDestroyDecisionModel(VabpnnDecisionModel* model) noexcept {
    if (!model) return;
    // Caller drains every consumer/update stream first; this covers initialization only.
    if (model->initialized) cudaEventSynchronize(model->initialized);
    vabpnnDestroyDenseBlockModel(model->dense);
    vabpnnDestroyMinuteModel(model->minutes);
    if(model->bayesianMotion)cudaFree(model->bayesianMotion);
    if (model->parameters) cudaFree(model->parameters);
    if (model->constraints) cudaFree(model->constraints);
    if (model->state) cudaFree(model->state);
    if (model->output) cudaFree(model->output);
    if (model->motion) cudaFree(model->motion);
    if (model->positionState) cudaFree(model->positionState);
    if (model->position) cudaFree(model->position);
    if (model->initialized) cudaEventDestroy(model->initialized);
    delete model;
}
std::uint64_t vabpnnDecisionModelBytes(const VabpnnDecisionModel* model) noexcept {
    return model ? DecisionBytes+vabpnnMinuteModelBytes(model->minutes)+vabpnnDenseBlockModelBytes(model->dense) : 0;
}
vabpnn::DeviceDecisionView vabpnnDecisionModelView(const VabpnnDecisionModel* model) {
    if (!model) throw std::invalid_argument("Decision model is not initialized");
    return {model->parameters, model->constraints, model->output, &model->output->command, model->position,
        vabpnnMinuteModelView(model->minutes)};
}
vabpnn::DecisionParameters* vabpnnMutableDecisionParameters(VabpnnDecisionModel* model, cudaStream_t stream) {
    if (!model) throw std::invalid_argument("Decision model is not initialized");
    check(cudaStreamWaitEvent(stream, model->initialized, 0), "wait before decision parameter update");
    return model->parameters;
}
vabpnn::DecisionConstraints* vabpnnMutableDecisionConstraints(VabpnnDecisionModel* model, cudaStream_t stream) {
    if (!model) throw std::invalid_argument("Decision model is not initialized");
    check(cudaStreamWaitEvent(stream, model->initialized, 0), "wait before action constraint update");
    return model->constraints;
}
vabpnn::DecisionPositionAdapter* vabpnnMutablePositionAdapter(VabpnnDecisionModel* model, cudaStream_t stream) {
    if(!model) throw std::invalid_argument("Decision model is not initialized");
    check(cudaStreamWaitEvent(stream,model->initialized,0),"wait before position adapter update");
    return model->position;
}
std::size_t vabpnnDecisionPositionCheckpointParts(VabpnnDecisionModel* model,vabpnn::ModelPart* parts) {
    parts[0]={model->position,sizeof(vabpnn::DecisionPositionAdapter)};
    return 1;
}
vabpnn::DecisionMinutePersistent* vabpnnMutableMinuteTimeline(VabpnnDecisionModel* model,cudaStream_t stream) {
    if(!model)throw std::invalid_argument("Decision model is not initialized");
    return vabpnnMutableMinuteModel(model->minutes,stream);
}
vabpnn::ModelPart vabpnnDecisionMinuteCheckpointPart(VabpnnDecisionModel* model) {
    return vabpnnMinuteModelCheckpointPart(model->minutes);
}
void vabpnnTrainMinuteDecision(VabpnnDecisionModel* model,const int* active,const std::uint32_t* label,cudaStream_t stream) {
    vabpnnTrainMinuteTimeline(model->minutes,model->parameters,model->output,active,label,stream);
}
void vabpnnTrainBayesianMotion(VabpnnDecisionModel* model,const int* active,
        const float* rateX,const float* rateY,cudaStream_t stream) {
    auto* p=vabpnnMutableMinuteTimeline(model,stream);
    trainBayesianMotion<<<1,128,0,stream>>>(&p->motion,model->bayesianMotion,model->output,active,rateX,rateY);
    check(cudaGetLastError(),"learn cached Bayesian JEPA displacement");
}
void vabpnnLaunchTemporalDecision(const VabpnnGpuFrame& frame,
                                  const std::int32_t* votes, VabpnnDecisionModel* model) {
    if (!model || !votes) throw std::invalid_argument("Invalid decision model or device vote vector");
    check(cudaStreamWaitEvent(frame.stream, model->initialized, 0), "wait for decision model");
    vabpnnLaunchMinuteTimeline(frame,votes,model->minutes);
    const auto minutes=vabpnnMinuteModelView(model->minutes);
    if(!frame.neuronModel)throw std::invalid_argument("Comparison requires real 7680 block votes");
    densePosition<<<1,1,0,frame.stream>>>(frame.position,frame.captureNanos,model->positionState,model->output);
    vabpnnLaunchDenseBlocks(model->dense,frame,vabpnnNeuronModelView(frame.neuronModel).blockVotes,
        model->output->positionFeatures);
    const auto dense=vabpnnDenseBlockModelView(model->dense);
    decide<<<1, 32, 0, frame.stream>>>(votes, model->parameters, model->constraints,
                                      model->motion, model->state, model->output, frame.frameId, frame.captureNanos,
                                      frame.position,model->positionState,model->position,minutes.output,minutes.persistent,dense.output);
    if(frame.jepaModel) {
        const auto jepa=vabpnnJepaModelView(frame.jepaModel);
        fuseBayesianMotion<<<1,1,0,frame.stream>>>(model->output,const_cast<JepaOutput*>(jepa.output),jepa.config,
            &minutes.persistent->motion,model->bayesianMotion,frame.evaluationMode);
    }
    check(cudaGetLastError(), "launch temporal Bayes decision and mouse conditioning");
}
vabpnn::MotionProposal* vabpnnMutableMotionProposal(VabpnnDecisionModel* model, cudaStream_t stream) {
    if (!model) throw std::invalid_argument("Decision model is not initialized");
    check(cudaStreamWaitEvent(stream, model->initialized, 0), "wait before motion proposal");
    return model->motion;
}
void vabpnnClearMotionProposal(VabpnnDecisionModel* model, cudaStream_t stream) {
    auto* motion = vabpnnMutableMotionProposal(model, stream);
    check(cudaMemsetAsync(motion, 0, sizeof(vabpnn::MotionProposal), stream), "clear current motion proposal");
}
void vabpnnResetDecisionState(VabpnnDecisionModel* model, cudaStream_t stream) {
    if (!model) throw std::invalid_argument("Decision model is not initialized");
    check(cudaStreamWaitEvent(stream, model->initialized, 0), "wait before decision reset");
    vabpnnClearMotionProposal(model, stream);
    vabpnnResetMinuteHistory(model->minutes,stream);
    vabpnnResetDenseBlocks(model->dense,stream);
    check(cudaMemsetAsync(model->bayesianMotion,0,sizeof(BayesianMotionCache),stream),"reset Bayesian mouse observation");
    check(cudaMemsetAsync(model->state, 0, sizeof(DecisionState), stream), "reset temporal state");
    check(cudaMemsetAsync(model->output, 0, sizeof(vabpnn::DecisionOutput), stream), "reset decision output");
    check(cudaMemsetAsync(model->positionState, 0, sizeof(PositionState), stream), "reset position history");
}



VabpnnDenseBlockModel* vabpnnDecisionDenseBlocks(VabpnnDecisionModel* model) {
    if(!model)throw std::invalid_argument("Decision comparison missing");return model->dense;
}
std::size_t vabpnnDecisionDenseCheckpointParts(VabpnnDecisionModel* model,vabpnn::ModelPart* parts) {
    return vabpnnDenseBlockCheckpointParts(model->dense,parts);
}
