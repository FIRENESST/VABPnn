#include "OnlineLearning.h"
#include "FrameOperator.h"
#include "SpikeConnections.h"
#include <cmath>
#include <stdexcept>
#include <string>
namespace {
using namespace vabpnn;
constexpr int WheelFeatures = DecisionWidth + VoteCount + 1;
struct Persistent {
    DecisionParameters first, second;
    float wheel[WheelFeatures], wheelFirst[WheelFeatures], wheelSecond[WheelFeatures];
    OnlineOutput output;
};
struct Cache {
    DemoSample previous;
    std::uint64_t nanos, frame;
    int ready, policyActive, pixelActive, motionActive;
    float dt, rateX, rateY, wheelRate;
    std::uint32_t label;
};
void check(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(code));
}
__device__ float sigmoid(float x) { return 1.0f/(1.0f+expf(-x)); }
__device__ bool conflict(std::uint32_t mask) {
    return ((mask & (1u<<7)) && (mask & ((1u<<8)|(1u<<11))))
        || ((mask & (1u<<8)) && (mask & (1u<<11)))
        || ((mask & (1u<<9)) && (mask & (1u<<10)));
}
__global__ void prepare(const DemoSample* now, Cache* c, Persistent* state,
                        const JepaOutput* jepa, const JepaConfig* config, const DecisionOutput* decision,
                        JepaMotionTarget* target, std::uint64_t id, std::uint64_t nanos, int sameSize) {
    c->policyActive = c->pixelActive = c->motionActive = 0;
    *target = JepaMotionTarget{};
    const auto& previous = c->previous;
    bool pair = c->ready && sameSize && now->active && previous.active
        && !now->modelControl && !previous.modelControl && now->epoch == previous.epoch
        && id > c->frame && nanos > c->nanos;
    const double dt = pair ? (nanos-c->nanos)*1e-9 : 0;
    pair = pair && dt >= 0.001 && dt <= 0.25;
    if (!pair) { if (c->ready) ++state->output.rejected; return; }
    const double dx = now->mouseX-previous.mouseX, dy = now->mouseY-previous.mouseY;
    const double wheel = now->wheel-previous.wheel;
    if (!isfinite(dx) || !isfinite(dy) || !isfinite(wheel)) { ++state->output.rejected; return; }
    // Keyboard holds and physical left/right holds survive between edge events.
    std::uint32_t label = now->heldKeys & 0x119f;
    const int keyIndex[6] = {0,1,2,3,4,12};
    for (int k=0;k<6;++k) {
        if (now->keyHeldNanos[k] < previous.keyHeldNanos[k]) { ++state->output.rejected; return; }
        // Any meaningful hold within the interval also teaches a short keyboard pulse.
        if ((now->keyHeldNanos[k]-previous.keyHeldNanos[k])*1e-9 > dt*0.1) label |= 1u<<keyIndex[k];
    }
    if (fabs(dx)>0.01) label |= 1u<<5;
    if (fabs(dy)>0.01) label |= 1u<<6;
    if (fabs(wheel)>0.001) label |= 1u<<11;
    for (int k=0;k<4;++k) {
        if (now->edge[k] < previous.edge[k]) { ++state->output.rejected; return; }
        if (now->edge[k] != previous.edge[k]) label |= 1u<<(7+k);
    }
    c->dt = static_cast<float>(dt); c->label = label;
    c->rateX = static_cast<float>(dx/dt); c->rateY = static_cast<float>(dy/dt);
    c->wheelRate = fminf(20.0f,fmaxf(-20.0f,static_cast<float>(wheel/dt)));
    c->policyActive = !conflict(label) && decision->valid && decision->frameId == c->frame;
    if (c->policyActive) {
        ++state->output.examples; ++state->output.decisionSteps;
        state->output.lastLabel = label; state->output.valid = 1;
        c->pixelActive = state->output.examples % 4 == 0;
        if (c->pixelActive) ++state->output.pixelSteps;
    } else ++state->output.rejected;
    c->motionActive = jepa->valid && jepa->pairReady && jepa->frameId == c->frame
        && now->pointerMode == previous.pointerMode
        && fabsf(c->rateX)<=config->maxMouseUnitsPerSecond
        && fabsf(c->rateY)<=config->maxMouseUnitsPerSecond;
    *target = JepaMotionTarget{c->frame,c->rateX,c->rateY,static_cast<std::uint32_t>(c->motionActive)};
}
__device__ void adam(float* parameter, float* first, float* second, float gradient,
                     float bias1, float bias2, bool decay) {
    gradient = fminf(1.0f,fmaxf(-1.0f,gradient));
    *first = 0.9f*(*first)+0.1f*gradient;
    *second = 0.999f*(*second)+0.001f*gradient*gradient;
    const float m = *first/bias1;
    const float v = *second/bias2;
    *parameter = *parameter*(decay ? 0.999997f:1.0f)-0.0003f*m/(sqrtf(v)+1e-8f);
}
__device__ float wheelFeature(const DecisionOutput* out,int d) {
    return d<DecisionWidth ? out->kFeatures[d]:
        d<DecisionWidth+VoteCount ? out->normalizedVotes[d-DecisionWidth]:1.0f;
}
__global__ void trainPolicy(DecisionParameters* p, DecisionPositionAdapter* position,
                            const DecisionOutput* out, Cache* c, Persistent* state) {
    if (!c->policyActive || !out->valid || out->frameId!=c->frame) return;
    // Each thread updates distinct parameters. All gradients and the old wheel
    // prediction are computed before any parameter is written.
    constexpr int ActionStride=3+2*DecisionWidth, KStride=1+DecisionKFeatures;
    constexpr int ActionParameters=VoteCount*ActionStride, KParameters=DecisionKChoices*KStride;
    constexpr int ExistingParameters=ActionParameters+KParameters+WheelFeatures;
    constexpr int ParameterCount=ExistingParameters+(VoteCount+DecisionKChoices)*PositionFeatures;
    __shared__ float evidence[VoteCount],threshold[VoteCount],bias1,bias2,wheelGradient,positionBias1,positionBias2;
    __shared__ int trainPosition;
    const int tid=threadIdx.x;
    if(tid<VoteCount) {
        const float y=(c->label>>tid)&1u;
        const float prob=fminf(1-1e-6f,fmaxf(1e-6f,out->posterior[tid]));
        evidence[tid]=(prob-y)/VoteCount;
        threshold[tid]=(out->threshold[tid]-(y ? 0.25f:0.75f))/VoteCount;
    }
    if(tid==0) {
        const float step=static_cast<float>(state->output.decisionSteps);
        bias1=1-powf(0.9f,step); bias2=1-powf(0.999f,step);
        trainPosition=out->positionValid;
        if(trainPosition) {
            const float positionStep=static_cast<float>(++position->steps);
            positionBias1=1-powf(0.9f,positionStep); positionBias2=1-powf(0.999f,positionStep);
        }
        // Keep the original scalar accumulation order for both losses.
        float loss=0;
        for(int i=0;i<VoteCount;++i) {
            const float y=(c->label>>i)&1u;
            const float prob=fminf(1-1e-6f,fmaxf(1e-6f,out->posterior[i]));
            loss+=-y*logf(prob)-(1-y)*logf(1-prob);
        }
        state->output.decisionLoss=loss/VoteCount-logf(fmaxf(1e-6f,out->kProbability[__popc(c->label)]));
        float z=0;
        for(int d=0;d<WheelFeatures;++d) z+=state->wheel[d]*wheelFeature(out,d);
        const float pred=tanhf(z),error=pred-c->wheelRate/20.0f;
        wheelGradient=error*(1-pred*pred);
        state->output.wheelLoss=0.5f*error*error;
    }
    __syncthreads();
    auto& m=state->first; auto& v=state->second;
    for(int index=tid;index<ParameterCount;index+=blockDim.x) {
        if(index<ActionParameters) {
            const int i=index/ActionStride,at=index%ActionStride;
            if(at==0) adam(&p->logLikelihoodBias[i],&m.logLikelihoodBias[i],&v.logLikelihoodBias[i],
                           evidence[i],bias1,bias2,false);
            else if(at==1) adam(&p->rawVoteWeight[i],&m.rawVoteWeight[i],&v.rawVoteWeight[i],
                                evidence[i]*out->normalizedVotes[i],bias1,bias2,true);
            else if(at==2) adam(&p->thresholdBias[i],&m.thresholdBias[i],&v.thresholdBias[i],
                                threshold[i],bias1,bias2,false);
            else if(at<3+DecisionWidth) {
                const int d=at-3;
                adam(&p->logLikelihoodWeight[i][d],&m.logLikelihoodWeight[i][d],&v.logLikelihoodWeight[i][d],
                     evidence[i]*out->hidden[i][d],bias1,bias2,true);
            } else {
                const int d=at-3-DecisionWidth;
                adam(&p->thresholdWeight[i][d],&m.thresholdWeight[i][d],&v.thresholdWeight[i][d],
                     threshold[i]*out->hidden[i][d],bias1,bias2,true);
            }
        } else if(index<ActionParameters+KParameters) {
            const int at=index-ActionParameters,k=at/KStride,d=at%KStride-1;
            const float gradient=out->kProbability[k]-(k==__popc(c->label));
            if(d<0) adam(&p->kBias[k],&m.kBias[k],&v.kBias[k],gradient,bias1,bias2,false);
            else adam(&p->kWeight[k][d],&m.kWeight[k][d],&v.kWeight[k][d],
                      gradient*out->kFeatures[d],bias1,bias2,true);
        } else if(index<ExistingParameters) {
            const int d=index-ActionParameters-KParameters;
            adam(&state->wheel[d],&state->wheelFirst[d],&state->wheelSecond[d],
                  wheelGradient*wheelFeature(out,d),bias1,bias2,d!=WheelFeatures-1);
        } else if(trainPosition) {
            const int at=index-ExistingParameters,node=at/PositionFeatures,f=at%PositionFeatures;
            // Cached previous frame only: the intervening demonstration never leaks future XYZ.
            if(node<VoteCount)
                adam(&position->evidence[node][f],&position->evidenceFirst[node][f],&position->evidenceSecond[node][f],
                    evidence[node]*out->positionFeatures[f],positionBias1,positionBias2,f!=PositionFeatures-1);
            else {
                const int k=node-VoteCount;
                const float gradient=out->kProbability[k]-(k==__popc(c->label));
                adam(&position->k[k][f],&position->kFirst[k][f],&position->kSecond[k][f],
                    gradient*out->positionFeatures[f],positionBias1,positionBias2,f!=PositionFeatures-1);
            }
        }
    }
}
__global__ void trainPixels(const unsigned char* pixels, float* master, TernaryWeight* weights, const Cache* c,
                            const BlockControl* controls) {
    if (!c->pixelActive) return;
    const int pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if (pixel>=NeuronCount) return;
    const float rgb[3]={pixels[pixel*3]/255.0f,pixels[pixel*3+1]/255.0f,pixels[pixel*3+2]/255.0f};
    const int tile=pixel/NeuronWidth/16*120+pixel%NeuronWidth/16;
    const float learningRate=.02f*(controls ? controls[tile].learningScale:1.0f);
    for(int k=0;k<VoteCount;++k) {
        float z=0;
        for(int channel=0;channel<3;++channel) z+=rgb[channel]*master[weightIndex(pixel,channel,k)];
        const float error=sigmoid(z)-static_cast<float>((c->label>>k)&1u);
        for(int channel=0;channel<3;++channel) {
            const auto at=weightIndex(pixel,channel,k);
            // Local supervised pixel loss, not a claim of gradient through the discrete sign.
            const float w=fminf(2.0f,fmaxf(-2.0f,master[at]-learningRate*error*rgb[channel]));
            master[at]=w; weights[at]=static_cast<TernaryWeight>((w>0.05f)-(w< -0.05f));
        }
    }
}
__global__ void cachePixels(VabpnnGpuFrame frame, unsigned char* pixels) {
    const int pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if(pixel>=NeuronCount) return;
    const int x=pixel%NeuronWidth,y=pixel/NeuronWidth;
    const int sx=static_cast<std::size_t>(2*x+1)*frame.width/(2*NeuronWidth);
    const int sy=static_cast<std::size_t>(2*y+1)*frame.height/(2*NeuronHeight);
    const auto rgba=frame.rgba+static_cast<std::size_t>(frame.height-1-sy)*frame.rgbaPitch+sx*4;
    for(int channel=0;channel<3;++channel) pixels[pixel*3+channel]=rgba[channel];
}
__global__ void finishCache(Cache* c, const DemoSample* sample, std::uint64_t nanos, std::uint64_t frame) {
    c->previous=*sample; c->nanos=nanos; c->frame=frame; c->ready=1;
}
__global__ void configureInference(const DemoSample* sample, JepaConfig* config) {
    config->motionEnabled=sample->active && sample->modelControl;
}
__global__ void wheelInference(const Persistent* state, const DemoSample* sample,
                               const std::int32_t* votes, const DecisionOutput* previous,
                               const JepaOutput* jepa, MotionProposal* proposal) {
    proposal->wheelSteps=0;
    if (!sample->active || !sample->modelControl || !jepa->pairReady) return;
    float z=state->wheel[WheelFeatures-1];
    // Use the previous graph context: never an output from a not-yet-launched current decision.
    for(int d=0;d<DecisionWidth;++d) z+=state->wheel[d]*previous->kFeatures[d];
    for(int k=0;k<VoteCount;++k)
        z+=state->wheel[DecisionWidth+k]*(static_cast<float>(votes[k])/NeuronCount);
    const float delta=tanhf(z)*20.0f*fminf(0.1f,jepa->deltaSeconds);
    proposal->wheelSteps=isfinite(delta)?delta:0;
}
}
struct VabpnnOnlineModel {
    vabpnn::DemoSample* sample=nullptr;
    Cache* cache=nullptr;
    Persistent* persistent=nullptr;
    float* master=nullptr;
    unsigned char* pixels=nullptr;
    int width=0,height=0;
    cudaEvent_t initialized=nullptr;
};
VabpnnOnlineModel* vabpnnCreateOnlineModel(cudaStream_t stream) {
    auto* m=new VabpnnOnlineModel();
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&m->sample),sizeof(DemoSample)),"allocate real input");
        check(cudaMalloc(reinterpret_cast<void**>(&m->cache),sizeof(Cache)),"allocate teaching cache");
        check(cudaMalloc(reinterpret_cast<void**>(&m->persistent),sizeof(Persistent)),"allocate online optimizer");
        check(cudaMalloc(reinterpret_cast<void**>(&m->master),ParameterCount*sizeof(float)),"allocate latent ternary weights");
        check(cudaMalloc(reinterpret_cast<void**>(&m->pixels),NeuronCount*3),"allocate prior observation on GPU");
        check(cudaMemsetAsync(m->sample,0,sizeof(DemoSample),stream),"zero real input");
        check(cudaMemsetAsync(m->cache,0,sizeof(Cache),stream),"zero teaching cache");
        check(cudaMemsetAsync(m->persistent,0,sizeof(Persistent),stream),"zero online optimizer");
        check(cudaMemsetAsync(m->master,0,ParameterCount*sizeof(float),stream),"zero latent weights");
        check(cudaEventCreateWithFlags(&m->initialized,cudaEventDisableTiming),"create online ready event");
        check(cudaEventRecord(m->initialized,stream),"record online initialization");
        return m;
    } catch (...) { vabpnnDestroyOnlineModel(m); throw; }
}
void vabpnnDestroyOnlineModel(VabpnnOnlineModel* m) noexcept {
    if(!m) return;
    if(m->initialized) cudaEventSynchronize(m->initialized);
    if(m->sample) cudaFree(m->sample);
    if(m->cache) cudaFree(m->cache);
    if(m->persistent) cudaFree(m->persistent);
    if(m->master) cudaFree(m->master);
    if(m->pixels) cudaFree(m->pixels);
    if(m->initialized) cudaEventDestroy(m->initialized);
    delete m;
}
std::uint64_t vabpnnOnlineModelBytes(const VabpnnOnlineModel* m) noexcept {
    return m ? sizeof(DemoSample)+sizeof(Cache)+sizeof(Persistent)+ParameterCount*sizeof(float)+NeuronCount*3 : 0;
}
const vabpnn::OnlineOutput* vabpnnOnlineOutput(const VabpnnOnlineModel* m) { return &m->persistent->output; }
const vabpnn::DemoSample* vabpnnOnlineDemoSample(const VabpnnOnlineModel* m) { return m ? m->sample:nullptr; }
void vabpnnSetDemoSample(VabpnnOnlineModel* m,const vabpnn::DemoSample* sample,cudaStream_t stream) {
    check(cudaStreamWaitEvent(stream,m->initialized,0),"wait for online model");
    check(cudaMemcpyAsync(m->sample,sample,sizeof(DemoSample),cudaMemcpyHostToDevice,stream),"upload real input metadata");
}
void vabpnnLearnPreviousDemo(const VabpnnGpuFrame& frame) {
    auto* m=frame.onlineModel;
    if(!m || !frame.jepaModel || !frame.decisionModel || !frame.neuronModel) return;
    auto j=vabpnnJepaModelView(frame.jepaModel);
    auto* target=vabpnnMutableJepaMotionTarget(frame.jepaModel,frame.stream);
    prepare<<<1,1,0,frame.stream>>>(m->sample,m->cache,m->persistent,j.output,j.config,
        vabpnnDecisionModelView(frame.decisionModel).output,target,frame.frameId,frame.captureNanos,frame.width==m->width && frame.height==m->height);
    vabpnnObserveSpikeContribution(frame.spikeModel,vabpnnDecisionModelView(frame.decisionModel).output,
        vabpnnDecisionModelView(frame.decisionModel).parameters,&m->cache->policyActive,&m->cache->label,frame.stream);
    vabpnnLearnBlockControllers(frame,&m->cache->policyActive,&m->cache->label);
    // Generate controls BEFORE pixel learning: its learning-rate output affects the observation later scored.
    vabpnnPrepareBlockControllers(frame);
    trainPolicy<<<1,256,0,frame.stream>>>(vabpnnMutableDecisionParameters(frame.decisionModel,frame.stream),
        vabpnnMutablePositionAdapter(frame.decisionModel,frame.stream),
        vabpnnDecisionModelView(frame.decisionModel).output,m->cache,m->persistent);
    trainPixels<<<(NeuronCount+255)/256,256,0,frame.stream>>>(m->pixels,m->master,
        vabpnnMutableNeuronWeights(frame.neuronModel,frame.stream),m->cache,
        frame.spikeModel ? vabpnnSpikeModelView(frame.spikeModel).blockControls:nullptr);
    vabpnnTrainJepaMotion(frame.jepaModel,frame.stream);
    configureInference<<<1,1,0,frame.stream>>>(m->sample,vabpnnMutableJepaConfig(frame.jepaModel,frame.stream));
    check(cudaGetLastError(),"learn previous real-input demonstration");
}
void vabpnnCacheDemoObservation(const VabpnnGpuFrame& frame) {
    auto* m=frame.onlineModel;
    cachePixels<<<(NeuronCount+255)/256,256,0,frame.stream>>>(frame,m->pixels);
    finishCache<<<1,1,0,frame.stream>>>(m->cache,m->sample,frame.captureNanos,frame.frameId);
    m->width=frame.width; m->height=frame.height;
    check(cudaGetLastError(),"cache prior pixels only in GPU memory");
}
void vabpnnOnlineWheel(const VabpnnGpuFrame& frame,vabpnn::MotionProposal* proposal) {
    auto* m=frame.onlineModel;
    wheelInference<<<1,1,0,frame.stream>>>(m->persistent,m->sample,
        vabpnnNeuronModelView(frame.neuronModel).output,vabpnnDecisionModelView(frame.decisionModel).output,
        vabpnnJepaModelView(frame.jepaModel).output,proposal);
    check(cudaGetLastError(),"infer wheel displacement");
}
void vabpnnResetOnlineHistory(VabpnnOnlineModel* m,cudaStream_t stream) {
    check(cudaMemsetAsync(m->cache,0,sizeof(Cache),stream),"reset teaching history");
    m->width=m->height=0;
}
std::size_t vabpnnOnlineCheckpointParts(VabpnnOnlineModel* m,vabpnn::ModelPart* parts) {
    parts[0]={m->master,ParameterCount*sizeof(float)};
    parts[1]={m->persistent,sizeof(Persistent)};
    return 2;
}
