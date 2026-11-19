#include "FullSelfLearning.h"
#include "PnnJepa.h"
#include <cmath>
#include <stdexcept>
#include <vector>
namespace {
using namespace vabpnn;
struct BatchState {
 float gradient[FullParameters],credit,bias1[7],bias2[7];
 int support[7],visualCount,visual[4],active,pixelActive,motionActive;
 const RewardBatch* original;
 std::uint64_t blockBefore,memeBefore,prototypeBefore;
 const BlockRewardObservation* blockTraces[4];
};
struct Buffers {
 FullLearningPersistent* persistent;FullRewardObservation* cache;FullRewardRoots* roots;
 BatchState* batch;unsigned char* pixels;BlockRewardObservation* blocks;
 FullLearningStatus* status;std::uint64_t cached,dropped;
 float *spatialScore, *pnnPatchScores;
 unsigned char* pnnPulses;
};
void check(cudaError_t c,const char* what){if(c!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(c));}
__device__ float sig(float x){return 1/(1+expf(-x));}
__device__ float sp(float x){return x>20?x:log1pf(expf(x));}
__device__ unsigned hash(unsigned x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__device__ float normal(unsigned seed){float u=(static_cast<float>(hash(seed)&0xffffffu)+1)/16777217.f,v=(static_cast<float>(hash(seed+97)&0xffffffu)+.5f)/16777216.f;return sqrtf(-2*logf(u))*cosf(6.28318530718f*v);}
// Exact location score of a clipped Gaussian, including its two boundary probability masses.
__device__ float gaussian(float mean,float sigma,float limit,unsigned seed,float& sample,float& logProbability){
 float epsilon=normal(seed),raw=mean+sigma*epsilon;sample=fminf(limit,fmaxf(-limit,raw));
 if(raw>=limit){float z=(limit-mean)/sigma,tail=fmaxf(1e-30f,.5f*erfcf(z*.70710678118f));logProbability=logf(tail);return .3989422804f*expf(-.5f*z*z)/(sigma*tail);}
 if(raw<=-limit){float z=(-limit-mean)/sigma,tail=fmaxf(1e-30f,.5f*erfcf(-z*.70710678118f));logProbability=logf(tail);return -.3989422804f*expf(-.5f*z*z)/(sigma*tail);}
 logProbability=-.5f*epsilon*epsilon-logf(sigma)-.9189385332f;return (sample-mean)/(sigma*sigma);
}
__global__ void cacheRoots(Buffers b,VabpnnGpuFrame frame,RewardCache* reward,const DecisionParameters* p,
 DecisionOutput* out,DeviceJepaView jepa,const DecisionMinutePersistent* minute,const float* wheelFeatures,const OnlinePersistent* online,const DemoSample* demo,DeviceLongMemoryView memory){
 const int slot=frame.frameId%RewardCacheSlots;auto& c=b.cache[slot];c={};*b.roots={};
 if(!reward->proposal.valid || reward->proposal.frameId!=frame.frameId || !out->valid)return;
 c.proposal=reward->proposal;c.decision=*out;c.visual=frame.frameId%FullVisualEvery==0;
 c.motionValid=frame.onlineModel && frame.jepaModel && jepa.output->valid && jepa.output->pairReady && jepa.output->motionActive;
 const auto& cfg=*jepa.config;float dt=fminf(jepa.output->deltaSeconds,cfg.maxMotionDtSeconds);
 c.motionValid=c.motionValid && dt>0 && frame.width>0 && demo->pointerMode==1; // Java/JNI: 1 = grabbed world view, 0 = absolute GUI cursor.
 // Absolute GUI coordinates can clamp; only relative world-view samples train the continuous actor.
 // The host's execution acknowledgement additionally requires exact delivered deltas.
 const float head[2]={jepa.output->headRateX/cfg.maxMouseUnitsPerSecond,jepa.output->headRateY/cfg.maxMouseUnitsPerSecond};
 for(int f=0;f<BayesianMotionFeatures;++f)c.bayesFeatures[f]=f<VoteCount?2*out->posterior[f]-1:f<2*VoteCount?2*out->prior[f-VoteCount]-1:f==2*VoteCount?head[0]:f==2*VoteCount+1?head[1]:1;
 for(int a=0;a<2;++a){c.headRate[a]=head[a];float predicted=head[a];for(int f=0;f<BayesianMotionFeatures;++f)predicted+=minute->motion.weights[a][f]*c.bayesFeatures[f];
  c.meanDerivative[a]=(fabsf(predicted)<1 && fabsf(predicted*cfg.maxMouseUnitsPerSecond*dt)<cfg.maxMouseDelta)?cfg.maxMouseUnitsPerSecond*dt:0;
  float mean=a?jepa.output->mouseDy:jepa.output->mouseDx;
  if(c.motionValid && (out->selectedMask&(1u<<(5+a)))){float sample=0,logp=0;c.motionScore[a]=gaussian(mean,fmaxf(.05f,24*dt),cfg.maxMouseDelta,hash(static_cast<unsigned>(frame.frameId)^static_cast<unsigned>(frame.captureNanos)^static_cast<unsigned>(reward->proposal.epoch)^(a?731:337)),sample,logp);
   reward->proposal.logProbability+=logp;if(a)out->command.mouseDy=sample;else out->command.mouseDx=sample;}
 }
 for(int a=0;a<VoteCount;++a){float extra=0,direct=0;for(int axis=0;axis<2;++axis){float g=c.motionScore[axis]*c.meanDerivative[axis];extra+=2*g*minute->motion.weights[axis][a];direct+=2*g*minute->motion.weights[axis][VoteCount+a];}
  float de=extra*out->posterior[a]*(1-out->posterior[a]);int at=a*RewardEvidenceStride;
  reward->score[at+DecisionWidth]+=de;reward->score[at+DecisionWidth+1]+=de*out->normalizedVotes[a];for(int d=0;d<DecisionWidth;++d)reward->score[at+d]+=de*out->hidden[a][d];
  b.roots->evidence[a]=reward->score[at+DecisionWidth];b.roots->prior[a]=b.roots->evidence[a]+direct*out->prior[a]*(1-out->prior[a]);b.roots->vote[a]=b.roots->evidence[a]*p->rawVoteWeight[a];
 }
 for(int k=0;k<DecisionKChoices;++k)b.roots->k[k]=reward->score[RewardEvidenceParameters+k*RewardKStride+DecisionKFeatures];
 float temperature=sp(p->logGateTemperature),tempScore=0;
 for(int a=0;a<VoteCount;++a){float chain=0;for(int k=0;k<DecisionKChoices;++k)chain+=b.roots->k[k]*p->kWeight[k][DecisionWidth+VoteCount+a];
  float gate=out->softEligibility[a],threshold=out->threshold[a],base=chain*gate*(1-gate);
  b.roots->threshold[a]=-base*threshold*(1-threshold)/temperature;tempScore-=base*(out->posterior[a]-threshold)/(temperature*temperature)*sig(p->logGateTemperature);
  for(int d=0;d<DecisionWidth;++d){float g=b.roots->evidence[a]*p->logLikelihoodWeight[a][d]+b.roots->threshold[a]*p->thresholdWeight[a][d];for(int k=0;k<DecisionKChoices;++k)g+=b.roots->k[k]*p->kWeight[k][d]/VoteCount;b.roots->hidden[a][d]=g;}
 }
 b.roots->temperature=tempScore;
 for(int a=0;a<2;++a){float g=0;for(int axis=0;axis<2;++axis)g+=c.motionScore[axis]*c.meanDerivative[axis]*((axis==a?1.f:0.f)+minute->motion.weights[axis][2*VoteCount+a]);c.rawMotionScore[a]=g*(1-head[a]*head[a]);}
 for(int f=0;f<JepaMotionInput;++f)c.motionInput[f]=jepa.motionInput[f];
 for(int h=0;h<JepaHidden;++h){float value=jepa.motionHidden[h];c.motionHidden[h]=value;float g=0;for(int a=0;a<2;++a)g+=c.rawMotionScore[a]*jepa.parameters->motion.outputWeight[h][a];c.motionHiddenScore[h]=g*(1-value*value);}
 for(int a=0;a<VoteCount;++a)for(int h=0;h<JepaHidden;++h)b.roots->vote[a]+=c.motionHiddenScore[h]*jepa.parameters->motion.inputWeight[3*JepaMoments+a][h];
 for(int f=0;f<FullWheelParameters;++f)c.wheelFeatures[f]=wheelFeatures[f];
 if(out->selectedMask&actionBit(Action::Wheel)){float mean=out->command.wheelSteps,sample=0,logp=0;c.wheelScore=gaussian(mean,.08f,2,hash(static_cast<unsigned>(frame.frameId)^static_cast<unsigned>(frame.captureNanos)^919),sample,logp);out->command.wheelSteps=sample;reward->proposal.logProbability+=logp;
  float scale=20*dt,ratio=scale>0?mean/scale:0;c.wheelScore*=scale*(1-ratio*ratio);for(int a=0;a<VoteCount;++a)b.roots->vote[a]+=c.wheelScore*online->wheel[DecisionWidth+a];}
 c.roots=*b.roots;if(memory.trace && memory.trace->frame==frame.frameId)c.memory=*memory.trace;
 c.proposal=reward->proposal;c.decision.command=out->command;
 reward->proposal.expectedDx=out->command.mouseDx;reward->proposal.expectedDy=out->command.mouseDy;reward->proposal.expectedWheel=out->command.wheelSteps;c.proposal=reward->proposal;
}
__global__ void cachePixels(VabpnnGpuFrame frame,unsigned char* pixels){if(frame.frameId%FullVisualEvery)return;int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=NeuronCount)return;
 unsigned slot=(frame.frameId%RewardCacheSlots)/FullVisualEvery;auto* dest=pixels+static_cast<std::size_t>(slot)*NeuronCount*3;
 int x=pixel%NeuronWidth,y=pixel/NeuronWidth,sx=static_cast<std::size_t>(2*x+1)*frame.width/(2*NeuronWidth),sy=static_cast<std::size_t>(2*y+1)*frame.height/(2*NeuronHeight);
 const auto* src=frame.rgba+static_cast<std::size_t>(frame.height-1-sy)*frame.rgbaPitch+sx*4;for(int f=0;f<3;++f)dest[pixel*3+f]=src[f];}
// Two bits per signed spike, sampled with its actual RGB frame.
constexpr std::size_t PnnPulseBytes=NeuronCount/4;
constexpr int PnnPatchScoreCount=JepaTokens*JepaInput;
static_assert(NeuronCount%4==0);
__global__ void cachePnnPulses(const std::int8_t* pulse, unsigned char* packed, FullRewardObservation* observation) {
    const std::size_t at=blockIdx.x*blockDim.x+threadIdx.x;
    int any=0;
    if(at<PnnPulseBytes) {
        unsigned int bits=0;
        for(int i=0;i<4;++i) {
            const int value=pulse[at*4+i];
            bits|=static_cast<unsigned int>(value+1)<<(2*i);
            any|=value!=0;
        }
        packed[at]=static_cast<unsigned char>(bits);
    }
    if(__syncthreads_or(any) && threadIdx.x==0) atomicOr(&observation->pnnVisualSupport,1u);
}
__global__ void validatePnnPatchScores(const float* scores, FullRewardObservation* observation) {
    bool finite=true, nonzero=false;
    for(int at=threadIdx.x;at<PnnPatchScoreCount;at+=blockDim.x) {
        finite=finite && isfinite(scores[at]);
        nonzero=nonzero || scores[at]!=0;
    }
    const bool valid=__syncthreads_and(finite);
    const bool credit=__syncthreads_or(nonzero);
    if(threadIdx.x==0) {
        if(valid && credit) observation->pnnVisualSupport|=2u;
        if(!valid) observation->visual=0;
    }
}
__global__ void finalizeCache(Buffers b,const RewardCache* reward,std::uint64_t frame){auto& c=b.cache[frame%RewardCacheSlots];bool valid=c.proposal.valid;
 for(int i=0;i<FullDecisionParameters;++i)valid=valid&&isfinite(c.seconds[i]);for(int i=0;i<MinuteParameterCount;++i)valid=valid&&isfinite(c.minutes[i]);
 for(int i=0;i<VoteCount;++i)valid=valid&&isfinite(c.voteScore[i]);if(!valid){c.proposal.valid=0;++b.status->dropped;return;}c.proposal=reward->proposal;++b.status->cached;}
__global__ void prepareBatch(Buffers b,const RewardBatch* batch,int* accepted,bool pnnJepa){auto& s=*b.batch;s.active=0;s.visualCount=0;s.pixelActive=0;s.motionActive=0;s.credit=0;s.original=batch;for(int i=0;i<7;++i)s.support[i]=0;
 if(!*accepted || !batch->credit)return;
 for(unsigned i=0;i<batch->count;++i){const auto& c=b.cache[batch->frames[i]%RewardCacheSlots];if(!c.proposal.valid || c.proposal.frameId!=batch->frames[i] || c.proposal.epoch!=batch->epoch || c.proposal.mask!=batch->masks[i]){*accepted=0;return;}}
 s.credit=batch->credit;int seen=0;for(unsigned i=0;i<batch->count;++i){int slot=batch->frames[i]%RewardCacheSlots;if(!b.cache[slot].visual)continue;++seen;
  if(seen<=4)s.visual[s.visualCount++]=slot;else{unsigned at=hash(static_cast<unsigned>(batch->rewardId)^static_cast<unsigned>(batch->frames[i]))%seen;if(at<4)s.visual[at]=slot;}}
 for(int i=0;i<s.visualCount;++i) {
  const auto& observation=b.cache[s.visual[i]];
  if(pnnJepa) s.pixelActive|=observation.pnnVisualSupport==3;
  else for(int a=0;a<VoteCount;++a)if(observation.voteScore[a]!=0)s.pixelActive=1;
 }
 for(int i=0;i<s.visualCount;++i)s.blockTraces[i]=b.blocks+static_cast<std::size_t>(s.visual[i]/FullVisualEvery)*BlockCount;s.active=1;
}
__device__ int group(int at){return at<FullMinuteOffset?0:at<FullPositionOffset?1:at<FullBayesOffset?2:at<FullMemoryOffset?3:at<FullMotionOffset?4:at<FullWheelOffset?5:6;}
__device__ bool coreParameter(int at){
 #define IN(field) (at>=static_cast<int>(offsetof(DecisionParameters,field)/4) && at<static_cast<int>((offsetof(DecisionParameters,field)+sizeof(DecisionParameters::field))/4))
 return at==0 || IN(logLikelihoodWeight)||IN(logLikelihoodBias)||IN(rawVoteWeight)||IN(kWeight)||IN(kBias);
 #undef IN
}
__global__ void aggregate(Buffers b,const RewardBatch* batch,const LongMemoryParameters* memory,int* accepted){int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=FullParameters || !b.batch->active)return;float score=0,aux=0;
 int gr=group(at);for(unsigned i=0;i<batch->count;++i){const auto& c=b.cache[batch->frames[i]%RewardCacheSlots];
  if(gr==0){if(!coreParameter(at))score+=c.seconds[at];}
  else if(gr==1)score+=c.minutes[at-FullMinuteOffset];
  else if(gr==2 && c.decision.positionValid){int v=at-FullPositionOffset,node=v/PositionFeatures,f=v%PositionFeatures;score+=(node<VoteCount?c.roots.evidence[node]:c.roots.k[node-VoteCount])*c.decision.positionFeatures[f];}
  else if(gr==3){int v=at-FullBayesOffset,axis=v/BayesianMotionFeatures,f=v%BayesianMotionFeatures;score+=c.motionScore[axis]*c.meanDerivative[axis]*c.bayesFeatures[f];}
  else if(gr==4 && c.memory.ready && !c.memory.frozen){int v=at-FullMemoryOffset;const auto& t=c.memory;float read=0;for(int a=0;a<VoteCount;++a)read+=c.roots.evidence[a]*.5f*tanhf(memory->evidenceGain[a])*t.memoryEvidence[a];for(int k=0;k<DecisionKChoices;++k)read+=c.roots.k[k]*.5f*tanhf(memory->kGain[k])*t.memoryK[k];
   if(v<8)score+=t.support?read*t.readGate*(1-t.readGate)*t.readFeatures[v]:0;
   else if(v<16){float target=fminf(1,.2f+.6f*t.novelty+.2f*fminf(1,10*fabsf(batch->credit)));aux+=.02f*(t.writeGate-target)*t.writeFeatures[v-8];}
   else if(v<24){float target=fminf(1,.2f+.8f*fminf(1,10*fabsf(batch->credit)));aux+=t.support?.02f*(t.retainGate-target)*t.retainFeatures[v-16]:0;}
   else if(v<24+VoteCount){int a=v-24;float gain=tanhf(memory->evidenceGain[a]);score+=t.support?c.roots.evidence[a]*.5f*(1-gain*gain)*t.readGate*t.memoryEvidence[a]:0;}
   else{int k=v-24-VoteCount;float gain=tanhf(memory->kGain[k]);score+=t.support?c.roots.k[k]*.5f*(1-gain*gain)*t.readGate*t.memoryK[k]:0;}}
  else if(gr==5 && c.motionValid){int v=at-FullMotionOffset;constexpr int Input=JepaMotionInput*JepaHidden;
   if(v<Input)score+=c.motionInput[v/JepaHidden]*c.motionHiddenScore[v%JepaHidden];
   else if(v<Input+JepaHidden)score+=c.motionHiddenScore[v-Input];
   else if(v<Input+JepaHidden+JepaHidden*2){int t=v-Input-JepaHidden;score+=c.motionHidden[t/2]*c.rawMotionScore[t%2];}
   else score+=c.rawMotionScore[v-Input-JepaHidden-JepaHidden*2];}
  else if(gr==6)score+=c.wheelScore*c.wheelFeatures[at-FullWheelOffset];
 }
 float g=(-batch->credit*score+aux)/batch->count;b.batch->gradient[at]=g;
 if(!isfinite(g))atomicExch(accepted,0);
 unsigned mask=__activemask(),same=__match_any_sync(mask,gr),nonzero=__ballot_sync(mask,isfinite(g)&&g!=0),candidates=same&nonzero;
 if(candidates && (threadIdx.x&31)==__ffs(candidates)-1)atomicExch(&b.batch->support[gr],1);
}
__global__ void finalizeSupport(Buffers b,const int* accepted){if(!*accepted){b.batch->active=0;return;}bool any=false;for(int i=0;i<7;++i)any=any||b.batch->support[i]!=0;b.batch->active=b.batch->active&&any;}
__device__ void update(float* p,float& first,float& second,float gradient,float bias1,float bias2){float g=fminf(1,fmaxf(-1,gradient));first=.9f*first+.1f*g;second=.999f*second+.001f*g*g;
 *p=*p*.999999f-.0001f*(first/bias1)/(sqrtf(second/bias2)+1e-8f);}
__global__ void countUpdates(Buffers b,const int* accepted,OnlinePersistent* online,DecisionPositionAdapter* position,DecisionMinutePersistent* minutes,LongMemoryPersistent* memory){
 if(!*accepted || !b.batch->active)return;auto& p=*b.persistent;++p.updates;
 if(b.batch->support[0]){++p.secondsSteps;++online->output.decisionSteps;}
 if(b.batch->support[1]){++p.minuteSteps;++minutes->steps;}
 if(b.batch->support[2]){++p.positionSteps;++position->steps;}
 if(b.batch->support[3]){++p.bayesSteps;++minutes->motion.steps;}
 if(b.batch->support[4]){++p.memorySteps;++memory->steps;}
 if(b.batch->support[5]){++p.motionSteps;b.batch->motionActive=1;}if(b.batch->support[6])++p.wheelSteps;
 if(b.batch->pixelActive){++p.pixelSteps;++online->output.pixelSteps;}
 // Each group skips moments when unsupported; its Adam clock must skip those batches too.
 const std::uint64_t steps[7]={p.secondsSteps,p.minuteSteps,p.positionSteps,p.bayesSteps,p.memorySteps,p.motionSteps,p.wheelSteps};
 for(int i=0;i<7;++i)if(b.batch->support[i]){b.batch->bias1[i]=1-powf(.9f,static_cast<float>(steps[i]));b.batch->bias2[i]=1-powf(.999f,static_cast<float>(steps[i]));}
}
__global__ void commit(Buffers b,const int* accepted,DecisionParameters* decision,DecisionMinutePersistent* minutes,DecisionPositionAdapter* position,JepaParameters* jepa,LongMemoryPersistent* memory,OnlinePersistent* online){
 int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=FullParameters || !*accepted || !b.batch->active || !b.batch->support[group(at)])return;
 float* p=nullptr;int v=at;
 if(at<FullMinuteOffset){if(coreParameter(at))return;p=reinterpret_cast<float*>(decision)+at;}
 else if(at<FullPositionOffset)p=reinterpret_cast<float*>(&minutes->parameters)+(at-FullMinuteOffset);
 else if(at<FullBayesOffset){v=at-FullPositionOffset;p=v<VoteCount*PositionFeatures?&position->evidence[0][0]+v:&position->k[0][0]+v-VoteCount*PositionFeatures;}
 else if(at<FullMemoryOffset)p=&minutes->motion.weights[0][0]+at-FullBayesOffset;
 else if(at<FullMotionOffset)p=reinterpret_cast<float*>(&memory->parameters)+at-FullMemoryOffset;
 else if(at<FullWheelOffset)p=reinterpret_cast<float*>(&jepa->motion)+at-FullMotionOffset;
 else p=online->wheel+at-FullWheelOffset;
 update(p,b.persistent->first[at],b.persistent->second[at],b.batch->gradient[at],b.batch->bias1[group(at)],b.batch->bias2[group(at)]);if(group(at)==4)*p=fminf(8,fmaxf(-8,*p));
}
template<bool PnnJepa>
__global__ void pixelLearning(Buffers b,const int* accepted,float* master,TernaryWeight* weights,SpikePixelStatistics* stats){int pixel=blockIdx.x*blockDim.x+threadIdx.x;
 if(pixel>=NeuronCount || !*accepted || !b.batch->active || !b.batch->visualCount || !b.batch->pixelActive)return;
 const int n=b.batch->visualCount;
 float rgb[4][3];
 const float* patchCredit[4]{};
 for(int j=0;j<n;++j) {
  const auto slot=static_cast<std::size_t>(b.batch->visual[j]/FullVisualEvery);
  const auto offset=slot*NeuronCount*3+pixel*3;
  float pulse=1;
  if constexpr(PnnJepa) {
   const auto packed=b.pnnPulses[slot*PnnPulseBytes+pixel/4];
   pulse=static_cast<int>((packed>>(2*(pixel%4)))&3)-1;
   const int x=pixel%NeuronWidth, y=pixel/NeuronWidth;
   const int patch=pnnJepaPatch(x,y), band=pnnJepaBand(y/16%JepaCells);
   patchCredit[j]=b.pnnPatchScores+slot*PnnPatchScoreCount+patch*JepaInput+band*PnnJepaBandFeatures;
   // As in the original pixel STE, constant pooling scale is absorbed into the local learning rate.
   pulse/=pnnJepaBandRows(band);
  }
  for(int c=0;c<3;++c) rgb[j][c]=b.pixels[offset+c]*(pulse/255.f);
 }
 float utility=0;
 for(int a=0;a<VoteCount;++a)for(int c=0;c<3;++c) {
  float score=0;
  for(int j=0;j<n;++j) {
   const float credit=PnnJepa ? patchCredit[j][a] : b.cache[b.batch->visual[j]].voteScore[a];
   score+=credit*rgb[j][c]/n;
  }
  const auto at=weightIndex(pixel,c,a);
  const float g=fminf(1,fmaxf(-1,-b.batch->credit*score));
  const float old=master[at], rate=PnnJepa ? .02f : .002f;
  const float w=fminf(2,fmaxf(-2,old-rate*g));
  master[at]=w;
  weights[at]=static_cast<TernaryWeight>((w>.05f)-(w<-.05f));
  utility+=b.batch->credit*score*old;
 }
 // Current-observation STE: the 1/N vote averaging is absorbed in this local SGD rate.
 // Real reward credit also informs dynamic placement, without brightness becoming its target.
 if(stats){stats[pixel].crossEntropy=.995f*stats[pixel].crossEntropy+.005f*fminf(1,fmaxf(-1,utility));stats[pixel].hit=.995f*stats[pixel].hit+.005f*(utility>0);}
}
__global__ void rewardMemory(Buffers b,const int* accepted,DeviceLongMemoryView memory){if(!*accepted || !b.batch->active || !memory.catalog)return;const auto* batch=b.batch->original;
 for(unsigned i=0;i<batch->count;++i){const auto& c=b.cache[batch->frames[i]%RewardCacheSlots];const auto& t=c.memory;if(!t.ready || t.frozen || t.writtenSlot<0 || t.writtenSlot>=LongMemoryCapacity)continue;
  auto& e=memory.entries[t.writtenSlot];if(!e.valid || e.worldHi!=t.context.worldHi || e.worldLo!=t.context.worldLo || e.ordinal>t.ordinal)continue;
  float alpha=.1f*batch->credit;for(int a=0;a<VoteCount;++a)if(c.proposal.mask&(1u<<a)){float p=fminf(1-1e-5f,fmaxf(1e-5f,e.action[a]));e.action[a]=sig(logf(p/(1-p))+alpha);}
  float sum=0;for(int k=0;k<DecisionKChoices;++k){if(k==c.proposal.k)e.kMass[k]*=expf(alpha);sum+=e.kMass[k];}for(int k=0;k<DecisionKChoices;++k)e.kMass[k]/=fmaxf(1e-6f,sum);
  e.quality=.99f*e.quality+.01f*fminf(1,10*fabsf(batch->credit));e.labels=min(1000000u,e.labels+1);++memory.catalog->revision;
 }
}
__global__ void blockBefore(Buffers b,const BlockCounters* block,DeviceMemeView meme){b.batch->blockBefore=block->actorUpdates;b.batch->memeBefore=meme.persistent->counters.projectionUpdates;b.batch->prototypeBefore=meme.persistent->prototypes[0].support;}
__global__ void blockAfter(Buffers b,const int* active,const BlockCounters* block,DeviceMemeView meme){if(!*active)return;if(block->actorUpdates>b.batch->blockBefore)++b.persistent->blockSteps;if(meme.persistent->counters.projectionUpdates>b.batch->memeBefore || meme.persistent->prototypes[0].support>b.batch->prototypeBefore)++b.persistent->memeSteps;}
__global__ void commitGate(Buffers b,const int* active){if(!*active){b.batch->visualCount=0;b.batch->credit=0;b.batch->motionActive=0;b.batch->pixelActive=0;}}
__global__ void exportStatus(Buffers b,std::uint64_t bytes){auto cached=b.status->cached,dropped=b.status->dropped;const auto& p=*b.persistent;*b.status={p.updates,p.secondsSteps,p.pixelSteps,p.minuteSteps,p.positionSteps,p.bayesSteps,p.motionSteps,p.memorySteps,p.blockSteps,p.memeSteps,p.wheelSteps,cached,dropped,bytes};}
}
struct VabpnnFullLearning {Buffers b{};void* secondsScratch=nullptr;std::uint64_t bytes=0;cudaEvent_t initialized=nullptr;std::vector<void*> allocations;};
namespace {
template<class T>void allocate(VabpnnFullLearning* m,T*& p,std::size_t count,cudaStream_t stream){check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)),"allocate full-network learning");m->allocations.push_back(p);m->bytes+=count*sizeof(T);check(cudaMemsetAsync(p,0,count*sizeof(T),stream),"initialize full-network learning");}
}
VabpnnFullLearning* vabpnnCreateFullLearning(cudaStream_t stream){auto* m=new VabpnnFullLearning;try{allocate(m,m->b.persistent,1,stream);allocate(m,m->b.status,1,stream);check(cudaEventCreateWithFlags(&m->initialized,cudaEventDisableTiming),"full learning ready");check(cudaEventRecord(m->initialized,stream),"full ready event");return m;}catch(...){vabpnnDestroyFullLearning(m);throw;}}
void vabpnnEnableFullLearning(VabpnnFullLearning* m,cudaStream_t stream,bool pnnJepa) {
 if (!m->b.cache) {
  check(cudaStreamWaitEvent(stream,m->initialized,0),"wait full training");
  allocate(m,m->b.cache,RewardCacheSlots,stream);allocate(m,m->b.roots,1,stream);allocate(m,m->b.batch,1,stream);
  allocate(m,m->b.pixels,static_cast<std::size_t>(FullVisualSlots)*NeuronCount*3,stream);
  allocate(m,m->b.blocks,static_cast<std::size_t>(FullVisualSlots)*BlockCount,stream);
  check(cudaMalloc(&m->secondsScratch,vabpnnSecondsScratchBytes()),"allocate seconds backward scratch");
  m->allocations.push_back(m->secondsScratch);m->bytes+=vabpnnSecondsScratchBytes();
 }
 if(pnnJepa && !m->b.pnnPatchScores) {
  allocate(m,m->b.spatialScore,VoteCount*DecisionWidth,stream);
  allocate(m,m->b.pnnPulses,FullVisualSlots*PnnPulseBytes,stream);
  allocate(m,m->b.pnnPatchScores,FullVisualSlots*PnnPatchScoreCount,stream);
 }
}
void vabpnnDestroyFullLearning(VabpnnFullLearning* m)noexcept{if(!m)return;if(m->initialized)cudaEventSynchronize(m->initialized);for(void* p:m->allocations)cudaFree(p);if(m->initialized)cudaEventDestroy(m->initialized);delete m;}
void vabpnnResetFullLearning(VabpnnFullLearning* m,cudaStream_t stream){if(m->b.cache)check(cudaMemsetAsync(m->b.cache,0,RewardCacheSlots*sizeof(FullRewardObservation),stream),"discard old full reward receipts");}
std::uint64_t vabpnnFullLearningBytes(const VabpnnFullLearning* m)noexcept{return m?m->bytes:0;}
vabpnn::ModelPart vabpnnFullLearningCheckpointPart(VabpnnFullLearning* m){return {m->b.persistent,sizeof(vabpnn::FullLearningPersistent)};}
vabpnn::DeviceFullLearningView vabpnnFullLearningView(VabpnnFullLearning* m){return {m->b.persistent,m->b.cache,m->b.roots,m->b.batch?m->b.batch->gradient:nullptr,m->b.batch?&m->b.batch->active:nullptr};}
void vabpnnCacheFullReward(const VabpnnGpuFrame& frame,VabpnnFullLearning* m,vabpnn::RewardCache* reward){
 if(!frame.selfLearning || !m->b.cache)return;auto d=vabpnnDecisionModelView(frame.decisionModel);auto j=vabpnnJepaModelView(frame.jepaModel);auto memory=vabpnnLongMemoryView(frame.longMemory);
 auto* c=m->b.cache+(frame.frameId%RewardCacheSlots);
 cacheRoots<<<1,1,0,frame.stream>>>(m->b,frame,reward,d.parameters,const_cast<vabpnn::DecisionOutput*>(d.output),j,d.minutes.persistent,vabpnnOnlineWheelFeatures(frame.onlineModel),vabpnnOnlinePersistent(frame.onlineModel),vabpnnOnlineDemoSample(frame.onlineModel),memory);
 const bool visual=frame.frameId%FullVisualEvery==0;
 const bool pnnVisual=frame.pnnJepa && visual;
 if(pnnVisual && !m->b.pnnPatchScores)throw std::runtime_error("PNN reward memory was not initialized");
 auto* spatialScore=pnnVisual ? m->b.spatialScore : nullptr;
 vabpnnSecondsRewardScore(frame.decisionModel,m->b.roots,c->seconds,c->voteScore,m->secondsScratch,frame.stream,spatialScore);
 vabpnnMinuteRewardScore(frame.decisionModel,m->b.roots,c->minutes,c->voteScore,frame.stream);
 if(visual) {
  const auto slot=static_cast<std::size_t>(frame.frameId%RewardCacheSlots)/FullVisualEvery;
  cachePixels<<<(NeuronCount+255)/256,256,0,frame.stream>>>(frame,m->b.pixels);
  if(pnnVisual) {
   auto* scores=m->b.pnnPatchScores+slot*PnnPatchScoreCount;
   vabpnnScoreJepaPnnFeatures(frame.jepaModel,spatialScore,c->voteScore,scores,frame.stream);
   validatePnnPatchScores<<<1,256,0,frame.stream>>>(scores,c);
   cachePnnPulses<<<(PnnPulseBytes+255)/256,256,0,frame.stream>>>(vabpnnSpikeModelView(frame.spikeModel).spikes,
       m->b.pnnPulses+slot*PnnPulseBytes,c);
  }
  vabpnnCopyBlockRewardObservation(vabpnnSpikeBlockModel(frame.spikeModel),m->b.blocks+slot*BlockCount,frame.stream);
 }
 finalizeCache<<<1,1,0,frame.stream>>>(m->b,reward,frame.frameId);check(cudaGetLastError(),"cache causal full-network score eligibility");
}
void vabpnnPrepareFullReward(VabpnnFullLearning* m,const VabpnnGpuFrame& models,const vabpnn::RewardBatch* batch,int* accepted,cudaStream_t stream){
 if(!m->b.cache)throw std::runtime_error("Full training cache was not initialized");prepareBatch<<<1,1,0,stream>>>(m->b,batch,accepted,models.pnnJepa);
 aggregate<<<(vabpnn::FullParameters+255)/256,256,0,stream>>>(m->b,batch,&vabpnnLongMemoryView(models.longMemory).persistent->parameters,accepted);finalizeSupport<<<1,1,0,stream>>>(m->b,accepted);check(cudaGetLastError(),"validate full-network reward gradients");}
void vabpnnCommitFullReward(VabpnnFullLearning* m,const VabpnnGpuFrame& models,const int* accepted,cudaStream_t stream){
 auto* online=vabpnnOnlinePersistent(models.onlineModel);auto* minutes=vabpnnMutableMinuteTimeline(models.decisionModel,stream);auto* position=vabpnnMutablePositionAdapter(models.decisionModel,stream);auto memory=vabpnnLongMemoryView(models.longMemory);
 commitGate<<<1,1,0,stream>>>(m->b,accepted);
 countUpdates<<<1,1,0,stream>>>(m->b,accepted,online,position,minutes,memory.persistent);
 commit<<<(vabpnn::FullParameters+255)/256,256,0,stream>>>(m->b,accepted,vabpnnMutableDecisionParameters(models.decisionModel,stream),minutes,position,vabpnnMutableJepaParameters(models.jepaModel,stream),memory.persistent,online);
 auto* master=vabpnnOnlineMasterWeights(models.onlineModel);
 auto* weights=vabpnnMutableNeuronWeights(models.neuronModel,stream);
 auto* pixelStats=vabpnnMutableSpikeStatistics(models.spikeModel,stream);
 if(models.pnnJepa)
  pixelLearning<true><<<(NeuronCount+255)/256,256,0,stream>>>(m->b,accepted,master,weights,pixelStats);
 else
  pixelLearning<false><<<(NeuronCount+255)/256,256,0,stream>>>(m->b,accepted,master,weights,pixelStats);
 vabpnnFullJepaMotionStep(models.jepaModel,&m->b.batch->motionActive,stream);
 rewardMemory<<<1,1,0,stream>>>(m->b,accepted,memory);
 auto* blocks=vabpnnSpikeBlockModel(models.spikeModel);auto bv=vabpnnBlockModelView(blocks);auto mv=vabpnnMemeModelView(vabpnnBlockMemeModel(blocks));
 blockBefore<<<1,1,0,stream>>>(m->b,bv.counters,mv);
 vabpnnApplyBlockReward(blocks,m->b.batch->blockTraces,&m->b.batch->visualCount,&m->b.batch->credit,stream);
 blockAfter<<<1,1,0,stream>>>(m->b,accepted,bv.counters,mv);
 // Batch ownership remains with RewardPolicy until all queued consumers finish on this stream.
 check(cudaGetLastError(),"update all neural reward branches");
}
void vabpnnReadFullLearningStatus(VabpnnFullLearning* m,vabpnn::FullLearningStatus* host,cudaStream_t stream){exportStatus<<<1,1,0,stream>>>(m->b,m->bytes);check(cudaMemcpyAsync(host,m->b.status,sizeof(*host),cudaMemcpyDeviceToHost,stream),"read bounded full-learning counters");}
