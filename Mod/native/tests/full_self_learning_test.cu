#include "FullSelfLearning.h"
#include "FrameOperator.h"
#include "full_learning_reference.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include <stdexcept>
using namespace vabpnn;
static void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
static void require(bool v,const char* text){if(!v)throw std::runtime_error(text);}
template<class T>T read(const T* p){T value;check(cudaMemcpy(&value,p,sizeof(T),cudaMemcpyDeviceToHost));return value;}
template<class T>std::unique_ptr<T> heapRead(const T* p){auto value=std::make_unique<T>();check(cudaMemcpy(value.get(),p,sizeof(T),cudaMemcpyDeviceToHost));return value;}
template<class T>void write(T* p,const T& value){check(cudaMemcpy(p,&value,sizeof(T),cudaMemcpyHostToDevice));}
__global__ void image(unsigned char* rgba){int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=NeuronCount)return;rgba[4*pixel]=40+pixel%190;rgba[4*pixel+1]=30+(pixel/NeuronWidth)%170;rgba[4*pixel+2]=20+((pixel*17)%210);rgba[4*pixel+3]=255;}
__global__ void seedPixels(float* master,TernaryWeight* weights){int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=NeuronCount)return;for(int c=0;c<3;++c)for(int a=0;a<VoteCount;++a){auto at=weightIndex(pixel,c,a);float w=((pixel+c+a)%3==0)?.050001f:.049999f;master[at]=w;weights[at]=static_cast<TernaryWeight>(w>.05f);}}
__global__ void seedMemes(DeviceMemeView m){int tile=blockIdx.x;auto& p=m.persistent->prototypes[tile];p.support=64;p.revision=1;p.origin=tile;p.utility=.5f;p.confidence=1;p.coverage=1;for(int f=0;f<MemeContext;++f)p.context[f]=m.observations[tile].context[f];for(int a=0;a<MemeChannels;++a)p.evidence[a]=(a%2?.5f:-.5f);}
__global__ void fingerprint(const unsigned char* bytes,std::size_t count,unsigned long long* result){unsigned long long hash=0;for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x){unsigned long long x=static_cast<unsigned long long>(bytes[i])^(i*0x9e3779b97f4a7c15ull);x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;hash^=x^(x>>31);}atomicXor(result,hash);}
static unsigned long long hashBytes(const void* p,std::size_t n){unsigned long long *out=nullptr,value=0;check(cudaMalloc(&out,sizeof(value)));check(cudaMemset(out,0,sizeof(value)));fingerprint<<<256,256>>>(static_cast<const unsigned char*>(p),n,out);check(cudaMemcpy(&value,out,sizeof(value),cudaMemcpyDeviceToHost));cudaFree(out);return value;}
__global__ void utilityOnly(RewardCache* policy,FullRewardObservation* cache){for(int i=0;i<RewardPolicyParameters;++i)policy->score[i]=0;for(int i=0;i<FullDecisionParameters;++i)cache->seconds[i]=0;for(int i=0;i<MinuteParameterCount;++i)cache->minutes[i]=0;for(int i=0;i<VoteCount;++i)cache->voteScore[i]=0;cache->roots={};cache->motionValid=0;cache->wheelScore=0;cache->visual=0;cache->motionScore[0]=cache->motionScore[1]=0;}
int main()try{
 VabpnnGpuFrame f{};f.width=NeuronWidth;f.height=NeuronHeight;f.rgbaPitch=NeuronWidth*4;f.selfLearning=true;
 unsigned char* rgba=nullptr;check(cudaMalloc(&rgba,NeuronCount*4));f.rgba=rgba;check(cudaMalloc(&f.rgbNchw,NeuronCount*3*sizeof(float)));image<<<(NeuronCount+255)/256,256>>>(rgba);
 f.neuronModel=vabpnnCreateNeuronModel(nullptr);f.decisionModel=vabpnnCreateDecisionModel(nullptr);f.jepaModel=vabpnnCreateJepaModel(nullptr);f.onlineModel=vabpnnCreateOnlineModel(nullptr);f.spikeModel=vabpnnCreateSpikeModel(nullptr);f.longMemory=vabpnnCreateLongMemory(nullptr);f.fullLearning=vabpnnCreateFullLearning(nullptr);auto* policy=vabpnnCreateRewardPolicy(nullptr);
 vabpnnEnableFullLearning(f.fullLearning,nullptr);vabpnnConfigureRewardPolicy(policy,7,true,nullptr);
 seedPixels<<<(NeuronCount+255)/256,256>>>(vabpnnOnlineMasterWeights(f.onlineModel),vabpnnMutableNeuronWeights(f.neuronModel,nullptr));
 auto parameters=std::make_unique<DecisionParameters>(read(vabpnnDecisionModelView(f.decisionModel).parameters));
 for(int a=0;a<VoteCount;++a){parameters->thresholdBias[a]=-1.4f;parameters->rawVoteWeight[a]=.1f;for(int d=0;d<DecisionWidth;++d){parameters->logLikelihoodWeight[a][d]=.08f*std::sin(float(1+a*7+d));parameters->thresholdWeight[a][d]=.02f*std::cos(float(a+d));}}
 for(int k=0;k<DecisionKChoices;++k){parameters->kBias[k]=k==3?3.f:0;for(int d=0;d<DecisionKFeatures;++d)parameters->kWeight[k][d]=.03f*std::sin(float(k*13+d+2));}
 write(vabpnnMutableDecisionParameters(f.decisionModel,nullptr),*parameters);
 auto minute=read(vabpnnDecisionModelView(f.decisionModel).minutes.persistent);for(int a=0;a<VoteCount;++a)for(int d=0;d<MinuteWidth;++d)minute.parameters.evidence[a][d]=.04f*std::sin(float(a+d));for(int k=0;k<DecisionKChoices;++k)for(int d=0;d<MinuteWidth;++d)minute.parameters.k[k][d]=.02f*std::cos(float(k+d));
 for(int a=0;a<2;++a)for(int d=0;d<BayesianMotionFeatures;++d)minute.motion.weights[a][d]=.002f*std::sin(float(a*7+d));write(vabpnnMutableMinuteTimeline(f.decisionModel,nullptr),minute);
 auto block=read(vabpnnBlockModelView(vabpnnSpikeBlockModel(f.spikeModel)).config);block.warmup=0;vabpnnConfigureBlockModel(vabpnnSpikeBlockModel(f.spikeModel),block,nullptr);
 auto gate=read(vabpnnLongMemoryView(f.longMemory).persistent);for(int a=0;a<VoteCount;++a)gate.parameters.evidenceGain[a]=.2f;for(int k=0;k<DecisionKChoices;++k)gate.parameters.kGain[k]=.15f;write(vabpnnLongMemoryView(f.longMemory).persistent,gate);
 DemoSample sample{};sample.active=1;sample.modelControl=1;sample.pointerMode=1;sample.epoch=1;vabpnnSetDemoSample(f.onlineModel,&sample,nullptr);
 f.longMemoryContext={11,22,33,44,0,1790000000000ull};
 cudaEvent_t begin=nullptr,end=nullptr;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));std::vector<float> timings;
 RewardBatch batch{};batch.epoch=7;batch.rewardId=1;batch.credit=.4f;int mouseCount=0,wheelCount=0;float maxMotionFd=0;std::vector<std::unique_ptr<FullRewardObservation>> snapshots;
 for(int id=1;id<=48;++id){f.frameId=id;f.captureNanos=1000000000ull+id*33333333ull;f.position={f.captureNanos,1,id*.01,70,2};f.longMemoryContext.sampleNanos=f.captureNanos;f.longMemoryContext.wallMillis+=33;
  check(cudaEventRecord(begin));vabpnnLaunchOperator(f);auto d=vabpnnDecisionModelView(f.decisionModel);auto j=vabpnnJepaModelView(f.jepaModel);vabpnnProposeRewardAction(policy,d.parameters,d.constraints,const_cast<DecisionOutput*>(d.output),j.output,vabpnnMutableMotionProposal(f.decisionModel,nullptr),nullptr);
  auto* reward=vabpnnRewardPolicyView(policy).cache+(id%RewardCacheSlots);vabpnnCacheFullReward(f,f.fullLearning,reward);check(cudaEventRecord(end));check(cudaDeviceSynchronize());float elapsed=0;check(cudaEventElapsedTime(&elapsed,begin,end));if(id>12)timings.push_back(elapsed);
  auto c=std::make_unique<FullRewardObservation>(read(vabpnnFullLearningView(f.fullLearning).cache+id%RewardCacheSlots));require(c->proposal.valid,"full score cache must remain valid");if(id>1)require(c->motionValid,"grabbed world-view pointer (1) must allow continuous reward learning");
  if(id==1){seedMemes<<<MemeBlocks,1>>>(vabpnnMemeModelView(vabpnnBlockMemeModel(vabpnnSpikeBlockModel(f.spikeModel))));auto memory=vabpnnLongMemoryView(f.longMemory);auto entry=read(memory.entries);auto trace=read(memory.trace);for(int d=0;d<64;++d)entry.key[d]=trace.key[d];entry.valid=1;entry.labels=8;entry.worldHi=11;entry.worldLo=22;entry.sessionHi=33;entry.sessionLo=44;entry.ordinal=trace.ordinal;entry.wallMillis=f.longMemoryContext.wallMillis;entry.quality=.7f;for(int a=0;a<VoteCount;++a)entry.action[a]=a%2?.7f:.3f;for(int k=0;k<DecisionKChoices;++k)entry.kMass[k]=k==3?.7f:.3f/15;write(memory.entries,entry);}
  if(id>1){snapshots.push_back(std::make_unique<FullRewardObservation>(*c));batch.frames[batch.count]=id;batch.masks[batch.count]=c->proposal.mask;++batch.count;}
  for(int axis=0;axis<2;++axis)if(c->proposal.mask&(1u<<(5+axis))){++mouseCount;float sigma=std::max(.05f,24*std::min(c->decision.deltaSeconds,.1f));float actual=axis?c->decision.command.mouseDy:c->decision.command.mouseDx;float mean=axis?read(j.output).mouseDy:read(j.output).mouseDx;
   if(std::abs(actual)<31.9f){auto logDensity=[&](double mu){return -.5*std::pow((actual-mu)/sigma,2)-std::log(sigma)-.9189385332046727;};float fd=(logDensity(mean+.0001)-logDensity(mean-.0001))/.0002;maxMotionFd=std::max(maxMotionFd,std::abs(fd-c->motionScore[axis]));}}
  if(c->proposal.mask&actionBit(Action::Wheel))++wheelCount;
 }
 require(mouseCount>0&&wheelCount>0,"fixture must sample real continuous permissions");require(maxMotionFd<.003f,"continuous Gaussian score finite difference");
 
 auto lastState=read(vabpnnDecisionModelView(f.decisionModel).learningState);auto last=*snapshots.back();
 auto pfd=std::make_unique<DecisionParameters>(read(vabpnnDecisionModelView(f.decisionModel).parameters));
 double maxSeconds=0,maxMinutes=0,maxMotion=0;int secondsChecked=0;
 const auto* original=reinterpret_cast<const float*>(pfd.get());
 for(int at=1;at<FullDecisionParameters;++at){bool checkIndex=at%37==0 || at==int(offsetof(DecisionParameters,logClockFrequency)/4) || at==int(offsetof(DecisionParameters,logAttentionDecay)/4) || at==int(offsetof(DecisionParameters,logOnRate)/4) || at==int(offsetof(DecisionParameters,thresholdBias)/4) || at==int(offsetof(DecisionParameters,logGateTemperature)/4);if(!checkIndex)continue;
  if(at>=int(offsetof(DecisionParameters,logLikelihoodWeight)/4) && at<int(offsetof(DecisionParameters,thresholdWeight)/4))continue;
  if(at>=int(offsetof(DecisionParameters,kWeight)/4) && at<int(offsetof(DecisionParameters,initialPriorLogit)/4))continue;
  float* value=reinterpret_cast<float*>(pfd.get())+at;float old=*value,h=.0005f;*value=old+h;double plus=full_reference::secondsObjective(*pfd,lastState,last.roots,last.decision);float actualPlus=*value;*value=old-h;double minus=full_reference::secondsObjective(*pfd,lastState,last.roots,last.decision);float actualMinus=*value;*value=old;
  double fd=(plus-minus)/(actualPlus-actualMinus),error=std::abs(fd-last.seconds[at]);if(error>maxSeconds)maxSeconds=error;if(error>.0008){std::fprintf(stderr,"seconds FD index %d analytic %.9g numerical %.9g\n",at,last.seconds[at],fd);throw std::runtime_error("seconds graph analytic gradient");}++secondsChecked;
 }
 auto mpfd=read(vabpnnDecisionModelView(f.decisionModel).minutes.persistent);auto minuteState=read(vabpnnDecisionModelView(f.decisionModel).minutes.output);
 for(int at=0;at<MinuteParameterCount;at+=11){float* value=reinterpret_cast<float*>(&mpfd.parameters)+at;float old=*value,h=.0005f;*value=old+h;double plus=full_reference::minuteObjective(mpfd.parameters,minuteState,last.roots);float ap=*value;*value=old-h;double minus=full_reference::minuteObjective(mpfd.parameters,minuteState,last.roots);float am=*value;*value=old;double fd=(plus-minus)/(ap-am),error=std::abs(fd-last.minutes[at]);maxMinutes=std::max(maxMinutes,error);require(error<.0003,"minute reward gradient finite difference");}
 const auto* motionSample=snapshots.front().get();for(const auto& c:snapshots)if(c->motionValid&&(c->proposal.mask&96)){motionSample=c.get();break;}
 auto jfd=heapRead(vabpnnJepaModelView(f.jepaModel).parameters);for(int at:std::vector<int>{0,17,JepaHidden*33+5,JepaHidden*100+27,JepaMotionInput*JepaHidden+3,JepaMotionInput*JepaHidden+JepaHidden+12,FullMotionParameters-1}){
  float* value=reinterpret_cast<float*>(&jfd->motion)+at;float old=*value,h=.00001f;*value=old+h;double plus=full_reference::motionObjective(jfd->motion,mpfd.motion,*motionSample);float ap=*value;*value=old-h;double minus=full_reference::motionObjective(jfd->motion,mpfd.motion,*motionSample);float am=*value;*value=old;
  double fd=(plus-minus)/(ap-am),analytic=0;int input=JepaMotionInput*JepaHidden;
  if(at<input)analytic=motionSample->motionInput[at/JepaHidden]*motionSample->motionHiddenScore[at%JepaHidden];else if(at<input+JepaHidden)analytic=motionSample->motionHiddenScore[at-input];else if(at<input+JepaHidden+JepaHidden*2)analytic=motionSample->motionHidden[(at-input-JepaHidden)/2]*motionSample->rawMotionScore[(at-input-JepaHidden)%2];else analytic=motionSample->rawMotionScore[at-input-JepaHidden-JepaHidden*2];
  double error=std::abs(fd-analytic)/(1+std::abs(fd));maxMotion=std::max(maxMotion,error);if(error>.003){std::fprintf(stderr,"motion FD %d analytic %.9g numerical %.9g\n",at,analytic,fd);throw std::runtime_error("motion MLP analytic gradient");}
 }
 const FullRewardObservation* wheelSample=nullptr;for(const auto& c:snapshots)if(c->proposal.mask&actionBit(Action::Wheel)){wheelSample=c.get();break;}
 require(wheelSample,"wheel fixture has an actually sampled wheel permission");auto wheelParameters=read(vabpnnOnlinePersistent(f.onlineModel));double maxWheel=0;
 for(int at=0;at<FullWheelParameters;++at){auto objective=[&](){double z=0;for(int j=0;j<FullWheelParameters;++j)z+=wheelParameters.wheel[j]*wheelSample->wheelFeatures[j];double mean=std::tanh(z)*20*std::min(double(wheelSample->decision.deltaSeconds),.1);return full_reference::logGaussian(wheelSample->decision.command.wheelSteps,mean,.08,2);};
  float old=wheelParameters.wheel[at],h=.00001f;wheelParameters.wheel[at]=old+h;double plus=objective();float ap=wheelParameters.wheel[at];wheelParameters.wheel[at]=old-h;double minus=objective();float am=wheelParameters.wheel[at];wheelParameters.wheel[at]=old;double fd=(plus-minus)/(ap-am),analytic=wheelSample->wheelScore*wheelSample->wheelFeatures[at];double error=std::abs(fd-analytic)/(1+std::abs(fd));maxWheel=std::max(maxWheel,error);require(error<.0003,"wheel cached causal features and tanh/Gaussian score gradient");}
 std::printf("PASS: all 32 wheel gradients versus independent FP64 objective, max relative %.3g.\n",maxWheel);
 std::printf("PASS: independent FP64 finite differences: seconds %d parameters max %.3g; minutes max %.3g; JEPA motion chain relative %.3g.\n",secondsChecked,maxSeconds,maxMinutes,maxMotion);

auto before=std::make_unique<DecisionParameters>(read(vabpnnDecisionModelView(f.decisionModel).parameters));auto jepaBefore=heapRead(vabpnnJepaModelView(f.jepaModel).parameters);
 auto pixelHash=hashBytes(vabpnnOnlineMasterWeights(f.onlineModel),ParameterCount*sizeof(float));auto memBefore=read(vabpnnLongMemoryView(f.longMemory).persistent);
 auto bh=hashBytes(vabpnnBlockCheckpointPart(vabpnnSpikeBlockModel(f.spikeModel)).device,sizeof(BlockPersistent));auto mh=hashBytes(vabpnnMemeCheckpointPart(vabpnnBlockMemeModel(vabpnnSpikeBlockModel(f.spikeModel))).device,sizeof(MemePersistent));
 check(cudaEventRecord(begin));vabpnnApplyRewardBatch(policy,vabpnnMutableDecisionParameters(f.decisionModel,nullptr),batch,nullptr,&f);check(cudaEventRecord(end));check(cudaDeviceSynchronize());float updateMs=0;check(cudaEventElapsedTime(&updateMs,begin,end));
 
 std::vector<float> fullGrad(FullParameters);check(cudaMemcpy(fullGrad.data(),vabpnnFullLearningView(f.fullLearning).gradient,FullParameters*sizeof(float),cudaMemcpyDeviceToHost));
 auto gatefd=memBefore.parameters;double maxMemory=0;for(int at=0;at<LongMemoryParameterCount;++at){float* value=reinterpret_cast<float*>(&gatefd)+at;float old=*value,h=.0001f;*value=old+h;double plus=0,minus=0;for(const auto& c:snapshots)plus+=full_reference::memoryObjective(gatefd,*c,batch.credit);float ap=*value;*value=old-h;for(const auto& c:snapshots)minus+=full_reference::memoryObjective(gatefd,*c,batch.credit);float am=*value;*value=old;double fd=(plus-minus)/((ap-am)*snapshots.size()),error=std::abs(fd-fullGrad[FullMemoryOffset+at]);maxMemory=std::max(maxMemory,error);require(error<.0001,"memory reward/utility gradients finite difference");}
 std::printf("PASS: all 55 memory gate/retrieval/utility reward gradients vs independent FP64 objective, max %.3g.\n",maxMemory);

auto persistent=heapRead(vabpnnFullLearningView(f.fullLearning).persistent);
 std::printf("FULL_BRANCHES updates=%llu seconds=%llu pixels=%llu minutes=%llu position=%llu bayes=%llu motion=%llu memory=%llu blocks=%llu memes=%llu wheel=%llu\n",persistent->updates,persistent->secondsSteps,persistent->pixelSteps,persistent->minuteSteps,persistent->positionSteps,persistent->bayesSteps,persistent->motionSteps,persistent->memorySteps,persistent->blockSteps,persistent->memeSteps,persistent->wheelSteps);
 require(persistent->updates==1&&persistent->secondsSteps&&persistent->pixelSteps&&persistent->minuteSteps&&persistent->positionSteps&&persistent->bayesSteps&&persistent->motionSteps&&persistent->memorySteps&&persistent->blockSteps&&persistent->memeSteps&&persistent->wheelSteps,"every trainable branch must actually update");
 auto after=std::make_unique<DecisionParameters>(read(vabpnnDecisionModelView(f.decisionModel).parameters));
 require(std::memcmp(before->query,after->query,sizeof(before->query))&&std::memcmp(before->key,after->key,sizeof(before->key))&&std::memcmp(before->value,after->value,sizeof(before->value))&&std::memcmp(before->feedForwardUp,after->feedForwardUp,sizeof(before->feedForwardUp)),"seconds multihead Q/K/V and feed-forward weights change");
 require(pixelHash!=hashBytes(vabpnnOnlineMasterWeights(f.onlineModel),ParameterCount*sizeof(float)),"full-resolution latent ternary weights learn");
 auto jepaAfter=heapRead(vabpnnJepaModelView(f.jepaModel).parameters);require(std::memcmp(&jepaBefore->motion,&jepaAfter->motion,sizeof(JepaMotionHead)),"JEPA displacement head receives real reward gradient");
 require(bh!=hashBytes(vabpnnBlockCheckpointPart(vabpnnSpikeBlockModel(f.spikeModel)).device,sizeof(BlockPersistent))&&mh!=hashBytes(vabpnnMemeCheckpointPart(vabpnnBlockMemeModel(vabpnnSpikeBlockModel(f.spikeModel))).device,sizeof(MemePersistent)),"block and meme learning are live");
 require(read(vabpnnOnlineOutput(f.onlineModel)).examples==0,"model actions never become human demonstrations");
 
 vabpnnConfigureRewardPolicy(policy,8,true,nullptr);++f.frameId;f.captureNanos+=33333333;f.longMemoryContext.sampleNanos=f.captureNanos;vabpnnLaunchOperator(f);
 auto dnow=vabpnnDecisionModelView(f.decisionModel);vabpnnProposeRewardAction(policy,dnow.parameters,dnow.constraints,const_cast<DecisionOutput*>(dnow.output),vabpnnJepaModelView(f.jepaModel).output,vabpnnMutableMotionProposal(f.decisionModel,nullptr),nullptr);
 auto* auxPolicy=vabpnnRewardPolicyView(policy).cache+f.frameId%RewardCacheSlots;vabpnnCacheFullReward(f,f.fullLearning,auxPolicy);auto* auxFull=vabpnnFullLearningView(f.fullLearning).cache+f.frameId%RewardCacheSlots;utilityOnly<<<1,1>>>(auxPolicy,auxFull);check(cudaDeviceSynchronize());
 RewardBatch utility{};utility.epoch=8;utility.rewardId=1;utility.credit=.4f;utility.count=1;utility.frames[0]=f.frameId;utility.masks[0]=read(auxPolicy).proposal.mask;
 auto priorMemorySteps=heapRead(vabpnnFullLearningView(f.fullLearning).persistent)->memorySteps;
 vabpnnApplyRewardBatch(policy,vabpnnMutableDecisionParameters(f.decisionModel,nullptr),utility,nullptr,&f);check(cudaDeviceSynchronize());
 require(heapRead(vabpnnFullLearningView(f.fullLearning).persistent)->memorySteps==priorMemorySteps+1,"memory utility learns even when the sampled action head has zero score support");
 std::puts("PASS: auxiliary full-network learning is not blocked by zero head-only policy support.");

 // Mirror Minecraft's real pointer protocol, including absolute inventory cursors.
 sample.pointerMode=0;vabpnnSetDemoSample(f.onlineModel,&sample,nullptr);++f.frameId;f.captureNanos+=33333333;f.longMemoryContext.sampleNanos=f.captureNanos;vabpnnLaunchOperator(f);
 auto guiDecision=vabpnnDecisionModelView(f.decisionModel);vabpnnProposeRewardAction(policy,guiDecision.parameters,guiDecision.constraints,const_cast<DecisionOutput*>(guiDecision.output),vabpnnJepaModelView(f.jepaModel).output,vabpnnMutableMotionProposal(f.decisionModel,nullptr),nullptr);
 auto* guiPolicy=vabpnnRewardPolicyView(policy).cache+f.frameId%RewardCacheSlots;vabpnnCacheFullReward(f,f.fullLearning,guiPolicy);check(cudaDeviceSynchronize());
 auto gui=std::make_unique<FullRewardObservation>(read(vabpnnFullLearningView(f.fullLearning).cache+f.frameId%RewardCacheSlots));
 require(!gui->motionValid && gui->motionScore[0]==0 && gui->motionScore[1]==0,"ungrabbed GUI pointer (0) is excluded from relative mouse reward training");
 sample.pointerMode=1;vabpnnSetDemoSample(f.onlineModel,&sample,nullptr);
 std::puts("PASS: actual Java pointer protocol: grabbed world=1 trains mouse; GUI=0 excludes relative mouse samples.");

auto frozenFull=hashBytes(vabpnnFullLearningView(f.fullLearning).persistent,sizeof(FullLearningPersistent));auto frozenP=hashBytes(vabpnnDecisionModelView(f.decisionModel).parameters,sizeof(DecisionParameters));
 auto memory=vabpnnLongMemoryView(f.longMemory);auto frozenMemory=hashBytes(memory.catalog,sizeof(LongMemoryCatalog))^hashBytes(memory.entries,LongMemoryCapacity*sizeof(LongMemoryEntry))^hashBytes(memory.persistent,sizeof(LongMemoryPersistent));
 auto frozenJepa=read(vabpnnJepaModelView(f.jepaModel).output);f.selfLearning=false;f.evaluationMode=true;
 for(int i=0;i<12;++i){++f.frameId;f.captureNanos+=33333333;f.longMemoryContext.sampleNanos=f.captureNanos;vabpnnLaunchOperator(f);}check(cudaDeviceSynchronize());
 require(frozenFull==hashBytes(vabpnnFullLearningView(f.fullLearning).persistent,sizeof(FullLearningPersistent))&&frozenP==hashBytes(vabpnnDecisionModelView(f.decisionModel).parameters,sizeof(DecisionParameters)),"F8/evaluation freezes full optimizer and graph weights");
 require(frozenMemory==(hashBytes(memory.catalog,sizeof(LongMemoryCatalog))^hashBytes(memory.entries,LongMemoryCapacity*sizeof(LongMemoryEntry))^hashBytes(memory.persistent,sizeof(LongMemoryPersistent))),"F8 freezes every external memory byte");
 auto stoppedJepa=read(vabpnnJepaModelView(f.jepaModel).output);require(stoppedJepa.selfSupervisedSteps==frozenJepa.selfSupervisedSteps&&stoppedJepa.motionSteps==frozenJepa.motionSteps,"F8 freezes JEPA SSL/EMA and reward motion");
 auto r=read(vabpnnRewardPolicyView(policy).persistent);vabpnnApplyRewardBatch(policy,vabpnnMutableDecisionParameters(f.decisionModel,nullptr),batch,nullptr,&f);check(cudaDeviceSynchronize());require(read(vabpnnRewardPolicyView(policy).persistent).steps==r.steps,"duplicate reward cannot mutate any branch");
 std::printf("PASS: all-network rewards, motion score finite differences max %.6g, real GPU weight changes, no self-labeling, full F8/memory freeze and duplicate protection. Cache %.1f MiB; SSL=%llu.\n",maxMotionFd,vabpnnFullLearningBytes(f.fullLearning)/1048576.,frozenJepa.selfSupervisedSteps);
 float average=0,maximum=0;for(float t:timings){average+=t/timings.size();maximum=std::max(maximum,t);}
 std::printf("FULL_TIMING measured=%zu frame_mean_ms=%.4f frame_max_ms=%.4f reward_update_ms=%.4f\n",timings.size(),average,maximum,updateMs);cudaEventDestroy(begin);cudaEventDestroy(end);
 vabpnnDestroyRewardPolicy(policy);vabpnnDestroyFullLearning(f.fullLearning);vabpnnDestroyLongMemory(f.longMemory);vabpnnDestroySpikeModel(f.spikeModel);vabpnnDestroyOnlineModel(f.onlineModel);vabpnnDestroyJepaModel(f.jepaModel);vabpnnDestroyDecisionModel(f.decisionModel);vabpnnDestroyNeuronModel(f.neuronModel);cudaFree(f.rgbNchw);cudaFree(rgba);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
