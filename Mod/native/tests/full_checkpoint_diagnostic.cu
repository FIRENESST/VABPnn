// Offline diagnosis on a read-only v15 checkpoint. Synthetic frames/rewards never reach the game or disk.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "FullSelfLearning.h"
#include "FrameOperator.h"
#include "../model_checkpoint.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>
using namespace vabpnn;
static void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
template<class T> static T read(const T* p){T v;check(cudaMemcpy(&v,p,sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T> static std::unique_ptr<T> heapRead(const T* p){auto v=std::make_unique<T>();check(cudaMemcpy(v.get(),p,sizeof(T),cudaMemcpyDeviceToHost));return v;}
__global__ void pattern(unsigned char* p){int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=NeuronCount)return;int x=at%NeuronWidth,y=at/NeuronWidth;p[4*at]=(x*3+y*7+91)&255;p[4*at+1]=(x+y*11+43)&255;p[4*at+2]=(x*13+y+29)&255;p[4*at+3]=255;}
struct Sample{float forward,cache,reward,total;bool ssl,rewardFrame;};
static void report(const char* name,const std::vector<Sample>& a){
 std::vector<float> total;double forward=0,cache=0;int over=0;float rewardMax=0;double ssl=0,plain=0;int ns=0,np=0;
 for(auto& s:a){total.push_back(s.total);forward+=s.forward;cache+=s.cache;over+=s.total>10;rewardMax=std::max(rewardMax,s.reward);if(s.ssl){ssl+=s.forward;++ns;}else{plain+=s.forward;++np;}}
 std::sort(total.begin(),total.end());auto p=[&](float q){return total[size_t((total.size()-1)*q)];};
 std::printf("DIAG_PERF %s n=%zu forward_mean=%.4f ssl_forward_mean=%.4f plain_forward_mean=%.4f cache_mean=%.4f reward_max=%.4f total_mean=%.4f p95=%.4f p99=%.4f max=%.4f over10=%d\n",name,a.size(),forward/a.size(),ns?ssl/ns:0,np?plain/np:0,cache/a.size(),rewardMax,std::accumulate(total.begin(),total.end(),0.)/total.size(),p(.95),p(.99),total.back(),over);
}
int main()try{
 const char* path=std::getenv("VABPNN_DIAG_CHECKPOINT");if(!path)throw std::runtime_error("Set VABPNN_DIAG_CHECKPOINT to a read-only v15 snapshot");
 VabpnnGpuFrame f{};f.width=NeuronWidth;f.height=NeuronHeight;f.rgbaPitch=NeuronWidth*4;f.selfLearning=true;
 unsigned char* rgba=nullptr;check(cudaMalloc(&rgba,NeuronCount*4));f.rgba=rgba;check(cudaMalloc(&f.rgbNchw,NeuronCount*3*sizeof(float)));pattern<<<(NeuronCount+255)/256,256>>>(rgba);
 f.neuronModel=vabpnnCreateNeuronModel(nullptr);f.decisionModel=vabpnnCreateDecisionModel(nullptr);f.jepaModel=vabpnnCreateJepaModel(nullptr);f.onlineModel=vabpnnCreateOnlineModel(nullptr);f.spikeModel=vabpnnCreateSpikeModel(nullptr);f.longMemory=vabpnnCreateLongMemory(nullptr);f.fullLearning=vabpnnCreateFullLearning(nullptr);auto* policy=vabpnnCreateRewardPolicy(nullptr);
 std::vector<ModelPart> parts{{vabpnnMutableNeuronWeights(f.neuronModel,nullptr),ParameterCount},{vabpnnMutableDecisionParameters(f.decisionModel,nullptr),sizeof(DecisionParameters)}};ModelPart extra[8];
 auto add=[&](std::size_t n){parts.insert(parts.end(),extra,extra+n);};add(vabpnnJepaCheckpointParts(f.jepaModel,extra));add(vabpnnOnlineCheckpointParts(f.onlineModel,extra));add(vabpnnSpikeCheckpointParts(f.spikeModel,extra));add(vabpnnDecisionPositionCheckpointParts(f.decisionModel,extra));
 parts.push_back(vabpnnBlockCheckpointPart(vabpnnSpikeBlockModel(f.spikeModel)));parts.push_back(vabpnnMemeCheckpointPart(vabpnnBlockMemeModel(vabpnnSpikeBlockModel(f.spikeModel))));parts.push_back(vabpnnSpikePlacementCheckpointPart(f.spikeModel));parts.push_back(vabpnnDecisionMinuteCheckpointPart(f.decisionModel));parts.push_back(vabpnnRewardCheckpointPart(policy));parts.push_back(vabpnnLongMemoryCheckpointPart(f.longMemory));parts.push_back(vabpnnFullLearningCheckpointPart(f.fullLearning));
 check(cudaDeviceSynchronize());std::filesystem::path checkpoint(path);if(!loadModelCheckpoint(checkpoint.wstring(),parts,false,15))throw std::runtime_error("Snapshot not found");
 auto db=checkpoint.parent_path()/"experience-v1.vdb";if(std::filesystem::exists(db)){ModelPart store[2];vabpnnLongMemoryStoreParts(f.longMemory,store);loadModelCheckpoint(db.wstring(),{store,store+2},false,14);}
 vabpnnResumeJepaOptimizer(f.jepaModel,nullptr);vabpnnResumeSpikeModel(f.spikeModel,nullptr);vabpnnResetDecisionState(f.decisionModel,nullptr);vabpnnResetJepaHistory(f.jepaModel,nullptr);vabpnnResetOnlineHistory(f.onlineModel,nullptr);vabpnnResetLongMemoryTransient(f.longMemory,nullptr);
 vabpnnEnableFullLearning(f.fullLearning,nullptr);vabpnnConfigureRewardPolicy(policy,71,true,nullptr);
 auto entry=read(vabpnnLongMemoryView(f.longMemory).entries);f.longMemoryContext={entry.worldHi,entry.worldLo,entry.sessionHi,entry.sessionLo,0,1792154269000ull};
 DemoSample demo{};demo.active=1;demo.modelControl=1;demo.pointerMode=1;demo.epoch=1;vabpnnSetDemoSample(f.onlineModel,&demo,nullptr);
 auto old=heapRead(vabpnnFullLearningView(f.fullLearning).persistent);auto oldOnline=read(vabpnnOnlineOutput(f.onlineModel));
 std::array<cudaEvent_t,4> event{};for(auto& e:event)check(cudaEventCreate(&e));int relativeValid=0,selectedMouse=0,selectedWheel=0,wheelEligible=0;RewardBatch batch{};batch.epoch=71;batch.credit=-.1f;std::uint64_t rewardId=0;
 for(int phase=0;phase<3;++phase){
  bool frozen=phase==1;f.evaluationMode=frozen;f.selfLearning=!frozen;demo.pointerMode=1;vabpnnSetDemoSample(f.onlineModel,&demo,nullptr);
  if(phase==2){ // Controlled opportunity: ONLY the wheel gate/k-head changes. Real wheel displacement parameters stay intact.
   auto p=read(vabpnnDecisionModelView(f.decisionModel).parameters);for(int a=0;a<VoteCount;++a){p.logLikelihoodBias[a]=a==11?15:-15;p.rawVoteWeight[a]=0;p.thresholdBias[a]=a==11?-15:15;for(int d=0;d<DecisionWidth;++d){p.logLikelihoodWeight[a][d]=0;p.thresholdWeight[a][d]=0;}}
   for(int k=0;k<DecisionKChoices;++k){p.kBias[k]=k==1?15:-15;for(int d=0;d<DecisionKFeatures;++d)p.kWeight[k][d]=0;}check(cudaMemcpy(vabpnnMutableDecisionParameters(f.decisionModel,nullptr),&p,sizeof(p),cudaMemcpyHostToDevice));vabpnnConfigureRewardPolicy(policy,72,true,nullptr);batch.epoch=72;
  }
  std::vector<Sample> timings;int frames=phase==0?180:phase==1?64:64;
  for(int i=1;i<=frames;++i){++f.frameId;f.captureNanos=1'000'000'000ull+f.frameId*33'333'333ull;f.position={f.captureNanos,1,-15.3,66,17.3};f.longMemoryContext.sampleNanos=f.captureNanos;f.longMemoryContext.wallMillis+=33;
   auto j=vabpnnJepaModelView(f.jepaModel);auto before=read(j.output).selfSupervisedSteps;check(cudaEventRecord(event[0]));vabpnnLaunchOperator(f);check(cudaEventRecord(event[1]));
   auto d=vabpnnDecisionModelView(f.decisionModel);if(!frozen){vabpnnProposeRewardAction(policy,d.parameters,d.constraints,const_cast<DecisionOutput*>(d.output),j.output,vabpnnMutableMotionProposal(f.decisionModel,nullptr),nullptr);vabpnnCacheFullReward(f,f.fullLearning,vabpnnRewardPolicyView(policy).cache+f.frameId%RewardCacheSlots);}
   check(cudaEventRecord(event[2]));check(cudaEventSynchronize(event[2]));auto mouse=read(j.output);
   if(!frozen){auto c=heapRead(vabpnnFullLearningView(f.fullLearning).cache+f.frameId%RewardCacheSlots);if(c->proposal.valid){relativeValid+=c->motionValid;selectedMouse+=!!(c->proposal.mask&96);selectedWheel+=!!(c->proposal.mask&(1u<<11));wheelEligible+=!!(c->proposal.eligibleMask&(1u<<11));batch.frames[batch.count]=f.frameId;batch.masks[batch.count]=c->proposal.mask;++batch.count;}
    if((phase==0&&i==24)||(phase==2&&i==24))std::printf("DIAG_SAMPLE phase=%d pointer=%u pair=%d active=%d valid=%u mask=%x eligible=%x head=%.5f/%.5f dmean=%.5f/%.5f wheel_score=%.5f dxdy=%.5f/%.5f wheel=%.5f wheel_post=%.6f wheel_thr=%.6f\n",phase,demo.pointerMode,mouse.pairReady,mouse.motionActive,c->motionValid,c->proposal.mask,c->proposal.eligibleMask,c->headRate[0],c->headRate[1],c->meanDerivative[0],c->meanDerivative[1],c->wheelScore,c->decision.command.mouseDx,c->decision.command.mouseDy,c->decision.command.wheelSteps,c->decision.posterior[11],c->decision.threshold[11]);
   }
   bool rewardFrame=!frozen&&batch.count&&i%30==0;if(rewardFrame){batch.rewardId=++rewardId;vabpnnApplyRewardBatch(policy,vabpnnMutableDecisionParameters(f.decisionModel,nullptr),batch,nullptr,&f);batch.count=0;}
   check(cudaEventRecord(event[3]));check(cudaEventSynchronize(event[3]));Sample s{};check(cudaEventElapsedTime(&s.forward,event[0],event[1]));check(cudaEventElapsedTime(&s.cache,event[1],event[2]));check(cudaEventElapsedTime(&s.reward,event[2],event[3]));check(cudaEventElapsedTime(&s.total,event[0],event[3]));s.ssl=mouse.selfSupervisedSteps>before;s.rewardFrame=rewardFrame;if(i>16)timings.push_back(s);
  }batch.count=0;report(phase==0?"restored_full":phase==1?"restored_frozen":"controlled_wheel",timings);
 }
 auto now=heapRead(vabpnnFullLearningView(f.fullLearning).persistent);auto examples=read(vabpnnOnlineOutput(f.onlineModel)).examples;
 std::printf("DIAG_UPDATES delta=%llu seconds=%llu pixels=%llu bayes=%llu motion=%llu wheel=%llu relative_samples=%d mouse_selected=%d wheel_selected=%d wheel_eligible=%d human_examples_delta=%lld\n",now->updates-old->updates,now->secondsSteps-old->secondsSteps,now->pixelSteps-old->pixelSteps,now->bayesSteps-old->bayesSteps,now->motionSteps-old->motionSteps,now->wheelSteps-old->wheelSteps,relativeValid,selectedMouse,selectedWheel,wheelEligible,static_cast<long long>(examples-oldOnline.examples));
 if(!relativeValid||now->motionSteps==old->motionSteps||now->bayesSteps==old->bayesSteps||now->wheelSteps==old->wheelSteps||examples!=oldOnline.examples)throw std::runtime_error("Restored checkpoint has an inactive required reward branch");
 std::puts("PASS: trained v15 snapshot supports relative mouse/Bayesian learning and controlled wheel opportunities; no checkpoint save or game launch occurred.");
 return 0;
}catch(const std::exception& e){std::fprintf(stderr,"DIAG_FAIL: %s\n",e.what());return 1;}
