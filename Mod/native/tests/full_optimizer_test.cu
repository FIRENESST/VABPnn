#include "FullSelfLearning.h"
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
using namespace vabpnn;
static void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
template<class T> static std::unique_ptr<T> get(const T* p){auto v=std::make_unique<T>();check(cudaMemcpy(v.get(),p,sizeof(T),cudaMemcpyDeviceToHost));return v;}
int main()try{
 VabpnnGpuFrame f{};f.neuronModel=vabpnnCreateNeuronModel(nullptr);f.onlineModel=vabpnnCreateOnlineModel(nullptr);f.jepaModel=vabpnnCreateJepaModel(nullptr);f.decisionModel=vabpnnCreateDecisionModel(nullptr);f.spikeModel=vabpnnCreateSpikeModel(nullptr);f.longMemory=vabpnnCreateLongMemory(nullptr);f.fullLearning=vabpnnCreateFullLearning(nullptr);vabpnnEnableFullLearning(f.fullLearning,nullptr);
 auto view=vabpnnFullLearningView(f.fullLearning);auto p=get(view.persistent);p->updates=1000;p->wheelSteps=0;check(cudaMemcpy(view.persistent,p.get(),sizeof(*p),cudaMemcpyHostToDevice));
 auto online=get(vabpnnOnlinePersistent(f.onlineModel));online->wheel[FullWheelParameters-1]=0;check(cudaMemcpy(vabpnnOnlinePersistent(f.onlineModel),online.get(),sizeof(*online),cudaMemcpyHostToDevice));
 auto c=std::make_unique<FullRewardObservation>();c->proposal.valid=1;c->proposal.frameId=3;c->proposal.epoch=7;c->proposal.mask=1u<<11;c->wheelScore=1;c->wheelFeatures[FullWheelParameters-1]=1;check(cudaMemcpy(view.cache+3,c.get(),sizeof(*c),cudaMemcpyHostToDevice));
 RewardBatch batch{};batch.epoch=7;batch.rewardId=1;batch.frames[0]=3;batch.masks[0]=1u<<11;batch.count=1;batch.credit=.5f;
 RewardBatch* deviceBatch=nullptr;int* active=nullptr;check(cudaMalloc(&deviceBatch,sizeof(batch)));check(cudaMemcpy(deviceBatch,&batch,sizeof(batch),cudaMemcpyHostToDevice));check(cudaMalloc(&active,sizeof(int)));int one=1;check(cudaMemcpy(active,&one,sizeof(one),cudaMemcpyHostToDevice));
 vabpnnPrepareFullReward(f.fullLearning,f,deviceBatch,active,nullptr);vabpnnCommitFullReward(f.fullLearning,f,active,nullptr);check(cudaDeviceSynchronize());
 auto after=get(vabpnnFullLearningView(f.fullLearning).persistent);auto learned=get(vabpnnOnlinePersistent(f.onlineModel));float value=learned->wheel[FullWheelParameters-1];
 // First nonzero wheel gradient is -0.5. Its own Adam time is ONE, even after 1000 unrelated updates.
 double g=-.5,m=.1*g,v=.001*g*g,expected=-.0001*(m/(1-.9))/(std::sqrt(v/(1-.999))+1e-8);
 std::printf("LATE_BRANCH global=%llu wheel=%llu actual=%.9g expected_first_step=%.9g\n",after->updates,after->wheelSteps,value,expected);
 if(after->updates!=1001||after->wheelSteps!=1||std::abs(value-expected)>1e-7)throw std::runtime_error("newly activated branch must use its own Adam step for bias correction");
 std::puts("PASS: branch-local Adam bias correction after a long period without wheel gradients.");
 return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
