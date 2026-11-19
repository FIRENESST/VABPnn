#include "LongTermMemory.h"
#include "JepaMotion.h"
#include <cstdio>
#include <memory>
#include <cstring>
#include <stdexcept>
using namespace vabpnn;
void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
template<class T>T get(const T* p){T value;check(cudaMemcpy(&value,p,sizeof(T),cudaMemcpyDeviceToHost));return value;}
int main()try{
    auto* decision=vabpnnCreateDecisionModel(nullptr);auto* jepa=vabpnnCreateJepaModel(nullptr);auto* memory=vabpnnCreateLongMemory(nullptr);
    std::int32_t* votes=nullptr;check(cudaMalloc(&votes,VoteCount*sizeof(*votes)));check(cudaMemset(votes,0,VoteCount*sizeof(*votes)));
    auto p=std::make_unique<DecisionParameters>(get(vabpnnDecisionModelView(decision).parameters));
    p->enabledKChoices=DecisionKChoices;p->kBias[2]=4;
    for(int a=0;a<VoteCount;++a)p->thresholdBias[a]=a==0 ? 0:a==5 ? -10:30;
    check(cudaMemcpy(vabpnnMutableDecisionParameters(decision,nullptr),p.get(),sizeof(*p),cudaMemcpyHostToDevice));
    auto minutes=get(vabpnnDecisionModelView(decision).minutes.persistent);minutes.motion.weights[0][0]=.1f;
    check(cudaMemcpy(vabpnnMutableMinuteTimeline(decision,nullptr),&minutes,sizeof(minutes),cudaMemcpyHostToDevice));
    VabpnnGpuFrame frame{};frame.decisionModel=decision;frame.jepaModel=jepa;frame.evaluationMode=true;
    frame.frameId=1;frame.captureNanos=1000000000;frame.longMemoryContext={11,22,33,44,1000000000,1790000000000};
    auto run=[&](bool attached){frame.longMemory=attached ? memory:nullptr;vabpnnResetDecisionState(decision,nullptr);
        JepaOutput mouse{};mouse.frameId=frame.frameId;mouse.captureNanos=frame.captureNanos;mouse.valid=mouse.pairReady=mouse.motionActive=1;
        mouse.headRateX=100;mouse.deltaSeconds=.02f;
        check(cudaMemcpy(const_cast<JepaOutput*>(vabpnnJepaModelView(jepa).output),&mouse,sizeof(mouse),cudaMemcpyHostToDevice));
        vabpnnLaunchTemporalDecision(frame,votes,decision);check(cudaDeviceSynchronize());return get(vabpnnDecisionModelView(decision).output);};
    const auto neutral=run(true),base=run(false);
    require(!std::memcmp(&neutral,&base,sizeof(base)),"empty external memory preserves the complete existing decision output exactly");
    auto view=vabpnnLongMemoryView(memory);auto entry=get(view.entries);entry.labels=4;entry.valid=1;
    const auto query=get(view.trace);for(int d=0;d<64;++d)entry.key[d]=query.key[d];
    entry.worldHi=11;entry.worldLo=22;entry.sessionHi=33;entry.sessionLo=44;entry.ordinal=1;entry.wallMillis=frame.longMemoryContext.wallMillis;
    auto catalog=get(view.catalog);catalog.used=1;catalog.ordinal=1;
    check(cudaMemcpy(view.catalog,&catalog,sizeof(catalog),cudaMemcpyHostToDevice));
    for(int a=0;a<15;++a)entry.action[a]=a==0||a==5 ? 1:0;
    for(int k=0;k<16;++k)entry.kMass[k]=k==2 ? .8f:.2f/15;
    check(cudaMemcpy(view.entries,&entry,sizeof(entry),cudaMemcpyHostToDevice));
    auto gate=get(view.persistent);gate.parameters.evidenceGain[0]=gate.parameters.evidenceGain[5]=1;gate.parameters.kGain[2]=.5f;
    check(cudaMemcpy(view.persistent,&gate,sizeof(gate),cudaMemcpyHostToDevice));
    frame.frameId=2;frame.captureNanos+=20000000;frame.longMemoryContext.sampleNanos=frame.captureNanos;frame.longMemoryContext.wallMillis+=20;
    const auto without=run(false),with=run(true);
    require(with.posterior[0]>without.posterior[0]&&with.selectedMask==33&&with.selectedCount==2,"retrieved human action evidence reaches Bayes and original top-k selection");
    require(with.command.mouseDx>without.command.mouseDx+.05f,"JEPA mouse residual consumes memory-conditioned Bayesian posterior");
    require(with.kContextLogits[2]!=without.kContextLogits[2],"context offsets include the memory k head for reward sampling");
    p->thresholdBias[0]=30;check(cudaMemcpy(vabpnnMutableDecisionParameters(decision,nullptr),p.get(),sizeof(*p),cudaMemcpyHostToDevice));
    frame.frameId++;frame.captureNanos+=20000000;frame.longMemoryContext.sampleNanos=frame.captureNanos;
    const auto thresholded=run(true);require(!(thresholded.selectedMask&1)&&thresholded.command.valid,"memory cannot bypass a learned confidence threshold");
    const auto frozenCatalog=get(view.catalog);require(!std::memcmp(&catalog,&frozenCatalog,sizeof(catalog)),"Bayesian/JEPA inference leaves the complete persistent memory catalog frozen");
    const auto frozen=get(view.persistent);require(!std::memcmp(&frozen,&gate,sizeof(gate)),"retrieval changes zero memory weights or AdamW state in F8");
    std::printf("PASS: neutral exact decision parity; external retrieval -> Bayes/k -> thresholds; JEPA mouse %.6f -> %.6f uses final memory posterior; frozen neural/optimizer state.\n",without.command.mouseDx,with.command.mouseDx);
    cudaFree(votes);vabpnnDestroyLongMemory(memory);vabpnnDestroyJepaModel(jepa);vabpnnDestroyDecisionModel(decision);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
