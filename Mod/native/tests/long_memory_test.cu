#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "LongTermMemory.h"
#include "model_checkpoint.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <filesystem>
using namespace vabpnn;
void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
void require(bool v,const char* text){if(!v)throw std::runtime_error(text);}
template<class T>struct Device{T* p=nullptr;Device(){check(cudaMalloc(&p,sizeof(T)));}~Device(){cudaFree(p);}void set(const T& v){check(cudaMemcpy(p,&v,sizeof(v),cudaMemcpyHostToDevice));}T get(){T v;check(cudaMemcpy(&v,p,sizeof(v),cudaMemcpyDeviceToHost));return v;}};
template<class T>T get(const T* p){T v;check(cudaMemcpy(&v,p,sizeof(v),cudaMemcpyDeviceToHost));return v;}
int countBits(unsigned v){int n=0;while(v){n+=v&1;v>>=1;}return n;}
double sigmoid(double x){return 1/(1+std::exp(-x));}
double ce(double p,bool y){p=std::clamp(p,1e-5,1-1e-5);return -std::log(y ? p:1-p);}
struct Fixture{
    VabpnnDecisionModel* decision=vabpnnCreateDecisionModel(nullptr);
    VabpnnLongMemory* memory=vabpnnCreateLongMemory(nullptr);
    Device<DecisionOutput> d;Device<std::uint64_t> previous;Device<int> active;Device<unsigned> label;
    VabpnnGpuFrame frame{};
    Fixture(){check(cudaDeviceSynchronize());frame.decisionModel=decision;frame.longMemory=memory;frame.frameId=1;
        frame.captureNanos=1000000000;frame.longMemoryContext={11,22,33,44,1000000000,1790000000000};active.set(1);label.set(1);}
    ~Fixture(){cudaDeviceSynchronize();vabpnnDestroyLongMemory(memory);vabpnnDestroyDecisionModel(decision);}
    void next(double seconds=61){DecisionOutput out{};out.valid=out.status.valid=1;out.frameId=frame.frameId;
        out.captureNanos=frame.captureNanos;out.status.elapsedMinutes=seconds/60;
        for(int a=0;a<VoteCount;++a){out.posterior[a]=.5f;out.softEligibility[a]=.5f;for(int j=0;j<16;++j)out.hidden[a][j]=.2f*std::sin(j+1);}
        for(int k=0;k<16;++k)out.kProbability[k]=1.f/16;
        d.set(out);frame.longMemoryContext.sampleNanos=frame.captureNanos;
        vabpnnRecallLongMemory(frame,d.p,nullptr,memory);vabpnnRecordLongMemory(frame,d.p,memory);check(cudaDeviceSynchronize());}
    void teach(){previous.set(frame.frameId);vabpnnTrainLongMemory(memory,vabpnnDecisionModelView(decision).parameters,previous.p,active.p,label.p,frame.longMemoryContext,nullptr);check(cudaDeviceSynchronize());}
    void advance(std::uint64_t nanos=20000000){frame.frameId++;frame.captureNanos+=nanos;frame.longMemoryContext.wallMillis+=nanos/1000000;}
};
void temporalStore(){Fixture f;const auto v=vabpnnLongMemoryView(f.memory);f.next();
    require(get(v.catalog).used==1&&!get(v.output).active&&!get(v.catalog).supervised,"neutral unlabeled visual record, no fabricated target");
    f.teach();require(get(v.catalog).supervised==1&&get(v.persistent).steps==1&&get(v.entries).action[0]==1,"causal human label and write gate learning");
    const auto learned=get(v.persistent);const auto frozenCatalog=get(v.catalog);
    std::vector<LongMemoryEntry> frozenEntries(LongMemoryCapacity),checkedEntries(LongMemoryCapacity);
    check(cudaMemcpy(frozenEntries.data(),v.entries,frozenEntries.size()*sizeof(LongMemoryEntry),cudaMemcpyDeviceToHost));
    f.frame.evaluationMode=true;f.advance();f.next(61.02);f.teach();
    auto o=get(v.output);require(o.status.retrieved==1&&!o.active&&o.status.ageSeconds>.019f,"frozen retrieval and zero-initialized neutral fusion");
    const auto frozen=get(v.persistent);require(!std::memcmp(&learned,&frozen,sizeof(learned)),"F8 freezes every memory gate/moment/step");
    auto parameters=learned;parameters.parameters.evidenceGain[0]=1;parameters.parameters.kGain[0]=.5f;
    check(cudaMemcpy(v.persistent,&parameters,sizeof(parameters),cudaMemcpyHostToDevice));
    f.advance();f.next(61.04);o=get(v.output);require(o.active&&o.evidence[0]>0&&o.kLogits[0]<0,"learned gate reads real supported action evidence");
    const auto afterFrozen=get(v.catalog);
    check(cudaMemcpy(checkedEntries.data(),v.entries,checkedEntries.size()*sizeof(LongMemoryEntry),cudaMemcpyDeviceToHost));
    require(!std::memcmp(&frozenCatalog,&afterFrozen,sizeof(frozenCatalog))&&!std::memcmp(frozenEntries.data(),checkedEntries.data(),frozenEntries.size()*sizeof(LongMemoryEntry)),"F8 freezes complete catalog and every vector/action/k/time/visit byte while recall continues");
    const auto before=get(v.catalog);f.next(61.04);require(get(v.catalog).queries==before.queries&&!get(v.trace).ready,"duplicate frame cannot observe/write twice");
    f.advance(70000000000ull);f.frame.longMemoryContext.sessionLo++;f.next(131.04);o=get(v.output);
    require(o.status.retrieved==1&&o.status.ageSeconds>70&&std::abs(o.status.ageMinutes-o.status.ageSeconds/60)<1e-5,"cross-life retained experience has linked second/minute age");
    f.frame.evaluationMode=false;f.advance();f.next(131.06);f.teach();
    require(get(v.catalog).queries>frozenCatalog.queries&&get(v.catalog).supervised>frozenCatalog.supervised,"return to human demonstration resumes memory writes and causal labels");
    f.frame.evaluationMode=true;f.advance();f.frame.longMemoryContext.worldLo++;f.next(131.08);o=get(v.output);
    require(!o.status.retrieved&&!o.active,"world/dimension/player identity isolation");
    f.advance();f.frame.captureNanos-=100000000;f.next();require(!get(v.output).valid,"clock reversal cannot retrieve future experiences");
    ModelPart parts[2];vabpnnLongMemoryStoreParts(f.memory,parts);std::vector<ModelPart> layout(parts,parts+2);
    const auto path=(std::filesystem::absolute(L"build/native")/L"long-memory-isolated.vdb").wstring();
    saveModelCheckpoint(path,layout,14);const auto original=get(v.catalog);
    check(cudaMemset(v.catalog,0,sizeof(LongMemoryCatalog)));require(loadModelCheckpoint(path,layout,false,14),"independent vector file restores");
    require(get(v.catalog).queries==original.queries&&get(v.catalog).supervised==original.supervised,"catalog/time/labels persist outside neural checkpoint");
    std::fstream file(std::filesystem::path(path),std::ios::in|std::ios::out|std::ios::binary);file.seekg(300);char c;file.read(&c,1);c^=1;file.seekp(300);file.write(&c,1);file.close();
    bool rejected=false;try{loadModelCheckpoint(path,layout,false,14);}catch(...){rejected=true;}
    require(rejected&&get(v.catalog).queries==original.queries,"corrupt vector file rejected before changing GPU data");
    std::puts("PASS: external vector writes/human labels, read-only frozen gates/catalog/all vector bytes, resumed human learning, exact duplicate rejection, world isolation, cross-life seconds/minutes age, independent SHA-checked v14 persistence.");
}
struct CpuForward{double posterior[15],k[16],soft[15],read,write,retain,loss;};
CpuForward loss(const LongMemoryParameters& p,const LongMemoryTrace& t,const DecisionParameters& base,unsigned target){
    CpuForward f{};auto g=[](const float* w,const float* x){double s=0;for(int j=0;j<8;++j)s+=w[j]*x[j];return sigmoid(s);};
    f.read=g(p.read,t.readFeatures);f.write=g(p.write,t.writeFeatures);f.retain=g(p.retain,t.retainFeatures);
    const double temp=std::log1p(std::exp(base.logGateTemperature));double baseLoss=0,memoryLoss=0;
    for(int a=0;a<15;++a){const bool y=target&(1u<<a);double old=t.basePosterior[a];
        const double delta=.5*std::tanh(p.evidenceGain[a])*f.read*t.memoryEvidence[a];
        f.posterior[a]=sigmoid(std::log(old/(1-old))+delta);f.soft[a]=sigmoid((f.posterior[a]-.5)/temp);
        f.loss+=ce(f.posterior[a],y)/15;baseLoss+=ce(old,y)/15;memoryLoss+=ce(sigmoid(t.memoryEvidence[a]),y)/15;}
    double maximum=-INFINITY;for(int k=0;k<16;++k){double value=base.kBias[k]+.5*std::tanh(p.kGain[k])*f.read*t.memoryK[k];
        for(int a=0;a<15;++a)value+=base.kWeight[k][16+a]*f.posterior[a]+base.kWeight[k][31+a]*f.soft[a];
        f.k[k]=value;maximum=std::max(maximum,value);}
    double total=0;for(auto& q:f.k){q=std::exp(q-maximum);total+=q;}for(auto& q:f.k)q/=total;
    f.loss-=std::log(f.k[countBits(target)]);
    f.loss+=.02*ce(f.read,memoryLoss<baseLoss)+.02*ce(f.write,t.novelty>.015f||baseLoss>.35);
    const double retainTarget=memoryLoss<.7 ? 1:.2;
    f.loss+=.02*(-retainTarget*std::log(f.retain)-(1-retainTarget)*std::log(1-f.retain));return f;
}
void gradients(){Fixture f;auto v=vabpnnLongMemoryView(f.memory);LongMemoryPersistent learned{};
    auto* values=reinterpret_cast<float*>(&learned.parameters);for(int i=0;i<55;++i)values[i]=.1f*std::sin(i*.37f+.2f);
    auto base=std::make_unique<DecisionParameters>();base->logGateTemperature=-1;
    for(int k=0;k<16;++k){base->kBias[k]=.03f*std::cos(k);for(int a=0;a<15;++a){base->kWeight[k][16+a]=.07f*std::sin(k+a);base->kWeight[k][31+a]=.06f*std::cos(k-a);}}
    LongMemoryTrace trace{};trace.context=f.frame.longMemoryContext;trace.frame=4;trace.ready=1;trace.support=1;trace.writtenSlot=-1;trace.novelty=.1f;
    for(int i=0;i<8;++i){trace.readFeatures[i]=.5f+.1f*i;trace.writeFeatures[i]=.4f-.03f*i;trace.retainFeatures[i]=.2f+.07f*i;}
    for(int a=0;a<15;++a){trace.basePosterior[a]=.35f+.01f*a;trace.memoryEvidence[a]=.7f*std::sin(a+.3f);}
    for(int k=0;k<16;++k)trace.memoryK[k]=-.3f-.03f*k;
    const unsigned target=(1u<<0)|(1u<<12);const auto expected=loss(learned.parameters,trace,*base,target);
    trace.readGate=expected.read;trace.writeGate=expected.write;trace.retainGate=expected.retain;
    const double temp=std::log1p(std::exp(base->logGateTemperature));
    for(int a=0;a<15;++a){trace.posterior[a]=expected.posterior[a];trace.softDerivative[a]=expected.soft[a]*(1-expected.soft[a])/temp;}
    for(int k=0;k<16;++k)trace.kProbability[k]=expected.k[k];
    check(cudaMemcpy(v.persistent,&learned,sizeof(learned),cudaMemcpyHostToDevice));check(cudaMemcpy(v.trace,&trace,sizeof(trace),cudaMemcpyHostToDevice));
    check(cudaMemcpy(vabpnnMutableDecisionParameters(f.decision,nullptr),base.get(),sizeof(*base),cudaMemcpyHostToDevice));
    f.previous.set(4);f.label.set(target);vabpnnTrainLongMemory(f.memory,vabpnnDecisionModelView(f.decision).parameters,f.previous.p,f.active.p,f.label.p,f.frame.longMemoryContext,nullptr);check(cudaDeviceSynchronize());
    float actual[55];check(cudaMemcpy(actual,v.gradients,sizeof(actual),cudaMemcpyDeviceToHost));double maximum=0;int nonzero=0;
    for(int i=0;i<55;++i){auto plus=learned.parameters,minus=plus;auto* a=reinterpret_cast<float*>(&plus);auto* b=reinterpret_cast<float*>(&minus);
        const float eps=.001f;a[i]+=eps;b[i]-=eps;const double reference=(loss(plus,trace,*base,target).loss-loss(minus,trace,*base,target).loss)/(a[i]-b[i]);
        maximum=std::max(maximum,std::abs(actual[i]-reference));nonzero+=std::abs(reference)>1e-7;
        if(std::abs(actual[i]-reference)>2e-5){std::printf("MEMORY_GRAD id=%d CUDA=%g CPU=%g\n",i,actual[i],reference);throw std::runtime_error("Memory gradient does not match independent scalar loss");}}
    const auto after=get(v.persistent);require(after.steps==1,"one causal memory AdamW step");
    vabpnnTrainLongMemory(f.memory,vabpnnDecisionModelView(f.decision).parameters,f.previous.p,f.active.p,f.label.p,f.frame.longMemoryContext,nullptr);check(cudaDeviceSynchronize());const auto duplicate=get(v.persistent);
    require(!std::memcmp(&after,&duplicate,sizeof(after)),"same human observation updates gate optimizer once");
    std::printf("PASS: independent 55-parameter memory gate/evidence/k finite differences, %d nonzero, max error %.3g; exact causal update deduplication.\n",nonzero,maximum);
}

void boundedIndex(){Fixture f;const auto v=vabpnnLongMemoryView(f.memory);f.next();const auto trace=get(v.trace);
    std::vector<LongMemoryEntry> records(LongMemoryCapacity);
    for(int i=0;i<LongMemoryCapacity;++i){auto& e=records[i];for(int d=0;d<64;++d)e.key[d]=trace.key[d];
        e.worldHi=11;e.worldLo=22;e.sessionHi=33;e.sessionLo=44;e.ordinal=i+1;e.wallMillis=f.frame.longMemoryContext.wallMillis;
        e.labels=4;e.quality=.5f;e.valid=1;for(int a=0;a<15;++a)e.action[a]=a==0 ? .8f:.1f;
        for(int k=0;k<16;++k)e.kMass[k]=k==1 ? .8f:.2f/15;}
    auto catalog=get(v.catalog);catalog.used=LongMemoryCapacity;catalog.ordinal=LongMemoryCapacity+1;
    check(cudaMemcpy(v.catalog,&catalog,sizeof(catalog),cudaMemcpyHostToDevice));
    check(cudaMemcpy(v.entries,records.data(),records.size()*sizeof(records[0]),cudaMemcpyHostToDevice));
    f.advance();f.frame.evaluationMode=true;f.next(61.02);
    const auto selected=get(v.trace);require(selected.count==8,"full vector index performs actual top-8 supported recall");
    for(int i=0;i<8;++i)require(selected.selected[i]==i,"exact cosine/timestamp tie gives deterministic index order");
    f.frame.evaluationMode=false;f.advance();f.frame.longMemoryContext.worldLo++;f.next(61.04);
    const auto after=get(v.catalog);require(after.used==8192&&after.evictionCursor==64,"bounded 64-candidate retention eviction at fixed capacity");
    require(get(v.entries).worldLo==23,"frozen recall never increments visits; equal retention scores evict the first candidate on learning resume");
    Fixture boundary;boundary.next();boundary.frame.longMemoryContext.sessionLo++;boundary.teach();
    require(!get(vabpnnLongMemoryView(boundary.memory).persistent).steps,"cross-life human target cannot annotate old cached memory");
    std::puts("PASS: full 8192-entry exact top-8/ties, separate observation novelty, constant-capacity retention eviction, cross-life target isolation.");
}

int main()try{temporalStore();boundedIndex();gradients();return 0;}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
