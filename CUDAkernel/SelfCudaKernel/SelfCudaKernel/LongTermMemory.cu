#include "LongTermMemory.h"
#include "TemporalDecision.h"
#include "JepaMotion.h"
#include <cmath>
#include <stdexcept>
#include <string>
namespace {
using namespace vabpnn;
struct Runtime {std::uint64_t lastFrame,lastNanos,lastWrite;LongMemoryContext context;};
struct Buffers {LongMemoryPersistent* learned;LongMemoryCatalog* catalog;LongMemoryEntry* entries;
    LongMemoryOutput* output;LongMemoryTrace* trace;Runtime* runtime;float* scores;float* nearestScores;float* gradient;};
void check(cudaError_t c,const char* op){if(c!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(c));}
__device__ float sigmoid(float x){return 1/(1+expf(-fminf(30,fmaxf(-30,x))));}
__device__ float probability(float x){return fminf(1-1e-5f,fmaxf(1e-5f,x));}
__device__ float logit(float x){x=probability(x);return logf(x/(1-x));}
__device__ float gate(const float* p,const float* f){float x=0;for(int i=0;i<8;++i)x+=p[i]*f[i];return sigmoid(x);}
__device__ bool world(const LongMemoryEntry& e,const LongMemoryContext& c){return e.worldHi==c.worldHi&&e.worldLo==c.worldLo;}
__device__ float ageSeconds(const LongMemoryEntry& e,const LongMemoryTrace& t){
    const double observed=fmax(0.0,t.observedSeconds-e.observedSeconds);
    const double wall=t.context.wallMillis>=e.wallMillis ? (t.context.wallMillis-e.wallMillis)*.001:0;
    return static_cast<float>(fmin(315360000.0,fmax(observed,wall)));
}
__device__ void ageFeatures(float* f,float age,float similarity,float quality,float labels,float visits,float phase){
    f[0]=1;f[1]=fminf(4,log1pf(age)/logf(61.f));
    f[2]=fminf(4,log1pf(age/60)/logf(61.f));f[3]=similarity;
    f[4]=quality;f[5]=fminf(1,labels/8);f[6]=fminf(1,log1pf(visits)/8);f[7]=phase;
}
__global__ void initialize(Buffers b){
    b.learned->parameters.read[0]=-1;
    b.learned->parameters.read[3]=1;
    b.learned->parameters.write[0]=0;b.learned->parameters.write[1]=2;
    b.learned->parameters.retain[0]=2;b.learned->parameters.retain[1]=-.15f;
    b.learned->parameters.retain[2]=-.1f;b.learned->parameters.retain[4]=1;
    b.output->status.enabled=1;b.output->status.capacity=LongMemoryCapacity;
}
__global__ void prepare(Buffers b,const DecisionOutput* d,const float* latents,
        std::uint64_t frame,std::uint64_t nanos,LongMemoryContext context,bool frozen){
    __shared__ float square[64];__shared__ int valid;
    const int tid=threadIdx.x;auto& t=*b.trace;auto& rt=*b.runtime;
    if(tid==0){
        *b.output={};*b.trace={};t.writtenSlot=t.nearestSlot=-1;t.frame=frame;t.captureNanos=nanos;
        t.context=context;t.seconds=d->status.elapsedMinutes*60;t.minutes=d->status.elapsedMinutes;t.frozen=frozen;
        valid=d->valid && d->frameId==frame && frame>rt.lastFrame && nanos>rt.lastNanos
            && context.sampleNanos && context.sampleNanos<=nanos && nanos-context.sampleNanos<=250000000ull
            && context.wallMillis && (context.worldHi||context.worldLo) && (context.sessionHi||context.sessionLo)
            && isfinite(t.seconds) && t.seconds>=0;
        b.output->status={frame,b.catalog->queries,b.catalog->writes,b.catalog->hits,b.catalog->supervised,
            b.learned->steps,b.catalog->revision,b.catalog->used,LongMemoryCapacity,0,1,0};
        if(valid){
            const bool same=rt.lastNanos && rt.context.worldHi==context.worldHi && rt.context.worldLo==context.worldLo
                && rt.context.sessionHi==context.sessionHi && rt.context.sessionLo==context.sessionLo;
            if(same && !frozen)b.catalog->observedSeconds+=fmin(1.0,(nanos-rt.lastNanos)*1e-9);
            else rt.lastWrite=0;
            rt.lastFrame=frame;rt.lastNanos=nanos;rt.context=context;
            t.observedSeconds=b.catalog->observedSeconds;t.ordinal=b.catalog->ordinal+1;
            // Frozen evaluation may read old records but cannot mutate the persistent catalog.
            if(!frozen){b.catalog->ordinal=t.ordinal;++b.catalog->queries;}
            t.context.wallMillis+=(nanos-context.sampleNanos)/1000000;
            t.ready=1;
            for(int a=0;a<VoteCount;++a)t.basePosterior[a]=d->posterior[a];
        }
    }
    __syncthreads();
    float x=0;
    if(valid){
        if(tid<32){
            if(latents){for(int token=0;token<JepaTokens;++token)for(int j=0;j<4;++j)x+=latents[token*JepaWidth+tid*4+j];x/=JepaTokens*4;}
            else {for(int a=0;a<VoteCount;++a)x+=d->hidden[a][tid%DecisionWidth];x/=VoteCount;}
            x=.8f*tanhf(x);
        }else if(tid<47)x=.15f*d->normalizedVotes[tid-32];
        else if(tid<55){const int indices[8]={3,4,5,6,7,8,9,10};x=d->positionValid ? .15f*d->positionFeatures[indices[tid-47]]:0;}
        else {
            const float seconds=t.seconds,minutes=t.minutes;
            const float time[9]={sinf(seconds*.1f),cosf(seconds*.1f),sinf(minutes*.1f),cosf(minutes*.1f),
                sinf(minutes/60),cosf(minutes/60),tanhf(d->deltaSeconds),tanhf(minutes/60),1};
            x=.025f*time[tid-55];
        }
        if(!isfinite(x))atomicExch(&valid,0);
    }
    t.key[tid]=x;square[tid]=x*x;__syncthreads();
    for(int s=32;s;s>>=1){if(tid<s)square[tid]+=square[tid+s];__syncthreads();}
    const float norm=square[0];if(valid && isfinite(norm) && norm>1e-12f)t.key[tid]*=rsqrtf(norm);
    else if(tid==0)t.ready=0;
}
// One warp per record; fully coalesced 64-float exact cosine, no CPU/foreign service.
__global__ void score(Buffers b){
    const int id=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(id>=LongMemoryCapacity)return;const auto& t=*b.trace;const auto& e=b.entries[id];
    float value=0;
    const bool valid=t.ready && id<static_cast<int>(b.catalog->used) && e.valid && world(e,t.context) && e.ordinal<t.ordinal;
    if(valid)value=t.key[lane]*e.key[lane]+t.key[lane+32]*e.key[lane+32];
    for(int offset=16;offset;offset>>=1)value+=__shfl_down_sync(0xffffffff,value,offset);
    if(lane==0){
        b.nearestScores[id]=valid && isfinite(value) && value>=.70f ? value:-INFINITY;
        if(!valid || !isfinite(value) || value<.70f || !e.labels){b.scores[id]=-INFINITY;return;}
        float features[8];const float age=ageSeconds(e,t);
        const float phase=fabsf(sinf(t.seconds*.1f)-sinf(e.seconds*.1f))*.5f
            +fabsf(sinf(t.minutes*.1f)-sinf(e.minutes*.1f))*.5f;
        ageFeatures(features,age,value,e.quality,e.labels,e.visits,phase);
        const float retain=gate(b.learned->parameters.retain,features);
        // A retention floor leaves very old but semantically relevant experiences retrievable.
        b.scores[id]=value+.025f*logf(.05f+.95f*retain);
    }
}
__global__ void retrieve(Buffers b){
    __shared__ float scores[256];__shared__ int ids[256];
    const int tid=threadIdx.x;auto& t=*b.trace;
    if(!t.ready)return;
    // Unlabeled observations participate in novelty/dedup, never crowd out supported memories.
    float nearest=-INFINITY;int nearId=-1;
    for(int id=tid;id<LongMemoryCapacity;id+=256)if(b.nearestScores[id]>nearest){nearest=b.nearestScores[id];nearId=id;}
    scores[tid]=nearest;ids[tid]=nearId;__syncthreads();
    for(int width=128;width;width>>=1){if(tid<width && (scores[tid+width]>scores[tid]
        || scores[tid+width]==scores[tid] && ids[tid+width]>=0 && (ids[tid]<0||ids[tid+width]<ids[tid]))){
        scores[tid]=scores[tid+width];ids[tid]=ids[tid+width];}__syncthreads();}
    if(tid==0){t.nearestSlot=ids[0];t.maximumSimilarity=ids[0]>=0 ? scores[0]:-1;}__syncthreads();
    for(int rank=0;rank<LongMemoryTopK;++rank){
        float best=-INFINITY;int index=-1;
        for(int id=tid;id<LongMemoryCapacity;id+=256){
            bool seen=false;for(int j=0;j<rank;++j)seen|=t.selected[j]==id;
            if(!seen && (b.scores[id]>best || b.scores[id]==best && index>=0 && id<index)){best=b.scores[id];index=id;}
        }
        scores[tid]=best;ids[tid]=index;__syncthreads();
        for(int width=128;width;width>>=1){if(tid<width && (scores[tid+width]>scores[tid]
            || scores[tid+width]==scores[tid] && ids[tid+width]>=0 && (ids[tid]<0||ids[tid+width]<ids[tid]))){
            scores[tid]=scores[tid+width];ids[tid]=ids[tid+width];}__syncthreads();}
        if(tid==0)t.selected[rank]=ids[0];__syncthreads();
    }
    if(tid!=0)return;
    auto& p=b.learned->parameters;auto& o=*b.output;
    float sum=0,age=0,maximum=t.maximumSimilarity,quality=0,labelSupport=0,visits=0,phase=0;
    for(int rank=0;rank<LongMemoryTopK;++rank){const int id=t.selected[rank];if(id<0)continue;
        auto& e=b.entries[id];float cosine=0;for(int d=0;d<64;++d)cosine+=t.key[d]*e.key[d];
        if(!t.frozen)++e.visits;
        if(!e.labels)continue;
        const float weight=expf(8*(b.scores[id]-1))*fminf(1.f,e.labels/4.f);sum+=weight;++t.count;
        age+=weight*ageSeconds(e,t);quality+=weight*e.quality;labelSupport+=weight*fminf(8.f,e.labels);
        visits+=weight*e.visits;phase+=weight*(fabsf(sinf(t.seconds*.1f)-sinf(e.seconds*.1f))*.5f
            +fabsf(sinf(t.minutes*.1f)-sinf(e.minutes*.1f))*.5f);
        for(int a=0;a<VoteCount;++a)t.memoryEvidence[a]+=weight*e.action[a];
        for(int k=0;k<LongMemoryKChoices;++k)t.memoryK[k]+=weight*e.kMass[k];
    }
    t.maximumSimilarity=maximum;t.novelty=maximum<0 ? 1:fmaxf(0,1-maximum);
    t.support=sum>1e-12f;
    if(t.support){
        age/=sum;quality/=sum;labelSupport/=sum;visits/=sum;phase/=sum;
        for(int a=0;a<VoteCount;++a)t.memoryEvidence[a]=fminf(4,fmaxf(-4,logit(t.memoryEvidence[a]/sum)));
        for(int k=0;k<LongMemoryKChoices;++k)t.memoryK[k]=logf(fmaxf(1e-5f,t.memoryK[k]/sum));
        ageFeatures(t.readFeatures,age,maximum,quality,labelSupport,visits,phase);
        for(int j=0;j<8;++j)t.retainFeatures[j]=t.readFeatures[j];
        t.readGate=gate(p.read,t.readFeatures);t.retainGate=gate(p.retain,t.retainFeatures);
        for(int a=0;a<VoteCount;++a){o.evidence[a]=.5f*tanhf(p.evidenceGain[a])*t.readGate*t.memoryEvidence[a];o.active|=o.evidence[a]!=0;}
        for(int k=0;k<LongMemoryKChoices;++k){o.kLogits[k]=.5f*tanhf(p.kGain[k])*t.readGate*t.memoryK[k];o.active|=o.kLogits[k]!=0;}
        if(!t.frozen)++b.catalog->hits;
    }
    const float gap=b.runtime->lastWrite ? fminf(10,(t.captureNanos-b.runtime->lastWrite)*1e-9f):10;
    const float wf[8]={1,t.novelty,gap/10,t.support,maximum<0 ? 0:maximum,sinf(t.seconds*.1f),sinf(t.minutes*.1f),t.frozen ? 1.f:0.f};
    for(int j=0;j<8;++j)t.writeFeatures[j]=wf[j];
    t.writeGate=gate(p.write,t.writeFeatures);o.valid=1;
    o.status={t.frame,b.catalog->queries,b.catalog->writes,b.catalog->hits,b.catalog->supervised,
        b.learned->steps,b.catalog->revision,b.catalog->used,LongMemoryCapacity,static_cast<unsigned>(t.count),1,1,
        t.seconds,t.minutes,age,age/60,maximum,t.readGate,t.writeGate,t.retainGate,0};
    for(int a=0;a<VoteCount;++a)o.status.strength=fmaxf(o.status.strength,fabsf(o.evidence[a]));
}
__global__ void record(Buffers b,const DecisionOutput* d,const DecisionParameters* p){
    auto& t=*b.trace;if(t.frozen)return;
    if(!t.ready || !d->valid || d->frameId!=t.frame){t.ready=0;return;}
    for(int a=0;a<VoteCount;++a)t.posterior[a]=d->posterior[a];
    for(int k=0;k<LongMemoryKChoices;++k)t.kProbability[k]=d->kProbability[k];
    const float temperature=p->logGateTemperature>20 ? p->logGateTemperature:log1pf(expf(p->logGateTemperature));
    for(int a=0;a<VoteCount;++a)t.softDerivative[a]=d->softEligibility[a]*(1-d->softEligibility[a])/temperature;
    const bool gap=!b.runtime->lastWrite || t.captureNanos-b.runtime->lastWrite>=500000000ull;
    int closest=t.nearestSlot;
    if(closest>=0 && t.maximumSimilarity>=.985f){
        t.writtenSlot=closest;b.entries[closest].lastSeenMillis=t.context.wallMillis;
    }else if(gap && t.writeGate>=.45f){
        auto& catalog=*b.catalog;int slot;
        if(catalog.used<LongMemoryCapacity)slot=catalog.used++;
        else {
            float lowest=INFINITY;slot=catalog.evictionCursor%LongMemoryCapacity;
            // Bounded retention-aware eviction: examine 64 slots, never sort/allocate.
            for(int j=0;j<64;++j){const int id=(catalog.evictionCursor+j)%LongMemoryCapacity;auto& e=b.entries[id];
                float f[8];ageFeatures(f,ageSeconds(e,t),0,e.quality,e.labels,e.visits,0);
                const float importance=gate(b.learned->parameters.retain,f)*(1+log1pf(e.labels))+.01f*log1pf(e.visits);
                if(importance<lowest){lowest=importance;slot=id;}}
            catalog.evictionCursor=(catalog.evictionCursor+64)%LongMemoryCapacity;
        }
        auto& e=b.entries[slot];e={};
        for(int d=0;d<64;++d)e.key[d]=t.key[d];
        for(int a=0;a<VoteCount;++a)e.action[a]=.5f;
        for(int k=0;k<LongMemoryKChoices;++k)e.kMass[k]=1.f/LongMemoryKChoices;
        e.worldHi=t.context.worldHi;e.worldLo=t.context.worldLo;e.sessionHi=t.context.sessionHi;e.sessionLo=t.context.sessionLo;
        e.ordinal=t.ordinal;e.wallMillis=e.lastSeenMillis=t.context.wallMillis;e.observedSeconds=t.observedSeconds;
        e.seconds=t.seconds;e.minutes=t.minutes;e.quality=.5f;e.valid=1;
        t.writtenSlot=slot;b.runtime->lastWrite=t.captureNanos;++catalog.writes;++catalog.revision;
    }
    auto& s=b.output->status;s.used=b.catalog->used;s.writes=b.catalog->writes;s.revision=b.catalog->revision;s.steps=b.learned->steps;
}
__device__ float ce(float p,bool y){return -logf(probability(y ? p:1-p));}
__global__ void train(Buffers b,const DecisionParameters* base,const std::uint64_t* frame,const int* active,const std::uint32_t* label,LongMemoryContext current){
    auto& t=*b.trace;
    if(!*active || !t.ready || t.frozen || t.consumed || t.frame!=*frame)return;
    if(t.context.worldHi!=current.worldHi||t.context.worldLo!=current.worldLo
        ||t.context.sessionHi!=current.sessionHi||t.context.sessionLo!=current.sessionLo)return;
    t.consumed=1;const unsigned target=*label;const int count=__popc(target);auto& persistent=*b.learned;
    auto& p=persistent.parameters;
    float baseLoss=0,memoryLoss=0,readChain=0;
    const int activeK=static_cast<int>(fminf(LongMemoryKChoices,fmaxf(1,base->enabledKChoices)));
    float dk[LongMemoryKChoices];for(int k=0;k<LongMemoryKChoices;++k)dk[k]=count<activeK && k<activeK ? t.kProbability[k]-(k==count ? 1.f:0.f):0;
    float gradients[LongMemoryParameterCount]{};
    for(int a=0;a<VoteCount;++a){
        const bool y=(target&(1u<<a))!=0;baseLoss+=ce(t.basePosterior[a],y)/VoteCount;
        memoryLoss+=ce(sigmoid(t.memoryEvidence[a]),y)/VoteCount;
        float de=(t.posterior[a]-(y ? 1.f:0.f))/VoteCount;
        // Full k-feature path, evaluated at the cached causal forward, before base head updates.
        const float gateDerivative=t.softDerivative[a];
        for(int k=0;k<LongMemoryKChoices;++k)
            de+=dk[k]*(base->kWeight[k][DecisionWidth+a]
                +base->kWeight[k][DecisionWidth+VoteCount+a]*gateDerivative)*t.posterior[a]*(1-t.posterior[a]);
        const float gain=tanhf(p.evidenceGain[a]);
        gradients[24+a]=t.support ? de*.5f*(1-gain*gain)*t.readGate*t.memoryEvidence[a]:0;
        readChain+=t.support ? de*.5f*gain*t.memoryEvidence[a]:0;
    }
    for(int k=0;k<LongMemoryKChoices;++k){const float gain=tanhf(p.kGain[k]);
        gradients[24+VoteCount+k]=t.support ? dk[k]*.5f*(1-gain*gain)*t.readGate*t.memoryK[k]:0;
        readChain+=t.support ? dk[k]*.5f*gain*t.memoryK[k]:0;}
    const float useful=memoryLoss<baseLoss ? 1.f:0.f;
    const float writeTarget=t.novelty>.015f || baseLoss>.35f ? 1.f:0.f;
    const float retainTarget=memoryLoss<.7f ? 1.f:.2f;
    const float dr=t.support ? readChain*t.readGate*(1-t.readGate)+.02f*(t.readGate-useful):0;
    const float dw=.02f*(t.writeGate-writeTarget);
    const float df=t.support ? .02f*(t.retainGate-retainTarget):0;
    for(int i=0;i<8;++i){gradients[i]=dr*t.readFeatures[i];gradients[8+i]=dw*t.writeFeatures[i];gradients[16+i]=df*t.retainFeatures[i];}
    const float step=static_cast<float>(++persistent.steps),bc1=1-powf(.9f,step),bc2=1-powf(.999f,step);
    auto* weights=reinterpret_cast<float*>(&p);auto* first=reinterpret_cast<float*>(&persistent.first);auto* second=reinterpret_cast<float*>(&persistent.second);
    for(int i=0;i<LongMemoryParameterCount;++i){
        const float raw=isfinite(gradients[i]) ? gradients[i]:0;b.gradient[i]=raw;
        const float g=fminf(1,fmaxf(-1,raw));first[i]=.9f*first[i]+.1f*g;second[i]=.999f*second[i]+.001f*g*g;
        weights[i]=fminf(8,fmaxf(-8,weights[i]*.999997f-.0003f*(first[i]/bc1)/(sqrtf(second[i]/bc2)+1e-8f)));
    }
    // Real human labels annotate the matching old visual experience. Predictions never become labels.
    if(t.writtenSlot>=0){auto& e=b.entries[t.writtenSlot];
        if(e.valid && world(e,t.context)){
            const float alpha=1.f/fminf(256,static_cast<float>(e.labels+1));
            for(int a=0;a<VoteCount;++a)e.action[a]+=alpha*(((target&(1u<<a)) ? 1.f:0.f)-e.action[a]);
            for(int k=0;k<LongMemoryKChoices;++k)e.kMass[k]+=alpha*((k==count ? 1.f:0.f)-e.kMass[k]);
            e.labels=min(1000000u,e.labels+1);e.quality=.99f*e.quality+.01f*(1.f/(1+memoryLoss));
            ++b.catalog->supervised;++b.catalog->revision;
        }
    }
}
__global__ void reset(Buffers b,bool enabled){*b.runtime={};*b.trace={};*b.output={};
    b.output->status={0,b.catalog->queries,b.catalog->writes,b.catalog->hits,b.catalog->supervised,
        b.learned->steps,b.catalog->revision,b.catalog->used,LongMemoryCapacity,0,static_cast<unsigned>(enabled),0};}
}
struct VabpnnLongMemory {Buffers b{};cudaEvent_t initialized=nullptr;bool enabled=true;};
namespace {void wait(VabpnnLongMemory* m,cudaStream_t s){if(!m)throw std::invalid_argument("Long memory is not initialized");check(cudaStreamWaitEvent(s,m->initialized,0),"wait for external memory");}}
VabpnnLongMemory* vabpnnCreateLongMemory(cudaStream_t stream){
    auto* m=new VabpnnLongMemory;
    try{
        auto allocate=[&](auto** p,std::size_t bytes){check(cudaMalloc(reinterpret_cast<void**>(p),bytes),"allocate external memory");check(cudaMemsetAsync(*p,0,bytes,stream),"initialize external memory");};
        allocate(&m->b.learned,sizeof(LongMemoryPersistent));allocate(&m->b.catalog,sizeof(LongMemoryCatalog));
        allocate(&m->b.entries,LongMemoryCapacity*sizeof(LongMemoryEntry));allocate(&m->b.output,sizeof(LongMemoryOutput));
        allocate(&m->b.trace,sizeof(LongMemoryTrace));allocate(&m->b.runtime,sizeof(Runtime));
        allocate(&m->b.scores,LongMemoryCapacity*sizeof(float));allocate(&m->b.nearestScores,LongMemoryCapacity*sizeof(float));allocate(&m->b.gradient,LongMemoryParameterCount*sizeof(float));
        initialize<<<1,1,0,stream>>>(m->b);check(cudaEventCreateWithFlags(&m->initialized,cudaEventDisableTiming),"create memory initialization event");
        check(cudaEventRecord(m->initialized,stream),"record memory initialization");return m;
    }catch(...){vabpnnDestroyLongMemory(m);throw;}
}
void vabpnnDestroyLongMemory(VabpnnLongMemory* m) noexcept{if(!m)return;if(m->initialized)cudaEventSynchronize(m->initialized);
    cudaFree(m->b.learned);cudaFree(m->b.catalog);cudaFree(m->b.entries);cudaFree(m->b.output);cudaFree(m->b.trace);cudaFree(m->b.runtime);cudaFree(m->b.scores);cudaFree(m->b.nearestScores);cudaFree(m->b.gradient);
    if(m->initialized)cudaEventDestroy(m->initialized);delete m;}
std::uint64_t vabpnnLongMemoryBytes(const VabpnnLongMemory* m) noexcept{return m ? sizeof(LongMemoryPersistent)+sizeof(LongMemoryCatalog)
    +LongMemoryCapacity*(sizeof(LongMemoryEntry)+2*sizeof(float))+sizeof(LongMemoryOutput)+sizeof(LongMemoryTrace)+sizeof(Runtime)+LongMemoryParameterCount*sizeof(float):0;}
bool vabpnnLongMemoryEnabled(const VabpnnLongMemory* m) noexcept{return m&&m->enabled;}
vabpnn::DeviceLongMemoryView vabpnnLongMemoryView(VabpnnLongMemory* m){if(!m)throw std::invalid_argument("No memory model");return {m->b.learned,m->b.catalog,m->b.entries,m->b.output,m->b.trace,m->b.gradient};}
vabpnn::ModelPart vabpnnLongMemoryCheckpointPart(VabpnnLongMemory* m){wait(m,nullptr);return {m->b.learned,sizeof(LongMemoryPersistent)};}
std::size_t vabpnnLongMemoryStoreParts(VabpnnLongMemory* m,vabpnn::ModelPart* parts){wait(m,nullptr);parts[0]={m->b.catalog,sizeof(LongMemoryCatalog)};parts[1]={m->b.entries,LongMemoryCapacity*sizeof(LongMemoryEntry)};return 2;}
void vabpnnConfigureLongMemory(VabpnnLongMemory* m,bool enabled,cudaStream_t stream){wait(m,stream);m->enabled=enabled;reset<<<1,1,0,stream>>>(m->b,enabled);check(cudaGetLastError(),"configure memory");}
void vabpnnResetLongMemoryTransient(VabpnnLongMemory* m,cudaStream_t stream){if(!m)return;wait(m,stream);reset<<<1,1,0,stream>>>(m->b,m->enabled);}
void vabpnnRecallLongMemory(const VabpnnGpuFrame& frame,const DecisionOutput* d,const float* latents,VabpnnLongMemory* m){if(!vabpnnLongMemoryEnabled(m))return;wait(m,frame.stream);
    prepare<<<1,64,0,frame.stream>>>(m->b,d,latents,frame.frameId,frame.captureNanos,frame.longMemoryContext,frame.evaluationMode);
    score<<<LongMemoryCapacity/4,128,0,frame.stream>>>(m->b);retrieve<<<1,256,0,frame.stream>>>(m->b);check(cudaGetLastError(),"retrieve timestamped long memory");}
void vabpnnRecordLongMemory(const VabpnnGpuFrame& frame,const DecisionOutput* d,VabpnnLongMemory* m){if(!vabpnnLongMemoryEnabled(m)||frame.evaluationMode)return;
    record<<<1,1,0,frame.stream>>>(m->b,d,vabpnnDecisionModelView(frame.decisionModel).parameters);check(cudaGetLastError(),"cache causal memory result");}
void vabpnnTrainLongMemory(VabpnnLongMemory* m,const DecisionParameters* p,const std::uint64_t* frame,const int* active,const std::uint32_t* label,const LongMemoryContext& context,cudaStream_t stream){
    if(!vabpnnLongMemoryEnabled(m))return;if(!p||!frame||!active||!label)throw std::invalid_argument("Memory training requires causal device targets");
    wait(m,stream);train<<<1,1,0,stream>>>(m->b,p,frame,active,label,context);check(cudaGetLastError(),"learn memory gates from human demonstration");}
