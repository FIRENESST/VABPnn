#include "DenseBlockTransformer.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace vabpnn;
constexpr float HeadScale=0.3535533905932738f,ReadoutScale=0.25f;
constexpr int ReduceThreads=128;
struct DenseState {
    std::uint64_t origin,lastNanos,frameId;
    int started,count,head,overrideFeatures,trainActive,policyActive;
    float elapsedSeconds,elapsedMinutes;
    float time[DenseTimeFeatures],bias1,bias2;
    float deltaSeconds,gateBias1,gateBias2;int resetMemory;
};
struct Buffers {
    int n;
    DenseBlockPersistent* persistent;
    float *bias,*first,*second,*gradient,*probability;
    float *memory,*previousMemory,*gateDerivative[DenseMemoryGates],*gateRows,*gateGainParts,*memoryPartials;
    DenseBlockParameters* smallGradient;
    DenseState* state;
    std::int32_t* history;
    float *features,*encoded,*query,*key,*value,*context,*y1,*y2,*xhat1,*xhat2,*inv1,*inv2,*ff;
    float *readout,*hiddenGradient,*dy2,*dz2,*dy1,*dz1,*dff,*dcontext,*dencoded,*dq,*dk,*dv;
    DenseBlockOutput* output;
    DenseBlockStatus* status;
};
void check(cudaError_t c,const char* op) {
    if(c!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(c));
}
__device__ float softplus(float x) {return x>20 ? x:log1pf(expf(x));}
__device__ float sigmoid(float x) {return 1/(1+expf(-x));}
__device__ float gelu(float x) {
    return .5f*x*(1+tanhf(.7978845608f*(x+.044715f*x*x*x)));
}
__device__ float geluDerivative(float x) {
    const float t=tanhf(.7978845608f*(x+.044715f*x*x*x));
    return .5f*(1+t)+.5f*x*(1-t*t)*.7978845608f*(1+3*.044715f*x*x);
}
__device__ float seeded(unsigned i,float scale) {
    i+=113; i^=i>>16; i*=0x7feb352du; i^=i>>15; i*=0x846ca68bu; i^=i>>16;
    return (static_cast<float>(i&65535u)/32767.5f-1)*scale;
}
__global__ void initializeSmall(DenseBlockPersistent* persistent) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=DenseSmallParameters)return;
    auto* p=&persistent->parameters;float v=0;
    const unsigned nodeEnd=offsetof(DenseBlockParameters,input)/4;
    if(i<nodeEnd)v=seeded(i,.1f);
    else if(i>=offsetof(DenseBlockParameters,input)/4 && i<offsetof(DenseBlockParameters,attentionGain)/4)
        v=seeded(i,i<offsetof(DenseBlockParameters,query)/4 ? .04f:.25f);
    else if((i>=offsetof(DenseBlockParameters,up)/4 && i<offsetof(DenseBlockParameters,upBias)/4)
         ||(i>=offsetof(DenseBlockParameters,down)/4 && i<offsetof(DenseBlockParameters,downBias)/4))
        v=seeded(i,.2f);
    else if(i>=offsetof(DenseBlockParameters,readout)/4 && i<offsetof(DenseBlockParameters,logSecondsFrequency)/4)
        v=seeded(i,.1f);
    else if((i>=offsetof(DenseBlockParameters,attentionGain)/4 && i<offsetof(DenseBlockParameters,attentionBias)/4)
         ||(i>=offsetof(DenseBlockParameters,outputGain)/4 && i<offsetof(DenseBlockParameters,outputBias)/4))v=1;
    if(i>=offsetof(DenseBlockParameters,memoryGate)/4) {
        const int at=i-static_cast<unsigned>(offsetof(DenseBlockParameters,memoryGate)/4);
        if(at<DenseHeads*DenseGateWeightsPerHead) {
            const int gate=at/DenseMemoryFeatures%DenseMemoryGates,col=at%DenseMemoryFeatures;
            v=gate==0 && col==DenseMemoryFeatures-1 ? -2.252168461f:0; // Initial forget rate 0.1/s.
        } else v=1; // One gain per head; zero first-frame memory preserves beta output.
    }
    reinterpret_cast<float*>(p)[i]=v;
}
__global__ void prepareFrame(Buffers b,std::uint64_t frame,std::uint64_t nanos,int overrideFeatures) {
    auto* s=b.state;
    float dt=0;
    if(s->started && nanos>s->lastNanos)dt=static_cast<float>((nanos-s->lastNanos)*1e-9);
    s->resetMemory=!s->started || nanos<s->lastNanos;
    s->deltaSeconds=s->resetMemory ? 0:dt;
    if(s->resetMemory) {
        s->origin=nanos;s->count=0;s->head=-1;s->started=1;
    }
    s->head=(s->head+1)%DenseVoteHistory;
    s->count=s->count<DenseVoteHistory ? s->count+1:DenseVoteHistory;
    s->lastNanos=nanos;s->frameId=frame;s->overrideFeatures=overrideFeatures;
    s->elapsedSeconds=static_cast<float>((nanos-s->origin)*1e-9);
    s->elapsedMinutes=s->elapsedSeconds/60;
    const float fs=softplus(b.persistent->parameters.logSecondsFrequency);
    const float fm=softplus(b.persistent->parameters.logMinutesFrequency);
    s->time[0]=log1pf(dt);s->time[1]=sinf(s->elapsedSeconds*fs);
    s->time[2]=cosf(s->elapsedSeconds*fs);s->time[3]=log1pf(s->elapsedSeconds);
    s->time[4]=log1pf(dt/60);s->time[5]=sinf(s->elapsedMinutes*fm);
    s->time[6]=cosf(s->elapsedMinutes*fm);s->time[7]=log1pf(s->elapsedMinutes);
    b.output->frameId=frame;b.output->captureNanos=nanos;b.output->valid=1;
    b.status->frameId=frame;b.status->steps=b.persistent->steps;b.status->gateSteps=b.persistent->memorySteps;b.status->valid=1;
}
__global__ void encodeFeatures(Buffers b,const std::int32_t* votes,const float* position) {
    const int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=b.n)return;
    const auto* s=b.state;const int slot=s->head;
    for(int k=0;k<VoteCount;++k)b.history[(slot*b.n+node)*VoteCount+k]=votes[k*b.n+node];
    auto* x=b.features+node*DenseFeatures;
    for(int age=0;age<DenseVoteHistory;++age)for(int k=0;k<VoteCount;++k) {
        const int at=(slot-age+DenseVoteHistory)%DenseVoteHistory;
        x[age*VoteCount+k]=age<s->count ? static_cast<float>(b.history[(at*b.n+node)*VoteCount+k])/256:0;
    }
    for(int f=0;f<DenseTimeFeatures;++f)x[VoteCount*DenseVoteHistory+f]=s->time[f];
    for(int f=0;f<PositionFeatures;++f)x[VoteCount*DenseVoteHistory+DenseTimeFeatures+f]=position ? position[f]:0;
    x[DenseFeatures-2]=static_cast<float>(node%120)/119*2-1;
    x[DenseFeatures-1]=static_cast<float>(node/120)/63*2-1;
}
__global__ void project(Buffers b) {
    const int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=b.n)return;
    const auto& p=b.persistent->parameters;float x[DenseWidth];
    for(int d=0;d<DenseWidth;++d) {
        float v=p.nodeEmbedding[node][d];
        for(int f=0;f<DenseFeatures;++f)v+=b.features[node*DenseFeatures+f]*p.input[f][d];
        x[d]=v;b.encoded[node*DenseWidth+d]=v;
    }
    for(int d=0;d<DenseWidth;++d) {
        float q=0,k=0,v=0;for(int e=0;e<DenseWidth;++e) {
            q+=x[e]*p.query[e][d];k+=x[e]*p.key[e][d];v+=x[e]*p.value[e][d];
        }
        const int index=(d/DenseHeadWidth*b.n+node)*DenseHeadWidth+d%DenseHeadWidth;
        b.query[index]=q;b.key[index]=k;b.value[index]=v;
    }
}
struct MemoryValues {float forgetLogit,retention,write,read,candidate,memory;};
__device__ void memoryFeatures(Buffers b,int h,int row,int col,float previous,float* x) {
    for(int d=0;d<DenseHeadWidth;++d) {
        x[d]=b.query[(h*b.n+row)*DenseHeadWidth+d];
        x[DenseHeadWidth+d]=b.key[(h*b.n+col)*DenseHeadWidth+d];
    }
    x[DenseHeadWidth*2]=previous;x[DenseHeadWidth*2+1]=b.state->time[0];x[DenseMemoryFeatures-1]=1;
}
__device__ MemoryValues memoryValues(Buffers b,int h,int row,int col,float raw,float previous) {
    float x[DenseMemoryFeatures],z[DenseMemoryGates]{};memoryFeatures(b,h,row,col,previous,x);
    for(int g=0;g<DenseMemoryGates;++g)for(int f=0;f<DenseMemoryFeatures;++f)
        z[g]+=x[f]*b.persistent->parameters.memoryGate.weight[h][g][f];
    MemoryValues v{};v.forgetLogit=z[0];v.retention=expf(-softplus(z[0])*b.state->deltaSeconds);
    v.write=sigmoid(z[1]);v.read=sigmoid(z[2]);v.candidate=tanhf(raw);
    v.memory=v.retention*previous+(1-v.retention)*v.write*v.candidate;return v;
}
__global__ void scores(Buffers b) {
    const int row=blockIdx.y%b.n,h=blockIdx.y/b.n,col=blockIdx.x*256+threadIdx.x,t=threadIdx.x;
    __shared__ float sums[256],maxima[256];float absolute=0;
    if(col<b.n) {
        const auto index=static_cast<std::size_t>(h*b.n+row)*b.n+col;
        float dot=0;for(int d=0;d<DenseHeadWidth;++d)
            dot+=b.query[(h*b.n+row)*DenseHeadWidth+d]*b.key[(h*b.n+col)*DenseHeadWidth+d];
        const float raw=HeadScale*dot+b.bias[index];
        const float previous=b.state->resetMemory ? 0:b.memory[index];b.previousMemory[index]=previous;
        const auto v=memoryValues(b,h,row,col,raw,previous);b.memory[index]=v.memory;
        b.probability[index]=raw+b.persistent->parameters.memoryGate.gain[h]*v.read*v.memory;
        absolute=fabsf(v.memory);
    }
    sums[t]=maxima[t]=absolute;__syncthreads();
    for(int stride=128;stride;stride/=2) {
        if(t<stride){sums[t]+=sums[t+stride];maxima[t]=fmaxf(maxima[t],maxima[t+stride]);}__syncthreads();
    }
    if(t==0) {
        const int part=blockIdx.y*((b.n+255)/256)+blockIdx.x;
        b.memoryPartials[part*2]=sums[0];b.memoryPartials[part*2+1]=maxima[0];
    }
}
__global__ void softmaxRows(float* matrix,int rows,int cols) {
    const int row=blockIdx.x,t=threadIdx.x;
    __shared__ float reduction[ReduceThreads];
    auto* values=matrix+static_cast<std::size_t>(row)*cols;
    float m=-INFINITY;for(int j=t;j<cols;j+=ReduceThreads)m=fmaxf(m,values[j]);
    reduction[t]=m;__syncthreads();
    for(int d=ReduceThreads/2;d;d/=2) {if(t<d)reduction[t]=fmaxf(reduction[t],reduction[t+d]);__syncthreads();}
    m=reduction[0];float sum=0;
    for(int j=t;j<cols;j+=ReduceThreads){values[j]=expf(values[j]-m);sum+=values[j];}
    reduction[t]=sum;__syncthreads();
    for(int d=ReduceThreads/2;d;d/=2) {if(t<d)reduction[t]+=reduction[t+d];__syncthreads();}
    sum=reduction[0];
    for(int j=t;j<cols;j+=ReduceThreads)values[j]/=sum;
}
__global__ void context(Buffers b) {
    const int row=blockIdx.x%b.n,h=blockIdx.x/b.n,t=threadIdx.x;
    __shared__ float reduce[DenseHeadWidth][ReduceThreads];float sum[DenseHeadWidth]{};
    const auto* p=b.probability+static_cast<std::size_t>(h*b.n+row)*b.n;
    for(int j=t;j<b.n;j+=ReduceThreads) {
        const float w=p[j];for(int d=0;d<DenseHeadWidth;++d)sum[d]+=w*b.value[(h*b.n+j)*DenseHeadWidth+d];
    }
    for(int d=0;d<DenseHeadWidth;++d)reduce[d][t]=sum[d];__syncthreads();
    for(int stride=ReduceThreads/2;stride;stride/=2) {
        if(t<stride)for(int d=0;d<DenseHeadWidth;++d)reduce[d][t]+=reduce[d][t+stride];__syncthreads();
    }
    if(t<DenseHeadWidth)b.context[row*DenseWidth+h*DenseHeadWidth+t]=reduce[t][0];
}
__device__ void normalize(float* x,float* hat,float* inverse,const float* gain,const float* bias) {
    float mean=0,v=0;for(int d=0;d<DenseWidth;++d)mean+=x[d];mean/=DenseWidth;
    for(int d=0;d<DenseWidth;++d){const float delta=x[d]-mean;v+=delta*delta;}
    *inverse=rsqrtf(v/DenseWidth+1e-5f);
    for(int d=0;d<DenseWidth;++d){hat[d]=(x[d]-mean)*(*inverse);x[d]=hat[d]*gain[d]+bias[d];}
}
__global__ void postAttention(Buffers b) {
    const int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=b.n)return;
    const auto& p=b.persistent->parameters;float a[DenseWidth],f[DenseHidden],y[DenseWidth];
    for(int d=0;d<DenseWidth;++d) {
        float z=b.encoded[node*DenseWidth+d];
        for(int e=0;e<DenseWidth;++e)z+=b.context[node*DenseWidth+e]*p.attentionOutput[e][d];a[d]=z;
    }
    normalize(a,b.xhat1+node*DenseWidth,b.inv1+node,p.attentionGain,p.attentionBias);
    for(int d=0;d<DenseWidth;++d)b.y1[node*DenseWidth+d]=a[d];
    for(int d=0;d<DenseHidden;++d) {
        float z=p.upBias[d];for(int e=0;e<DenseWidth;++e)z+=a[e]*p.up[e][d];
        b.ff[node*DenseHidden+d]=z;f[d]=gelu(z);
    }
    for(int d=0;d<DenseWidth;++d) {
        float z=a[d]+p.downBias[d];for(int e=0;e<DenseHidden;++e)z+=f[e]*p.down[e][d];y[d]=z;
    }
    normalize(y,b.xhat2+node*DenseWidth,b.inv2+node,p.outputGain,p.outputBias);
    for(int d=0;d<DenseWidth;++d) {
        b.y2[node*DenseWidth+d]=y[d];
        if(!isfinite(y[d])){atomicExch(&b.output->valid,0u);atomicExch(&b.status->valid,0u);}
    }
}
__global__ void readoutScores(Buffers b) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=VoteCount*b.n)return;
    const int action=index/b.n,node=index%b.n;float z=0;
    for(int d=0;d<DenseWidth;++d)z+=b.persistent->parameters.readout[action][d]*b.y2[node*DenseWidth+d];
    b.readout[index]=ReadoutScale*z;
}
__global__ void readout(Buffers b) {
    const int action=blockIdx.x,t=threadIdx.x;
    __shared__ float reduce[DenseWidth][ReduceThreads];float sum[DenseWidth]{};
    for(int j=t;j<b.n;j+=ReduceThreads)for(int d=0;d<DenseWidth;++d)
        sum[d]+=b.readout[action*b.n+j]*b.y2[j*DenseWidth+d];
    for(int d=0;d<DenseWidth;++d)reduce[d][t]=sum[d];__syncthreads();
    for(int stride=ReduceThreads/2;stride;stride/=2){
        if(t<stride)for(int d=0;d<DenseWidth;++d)reduce[d][t]+=reduce[d][t+stride];__syncthreads();
    }
    if(t<DenseWidth)b.output->hidden[action][t]=reduce[t][0];
}
__global__ void beginBackward(Buffers b,const int* active) {
    b.state->trainActive=*active!=0 && b.output->valid && b.output->frameId==b.state->frameId;
}
__global__ void policyGradient(Buffers b,const DecisionParameters* p,const DecisionOutput* out,
                               const int* active,const std::uint32_t* label) {
    const int i=threadIdx.x;if(i>=VoteCount)return;
    const bool valid=*active && out->valid && out->frameId==b.output->frameId && b.output->valid;
    if(i==0)b.state->policyActive=valid;
    if(!valid)return;
    const float y=(*label>>i)&1u;
    const float post=out->posterior[i],gate=out->softEligibility[i],threshold=out->threshold[i];
    const float temperature=softplus(p->logGateTemperature);
    float dp=0,dgate=0,dmean[DenseWidth]{};
    for(int k=0;k<DecisionKChoices;++k) {
        const float g=out->kProbability[k]-(k==__popc(*label));
        dp+=g*p->kWeight[k][DenseWidth+i];dgate+=g*p->kWeight[k][DenseWidth+VoteCount+i];
        for(int d=0;d<DenseWidth;++d)dmean[d]+=g*p->kWeight[k][d]/VoteCount;
    }
    const float gateGradient=dgate*gate*(1-gate)/temperature;
    const float evidence=(post-y)/VoteCount+(dp+gateGradient)*post*(1-post);
    const float thresholdGradient=(threshold-(y ? .25f:.75f))/VoteCount-gateGradient*threshold*(1-threshold);
    for(int d=0;d<DenseWidth;++d)b.hiddenGradient[i*DenseWidth+d]=
        evidence*p->logLikelihoodWeight[i][d]+thresholdGradient*p->thresholdWeight[i][d]+dmean[d];
}
__global__ void backwardReadout(Buffers b,const float* gradient) {
    const int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=b.n || !b.state->trainActive)return;
    float dy[DenseWidth]{};
    for(int a=0;a<VoteCount;++a) {
        float dot=0;for(int d=0;d<DenseWidth;++d)dot+=gradient[a*DenseWidth+d]
            *(b.y2[node*DenseWidth+d]-b.output->hidden[a][d]);
        const float alpha=b.readout[a*b.n+node],ds=alpha*dot*ReadoutScale;
        for(int d=0;d<DenseWidth;++d)dy[d]+=alpha*gradient[a*DenseWidth+d]
            +ds*b.persistent->parameters.readout[a][d];
    }
    for(int d=0;d<DenseWidth;++d)b.dy2[node*DenseWidth+d]=dy[d];
}
__device__ void backwardNorm(const float* dy,const float* hat,float inv,const float* gain,float* dx) {
    float a=0,b=0;for(int d=0;d<DenseWidth;++d){const float v=dy[d]*gain[d];a+=v;b+=v*hat[d];}
    for(int d=0;d<DenseWidth;++d)dx[d]=inv*(dy[d]*gain[d]-(a+hat[d]*b)/DenseWidth);
}
__global__ void backwardPost(Buffers b) {
    const int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=b.n || !b.state->trainActive)return;
    const auto& p=b.persistent->parameters;
    float gz2[DenseWidth],gff[DenseHidden],gy1[DenseWidth],gz1[DenseWidth];
    backwardNorm(b.dy2+node*DenseWidth,b.xhat2+node*DenseWidth,b.inv2[node],p.outputGain,gz2);
    for(int f=0;f<DenseHidden;++f) {
        float v=0;for(int d=0;d<DenseWidth;++d)v+=gz2[d]*p.down[f][d];
        gff[f]=v*geluDerivative(b.ff[node*DenseHidden+f]);b.dff[node*DenseHidden+f]=gff[f];
    }
    for(int d=0;d<DenseWidth;++d) {
        float v=gz2[d];for(int f=0;f<DenseHidden;++f)v+=gff[f]*p.up[d][f];
        gy1[d]=v;b.dy1[node*DenseWidth+d]=v;b.dz2[node*DenseWidth+d]=gz2[d];
    }
    backwardNorm(gy1,b.xhat1+node*DenseWidth,b.inv1[node],p.attentionGain,gz1);
    for(int d=0;d<DenseWidth;++d) {
        b.dz1[node*DenseWidth+d]=gz1[d];float v=0;
        for(int e=0;e<DenseWidth;++e)v+=gz1[e]*p.attentionOutput[d][e];
        b.dcontext[node*DenseWidth+d]=v;
    }
}
__global__ void pairGradient(Buffers b) {
    if(!b.state->trainActive)return;
    const int row=blockIdx.y%b.n,h=blockIdx.y/b.n,col=blockIdx.x*256+threadIdx.x,t=threadIdx.x;
    __shared__ float reduce[256];float gainGradient=0;
    if(col<b.n) {
        const auto index=static_cast<std::size_t>(h*b.n+row)*b.n+col;
        float derivative=0,rawDot=0;
        for(int d=0;d<DenseHeadWidth;++d) {
            derivative+=b.dcontext[row*DenseWidth+h*DenseHeadWidth+d]
                *(b.value[(h*b.n+col)*DenseHeadWidth+d]-b.context[row*DenseWidth+h*DenseHeadWidth+d]);
            rawDot+=b.query[(h*b.n+row)*DenseHeadWidth+d]*b.key[(h*b.n+col)*DenseHeadWidth+d];
        }
        const float ds=b.probability[index]*derivative,previous=b.previousMemory[index];
        const auto v=memoryValues(b,h,row,col,rawDot*HeadScale+b.bias[index],previous);
        const float gain=b.persistent->parameters.memoryGate.gain[h],dm=ds*gain*v.read;
        // Previous memory is the explicit one-step truncated recurrent boundary.
        b.gradient[index]=ds+dm*(1-v.retention)*v.write*(1-v.candidate*v.candidate);
        b.gateDerivative[0][index]=dm*(previous-v.write*v.candidate)
            *(-b.state->deltaSeconds*v.retention*sigmoid(v.forgetLogit));
        b.gateDerivative[1][index]=dm*(1-v.retention)*v.candidate*v.write*(1-v.write);
        b.gateDerivative[2][index]=ds*gain*v.memory*v.read*(1-v.read);
        gainGradient=ds*v.read*v.memory;
    }
    reduce[t]=gainGradient;__syncthreads();
    for(int stride=128;stride;stride/=2){if(t<stride)reduce[t]+=reduce[t+stride];__syncthreads();}
    if(t==0)b.gateGainParts[blockIdx.y*((b.n+255)/256)+blockIdx.x]=reduce[0];
}
__global__ void gateRowsGradient(Buffers b) {
    if(!b.state->trainActive)return;
    constexpr int Width=DenseMemoryGates*(DenseHeadWidth+2)+1;
    const int row=blockIdx.x%b.n,h=blockIdx.x/b.n,t=threadIdx.x;
    __shared__ float reduce[Width][ReduceThreads];float sum[Width]{};
    for(int j=t;j<b.n;j+=ReduceThreads) {
        const auto index=static_cast<std::size_t>(h*b.n+row)*b.n+j;
        for(int g=0;g<DenseMemoryGates;++g) {
            const float dg=b.gateDerivative[g][index];const int at=g*(DenseHeadWidth+2);
            for(int d=0;d<DenseHeadWidth;++d)sum[at+d]+=dg*b.key[(h*b.n+j)*DenseHeadWidth+d];
            sum[at+DenseHeadWidth]+=dg*b.previousMemory[index];sum[at+DenseHeadWidth+1]+=dg;
        }
    }
    const int chunks=(b.n+255)/256;
    for(int j=t;j<chunks;j+=ReduceThreads)sum[Width-1]+=b.gateGainParts[blockIdx.x*chunks+j];
    for(int d=0;d<Width;++d)reduce[d][t]=sum[d];__syncthreads();
    for(int stride=ReduceThreads/2;stride;stride/=2) {
        if(t<stride)for(int d=0;d<Width;++d)reduce[d][t]+=reduce[d][t+stride];__syncthreads();
    }
    if(t==0) {
        auto* out=b.gateRows+blockIdx.x*DenseGateRowValues;
        for(int g=0;g<DenseMemoryGates;++g) {
            const int at=g*(DenseHeadWidth+2),base=g*DenseMemoryFeatures;
            const float total=reduce[at+DenseHeadWidth+1][0];
            for(int d=0;d<DenseHeadWidth;++d) {
                out[base+d]=total*b.query[(h*b.n+row)*DenseHeadWidth+d];
                out[base+DenseHeadWidth+d]=reduce[at+d][0];
            }
            out[base+DenseHeadWidth*2]=reduce[at+DenseHeadWidth][0];
            out[base+DenseHeadWidth*2+1]=total*b.state->time[0];
            out[base+DenseMemoryFeatures-1]=total;
        }
        out[DenseGateRowValues-1]=reduce[Width-1][0];
    }
}
__global__ void backwardQuery(Buffers b) {
    const int row=blockIdx.x%b.n,h=blockIdx.x/b.n,t=threadIdx.x;if(!b.state->trainActive)return;
    __shared__ float reduce[DenseHeadWidth][ReduceThreads];float sum[DenseHeadWidth]{};
    for(int j=t;j<b.n;j+=ReduceThreads) {
        const float ds=b.gradient[static_cast<std::size_t>(h*b.n+row)*b.n+j];
        for(int d=0;d<DenseHeadWidth;++d)sum[d]+=ds*b.key[(h*b.n+j)*DenseHeadWidth+d]*HeadScale;
    }
    for(int d=0;d<DenseHeadWidth;++d)reduce[d][t]=sum[d];__syncthreads();
    for(int stride=ReduceThreads/2;stride;stride/=2){
        if(t<stride)for(int d=0;d<DenseHeadWidth;++d)reduce[d][t]+=reduce[d][t+stride];__syncthreads();
    }
    if(t<DenseHeadWidth) {
        float result=reduce[t][0];
        for(int g=0;g<DenseMemoryGates;++g)result+=b.gateRows[(h*b.n+row)*DenseGateRowValues+g*DenseMemoryFeatures+DenseMemoryFeatures-1]
            *b.persistent->parameters.memoryGate.weight[h][g][t];
        b.dq[(h*b.n+row)*DenseHeadWidth+t]=result;
    }
}
__global__ void backwardKeyValue(Buffers b) {
    const int node=blockIdx.x*16+threadIdx.x,d=threadIdx.y,h=blockIdx.y;
    if(node>=b.n || !b.state->trainActive)return;
    float dk=0,dv=0;
    for(int i=0;i<b.n;++i) {
        const auto index=static_cast<std::size_t>(h*b.n+i)*b.n+node;
        dk+=b.gradient[index]*b.query[(h*b.n+i)*DenseHeadWidth+d]*HeadScale;
        for(int g=0;g<DenseMemoryGates;++g)dk+=b.gateDerivative[g][index]
            *b.persistent->parameters.memoryGate.weight[h][g][DenseHeadWidth+d];
        dv+=b.probability[index]*b.dcontext[i*DenseWidth+h*DenseHeadWidth+d];
    }
    b.dk[(h*b.n+node)*DenseHeadWidth+d]=dk;b.dv[(h*b.n+node)*DenseHeadWidth+d]=dv;
}
__global__ void backwardEncoded(Buffers b) {
    const int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=b.n || !b.state->trainActive)return;
    const auto& p=b.persistent->parameters;
    for(int d=0;d<DenseWidth;++d) {
        float v=b.dz1[node*DenseWidth+d];
        for(int e=0;e<DenseWidth;++e) {
            const int index=(e/DenseHeadWidth*b.n+node)*DenseHeadWidth+e%DenseHeadWidth;
            v+=b.dq[index]*p.query[d][e]+b.dk[index]*p.key[d][e]+b.dv[index]*p.value[d][e];
        }
        b.dencoded[node*DenseWidth+d]=v;
    }
}
#define OFFSET(field) (offsetof(DenseBlockParameters,field)/sizeof(float))
__global__ void smallGradient(Buffers b,const float* hiddenGradient) {
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=DenseSmallParameters || !b.state->trainActive)return;
    float g=0;
    if(index<OFFSET(input)) {
        if(index<static_cast<unsigned>(b.n*DenseWidth))g=b.dencoded[index];
    } else if(index<OFFSET(query)) {
        const int at=index-OFFSET(input),f=at/DenseWidth,d=at%DenseWidth;
        for(int node=0;node<b.n;++node)g+=b.features[node*DenseFeatures+f]*b.dencoded[node*DenseWidth+d];
    } else if(index<OFFSET(attentionOutput)) {
        const int bank=index>=OFFSET(value) ? 2:index>=OFFSET(key) ? 1:0;
        const int start=bank==2 ? OFFSET(value):bank==1 ? OFFSET(key):OFFSET(query);
        const int at=index-start,e=at/DenseWidth,d=at%DenseWidth;
        const auto* gradient=bank==0 ? b.dq:bank==1 ? b.dk:b.dv;
        for(int node=0;node<b.n;++node)g+=b.encoded[node*DenseWidth+e]
            *gradient[(d/DenseHeadWidth*b.n+node)*DenseHeadWidth+d%DenseHeadWidth];
    } else if(index<OFFSET(attentionGain)) {
        const int at=index-OFFSET(attentionOutput),e=at/DenseWidth,d=at%DenseWidth;
        for(int node=0;node<b.n;++node)g+=b.context[node*DenseWidth+e]*b.dz1[node*DenseWidth+d];
    } else if(index<OFFSET(up)) {
        const bool gain=index<OFFSET(attentionBias);const int d=index-(gain ? OFFSET(attentionGain):OFFSET(attentionBias));
        for(int node=0;node<b.n;++node)g+=b.dy1[node*DenseWidth+d]*(gain ? b.xhat1[node*DenseWidth+d]:1);
    } else if(index<OFFSET(upBias)) {
        const int at=index-OFFSET(up),e=at/DenseHidden,f=at%DenseHidden;
        for(int node=0;node<b.n;++node)g+=b.y1[node*DenseWidth+e]*b.dff[node*DenseHidden+f];
    } else if(index<OFFSET(down)) {
        const int f=index-OFFSET(upBias);for(int node=0;node<b.n;++node)g+=b.dff[node*DenseHidden+f];
    } else if(index<OFFSET(downBias)) {
        const int at=index-OFFSET(down),f=at/DenseWidth,d=at%DenseWidth;
        for(int node=0;node<b.n;++node)g+=gelu(b.ff[node*DenseHidden+f])*b.dz2[node*DenseWidth+d];
    } else if(index<OFFSET(outputGain)) {
        const int d=index-OFFSET(downBias);for(int node=0;node<b.n;++node)g+=b.dz2[node*DenseWidth+d];
    } else if(index<OFFSET(readout)) {
        const bool gain=index<OFFSET(outputBias);const int d=index-(gain ? OFFSET(outputGain):OFFSET(outputBias));
        for(int node=0;node<b.n;++node)g+=b.dy2[node*DenseWidth+d]*(gain ? b.xhat2[node*DenseWidth+d]:1);
    } else if(index<OFFSET(logSecondsFrequency)) {
        const int at=index-OFFSET(readout),action=at/DenseWidth,d=at%DenseWidth;
        for(int node=0;node<b.n;++node) {
            float dot=0;for(int e=0;e<DenseWidth;++e)dot+=hiddenGradient[action*DenseWidth+e]
                *(b.y2[node*DenseWidth+e]-b.output->hidden[action][e]);
            g+=b.readout[action*b.n+node]*dot*ReadoutScale*b.y2[node*DenseWidth+d];
        }
    } else if(index<OFFSET(memoryGate) && !b.state->overrideFeatures) {
        const bool minutes=index==OFFSET(logMinutesFrequency);const int start=VoteCount*DenseVoteHistory+(minutes ? 5:1);
        const float elapsed=minutes ? b.state->elapsedMinutes:b.state->elapsedSeconds;
        const float raw=minutes ? b.persistent->parameters.logMinutesFrequency:b.persistent->parameters.logSecondsFrequency;
        for(int node=0;node<b.n;++node)for(int d=0;d<DenseWidth;++d)g+=b.dencoded[node*DenseWidth+d]
            *(b.persistent->parameters.input[start][d]*b.state->time[minutes ? 6:2]
              -b.persistent->parameters.input[start+1][d]*b.state->time[minutes ? 5:1])*elapsed*sigmoid(raw);
    }
    if(index>=OFFSET(memoryGate)) {
        const int at=index-OFFSET(memoryGate);
        const int head=at<DenseHeads*DenseGateWeightsPerHead ? at/DenseGateWeightsPerHead:at-DenseHeads*DenseGateWeightsPerHead;
        const int col=at<DenseHeads*DenseGateWeightsPerHead ? at%DenseGateWeightsPerHead:DenseGateRowValues-1;
        for(int node=0;node<b.n;++node)g+=b.gateRows[(head*b.n+node)*DenseGateRowValues+col];
    }
    reinterpret_cast<float*>(b.smallGradient)[index]=g;
}
__global__ void beginAdam(Buffers b) {
    if(!b.state->trainActive)return;
    const float step=static_cast<float>(++b.persistent->steps);
    b.state->bias1=1-powf(.9f,step);b.state->bias2=1-powf(.999f,step);
    b.status->steps=b.persistent->steps;
    const float gateStep=static_cast<float>(++b.persistent->memorySteps);
    b.state->gateBias1=1-powf(.9f,gateStep);b.state->gateBias2=1-powf(.999f,gateStep);
    b.status->gateSteps=b.persistent->memorySteps;
}
__device__ void adam(float& p,float& m,float& v,float g,float bias1,float bias2,bool decay) {
    g=fminf(1.0f,fmaxf(-1.0f,g));
    m=.9f*m+.1f*g;v=.999f*v+.001f*g*g;
    p=p*(decay ? .999997f:1)-.0003f*(m/bias1)/(sqrtf(v/bias2)+1e-8f);
}
__global__ void adamPairs(Buffers b) {
    const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=static_cast<std::size_t>(DenseHeads)*b.n*b.n || !b.state->trainActive)return;
    adam(b.bias[index],b.first[index],b.second[index],b.gradient[index],b.state->bias1,b.state->bias2,false);
}
__global__ void adamSmall(Buffers b) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=DenseSmallParameters || !b.state->trainActive)return;
    auto* p=reinterpret_cast<float*>(&b.persistent->parameters);
    auto* m=reinterpret_cast<float*>(&b.persistent->first);auto* v=reinterpret_cast<float*>(&b.persistent->second);
    bool decay=i<OFFSET(attentionGain) ||(i>=OFFSET(up) && i<OFFSET(upBias))
        ||(i>=OFFSET(down) && i<OFFSET(downBias))||(i>=OFFSET(readout) && i<OFFSET(logSecondsFrequency));
    if(i>=OFFSET(memoryGate)) {
        const unsigned at=i-OFFSET(memoryGate);
        decay=at<DenseHeads*DenseGateWeightsPerHead && at%DenseMemoryFeatures<DenseMemoryFeatures-1;
    }
    const bool gate=i>=OFFSET(memoryGate);
    adam(p[i],m[i],v[i],reinterpret_cast<const float*>(b.smallGradient)[i],
        gate ? b.state->gateBias1:b.state->bias1,gate ? b.state->gateBias2:b.state->bias2,decay);
}
#undef OFFSET
__global__ void publishMemory(Buffers b) {
    __shared__ float sum[ReduceThreads],maximum[ReduceThreads];
    const int t=threadIdx.x,parts=DenseHeads*b.n*((b.n+255)/256);float s=0,m=0;
    for(int i=t;i<parts;i+=ReduceThreads){s+=b.memoryPartials[i*2];m=fmaxf(m,b.memoryPartials[i*2+1]);}
    sum[t]=s;maximum[t]=m;__syncthreads();
    for(int stride=ReduceThreads/2;stride;stride/=2) {
        if(t<stride){sum[t]+=sum[t+stride];maximum[t]=fmaxf(maximum[t],maximum[t+stride]);}__syncthreads();
    }
    if(t==0) {
        if(b.output->valid)++b.status->memoryFrames;
        b.status->memoryDeltaSeconds=b.state->deltaSeconds;
        b.status->memoryMeanAbs=sum[0]/(static_cast<float>(DenseHeads)*b.n*b.n);
        b.status->memoryMaximumAbs=maximum[0];b.status->memoryReady=b.output->valid;
    }
}
void launchForward(Buffers b,cudaStream_t stream) {
    const unsigned nodeGrid=(b.n+31)/32;
    const unsigned pairGrid=static_cast<unsigned>((static_cast<std::size_t>(DenseHeads)*b.n*b.n+255)/256);
    project<<<nodeGrid,32,0,stream>>>(b);scores<<<dim3((b.n+255)/256,DenseHeads*b.n),256,0,stream>>>(b);
    softmaxRows<<<DenseHeads*b.n,ReduceThreads,0,stream>>>(b.probability,DenseHeads*b.n,b.n);
    context<<<DenseHeads*b.n,ReduceThreads,0,stream>>>(b);
    postAttention<<<nodeGrid,32,0,stream>>>(b);
    readoutScores<<<(VoteCount*b.n+255)/256,256,0,stream>>>(b);
    softmaxRows<<<VoteCount,ReduceThreads,0,stream>>>(b.readout,VoteCount,b.n);
    readout<<<VoteCount,ReduceThreads,0,stream>>>(b);
    publishMemory<<<1,ReduceThreads,0,stream>>>(b);
    check(cudaGetLastError(),"full two-head dense block Transformer forward");
}
}
struct VabpnnDenseBlockModel {
    Buffers b{};std::vector<void*> allocations;std::uint64_t bytes=0;cudaEvent_t initialized=nullptr;
    bool forwarded=false;std::uint64_t lastFrame=0,lastNanos=0;
};
namespace {
template<class T>void allocate(VabpnnDenseBlockModel* m,T*& p,std::size_t count,cudaStream_t stream) {
    const auto bytes=count*sizeof(T);check(cudaMalloc(reinterpret_cast<void**>(&p),bytes),"allocate dense comparison buffer");
    m->allocations.push_back(p);m->bytes+=bytes;check(cudaMemsetAsync(p,0,bytes,stream),"zero dense comparison buffer");
}
void wait(VabpnnDenseBlockModel* m,cudaStream_t stream) {
    if(!m)throw std::invalid_argument("Dense comparison model missing");
    check(cudaStreamWaitEvent(stream,m->initialized,0),"wait for dense comparison initialization");
}
}
VabpnnDenseBlockModel* vabpnnCreateDenseBlockModel(cudaStream_t stream,int n) {
    if(n<1 || n>vabpnn::DenseNodes)throw std::invalid_argument("Dense node count outside 1..7680");
    auto* m=new VabpnnDenseBlockModel();auto& b=m->b;b.n=n;
    try {
        const std::size_t pairs=static_cast<std::size_t>(vabpnn::DenseHeads)*n*n,nw=n*vabpnn::DenseWidth;
        allocate(m,b.persistent,1,stream);allocate(m,b.smallGradient,1,stream);
        for(auto p:{&b.bias,&b.first,&b.second,&b.gradient,&b.probability,&b.memory,&b.previousMemory})allocate(m,*p,pairs,stream);
        for(auto& p:b.gateDerivative)allocate(m,p,pairs,stream);
        allocate(m,b.gateRows,DenseHeads*n*DenseGateRowValues,stream);
        allocate(m,b.gateGainParts,DenseHeads*n*((n+255)/256),stream);
        allocate(m,b.memoryPartials,DenseHeads*n*((n+255)/256)*2,stream);
        allocate(m,b.state,1,stream);allocate(m,b.history,n*vabpnn::DenseVoteHistory*vabpnn::VoteCount,stream);
        allocate(m,b.features,n*vabpnn::DenseFeatures,stream);
        for(auto p:{&b.encoded,&b.query,&b.key,&b.value,&b.context,&b.y1,&b.y2,&b.xhat1,&b.xhat2,
                    &b.dy2,&b.dz2,&b.dy1,&b.dz1,&b.dcontext,&b.dencoded,&b.dq,&b.dk,&b.dv})allocate(m,*p,nw,stream);
        allocate(m,b.inv1,n,stream);allocate(m,b.inv2,n,stream);
        allocate(m,b.ff,n*vabpnn::DenseHidden,stream);allocate(m,b.dff,n*vabpnn::DenseHidden,stream);
        allocate(m,b.readout,n*vabpnn::VoteCount,stream);allocate(m,b.hiddenGradient,vabpnn::VoteCount*vabpnn::DenseWidth,stream);
        allocate(m,b.output,1,stream);allocate(m,b.status,1,stream);
        initializeSmall<<<(vabpnn::DenseSmallParameters+255)/256,256,0,stream>>>(b.persistent);
        vabpnn::DenseBlockStatus status{};status.parameters=pairs+vabpnn::DenseSmallParameters;
        status.pairs=pairs;status.allocatedBytes=m->bytes;status.nodes=n;status.heads=vabpnn::DenseHeads;
        // Pinned/stack lifetime is not allowed to race cudaMemcpyAsync on an initialization stream.
        check(cudaStreamSynchronize(stream),"complete dense comparison allocation");
        check(cudaMemcpy(b.status,&status,sizeof(status),cudaMemcpyHostToDevice),"initialize dense comparison status");
        check(cudaEventCreateWithFlags(&m->initialized,cudaEventDisableTiming),"create dense comparison event");
        check(cudaEventRecord(m->initialized,stream),"record dense comparison initialization");
        return m;
    }catch(...){vabpnnDestroyDenseBlockModel(m);throw;}
}
void vabpnnDestroyDenseBlockModel(VabpnnDenseBlockModel* m) noexcept {
    if(!m)return;if(m->initialized)cudaEventSynchronize(m->initialized);
    for(auto p:m->allocations)cudaFree(p);if(m->initialized)cudaEventDestroy(m->initialized);delete m;
}
std::uint64_t vabpnnDenseBlockModelBytes(const VabpnnDenseBlockModel* m) noexcept {return m ? m->bytes:0;}
vabpnn::DeviceDenseBlockView vabpnnDenseBlockModelView(VabpnnDenseBlockModel* m) {
    if(!m)throw std::invalid_argument("Dense comparison model missing");const auto& b=m->b;
    return {b.persistent,b.bias,b.first,b.second,b.gradient,b.smallGradient,b.probability,b.features,b.output,b.status,b.n,b.memory,b.previousMemory};
}
std::size_t vabpnnDenseBlockCheckpointParts(VabpnnDenseBlockModel* m,vabpnn::ModelPart* parts) {
    const auto& b=m->b;const std::size_t bytes=static_cast<std::size_t>(vabpnn::DenseHeads)*b.n*b.n*sizeof(float);
    parts[0]={b.persistent,sizeof(vabpnn::DenseBlockPersistent)};
    parts[1]={b.bias,bytes};parts[2]={b.first,bytes};parts[3]={b.second,bytes};return 4;
}
void vabpnnLaunchDenseBlocks(VabpnnDenseBlockModel* m,const VabpnnGpuFrame& frame,
    const std::int32_t* votes,const float* position) {
    wait(m,frame.stream);if(!votes)throw std::invalid_argument("Dense comparison needs per-block votes");
    if(m->forwarded && m->lastFrame==frame.frameId && m->lastNanos==frame.captureNanos)return;
    m->forwarded=true;m->lastFrame=frame.frameId;m->lastNanos=frame.captureNanos;
    prepareFrame<<<1,1,0,frame.stream>>>(m->b,frame.frameId,frame.captureNanos,0);
    encodeFeatures<<<(m->b.n+127)/128,128,0,frame.stream>>>(m->b,votes,position);launchForward(m->b,frame.stream);
}
void vabpnnDenseForwardFeatures(VabpnnDenseBlockModel* m,const float* features,
    std::uint64_t frame,std::uint64_t nanos,cudaStream_t stream) {
    wait(m,stream);
    if(m->forwarded && m->lastFrame==frame && m->lastNanos==nanos)return;
    m->forwarded=true;m->lastFrame=frame;m->lastNanos=nanos;
    prepareFrame<<<1,1,0,stream>>>(m->b,frame,nanos,1);
    check(cudaMemcpyAsync(m->b.features,features,m->b.n*vabpnn::DenseFeatures*sizeof(float),cudaMemcpyDeviceToDevice,stream),"copy verification features");
    launchForward(m->b,stream);
}
void vabpnnDenseBackwardHidden(VabpnnDenseBlockModel* m,const float* gradient,const int* active,cudaStream_t stream) {
    wait(m,stream);auto b=m->b;
    beginBackward<<<1,1,0,stream>>>(b,active);
    backwardReadout<<<(b.n+31)/32,32,0,stream>>>(b,gradient);backwardPost<<<(b.n+31)/32,32,0,stream>>>(b);
    pairGradient<<<dim3((b.n+255)/256,vabpnn::DenseHeads*b.n),256,0,stream>>>(b);
    gateRowsGradient<<<vabpnn::DenseHeads*b.n,ReduceThreads,0,stream>>>(b);
    backwardQuery<<<vabpnn::DenseHeads*b.n,ReduceThreads,0,stream>>>(b);
    backwardKeyValue<<<dim3((b.n+15)/16,vabpnn::DenseHeads),dim3(16,vabpnn::DenseHeadWidth),0,stream>>>(b);
    backwardEncoded<<<(b.n+31)/32,32,0,stream>>>(b);
    smallGradient<<<(vabpnn::DenseSmallParameters+255)/256,256,0,stream>>>(b,gradient);
    check(cudaGetLastError(),"full dense Transformer reverse-mode derivatives");
}
void vabpnnDenseApplyAdamW(VabpnnDenseBlockModel* m,cudaStream_t stream) {
    wait(m,stream);auto b=m->b;beginAdam<<<1,1,0,stream>>>(b);
    adamPairs<<<(static_cast<std::size_t>(vabpnn::DenseHeads)*b.n*b.n+255)/256,256,0,stream>>>(b);
    adamSmall<<<(vabpnn::DenseSmallParameters+255)/256,256,0,stream>>>(b);
    check(cudaGetLastError(),"dense Transformer AdamW");
}
void vabpnnTrainDenseBlocks(VabpnnDenseBlockModel* m,const vabpnn::DecisionParameters* base,
    const vabpnn::DecisionOutput* previous,const int* active,const std::uint32_t* label,cudaStream_t stream) {
    wait(m,stream);
    policyGradient<<<1,32,0,stream>>>(m->b,base,previous,active,label);
    vabpnnDenseBackwardHidden(m,m->b.hiddenGradient,&m->b.state->policyActive,stream);vabpnnDenseApplyAdamW(m,stream);
}
void vabpnnResetDenseBlocks(VabpnnDenseBlockModel* m,cudaStream_t stream) {
    wait(m,stream);m->forwarded=false;check(cudaMemsetAsync(m->b.state,0,sizeof(DenseState),stream),"clear comparison observation history");
    check(cudaMemsetAsync(m->b.history,0,m->b.n*vabpnn::DenseVoteHistory*vabpnn::VoteCount*sizeof(std::int32_t),stream),"clear comparison block vote history");
    check(cudaMemsetAsync(m->b.output,0,sizeof(vabpnn::DenseBlockOutput),stream),"clear comparison readout");
    const auto memoryBytes=static_cast<std::size_t>(vabpnn::DenseHeads)*m->b.n*m->b.n*sizeof(float);
    check(cudaMemsetAsync(m->b.memory,0,memoryBytes,stream),"clear episode matrix memory");
    check(cudaMemsetAsync(m->b.previousMemory,0,memoryBytes,stream),"clear cached previous matrix memory");
    check(cudaMemsetAsync(&m->b.status->memoryFrames,0,sizeof(vabpnn::DenseBlockStatus)-offsetof(vabpnn::DenseBlockStatus,memoryFrames),stream),"clear matrix memory telemetry");
}
