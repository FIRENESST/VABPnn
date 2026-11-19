#include "FullSelfLearning.h"
#include <cmath>
#include <stdexcept>
namespace {
using namespace vabpnn;
struct Scratch {
 float input[DecisionHistory][VoteCount][DecisionWidth],key[DecisionHistory][VoteCount][DecisionWidth],value[DecisionHistory][VoteCount][DecisionWidth];
 float queryInput[VoteCount][DecisionWidth];
 float query[VoteCount][DecisionWidth],attention[VoteCount][DecisionHeads][DecisionHistory*VoteCount],context[VoteCount][DecisionWidth];
 float ln1Pre[VoteCount][DecisionWidth],ln1[VoteCount][DecisionWidth],up[VoteCount][DecisionHiddenWidth],expanded[VoteCount][DecisionHiddenWidth],outPre[VoteCount][DecisionWidth];
 float dInput[DecisionHistory][VoteCount][DecisionWidth],dKey[DecisionHistory][VoteCount][DecisionWidth],dValue[DecisionHistory][VoteCount][DecisionWidth],dQuery[VoteCount][DecisionWidth];
 float dContext[VoteCount][DecisionWidth],dLn1Pre[VoteCount][DecisionWidth],dOutPre[VoteCount][DecisionWidth],dUp[VoteCount][DecisionHiddenWidth];
 float time[DecisionHistory][DecisionTimeFeatures],seconds[DecisionHistory],age[DecisionHistory],vote[DecisionHistory][VoteCount];
};
__device__ float sig(float x){return 1/(1+expf(-x));}
__device__ float sp(float x){return x>20?x:log1pf(expf(x));}
__device__ float gelu(float x){return .5f*x*(1+tanhf(.7978845608f*(x+.044715f*x*x*x)));}
__device__ float geluDerivative(float x){float t=tanhf(.7978845608f*(x+.044715f*x*x*x));return .5f*(1+t)+.5f*x*(1-t*t)*.7978845608f*(1+.134145f*x*x);}
__device__ void norm(const float* x,const float* gain,const float* bias,float* y){float m=0,v=0;for(int d=0;d<DecisionWidth;++d)m+=x[d];m/=DecisionWidth;
 for(int d=0;d<DecisionWidth;++d)v+=(x[d]-m)*(x[d]-m);float inv=rsqrtf(v/DecisionWidth+1e-5f);for(int d=0;d<DecisionWidth;++d)y[d]=(x[d]-m)*inv*gain[d]+bias[d];}
__device__ void normBack(const float* x,const float* gain,const float* dy,float* dx){float m=0,v=0;for(int d=0;d<DecisionWidth;++d)m+=x[d];m/=DecisionWidth;
 for(int d=0;d<DecisionWidth;++d)v+=(x[d]-m)*(x[d]-m);float inv=rsqrtf(v/DecisionWidth+1e-5f),sum=0,dot=0;
 for(int d=0;d<DecisionWidth;++d){float g=dy[d]*gain[d];sum+=g;dot+=g*(x[d]-m)*inv;}
 for(int d=0;d<DecisionWidth;++d)dx[d]=inv*(dy[d]*gain[d]-sum/DecisionWidth-(x[d]-m)*inv*dot/DecisionWidth);}
__global__ void forward(const DecisionParameters* p,const SecondsLearningState* state,Scratch* c,const int* active){
 if(active && !*active)return;
 int i=threadIdx.x,count=state->count;int slot=state->head;if(i>=VoteCount)return;
 for(int t=0;t<count;++t){int at=(slot-t+DecisionHistory)%DecisionHistory;float seconds=(state->historyNanos[at]-state->originNanos)*1e-9f;
  float time[4]={log1pf(state->historyDeltaSeconds[at]),sinf(seconds*sp(p->logClockFrequency)),cosf(seconds*sp(p->logClockFrequency)),log1pf(seconds)};
  if(i==0){c->seconds[t]=seconds;c->age[t]=(state->lastCaptureNanos-state->historyNanos[at])*1e-9f;for(int f=0;f<4;++f)c->time[t][f]=time[f];}
  float vote=static_cast<float>(state->historyVotes[at][i])/NeuronCount;c->vote[t][i]=vote;
  for(int d=0;d<DecisionWidth;++d){
   float x=p->nodeEmbedding[i][d]+vote*p->voteEmbedding[d];
   for(int f=0;f<4;++f)x+=time[f]*p->timeEmbedding[f][d];
   if(t==0)c->queryInput[i][d]=x;
   c->input[t][i][d]=x+(state->latentDecoder ? state->historyLatents[at][i][d]:0.f);
  }
  for(int d=0;d<DecisionWidth;++d){float q=0,k=0,v=0;for(int e=0;e<DecisionWidth;++e){float x=c->input[t][i][e];if(t==0)q+=c->queryInput[i][e]*p->query[e][d];k+=x*p->key[e][d];v+=x*p->value[e][d];}
   if(t==0)c->query[i][d]=q;c->key[t][i][d]=k;c->value[t][i][d]=v;}
 }
}
__global__ void attend(const DecisionParameters* p,const SecondsLearningState* state,Scratch* c,const int* active){
 if(active && !*active)return;
 int i=threadIdx.x;if(i>=VoteCount)return;int count=state->count;
 for(int h=0;h<DecisionHeads;++h){float maximum=-INFINITY;for(int t=0;t<count;++t)for(int j=0;j<VoteCount;++j){float x=0;for(int d=0;d<DecisionHeadWidth;++d){int e=h*DecisionHeadWidth+d;x+=c->query[i][e]*c->key[t][j][e];}
  x=x*.3535533906f+p->graphBias[h][i][j]-sp(p->logAttentionDecay[h])*c->age[t];c->attention[i][h][t*VoteCount+j]=x;maximum=fmaxf(maximum,x);}
  float sum=0;for(int t=0;t<count*VoteCount;++t){float a=expf(c->attention[i][h][t]-maximum);c->attention[i][h][t]=a;sum+=a;}
  for(int d=0;d<DecisionHeadWidth;++d)c->context[i][h*DecisionHeadWidth+d]=0;
  for(int t=0;t<count;++t)for(int j=0;j<VoteCount;++j){float a=c->attention[i][h][t*VoteCount+j]/sum;c->attention[i][h][t*VoteCount+j]=a;
   for(int d=0;d<DecisionHeadWidth;++d){int e=h*DecisionHeadWidth+d;c->context[i][e]+=a*c->value[t][j][e];}}
 }
 for(int d=0;d<DecisionWidth;++d){float x=c->queryInput[i][d];for(int e=0;e<DecisionWidth;++e)x+=c->context[i][e]*p->attentionOutput[e][d];c->ln1Pre[i][d]=x;}
 norm(c->ln1Pre[i],p->attentionGain,p->attentionBias,c->ln1[i]);
 for(int d=0;d<DecisionHiddenWidth;++d){float x=p->feedForwardUpBias[d];for(int e=0;e<DecisionWidth;++e)x+=c->ln1[i][e]*p->feedForwardUp[e][d];c->up[i][d]=x;c->expanded[i][d]=gelu(x);}
 for(int d=0;d<DecisionWidth;++d){float x=c->ln1[i][d]+p->feedForwardDownBias[d];for(int e=0;e<DecisionHiddenWidth;++e)x+=c->expanded[i][e]*p->feedForwardDown[e][d];c->outPre[i][d]=x;}
}
__global__ void reverse(const DecisionParameters* p,const SecondsLearningState* state,const FullRewardRoots* roots,
    Scratch* c,float* g,float* spatialGradient,const int* active){
 if(active && !*active)return;
 int i=threadIdx.x,count=state->count;
 for(int at=i;at<FullDecisionParameters;at+=blockDim.x)g[at]=0;
 for(int at=i;at<DecisionHistory*VoteCount*DecisionWidth;at+=blockDim.x){reinterpret_cast<float*>(c->dKey)[at]=0;reinterpret_cast<float*>(c->dValue)[at]=0;}
 if(i<VoteCount){
  normBack(c->outPre[i],p->outputGain,roots->hidden[i],c->dOutPre[i]);
  float dLn1[DecisionWidth];for(int d=0;d<DecisionWidth;++d)dLn1[d]=c->dOutPre[i][d];
  for(int e=0;e<DecisionHiddenWidth;++e){float x=0;for(int d=0;d<DecisionWidth;++d)x+=c->dOutPre[i][d]*p->feedForwardDown[e][d];x*=geluDerivative(c->up[i][e]);c->dUp[i][e]=x;for(int d=0;d<DecisionWidth;++d)dLn1[d]+=x*p->feedForwardUp[d][e];}
  normBack(c->ln1Pre[i],p->attentionGain,dLn1,c->dLn1Pre[i]);
  for(int e=0;e<DecisionWidth;++e){float x=0;for(int d=0;d<DecisionWidth;++d)x+=c->dLn1Pre[i][d]*p->attentionOutput[e][d];c->dContext[i][e]=x;c->dQuery[i][e]=0;}
 }
 __syncthreads();
 constexpr int G=offsetof(DecisionParameters,graphBias)/4,D=offsetof(DecisionParameters,logAttentionDecay)/4;
 if(i<VoteCount)for(int h=0;h<DecisionHeads;++h){float base=0,decay=0;for(int d=0;d<DecisionHeadWidth;++d){int e=h*DecisionHeadWidth+d;base+=c->dContext[i][e]*c->context[i][e];}
  for(int t=0;t<count;++t)for(int j=0;j<VoteCount;++j){float dot=0;for(int d=0;d<DecisionHeadWidth;++d){int e=h*DecisionHeadWidth+d;dot+=c->dContext[i][e]*c->value[t][j][e];}
   float a=c->attention[i][h][t*VoteCount+j],ds=a*(dot-base);g[G+(h*VoteCount+i)*VoteCount+j]+=ds;decay-=ds*c->age[t]*sig(p->logAttentionDecay[h]);
   for(int d=0;d<DecisionHeadWidth;++d){int e=h*DecisionHeadWidth+d;c->dQuery[i][e]+=ds*.3535533906f*c->key[t][j][e];atomicAdd(&c->dKey[t][j][e],ds*.3535533906f*c->query[i][e]);atomicAdd(&c->dValue[t][j][e],a*c->dContext[i][e]);}
  }atomicAdd(&g[D+h],decay);
 }
 __syncthreads();
 if(i<VoteCount)for(int t=0;t<count;++t)for(int e=0;e<DecisionWidth;++e){
  float memory=0;
  for(int d=0;d<DecisionWidth;++d)
   memory+=c->dKey[t][i][d]*p->key[e][d]+c->dValue[t][i][d]*p->value[e][d];
  float input=memory;
  if(t==0){
   if(spatialGradient)spatialGradient[i*DecisionWidth+e]=memory;
   input+=c->dLn1Pre[i][e];
   for(int d=0;d<DecisionWidth;++d)input+=c->dQuery[i][d]*p->query[e][d];
  }
  c->dInput[t][i][e]=input;
 }
}
__global__ void parameterGradient(const DecisionParameters* p,const SecondsLearningState* state,const DecisionOutput* out,
    const FullRewardRoots* r,const Scratch* c,float* g,float* votes,const int* active){
 if(active && !*active)return;
 int tid=threadIdx.x,count=state->count;
 #define OFF(field) (offsetof(DecisionParameters,field)/sizeof(float))
 for(int at=tid;at<DecisionWidth*DecisionWidth;at+=blockDim.x){int e=at/DecisionWidth,d=at%DecisionWidth;float q=0,k=0,v=0,o=0;
  for(int i=0;i<VoteCount;++i){q+=c->queryInput[i][e]*c->dQuery[i][d];o+=c->context[i][e]*c->dLn1Pre[i][d];for(int t=0;t<count;++t){k+=c->input[t][i][e]*c->dKey[t][i][d];v+=c->input[t][i][e]*c->dValue[t][i][d];}}
  g[OFF(query)+at]=q;g[OFF(key)+at]=k;g[OFF(value)+at]=v;g[OFF(attentionOutput)+at]=o;
 }
 for(int at=tid;at<DecisionWidth*DecisionHiddenWidth;at+=blockDim.x){int e=at/DecisionHiddenWidth,d=at%DecisionHiddenWidth;float x=0;for(int i=0;i<VoteCount;++i)x+=c->ln1[i][e]*c->dUp[i][d];g[OFF(feedForwardUp)+at]=x;}
 for(int at=tid;at<DecisionHiddenWidth*DecisionWidth;at+=blockDim.x){int e=at/DecisionWidth,d=at%DecisionWidth;float x=0;for(int i=0;i<VoteCount;++i)x+=c->expanded[i][e]*c->dOutPre[i][d];g[OFF(feedForwardDown)+at]=x;}
 for(int d=tid;d<DecisionHiddenWidth;d+=blockDim.x){float x=0;for(int i=0;i<VoteCount;++i)x+=c->dUp[i][d];g[OFF(feedForwardUpBias)+d]=x;}
 for(int d=tid;d<DecisionWidth;d+=blockDim.x){float a=0,b=0,v=0,oa=0,ob=0,down=0;float time[4]{};
  for(int i=0;i<VoteCount;++i){float m=0,var=0;for(int e=0;e<DecisionWidth;++e)m+=c->ln1Pre[i][e];m/=DecisionWidth;for(int e=0;e<DecisionWidth;++e)var+=(c->ln1Pre[i][e]-m)*(c->ln1Pre[i][e]-m);
   float inv=rsqrtf(var/DecisionWidth+1e-5f),dLn=c->dOutPre[i][d];for(int e=0;e<DecisionHiddenWidth;++e)dLn+=c->dUp[i][e]*p->feedForwardUp[d][e];a+=dLn*(c->ln1Pre[i][d]-m)*inv;b+=dLn;
   m=var=0;for(int e=0;e<DecisionWidth;++e)m+=c->outPre[i][e];m/=DecisionWidth;for(int e=0;e<DecisionWidth;++e)var+=(c->outPre[i][e]-m)*(c->outPre[i][e]-m);inv=rsqrtf(var/DecisionWidth+1e-5f);
   oa+=r->hidden[i][d]*(c->outPre[i][d]-m)*inv;ob+=r->hidden[i][d];down+=c->dOutPre[i][d];
   float node=0;for(int t=0;t<count;++t){float x=c->dInput[t][i][d];node+=x;v+=x*c->vote[t][i];for(int f=0;f<4;++f)time[f]+=x*c->time[t][f];}g[OFF(nodeEmbedding)+i*DecisionWidth+d]=node;
  }
  g[OFF(attentionGain)+d]=a;g[OFF(attentionBias)+d]=b;g[OFF(outputGain)+d]=oa;g[OFF(outputBias)+d]=ob;g[OFF(feedForwardDownBias)+d]=down;g[OFF(voteEmbedding)+d]=v;for(int f=0;f<4;++f)g[OFF(timeEmbedding)+f*DecisionWidth+d]=time[f];
 }
 if(tid<VoteCount){int i=tid;g[OFF(thresholdBias)+i]=r->threshold[i];for(int d=0;d<DecisionWidth;++d)g[OFF(thresholdWeight)+i*DecisionWidth+d]=r->threshold[i]*out->hidden[i][d];
  float prior=out->prior[i],dg=(prior>1e-6f && prior<1-1e-6f)?r->prior[i]/(prior*(1-prior)):0;
  if(state->first)g[OFF(initialPriorLogit)+i]=r->prior[i];else {float on=sp(p->logOnRate[i]),off=sp(p->logOffRate[i]),sum=on+off,stationary=on/sum,e=expf(-sum*out->deltaSeconds),common=-(state->priorSource[i]-stationary)*out->deltaSeconds*e;
   g[OFF(logOnRate)+i]=dg*(off/(sum*sum)*(1-e)+common)*sig(p->logOnRate[i]);g[OFF(logOffRate)+i]=dg*(-on/(sum*sum)*(1-e)+common)*sig(p->logOffRate[i]);}
  if(votes){
   float vote=r->vote[i];for(int d=0;d<DecisionWidth;++d)vote+=c->dInput[0][i][d]*p->voteEmbedding[d];votes[i]=vote;
  }
 }
 if(tid==0){float x=0;for(int t=0;t<count;++t)for(int i=0;i<VoteCount;++i)for(int d=0;d<DecisionWidth;++d)x+=c->dInput[t][i][d]*c->seconds[t]*(p->timeEmbedding[1][d]*c->time[t][2]-p->timeEmbedding[2][d]*c->time[t][1]);g[OFF(logClockFrequency)]=x*sig(p->logClockFrequency);g[OFF(logGateTemperature)]=r->temperature;}
 #undef OFF
}
}
std::size_t vabpnnSecondsScratchBytes(){return sizeof(Scratch);}
void vabpnnSecondsRewardScore(const VabpnnDecisionModel* model,const vabpnn::FullRewardRoots* roots,
    float* gradient, float* votes, void* scratch, cudaStream_t stream, float* spatialGradient, const int* active){
 auto v=vabpnnDecisionModelView(model);auto* c=static_cast<Scratch*>(scratch);
 forward<<<1,32,0,stream>>>(v.parameters,v.learningState,c,active);
 attend<<<1,32,0,stream>>>(v.parameters,v.learningState,c,active);
 reverse<<<1,32,0,stream>>>(v.parameters,v.learningState,roots,c,gradient,spatialGradient,active);
 parameterGradient<<<1,32,0,stream>>>(v.parameters,v.learningState,v.output,roots,c,gradient,votes,active);
 auto code=cudaGetLastError();if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
