#pragma once
#include "FullSelfLearning.h"
#include <array>
#include <cmath>
#include <algorithm>
#include <vector>
namespace full_reference {
using namespace vabpnn;
using Vec=std::array<double,DecisionWidth>;
inline double sigmoid(double x){return 1/(1+std::exp(-x));}
inline double softplus(double x){return x>20?x:std::log1p(std::exp(x));}
inline void norm(Vec& x,const float* gain,const float* bias){double mean=0,variance=0;for(double v:x)mean+=v;mean/=DecisionWidth;for(double v:x)variance+=(v-mean)*(v-mean);double inv=1/std::sqrt(variance/DecisionWidth+1e-5);for(int d=0;d<DecisionWidth;++d)x[d]=(x[d]-mean)*inv*gain[d]+bias[d];}
inline double secondsObjective(const DecisionParameters& p,const SecondsLearningState& state,const FullRewardRoots& roots,const DecisionOutput& baseline){
 std::array<std::array<Vec,VoteCount>,DecisionHistory> input{},key{},value{};std::array<Vec,VoteCount> query{},hidden{};
 double seconds[DecisionHistory]{},age[DecisionHistory]{};for(int t=0;t<state.count;++t){int at=(state.head-t+DecisionHistory)%DecisionHistory;seconds[t]=(state.historyNanos[at]-state.originNanos)*1e-9;age[t]=(state.lastCaptureNanos-state.historyNanos[at])*1e-9;
  double time[4]={std::log1p(state.historyDeltaSeconds[at]),std::sin(seconds[t]*softplus(p.logClockFrequency)),std::cos(seconds[t]*softplus(p.logClockFrequency)),std::log1p(seconds[t])};
  for(int i=0;i<VoteCount;++i){double vote=double(state.historyVotes[at][i])/NeuronCount;for(int d=0;d<DecisionWidth;++d){double x=p.nodeEmbedding[i][d]+vote*p.voteEmbedding[d];for(int f=0;f<4;++f)x+=time[f]*p.timeEmbedding[f][d];input[t][i][d]=x;}
   for(int d=0;d<DecisionWidth;++d)for(int e=0;e<DecisionWidth;++e){if(!t)query[i][d]+=input[t][i][e]*p.query[e][d];double latent=input[t][i][e]+(state.latentDecoder?state.historyLatents[at][i][e]:0);key[t][i][d]+=latent*p.key[e][d];value[t][i][d]+=latent*p.value[e][d];}}
 }
 double objective=roots.temperature*p.logGateTemperature;
 for(int i=0;i<VoteCount;++i){Vec context{};for(int h=0;h<DecisionHeads;++h){std::vector<double> score;double maximum=-INFINITY;for(int t=0;t<state.count;++t)for(int j=0;j<VoteCount;++j){double x=0;for(int d=0;d<DecisionHeadWidth;++d){int e=h*DecisionHeadWidth+d;x+=query[i][e]*key[t][j][e];}x=x*.3535533906+p.graphBias[h][i][j]-softplus(p.logAttentionDecay[h])*age[t];score.push_back(x);maximum=std::max(maximum,x);}
   double sum=0;for(double& x:score){x=std::exp(x-maximum);sum+=x;}for(int t=0;t<state.count;++t)for(int j=0;j<VoteCount;++j)for(int d=0;d<DecisionHeadWidth;++d){int e=h*DecisionHeadWidth+d;context[e]+=score[t*VoteCount+j]/sum*value[t][j][e];}}
  Vec encoded=input[0][i];for(int d=0;d<DecisionWidth;++d)for(int e=0;e<DecisionWidth;++e)encoded[d]+=context[e]*p.attentionOutput[e][d];norm(encoded,p.attentionGain,p.attentionBias);
  double expanded[DecisionHiddenWidth];for(int d=0;d<DecisionHiddenWidth;++d){double x=p.feedForwardUpBias[d];for(int e=0;e<DecisionWidth;++e)x+=encoded[e]*p.feedForwardUp[e][d];expanded[d]=.5*x*(1+std::tanh(.7978845608*(x+.044715*x*x*x)));}
  hidden[i]=encoded;for(int d=0;d<DecisionWidth;++d){hidden[i][d]+=p.feedForwardDownBias[d];for(int e=0;e<DecisionHiddenWidth;++e)hidden[i][d]+=expanded[e]*p.feedForwardDown[e][d];}norm(hidden[i],p.outputGain,p.outputBias);
  for(int d=0;d<DecisionWidth;++d)objective+=roots.hidden[i][d]*hidden[i][d];
  double threshold=p.thresholdBias[i];for(int d=0;d<DecisionWidth;++d)threshold+=p.thresholdWeight[i][d]*baseline.hidden[i][d];objective+=roots.threshold[i]*threshold;
  double prior;if(state.first)prior=sigmoid(p.initialPriorLogit[i]);else{double on=softplus(p.logOnRate[i]),off=softplus(p.logOffRate[i]),stationary=on/(on+off);prior=stationary+(state.priorSource[i]-stationary)*std::exp(-(on+off)*baseline.deltaSeconds);}
  prior=std::clamp(prior,1e-6,1-1e-6);objective+=roots.prior[i]*std::log(prior/(1-prior));
 }
 return objective;
}
inline double minuteObjective(const DecisionMinuteParameters& p,const DecisionMinuteOutput& state,const FullRewardRoots& roots){
 constexpr int W=MinuteWidth;using V=std::array<double,W>;std::array<std::array<V,VoteCount>,MinuteHistory> input{},key{},value{};std::array<V,VoteCount> current{},query{},hidden{};
 for(int t=0;t<state.buckets;++t){double phase=state.bucketMinutes[t]*softplus(p.logClockFrequency);double time[4]={std::log1p(state.coverageMinutes[t]),std::sin(phase),std::cos(phase),std::log1p(state.bucketMinutes[t])};
  for(int i=0;i<VoteCount;++i){for(int d=0;d<W;++d){double x=p.nodeEmbedding[i][d]+state.votes[t][i]*p.voteEmbedding[d];for(int f=0;f<4;++f)x+=time[f]*p.timeEmbedding[f][d];input[t][i][d]=x;if(t==0){current[i][d]=p.nodeEmbedding[i][d]+state.currentVotes[i]*p.voteEmbedding[d];for(int f=0;f<4;++f)current[i][d]+=time[f]*p.timeEmbedding[f][d];}}
   for(int d=0;d<W;++d)for(int e=0;e<W;++e){if(!t)query[i][d]+=current[i][e]*p.query[e][d];key[t][i][d]+=input[t][i][e]*p.key[e][d];value[t][i][d]+=input[t][i][e]*p.value[e][d];}}
 }
 double objective=0;V mean{};for(int i=0;i<VoteCount;++i){V context{};for(int h=0;h<MinuteHeads;++h){std::vector<double> score;double maximum=-INFINITY;for(int t=0;t<state.buckets;++t)for(int j=0;j<VoteCount;++j){double x=0;for(int d=0;d<MinuteHeadWidth;++d){int e=h*MinuteHeadWidth+d;x+=query[i][e]*key[t][j][e];}x=x*.5+p.graphBias[h][i][j]-softplus(p.logAttentionDecay[h])*state.ageMinutes[t];score.push_back(x);maximum=std::max(maximum,x);}double sum=0;for(double& x:score){x=std::exp(x-maximum);sum+=x;}for(int t=0;t<state.buckets;++t)for(int j=0;j<VoteCount;++j)for(int d=0;d<MinuteHeadWidth;++d){int e=h*MinuteHeadWidth+d;context[e]+=score[t*VoteCount+j]/sum*value[t][j][e];}}
  for(int d=0;d<W;++d){double x=current[i][d];for(int e=0;e<W;++e)x+=context[e]*p.attentionOutput[e][d];hidden[i][d]=std::tanh(x);objective+=roots.evidence[i]*hidden[i][d]*p.evidence[i][d];mean[d]+=hidden[i][d]/VoteCount;}}
 for(int k=0;k<DecisionKChoices;++k)for(int d=0;d<W;++d)objective+=roots.k[k]*p.k[k][d]*mean[d];return objective;
}
inline double logGaussian(double sample,double mean,double sigma,double limit){if(sample>=limit)return std::log(std::max(1e-30,.5*std::erfc((limit-mean)/sigma/std::sqrt(2.))));if(sample<=-limit)return std::log(std::max(1e-30,.5*std::erfc((limit+mean)/sigma/std::sqrt(2.))));return -.5*std::pow((sample-mean)/sigma,2)-std::log(sigma)-.9189385332046727;}
inline double motionObjective(const JepaMotionHead& p,const BayesianMotionPersistent& bayes,const FullRewardObservation& c){
 double hidden[JepaHidden];for(int h=0;h<JepaHidden;++h){double x=p.hiddenBias[h];for(int f=0;f<JepaMotionInput;++f)x+=p.inputWeight[f][h]*c.motionInput[f];hidden[h]=std::tanh(x);}
 double head[2];for(int a=0;a<2;++a){double x=p.outputBias[a];for(int h=0;h<JepaHidden;++h)x+=p.outputWeight[h][a]*hidden[h];head[a]=std::tanh(x);}
 double dt=std::min(double(c.decision.deltaSeconds),.1),objective=0;
 for(int a=0;a<2;++a)if(c.proposal.mask&(1u<<(5+a))){double rate=head[a];for(int f=0;f<BayesianMotionFeatures;++f){double feature=f==2*VoteCount?head[0]:f==2*VoteCount+1?head[1]:c.bayesFeatures[f];rate+=bayes.weights[a][f]*feature;}
  double mean=std::clamp(std::clamp(rate,-1.,1.)*1200*dt,-32.,32.),sample=a?c.decision.command.mouseDy:c.decision.command.mouseDx;objective+=logGaussian(sample,mean,std::max(.05,24*dt),32);}
 return objective;
}
inline double memoryObjective(const LongMemoryParameters& p,const FullRewardObservation& c,double credit){const auto& t=c.memory;if(!t.ready || t.frozen)return 0;
 auto gate=[](const float* w,const float* x){double z=0;for(int i=0;i<8;++i)z+=w[i]*x[i];return sigmoid(z);};double read=gate(p.read,t.readFeatures),write=gate(p.write,t.writeFeatures),retain=gate(p.retain,t.retainFeatures),objective=0;
 if(t.support){for(int a=0;a<VoteCount;++a)objective-=credit*c.roots.evidence[a]*.5*std::tanh(p.evidenceGain[a])*read*t.memoryEvidence[a];for(int k=0;k<DecisionKChoices;++k)objective-=credit*c.roots.k[k]*.5*std::tanh(p.kGain[k])*read*t.memoryK[k];}
 double writeTarget=std::min(1.,.2+.6*t.novelty+.2*std::min(1.,10*std::abs(credit))),retainTarget=std::min(1.,.2+.8*std::min(1.,10*std::abs(credit)));
 auto ce=[](double p,double y){return -y*std::log(p)-(1-y)*std::log1p(-p);};objective+=.02*ce(write,writeTarget);if(t.support)objective+=.02*ce(retain,retainTarget);return objective;
}
}
