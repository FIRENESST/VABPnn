#include "TemporalDecision.h"
#include "JepaMotion.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using namespace vabpnn;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
template<class T>T get(const T* p){T out;check(cudaMemcpy(&out,p,sizeof(out),cudaMemcpyDeviceToHost));return out;}
template<class T>void put(T* p,const T& x){check(cudaMemcpy(p,&x,sizeof(x),cudaMemcpyHostToDevice));}
void near(float a,float b,const char* what,float tolerance=2e-6f) {
    if(!std::isfinite(a) || fabsf(a-b)>tolerance){std::fprintf(stderr,"%s %.9g != %.9g\n",what,a,b);throw std::runtime_error(what);}
}
float logit(float p){return logf(p/(1-p));}
struct Fixture {
    cudaStream_t stream=nullptr;VabpnnDecisionModel* model=nullptr;VabpnnJepaModel* jepa=nullptr;
    std::int32_t* votes=nullptr;int* active=nullptr;float *x=nullptr,*y=nullptr;
    std::uint64_t id=0,nanos=1'000'000'000;
    Fixture() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));model=vabpnnCreateDecisionModel(stream);
        jepa=vabpnnCreateJepaModel(stream);check(cudaMalloc(reinterpret_cast<void**>(&votes),VoteCount*sizeof(std::int32_t)));
        check(cudaMemsetAsync(votes,0,VoteCount*sizeof(std::int32_t),stream));
        check(cudaMalloc(reinterpret_cast<void**>(&active),sizeof(int)));check(cudaMalloc(reinterpret_cast<void**>(&x),sizeof(float)));
        check(cudaMalloc(reinterpret_cast<void**>(&y),sizeof(float)));check(cudaStreamSynchronize(stream));
        JepaConfig cfg;cfg.motionEnabled=1;put(vabpnnMutableJepaConfig(jepa,stream),cfg);put(active,1);put(x,80.f);put(y,-20.f);
    }
    ~Fixture(){cudaStreamSynchronize(stream);cudaFree(votes);cudaFree(active);cudaFree(x);cudaFree(y);
        vabpnnDestroyDecisionModel(model);vabpnnDestroyJepaModel(jepa);cudaStreamDestroy(stream);}
    BayesianMotionPersistent persistent(){return get(vabpnnDecisionModelView(model).minutes.persistent).motion;}
    void persistent(const BayesianMotionPersistent& p){auto whole=get(vabpnnDecisionModelView(model).minutes.persistent);whole.motion=p;put(vabpnnMutableMinuteTimeline(model,stream),whole);}
    DecisionOutput run(float priorX,float priorY,bool frozen=false,bool pair=true,bool enabled=true) {
        auto base=get(vabpnnDecisionModelView(model).parameters);
        for(int k=0;k<VoteCount;++k) {base.logLikelihoodBias[k]=(k==5 || k==6) ? 0:-10;base.thresholdBias[k]=-8;}
        base.initialPriorLogit[5]=logit(priorX);base.initialPriorLogit[6]=logit(priorY);
        for(int k=0;k<DecisionKChoices;++k)base.kBias[k]=k==2 ? 10:-2;
        put(vabpnnMutableDecisionParameters(model,stream),base);vabpnnResetDecisionState(model,stream);
        auto cfg=get(vabpnnJepaModelView(jepa).config);cfg.motionEnabled=enabled;put(vabpnnMutableJepaConfig(jepa,stream),cfg);
        JepaOutput raw{};raw.frameId=++id;raw.captureNanos=nanos+=20'000'000;
        raw.valid=1;raw.pairReady=pair;raw.motionActive=pair && enabled;raw.deltaSeconds=.02f;
        raw.headRateX=120;raw.headRateY=-60;
        raw.rateX=raw.motionActive ? 120:0;raw.rateY=raw.motionActive ? -60:0;
        raw.mouseDx=raw.motionActive ? 2.4f:0;raw.mouseDy=raw.motionActive ? -1.2f:0;
        put(const_cast<JepaOutput*>(vabpnnJepaModelView(jepa).output),raw);
        MotionProposal proposal{raw.mouseDx,raw.mouseDy,0};put(vabpnnMutableMotionProposal(model,stream),proposal);
        VabpnnGpuFrame frame{};frame.stream=stream;frame.frameId=id;frame.captureNanos=nanos;frame.evaluationMode=frozen;frame.jepaModel=jepa;
        vabpnnLaunchTemporalDecision(frame,votes,model);check(cudaStreamSynchronize(stream));
        const auto out=get(vabpnnDecisionModelView(model).output);
        require(out.valid && out.selectedMask==((1u<<5)|(1u<<6)),"Only X/Y permissions selected by the ordinary Bayesian decision");
        return out;
    }
    void train(){vabpnnTrainBayesianMotion(model,active,x,y,stream);check(cudaStreamSynchronize(stream));}
};
int main() {
    try {
        Fixture f;auto out=f.run(.25f,.2f);
        near(out.command.mouseDx,2.4f,"Neutral Bayesian residual preserves original JEPA X");
        near(out.command.mouseDy,-1.2f,"Neutral Bayesian residual preserves original JEPA Y");
        auto p=f.persistent();p.weights[0][5]=.05f;p.weights[0][VoteCount+5]=.025f;
        p.weights[1][6]=-.04f;f.persistent(p);
        const auto low=f.run(.25f,.2f),high=f.run(.75f,.8f);
        near(low.command.mouseDx,1.5f,"Posterior and prior condition displacement value");
        near(high.command.mouseDx,3.3f,"Bayesian fusion can amplify rather than always suppress JEPA X");
        near(low.command.mouseDy,-.624f,"Bayesian Y displacement reference");
        near(high.command.mouseDy,-1.776f,"Y uses its probability independently");
        auto jepa=get(vabpnnJepaModelView(f.jepa).output);
        near(jepa.mouseDx,high.command.mouseDx,"JEPA diagnostic displacement reflects final Bayesian fusion");
        near(jepa.headRateX,120,"Unconditioned JEPA rate stays available for causal training");
        std::puts("PASS: neutral legacy mouse, real posterior/prior conditioning, independent X/Y, amplification and final JEPA diagnostics.");
        p={};f.persistent(p);out=f.run(.7f,.3f,false,true,false);f.train();const auto learned=f.persistent();
        require(learned.steps==1,"Real demonstration trains Bayesian motion even while execution is disabled");
        float features[BayesianMotionFeatures]{};
        for(int k=0;k<VoteCount;++k){features[k]=2*out.posterior[k]-1;features[VoteCount+k]=2*out.prior[k]-1;}
        features[VoteCount*2]=.1f;features[VoteCount*2+1]=-.05f;features[BayesianMotionFeatures-1]=1;
        const float head[2]={.1f,-.05f},target[2]={80.f/1200,-20.f/1200};
        for(int axis=0;axis<2;++axis)for(int at=0;at<BayesianMotionFeatures;++at) {
            const float gradient=.5f*(head[axis]-target[axis])*features[at];
            near(learned.first[axis][at],.1f*gradient,"Bayesian motion AdamW gradient from cached real target",2e-7f);
            near(learned.second[axis][at],.001f*gradient*gradient,"Bayesian motion second moment",1e-9f);
        }
        const auto before=learned;
        f.run(.7f,.3f,true);f.train();auto after=f.persistent();
        require(std::memcmp(&before,&after,sizeof(before))==0,"Evaluation freezes Bayesian motion weights/moments/steps");
        f.run(.7f,.3f,false,false);f.train();after=f.persistent();
        require(std::memcmp(&before,&after,sizeof(before))==0,"Missing JEPA pair cannot train a Bayesian displacement");
        f.run(.7f,.3f,false);put(f.active,0);f.train();after=f.persistent();
        require(std::memcmp(&before,&after,sizeof(before))==0,"Inactive human input cannot train Bayesian displacement");
        put(f.active,1);put(f.x,1500.f);f.train();after=f.persistent();
        require(std::memcmp(&before,&after,sizeof(before))==0,"Out-of-range mouse target is rejected for all CUDA lanes");
        put(f.x,80.f);f.train();require(f.persistent().steps==before.steps+1,"Thaw resumes Bayesian mouse learning");
        p={};p.weights[0][BayesianMotionFeatures-1]=10;p.weights[1][BayesianMotionFeatures-1]=-10;f.persistent(p);
        out=f.run(.5f,.5f);near(out.command.mouseDx,24,"Fusion keeps original rate cap");near(out.command.mouseDy,-24,"Negative rate cap");
        std::puts("PASS: all 66 cached real mouse gradients, no injected demonstration labels, freeze/thaw, pointer/pair/active/target guards and original motion limits.");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
