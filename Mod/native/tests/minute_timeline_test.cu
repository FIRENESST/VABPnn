#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "TemporalDecision.h"
#include "model_checkpoint.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>
using namespace vabpnn;
using Votes=std::array<std::int32_t,VoteCount>;
void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
template<class T>T get(const T* p){T x;check(cudaMemcpy(&x,p,sizeof(x),cudaMemcpyDeviceToHost));return x;}
template<class T>void put(T* p,const T& x){check(cudaMemcpy(p,&x,sizeof(x),cudaMemcpyHostToDevice));}
void assertNear(double actual,double expected,const char* what,double tolerance=2e-6) {
    if(!std::isfinite(actual) || std::abs(actual-expected)>tolerance) {
        std::fprintf(stderr,"%s: actual %.12g expected %.12g\n",what,actual,expected);throw std::runtime_error(what);
    }
}
double softplus(double x){return std::log1p(std::exp(x));}
double sigmoid(double x){return 1/(1+std::exp(-x));}
struct Fixture {
    cudaStream_t stream=nullptr;VabpnnDecisionModel* model=nullptr;
    std::int32_t* votes=nullptr;int* active=nullptr;std::uint32_t* label=nullptr;
    Fixture() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));model=vabpnnCreateDecisionModel(stream);
        check(cudaMalloc(reinterpret_cast<void**>(&votes),sizeof(Votes)));
        check(cudaMalloc(reinterpret_cast<void**>(&active),sizeof(int)));
        check(cudaMalloc(reinterpret_cast<void**>(&label),sizeof(std::uint32_t)));
        check(cudaStreamSynchronize(stream));put(active,1);put(label,std::uint32_t(1));
    }
    ~Fixture(){cudaStreamSynchronize(stream);cudaFree(votes);cudaFree(active);cudaFree(label);vabpnnDestroyDecisionModel(model);cudaStreamDestroy(stream);}
    DecisionOutput run(const Votes& input,std::uint64_t nanos,std::uint64_t id,bool frozen=false) {
        check(cudaMemcpyAsync(votes,input.data(),sizeof(input),cudaMemcpyHostToDevice,stream));
        VabpnnGpuFrame frame{};frame.stream=stream;frame.frameId=id;frame.captureNanos=nanos;frame.evaluationMode=frozen;
        vabpnnLaunchTemporalDecision(frame,votes,model);check(cudaStreamSynchronize(stream));
        auto out=get(vabpnnDecisionModelView(model).output);
        require(out.valid && out.command.frameId==id,"Current dual-axis result remains a valid command");return out;
    }
    DecisionMinuteOutput trace(){return get(vabpnnDecisionModelView(model).minutes.output);}
    DecisionMinutePersistent weights(){return get(vabpnnDecisionModelView(model).minutes.persistent);}
    void weights(const DecisionMinutePersistent& p){put(vabpnnMutableMinuteTimeline(model,stream),p);}
    void reset(){vabpnnResetDecisionState(model,stream);check(cudaStreamSynchronize(stream));}
    void train(){vabpnnTrainMinuteDecision(model,active,label,stream);check(cudaStreamSynchronize(stream));}
};
struct Reference {double hidden[VoteCount][MinuteWidth]{},evidence[VoteCount]{},k[MinuteKChoices]{};};
// Double precision reference uses only the published raw minute summaries and
// independently reconstructs embeddings/projections/softmax. No GPU cache reuse.
Reference reference(const DecisionMinuteParameters& p,const DecisionMinuteOutput& trace) {
    using Row=std::array<double,MinuteWidth>;
    std::array<Row,VoteCount> current{},query{};
    std::array<std::array<Row,VoteCount>,MinuteHistory> input{},key{},value{};
    const auto time=[&](double minutes,double coverage) {
        const double phase=minutes*softplus(p.logClockFrequency);
        return std::array<double,MinuteTimeFeatures>{std::log1p(coverage),std::sin(phase),std::cos(phase),std::log1p(minutes)};
    };
    const auto now=time(trace.elapsedMinutes,trace.coverageMinutes[0]);
    for(int j=0;j<VoteCount;++j) {
        for(int d=0;d<MinuteWidth;++d) {
            current[j][d]=p.nodeEmbedding[j][d]+trace.currentVotes[j]*static_cast<double>(p.voteEmbedding[d]);
            for(int f=0;f<MinuteTimeFeatures;++f)current[j][d]+=now[f]*p.timeEmbedding[f][d];
        }
        for(int d=0;d<MinuteWidth;++d)for(int e=0;e<MinuteWidth;++e)query[j][d]+=current[j][e]*p.query[e][d];
        for(int t=0;t<trace.buckets;++t) {
            const auto times=time(trace.bucketMinutes[t],trace.coverageMinutes[t]);
            for(int d=0;d<MinuteWidth;++d) {
                input[t][j][d]=p.nodeEmbedding[j][d]+trace.votes[t][j]*static_cast<double>(p.voteEmbedding[d]);
                for(int f=0;f<MinuteTimeFeatures;++f)input[t][j][d]+=times[f]*p.timeEmbedding[f][d];
            }
            for(int d=0;d<MinuteWidth;++d)for(int e=0;e<MinuteWidth;++e) {
                key[t][j][d]+=input[t][j][e]*p.key[e][d];value[t][j][d]+=input[t][j][e]*p.value[e][d];
            }
        }
    }
    Reference out;
    for(int i=0;i<VoteCount;++i) {
        Row context{};
        for(int h=0;h<MinuteHeads;++h) {
            std::vector<double> scores;double maximum=-INFINITY;
            for(int t=0;t<trace.buckets;++t)for(int j=0;j<VoteCount;++j) {
                double score=0;
                for(int d=0;d<MinuteHeadWidth;++d){const int e=h*MinuteHeadWidth+d;score+=query[i][e]*key[t][j][e];}
                score=score/std::sqrt(static_cast<double>(MinuteHeadWidth))+p.graphBias[h][i][j]
                    -softplus(p.logAttentionDecay[h])*trace.ageMinutes[t];
                scores.push_back(score);maximum=std::max(maximum,score);
            }
            double denominator=0;for(double& score:scores){score=std::exp(score-maximum);denominator+=score;}
            for(int t=0;t<trace.buckets;++t)for(int j=0;j<VoteCount;++j)for(int d=0;d<MinuteHeadWidth;++d) {
                const int e=h*MinuteHeadWidth+d;context[e]+=scores[t*VoteCount+j]/denominator*value[t][j][e];
            }
        }
        for(int d=0;d<MinuteWidth;++d) {
            double x=current[i][d];for(int e=0;e<MinuteWidth;++e)x+=context[e]*p.attentionOutput[e][d];
            out.hidden[i][d]=std::tanh(x);out.evidence[i]+=out.hidden[i][d]*p.evidence[i][d];
        }
    }
    for(int k=0;k<MinuteKChoices;++k)for(int d=0;d<MinuteWidth;++d) {
        double mean=0;for(int j=0;j<VoteCount;++j)mean+=out.hidden[j][d];
        out.k[k]+=mean/VoteCount*p.k[k][d];
    }
    return out;
}
double objective(const DecisionMinuteParameters& p,const DecisionMinuteOutput& trace,
        const DecisionParameters& base,const DecisionOutput& observed,std::uint32_t label) {
    const auto axis=reference(p,trace);
    std::array<double,DecisionKFeatures> features{};
    for(int d=0;d<DecisionWidth;++d)features[d]=observed.kFeatures[d];
    double loss=0;
    for(int j=0;j<VoteCount;++j) {
        const double prior=observed.prior[j];
        const double logit=std::log(prior/(1-prior))+observed.logLikelihoodRatio[j]-trace.evidence[j]+axis.evidence[j];
        const double posterior=sigmoid(logit);const double y=(label>>j)&1u;
        loss+=(-y*std::log(posterior)-(1-y)*std::log1p(-posterior))/VoteCount;
        features[DecisionWidth+j]=posterior;
        features[DecisionWidth+VoteCount+j]=sigmoid((posterior-observed.threshold[j])/softplus(base.logGateTemperature));
    }
    std::array<double,MinuteKChoices> logits{};double maximum=-INFINITY;
    for(int k=0;k<MinuteKChoices;++k) {
        logits[k]=base.kBias[k]+axis.k[k];
        for(int d=0;d<DecisionKFeatures;++d)logits[k]+=features[d]*base.kWeight[k][d];
        maximum=std::max(maximum,logits[k]);
    }
    double denominator=0;for(double x:logits)denominator+=std::exp(x-maximum);
    int count=0;for(auto bits=label;bits;bits>>=1)count+=bits&1u;
    return loss+maximum+std::log(denominator)-logits[count];
}
void histories() {
    Fixture f;Votes one{};one.fill(NeuronCount);
    auto out=f.run(one,1'000'000'000,1);auto trace=f.trace();
    assertNear(trace.elapsedMinutes,0,"Minute origin");require(trace.buckets==1,"One observed minute");
    Votes zero{};
    out=f.run(zero,60'000'000'000,2);trace=f.trace();
    assertNear(out.deltaSeconds,59,"Seconds clock still uses actual seconds");assertNear(trace.elapsedMinutes,59.0/60,"Minute clock has independent units");
    assertNear(trace.votes[0][0],(1.0/30)/(1.0/30+.25),"Duration weighting does not fill a 59 second unseen gap");
    const auto before=trace.votes[0][0];
    f.run(one,60'000'000'000,3);trace=f.trace();assertNear(trace.votes[0][0],before,"Duplicate timestamp adds no weighted observation");
    f.run(zero,61'000'000'000,4);trace=f.trace();require(trace.buckets==2,"Minute transition appends a summary");
    assertNear(trace.ageMinutes[1],1.0/60,"Actual historical sample age in minutes");
    for(int t=2;t<11;++t)f.run(one,1'000'000'000+t*MinuteNanos,10+t);
    trace=f.trace();require(trace.buckets==8,"Fixed eight minute history");
    assertNear(trace.bucketMinutes[7],3,"Old minutes expire with wall-clock time");
    f.run(one,1'000'000'000+20*MinuteNanos,99);require(f.trace().buckets==1,"Long gap retains only observed bins");
    f.run(one,900'000'000,100);trace=f.trace();
    assertNear(trace.elapsedMinutes,0,"Reverse clock resets minute origin");require(trace.buckets==1,"Reverse clock clears minute history");
    const auto p=f.weights();f.reset();const auto after=f.weights();
    require(std::memcmp(&p,&after,sizeof(p))==0,"Episode reset retains all minute weights and optimizer moments");
    require(f.trace().buckets==0,"Explicit reset clears minute history");
    std::puts("PASS: seconds/minutes units, 60-second boundaries, causal weighted summaries, duplicates, eight-bin capacity, gaps and episode resets.");
}
void longMemory() {
    Fixture f;auto p=f.weights();p.parameters={};
    p.parameters.voteEmbedding[0]=1;p.parameters.value[0][0]=1;p.parameters.attentionOutput[0][0]=1;
    p.parameters.evidence[0][0]=1;
    for(int h=0;h<MinuteHeads;++h)p.parameters.logAttentionDecay[h]=-30;
    f.weights(p);
    auto sequence=[&](int sign) {
        f.reset();Votes old{};old.fill(sign*NeuronCount);
        f.run(old,1'000'000'000,1);DecisionOutput out{};
        for(int step=0;step<18;++step)out=f.run(Votes{},61'000'000'000+step*20'000'000,step+2);
        return out;
    };
    const auto positive=sequence(1),negative=sequence(-1);
    require(std::memcmp(positive.hidden,negative.hidden,sizeof(positive.hidden))==0,
        "Same latest eight frames produce identical original seconds representations");
    require(positive.logLikelihoodRatio[0]>.45f && negative.logLikelihoodRatio[0]<-.45f,
        "Earlier minute affects evidence after the short seconds cache has rolled");
    p.parameters.value[0][0]=2;f.weights(p);
    auto changed=f.run(Votes{},61'400'000'000,30);const auto trace=f.trace();const auto cpu=reference(p.parameters,trace);
    assertNear(changed.logLikelihoodRatio[0],cpu.evidence[0],"Historical minute projections use new weights",4e-6);
    p.parameters.logAttentionDecay[0]=static_cast<float>(std::log(std::expm1(1.0)));f.weights(p);
    auto decayed=f.run(Votes{},61'420'000'000,31);const auto ref=reference(p.parameters,f.trace());
    assertNear(decayed.logLikelihoodRatio[0],ref.evidence[0],"Attention decay is measured per minute",4e-6);
    std::puts("PASS: genuine minute memory survives seconds-cache rollover; updated graph projections and per-minute decay match independent reference.");
}
void gradientsAndCheckpoint() {
    Fixture f;auto minute=f.weights();auto base=get(vabpnnDecisionModelView(f.model).parameters);
    // Nonzero heads and asymmetric graph expose all projection/time gradients.
    for(int i=0;i<VoteCount;++i)for(int d=0;d<MinuteWidth;++d)
        minute.parameters.evidence[i][d]=.13f*std::sin(static_cast<float>(i*11+d*3+1));
    for(int k=0;k<MinuteKChoices;++k)for(int d=0;d<MinuteWidth;++d)
        minute.parameters.k[k][d]=.21f*std::cos(static_cast<float>(k*7+d*5+2));
    for(int h=0;h<MinuteHeads;++h)for(int i=0;i<VoteCount;++i)for(int j=0;j<VoteCount;++j)
        minute.parameters.graphBias[h][i][j]=.15f*std::sin(static_cast<float>(h+i*2+j*3));
    for(int i=0;i<VoteCount;++i) {
        base.logLikelihoodBias[i]=.1f*std::sin(static_cast<float>(i));base.rawVoteWeight[i]=.25f;
        base.thresholdBias[i]=.2f*std::cos(static_cast<float>(i));
    }
    for(int k=0;k<MinuteKChoices;++k)for(int d=0;d<DecisionKFeatures;++d)
        base.kWeight[k][d]=.11f*std::sin(static_cast<float>(k*13+d*2));
    f.weights(minute);put(vabpnnMutableDecisionParameters(f.model,f.stream),base);f.reset();
    DecisionOutput observed{};
    for(int t=0;t<8;++t) {
        Votes input{};for(int i=0;i<VoteCount;++i)input[i]=static_cast<int>(NeuronCount*.7*std::sin(t*1.3+i*.9));
        observed=f.run(input,1'000'000'000+t*MinuteNanos+t*7'000'000'000ull,t+1);
    }
    const auto trace=f.trace();const auto cpu=reference(minute.parameters,trace);
    for(int i=0;i<VoteCount;++i)assertNear(trace.evidence[i],cpu.evidence[i],"Minute evidence double reference",3e-6);
    for(int k=0;k<MinuteKChoices;++k)assertNear(trace.kLogits[k],cpu.k[k],"Minute k fusion double reference",3e-6);
    for(int i=0;i<VoteCount;++i)for(int d=0;d<MinuteWidth;++d)
        assertNear(trace.hidden[i][d],cpu.hidden[i][d],"Minute graph context double reference",3e-6);
    constexpr std::uint32_t label=(1u<<0)|(1u<<6)|(1u<<13);
    put(f.label,label);f.train();const auto trained=f.weights();require(trained.steps==1,"One real label produces one minute AdamW step");
    const auto* actual=reinterpret_cast<const float*>(&trained.first);
    const auto* old=reinterpret_cast<const float*>(&minute.parameters);
    double maxError=0;int nonzero=0;
    for(int at=0;at<MinuteParameterCount;++at) {
        auto plus=minute.parameters,minus=minute.parameters;
        auto* hi=reinterpret_cast<float*>(&plus);auto* lo=reinterpret_cast<float*>(&minus);
        hi[at]=old[at]+.0001f;lo[at]=old[at]-.0001f;
        const double expected=(objective(plus,trace,base,observed,label)-objective(minus,trace,base,observed,label))/(hi[at]-lo[at]);
        const double unclipped=actual[at]/.1f;
        const double clipped=std::max(-1.0,std::min(1.0,expected));
        const double error=std::abs(unclipped-clipped);maxError=std::max(maxError,error);
        if(error>3e-6+std::abs(clipped)*.002) {
            std::fprintf(stderr,"gradient %d actual %.10g finite-difference %.10g\n",at,unclipped,clipped);
            throw std::runtime_error("Minute backward must match the full BCE + k CE objective");
        }
        nonzero+=std::abs(expected)>1e-8;
    }
    require(nonzero>900,"Gradients reach minute embeddings, multihead projections, graph bias, clock and fusion weights");
    require(trained.parameters.logClockFrequency!=minute.parameters.logClockFrequency
        && trained.parameters.logAttentionDecay[0]!=minute.parameters.logAttentionDecay[0]
        && trained.parameters.query[0][0]!=minute.parameters.query[0][0]
        && trained.parameters.key[0][0]!=minute.parameters.key[0][0],"Minute frequency/decay/Q/K are actually trainable");
    put(f.active,0);f.train();auto after=f.weights();
    require(std::memcmp(&trained,&after,sizeof(trained))==0,"Inactive supervision leaves all minute weights and moments unchanged");
    put(f.active,1);f.run(Votes{},observed.captureNanos+20'000'000,90,true);const auto frozen=f.weights();f.train();
    after=f.weights();require(std::memcmp(&frozen,&after,sizeof(frozen))==0,"Frozen minute forward rejects weight/moment updates even if a caller supplies a label");
    f.run(Votes{},observed.captureNanos+40'000'000,91,false);
    auto wrong=get(vabpnnDecisionModelView(f.model).output);wrong.frameId=0;
    put(const_cast<DecisionOutput*>(vabpnnDecisionModelView(f.model).output),wrong);f.train();after=f.weights();
    require(std::memcmp(&frozen,&after,sizeof(frozen))==0,"Stale frame cannot train a new minute observation");
    f.run(Votes{},observed.captureNanos+60'000'000,92,false);f.train();
    require(f.weights().steps==trained.steps+1,"Minute supervision resumes after thaw");
    const auto saved=f.weights();
    const auto path=std::filesystem::absolute("minute-timeline-test/release-v8.bin");
    const std::vector<ModelPart> parts{vabpnnDecisionMinuteCheckpointPart(f.model)};
    saveModelCheckpoint(path.wstring(),parts,8);
    check(cudaMemset(parts[0].device,0,parts[0].bytes));
    require(loadModelCheckpoint(path.wstring(),parts,false,8),"Version eight minute checkpoint loads");
    after=f.weights();require(std::memcmp(&saved,&after,sizeof(saved))==0,"Minute weights, both moments and steps restore exactly");
    const auto beforeReset=after;f.reset();after=f.weights();
    require(std::memcmp(&beforeReset,&after,sizeof(after))==0,"History reset does not reset the learned minute clock");
    {
        std::fstream file(path,std::ios::binary|std::ios::in|std::ios::out);file.seekg(-1,std::ios::end);
        char x;file.read(&x,1);x^=1;file.seekp(-1,std::ios::end);file.write(&x,1);
    }
    bool rejected=false;try{loadModelCheckpoint(path.wstring(),parts,false,8);}catch(const std::runtime_error&){rejected=true;}
    after=f.weights();require(rejected && std::memcmp(&saved,&after,sizeof(saved))==0,"Corrupt v8 rejected before any minute GPU write");
    std::filesystem::remove(path);
    std::printf("PASS: all %d minute gradients checked against independent central finite differences; %d nonzero, max absolute error %.3g.\n",
        MinuteParameterCount,nonzero,maxError);
    std::puts("PASS: real minute Q/K/clock/decay learning, freeze/thaw, label/frame gates, AdamW checkpoint/hash validation and weight-preserving history reset.");
}
int main() {
    try {histories();longMemory();gradientsAndCheckpoint();return 0;}
    catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
