#include "JepaMotion.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace vabpnn;
static void check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
static void require(bool value, const char* label) { if (!value) throw std::runtime_error(label); }
static void near(double actual, double expected, const char* label, double absolute=2e-6, double relative=2e-5) {
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual-expected) > absolute+relative*std::abs(expected)) {
        std::fprintf(stderr,"%s: got %.10g, expected %.10g\n",label,actual,expected);
        throw std::runtime_error(label);
    }
}
__host__ __device__ static float color(int x, int y, int channel, float phase) {
    return 0.05f + 0.85f * ((x/16 + 3*(y/16) + 7*channel) % 19) / 18.0f + 0.03f*phase;
}
__global__ static void fill(float* rgb, int width, int height, float phase) {
    const std::size_t index = blockIdx.x*blockDim.x+threadIdx.x;
    const std::size_t plane = static_cast<std::size_t>(width)*height;
    if (index >= plane) return;
    const int x = index%width, y = index/width;
    for (int c = 0; c < 3; ++c) rgb[c*plane+index] = color(x,y,c,phase);
}
struct Fixture {
    cudaStream_t stream = nullptr;
    VabpnnJepaModel* model = nullptr;
    float* rgb = nullptr;
    std::int32_t* votes = nullptr;
    MotionProposal* proposal = nullptr;
    Fixture() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        model = vabpnnCreateJepaModel(stream);
        check(cudaMalloc(reinterpret_cast<void**>(&rgb),3ull*NeuronWidth*NeuronHeight*sizeof(float)));
        check(cudaMalloc(reinterpret_cast<void**>(&votes),VoteCount*sizeof(std::int32_t)));
        check(cudaMemsetAsync(votes,0,VoteCount*sizeof(std::int32_t),stream));
        check(cudaMalloc(reinterpret_cast<void**>(&proposal),sizeof(MotionProposal)));
        MotionProposal p{123,456,7};
        check(cudaMemcpyAsync(proposal,&p,sizeof(p),cudaMemcpyHostToDevice,stream));
        check(cudaStreamSynchronize(stream));
    }
    ~Fixture() {
        if (stream) cudaStreamSynchronize(stream);
        vabpnnDestroyJepaModel(model);
        if (rgb) cudaFree(rgb);
        if (votes) cudaFree(votes);
        if (proposal) cudaFree(proposal);
        if (stream) cudaStreamDestroy(stream);
    }
    DeviceJepaView view() { return vabpnnJepaModelView(model); }
    float* parameters() { return reinterpret_cast<float*>(vabpnnMutableJepaParameters(model,stream)); }
    void configure(const JepaConfig& config) {
        check(cudaMemcpyAsync(vabpnnMutableJepaConfig(model,stream),&config,sizeof(config),cudaMemcpyHostToDevice,stream));
    }
    void set(std::size_t index, float value) {
        check(cudaMemcpyAsync(parameters()+index,&value,sizeof(value),cudaMemcpyHostToDevice,stream));
    }
    std::vector<float> read(const float* device, std::size_t count) {
        std::vector<float> values(count);
        check(cudaMemcpyAsync(values.data(),device,count*sizeof(float),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream)); return values;
    }
    JepaOutput output() {
        JepaOutput out{};
        check(cudaMemcpyAsync(&out,view().output,sizeof(out),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream)); return out;
    }
    MotionProposal motion() {
        MotionProposal p{};
        check(cudaMemcpyAsync(&p,proposal,sizeof(p),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream)); return p;
    }
    void forward(std::uint64_t id, std::uint64_t nanos, float phase=0, int width=NeuronWidth, int height=NeuronHeight) {
        fill<<<(static_cast<std::size_t>(width)*height+255)/256,256,0,stream>>>(rgb,width,height,phase);
        check(cudaGetLastError());
        VabpnnGpuFrame frame{}; frame.rgbNchw=rgb; frame.width=width; frame.height=height;
        frame.frameId=id; frame.captureNanos=nanos; frame.stream=stream; frame.jepaModel=model;
        vabpnnLaunchJepaForward(frame,votes,model,proposal);
    }
    void train() { vabpnnTrainJepaSelfSupervised(model,stream); }
    void reset() { vabpnnResetJepaHistory(model,stream); }
};
static bool matrix(std::size_t index) {
    int input=JepaInput, output=JepaWidth;
    if (index >= JepaSelfSupervisedParameters) { index-=JepaSelfSupervisedParameters; input=JepaMotionInput; output=2; }
    else if (index >= JepaEncoderParameters) { index-=JepaEncoderParameters; input=JepaPredictInput; }
    return index < static_cast<std::size_t>(input*JepaHidden)
        || (index >= static_cast<std::size_t>(input*JepaHidden+JepaHidden)
             && index < static_cast<std::size_t>(input*JepaHidden+JepaHidden+JepaHidden*output));
}
static bool selectedToken(int token, std::uint64_t frame) {
    const int x=frame%(JepaGridX-JepaMaskX+1), y=(frame/3)%(JepaGridY-JepaMaskY+1);
    return token%JepaGridX>=x && token%JepaGridX<x+JepaMaskX
        && token/JepaGridX>=y && token/JepaGridX<y+JepaMaskY;
}
static void lossReference(Fixture& f, const JepaConfig& config, const JepaOutput& out) {
    auto z=f.read(f.view().previousLatents,JepaTokens*JepaWidth);
    auto target=f.read(f.view().targetLatents,JepaTokens*JepaWidth);
    auto predicted=f.read(f.view().predictions,JepaTokens*JepaWidth);
    double squared=0, penalty=0;
    for (int d=0;d<JepaWidth;++d) {
        double mean=0, variance=0;
        for (int token=0;token<JepaTokens;++token) mean+=z[token*JepaWidth+d];
        mean/=JepaTokens;
        for (int token=0;token<JepaTokens;++token) {
            const double difference=z[token*JepaWidth+d]-mean;
            variance+=difference*difference;
            if (selectedToken(token,out.frameId)) {
                const double error=predicted[token*JepaWidth+d]-target[token*JepaWidth+d];
                squared+=0.5*error*error/(JepaTargets*JepaWidth);
            }
        }
        penalty+=std::max(0.0,config.varianceTarget-std::sqrt(variance/JepaTokens+1e-4))/JepaWidth;
    }
    near(out.predictionLoss,squared,"Independent latent MSE reference",1e-7);
    near(out.variancePenalty,penalty,"Independent variance regularizer reference",1e-7);
    near(out.jepaLoss,squared+config.varianceWeight*penalty,"Independent combined JEPA loss",1e-7);
}
static void adamReference(Fixture& f, const JepaConfig& config,
                           const std::vector<float>& before, const std::vector<float>& oldFirst,
                           const std::vector<float>& oldSecond, const std::vector<float>& oldTeacher,
                           std::size_t offset, std::size_t count, std::uint64_t step, bool ema) {
    const auto out=f.output();
    auto after=f.read(f.view().parameters?reinterpret_cast<const float*>(f.view().parameters):nullptr,JepaParameterCount);
    auto gradient=f.read(f.view().gradients,JepaParameterCount);
    auto first=f.read(f.view().firstMoment,JepaParameterCount), second=f.read(f.view().secondMoment,JepaParameterCount);
    double squared=0;
    for (std::size_t i=offset;i<offset+count;++i) squared+=static_cast<double>(gradient[i])*gradient[i];
    near(out.gradientNorm,std::sqrt(squared),"Gradient norm reference",1e-6);
    const double scale=out.gradientNorm>config.gradientClip?config.gradientClip/out.gradientNorm:1;
    const double beta1=config.beta1,beta2=config.beta2;
    for (std::size_t i=0;i<JepaParameterCount;++i) {
        if (i<offset || i>=offset+count) {
            require(after[i]==before[i] && first[i]==oldFirst[i] && second[i]==oldSecond[i],"Untargeted optimizer group is unchanged");
            continue;
        }
        const double g=gradient[i]*scale, m=beta1*oldFirst[i]+(1-beta1)*g;
        const double v=beta2*oldSecond[i]+(1-beta2)*g*g;
        const double expected=before[i]*(1-config.learningRate*(matrix(i)?config.weightDecay:0))
            -config.learningRate*(m/(1-std::pow(beta1,static_cast<double>(step))))
                /(std::sqrt(v/(1-std::pow(beta2,static_cast<double>(step))))+config.epsilon);
        near(first[i],m,"AdamW first moment",1e-9,3e-5);
        near(second[i],v,"AdamW second moment",1e-13,5e-5);
        near(after[i],expected,"AdamW bias correction, clipping and decoupled decay",5e-7,3e-6);
    }
    auto teacher=f.read(reinterpret_cast<const float*>(f.view().targetEncoder),JepaEncoderParameters);
    for (std::size_t i=0;i<JepaEncoderParameters;++i) {
        if (ema) near(teacher[i],config.emaDecay*oldTeacher[i]+(1-config.emaDecay)*after[i],"EMA target reference",2e-7,2e-6);
        else require(teacher[i]==oldTeacher[i],"Head trainer cannot change EMA encoder");
    }
}
static void poolingAndTraining() {
    Fixture f;
    require(JepaTotalParameterCount<50'000'000 && JepaParameterCount==510722,"JEPA below 0.05B parameter limit");
    const auto bytes=vabpnnJepaModelBytes(f.model);
    require(bytes>sizeof(JepaParameters)*4 && bytes<32'000'000,"Bounded JEPA/AdamW allocation");
    const auto stable=f.view();
    auto before=f.read(reinterpret_cast<const float*>(f.view().parameters),JepaParameterCount);
    auto teacher=f.read(reinterpret_cast<const float*>(f.view().targetEncoder),JepaEncoderParameters);
    require(std::equal(teacher.begin(),teacher.end(),before.begin()),"Initial EMA equals online encoder");
    JepaConfig config; config.trainEveryFrames=1; config.gradientClip=.05f; config.emaDecay=.9f;
    f.configure(config);
    f.forward(1,1'000'000'000ull);
    f.train();
    auto out=f.output();
    require(out.valid && !out.pairReady && out.selfSupervisedSteps==0 && out.motionSteps==0,"First frame has no future pair");
    auto patches=f.read(f.view().currentPatchFeatures,JepaTokens*JepaInput);
    for (int token=0;token<JepaTokens;++token)
        for (int cell=0;cell<JepaCells*JepaCells;++cell)
            for (int c=0;c<3;++c) {
                const int x=(token%JepaGridX*4+cell%4)*16, y=(token/JepaGridX*4+cell/4)*16;
                near(patches[token*JepaInput+cell*3+c],2*color(x,y,c,0)-1,"GPU RGB patch pooling/top-left origin",3e-6);
            }
    near(f.motion().mouseDx,0,"Untrained motion X is zero"); near(f.motion().mouseDy,0,"Untrained motion Y is zero");
    near(f.motion().wheelSteps,7,"JEPA preserves wheel proposal");
    std::vector<float> oldFirst(JepaParameterCount,0),oldSecond(JepaParameterCount,0);
    f.forward(2,1'016'000'000ull,.5f);
    out=f.output();
    require(out.valid && out.pairReady && out.jepaLoss>0,"Valid temporal JEPA pair/loss");
    lossReference(f,config,out);
    f.train(); out=f.output();
    require(out.lastLearningValid && out.selfSupervisedSteps==1 && out.motionSteps==0,"Actual encoder/predictor AdamW step");
    adamReference(f,config,before,oldFirst,oldSecond,teacher,0,JepaSelfSupervisedParameters,1,true);
    auto after=f.read(reinterpret_cast<const float*>(f.view().parameters),JepaParameterCount);
    require(after!=before,"JEPA weights actually changed");
    before=after; oldFirst=f.read(f.view().firstMoment,JepaParameterCount); oldSecond=f.read(f.view().secondMoment,JepaParameterCount);
    teacher=f.read(reinterpret_cast<const float*>(f.view().targetEncoder),JepaEncoderParameters);
    f.train();
    require(f.output().selfSupervisedSteps==1,"Duplicate training cannot reuse a frame");
    f.forward(3,1'032'000'000ull,.8f); f.train();
    require(f.output().selfSupervisedSteps==2,"Second temporal training step");
    adamReference(f,config,before,oldFirst,oldSecond,teacher,0,JepaSelfSupervisedParameters,2,true);
    require(f.view().parameters==stable.parameters && vabpnnJepaModelBytes(f.model)==bytes,"No per-frame model allocation");
    std::printf("PASS: %zu trainable parameters, %zu including EMA; GPU allocation %llu bytes; pooling/loss/AdamW/EMA references.\n",
        JepaParameterCount,JepaTotalParameterCount,static_cast<unsigned long long>(bytes));
}
static double pairLoss(Fixture& f, float currentPhase=.5f) {
    f.reset(); f.forward(1,2'000'000'000ull,0); f.forward(2,2'040'000'000ull,currentPhase);
    const auto out=f.output(); require(out.valid && out.pairReady,"Finite-difference pair validity");
    return out.jepaLoss;
}
static std::size_t largest(const std::vector<float>& gradient,std::size_t begin,std::size_t end) {
    auto at=begin;
    for (auto i=begin+1;i<end;++i) if (std::abs(gradient[i])>std::abs(gradient[at])) at=i;
    require(std::abs(gradient[at])>1e-7,"Nontrivial gradient probe");
    return at;
}
static void gradientsAndTemporalIsolation() {
    Fixture f;
    JepaConfig config; config.trainEveryFrames=1; config.learningRate=0; config.emaDecay=1;
    config.varianceTarget=.5f; config.varianceWeight=.2f;
    f.configure(config);
    pairLoss(f); lossReference(f,config,f.output()); f.train();
    auto gradient=f.read(f.view().gradients,JepaParameterCount);
    auto parameters=f.read(reinterpret_cast<const float*>(f.view().parameters),JepaParameterCount);
    const std::size_t encoderInput=JepaInput*JepaHidden, encoderHidden=encoderInput+JepaHidden;
    const std::size_t encoderOutput=encoderHidden+JepaHidden*JepaWidth;
    const std::size_t predictorBase=JepaEncoderParameters,predictorInput=predictorBase+JepaPredictInput*JepaHidden;
    const std::size_t predictorHidden=predictorInput+JepaHidden,predictorOutput=predictorHidden+JepaHidden*JepaWidth;
    const std::size_t probes[]={
        largest(gradient,0,encoderInput),largest(gradient,encoderInput,encoderHidden),
        largest(gradient,encoderHidden,encoderOutput),largest(gradient,encoderOutput,JepaEncoderParameters),
        largest(gradient,predictorBase,predictorInput),largest(gradient,predictorHidden,predictorOutput),
        largest(gradient,predictorOutput,JepaSelfSupervisedParameters)};
    for (auto index:probes) {
        const float plus=parameters[index]+.004f,minus=parameters[index]-.004f;
        f.set(index,plus); const double positive=pairLoss(f);
        f.set(index,minus); const double negative=pairLoss(f);
        f.set(index,parameters[index]);
        near(gradient[index],(positive-negative)/(plus-minus),"Backprop matches central finite difference",4e-5,.03);
    }
    pairLoss(f,.2f);
    auto prediction=f.read(f.view().predictions,JepaTokens*JepaWidth);
    auto target=f.read(f.view().targetLatents,JepaTokens*JepaWidth);
    pairLoss(f,.9f);
    require(f.read(f.view().predictions,JepaTokens*JepaWidth)==prediction,"Future image cannot leak into predictor");
    require(f.read(f.view().targetLatents,JepaTokens*JepaWidth)!=target,"Only target branch sees future image");
    auto current=f.read(f.view().currentLatents,JepaTokens*JepaWidth);
    f.set(encoderOutput,parameters[encoderOutput]+1);
    f.forward(3,2'056'000'000ull,.9f);
    auto previous=f.read(f.view().previousLatents,JepaTokens*JepaWidth);
    for (int token=0;token<JepaTokens;++token)
        near(previous[token*JepaWidth],current[token*JepaWidth]+1,"Historical features re-encoded with updated weights",4e-6);
    std::puts("PASS: encoder/predictor finite-difference gradients including variance loss; stop-gradient future targets and fresh historical projections.");
}
static void motionInterfaceAndLifecycle() {
    Fixture f;
    JepaConfig config; config.selfSupervisedEnabled=0; config.motionEnabled=1;
    f.configure(config);
    const auto before=f.read(reinterpret_cast<const float*>(f.view().parameters),JepaParameterCount);
    const auto base=JepaSelfSupervisedParameters;
    const auto bias=base+JepaMotionInput*JepaHidden+JepaHidden+JepaHidden*2;
    f.set(bias,std::atanh(.5f)); f.set(bias+1,std::atanh(-.25f));
    f.forward(1,3'000'000'000ull);
    require(!f.output().motionActive && f.motion().mouseDx==0,"No motion without temporal context");
    f.forward(2,3'016'000'000ull,.1f);
    auto out=f.output();
    require(out.motionActive,"Explicitly enabled trained-head inference");
    near(out.rateX,600,"Mouse X units/second",1e-4); near(out.rateY,-300,"Mouse Y units/second",1e-4);
    near(f.motion().mouseDx,9.6,"Seconds to per-command X displacement",1e-4);
    near(f.motion().mouseDy,-4.8,"Seconds to per-command Y displacement",1e-4);
    near(f.motion().wheelSteps,7,"Wheel still untouched");
    f.forward(3,3'216'000'000ull);
    near(f.motion().mouseDx,32,"Mouse displacement cap",1e-4); near(f.motion().mouseDy,-30,"Motion dt cap",1e-4);
    f.forward(4,4'000'000'000ull);
    require(!f.output().pairReady && f.motion().mouseDx==0,"Large capture gap stops mouse motion");
    f.forward(5,3'000'000'000ull);
    require(!f.output().pairReady,"Reversed clock discards history");
    f.forward(6,3'016'000'000ull,0,37,23);
    require(!f.output().pairReady,"Resize discards incompatible history");
    const auto saved=f.read(reinterpret_cast<const float*>(f.view().parameters),JepaParameterCount);
    const auto bytes=vabpnnJepaModelBytes(f.model);
    f.reset(); f.forward(7,3'032'000'000ull,0,37,23); f.train();
    require(!f.output().pairReady && f.read(reinterpret_cast<const float*>(f.view().parameters),JepaParameterCount)==saved,
        "Episode reset preserves trained weights");
    require(vabpnnJepaModelBytes(f.model)==bytes,"Resize/reset preserve bounded allocation");
    config.motionEnabled=0; f.configure(config); f.forward(8,3'048'000'000ull,0,37,23);
    require(!f.output().motionActive && f.motion().mouseDx==0,"Head disabled until training is supplied");
    config.beta1=std::numeric_limits<float>::quiet_NaN(); f.configure(config); f.forward(9,3'064'000'000ull,0,37,23); f.train();
    require(!f.output().valid && f.motion().mouseDx==0,"Nonfinite optimizer configuration rejected");
    std::puts("PASS: mouse units/seconds/caps, head standby, wheel preservation, resize/episode/clock history resets and invalid config.");
}
static void headTrainingHook() {
    Fixture f;
    JepaConfig config; config.selfSupervisedEnabled=0; config.learningRate=.001f;
    f.configure(config);
    f.forward(1,5'000'000'000ull); f.forward(2,5'016'000'000ull,.3f);
    auto before=f.read(reinterpret_cast<const float*>(f.view().parameters),JepaParameterCount);
    auto oldFirst=f.read(f.view().firstMoment,JepaParameterCount),oldSecond=f.read(f.view().secondMoment,JepaParameterCount);
    auto teacher=f.read(reinterpret_cast<const float*>(f.view().targetEncoder),JepaEncoderParameters);
    vabpnnTrainJepaMotion(f.model,f.stream);
    require(f.output().motionSteps==0,"Absent motion labels cannot train head");
    JepaMotionTarget target{1,600,-300,1};
    check(cudaMemcpyAsync(vabpnnMutableJepaMotionTarget(f.model,f.stream),&target,sizeof(target),cudaMemcpyHostToDevice,f.stream));
    vabpnnTrainJepaMotion(f.model,f.stream);
    require(f.output().motionSteps==0,"Stale motion labels rejected");
    target.frameId=2;
    check(cudaMemcpyAsync(vabpnnMutableJepaMotionTarget(f.model,f.stream),&target,sizeof(target),cudaMemcpyHostToDevice,f.stream));
    vabpnnTrainJepaMotion(f.model,f.stream);
    auto out=f.output();
    require(out.motionSteps==1 && out.selfSupervisedSteps==0 && out.lastLearningValid,"Optional supervised head-only AdamW step");
    near(out.motionLoss,.078125,"Head normalized MSE");
    adamReference(f,config,before,oldFirst,oldSecond,teacher,JepaSelfSupervisedParameters,
        JepaParameterCount-JepaSelfSupervisedParameters,1,false);
    vabpnnTrainJepaMotion(f.model,f.stream);
    require(f.output().motionSteps==1,"Head label consumed once");
    f.forward(3,5'032'000'000ull,.4f);
    vabpnnTrainJepaMotion(f.model,f.stream);
    require(f.output().motionSteps==1 && f.motion().mouseDx==0,"Labels clear each frame; learning does not enable input automatically");
    std::puts("PASS: future motion training target hook, matching-frame labels, head-only optimizer isolation and duplicate-label protection.");
}
int main() {
    try {
        poolingAndTraining(); gradientsAndTemporalIsolation(); motionInterfaceAndLifecycle(); headTrainingHook();
        std::puts("PASS: JEPA self-supervision + real GPU backprop + AdamW, no game-frame readback or frame files.");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}

