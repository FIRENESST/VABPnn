#include <windows.h>
#include "DenseBlockTransformer.h"
#include "model_checkpoint.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace vabpnn;
void require(bool v,const char* msg){if(!v)throw std::runtime_error(msg);}
void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
template<class T>struct Device {
    T* p=nullptr;std::size_t n;explicit Device(std::size_t count):n(count){check(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){cudaFree(p);}void set(const T* v){check(cudaMemcpy(p,v,n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
double gelu(double x){return .5*x*(1+std::tanh(.7978845608*(x+.044715*x*x*x)));}
void norm(std::vector<double>& x,const float* gain,const float* bias) {
    double mean=0,v=0;for(double a:x)mean+=a;mean/=DenseWidth;
    for(double a:x)v+=(a-mean)*(a-mean);const double s=1/std::sqrt(v/DenseWidth+1e-5);
    for(int d=0;d<DenseWidth;++d)x[d]=(x[d]-mean)*s*gain[d]+bias[d];
}
struct Reference {std::vector<double> attention,hidden;};
Reference reference(const DenseBlockParameters& p,const std::vector<float>& bias,
                    const std::vector<float>& features,int n) {
    std::vector<double>x(n*DenseWidth),q(x.size()),k(x.size()),v(x.size()),y(x.size());
    for(int i=0;i<n;++i)for(int d=0;d<DenseWidth;++d) {
        double z=p.nodeEmbedding[i][d];for(int f=0;f<DenseFeatures;++f)z+=features[i*DenseFeatures+f]*p.input[f][d];
        x[i*DenseWidth+d]=z;
    }
    for(int i=0;i<n;++i)for(int d=0;d<DenseWidth;++d)for(int e=0;e<DenseWidth;++e) {
        q[i*DenseWidth+d]+=x[i*DenseWidth+e]*p.query[e][d];
        k[i*DenseWidth+d]+=x[i*DenseWidth+e]*p.key[e][d];v[i*DenseWidth+d]+=x[i*DenseWidth+e]*p.value[e][d];
    }
    Reference out;out.attention.resize(DenseHeads*n*n);out.hidden.resize(VoteCount*DenseWidth);
    for(int i=0;i<n;++i) {
        std::vector<double> context(DenseWidth),a(DenseWidth),ff(DenseHidden),z(DenseWidth);
        for(int h=0;h<DenseHeads;++h) {
            double m=-INFINITY,sum=0;std::vector<double> score(n);
            for(int j=0;j<n;++j) {
                double s=bias[(h*n+i)*n+j];for(int d=0;d<DenseHeadWidth;++d)
                    s+=q[i*DenseWidth+h*DenseHeadWidth+d]*k[j*DenseWidth+h*DenseHeadWidth+d]/std::sqrt(8.);
                score[j]=s;m=std::max(m,s);
            }
            for(double& s:score){s=std::exp(s-m);sum+=s;}
            for(int j=0;j<n;++j) {
                const double prob=score[j]/sum;out.attention[(h*n+i)*n+j]=prob;
                for(int d=0;d<DenseHeadWidth;++d)context[h*DenseHeadWidth+d]+=prob*v[j*DenseWidth+h*DenseHeadWidth+d];
            }
        }
        for(int d=0;d<DenseWidth;++d) {
            a[d]=x[i*DenseWidth+d];for(int e=0;e<DenseWidth;++e)a[d]+=context[e]*p.attentionOutput[e][d];
        }
        norm(a,p.attentionGain,p.attentionBias);
        for(int f=0;f<DenseHidden;++f) {
            double s=p.upBias[f];for(int d=0;d<DenseWidth;++d)s+=a[d]*p.up[d][f];ff[f]=gelu(s);
        }
        for(int d=0;d<DenseWidth;++d) {
            z[d]=a[d]+p.downBias[d];for(int f=0;f<DenseHidden;++f)z[d]+=ff[f]*p.down[f][d];
        }
        norm(z,p.outputGain,p.outputBias);for(int d=0;d<DenseWidth;++d)y[i*DenseWidth+d]=z[d];
    }
    for(int a=0;a<VoteCount;++a) {
        std::vector<double> score(n);double m=-INFINITY,sum=0;
        for(int i=0;i<n;++i) {
            double s=0;for(int d=0;d<DenseWidth;++d)s+=y[i*DenseWidth+d]*p.readout[a][d]/4;
            score[i]=s;m=std::max(m,s);
        }
        for(double& s:score){s=std::exp(s-m);sum+=s;}
        for(int i=0;i<n;++i)for(int d=0;d<DenseWidth;++d)out.hidden[a*DenseWidth+d]+=score[i]/sum*y[i*DenseWidth+d];
    }
    return out;
}
double loss(const Reference& ref,const std::vector<float>& gradient) {
    double sum=0;for(std::size_t i=0;i<gradient.size();++i)sum+=ref.hidden[i]*gradient[i];return sum;
}
std::vector<unsigned char> bytes(void* p,std::size_t n) {
    std::vector<unsigned char> v(n);check(cudaMemcpy(v.data(),p,n,cudaMemcpyDeviceToHost));return v;
}
#include "dense_policy_checks.h"
int main()try {
    constexpr int n=4;auto* model=vabpnnCreateDenseBlockModel(nullptr,n);const auto view=vabpnnDenseBlockModelView(model);
    auto host=std::make_unique<DenseBlockPersistent>();check(cudaMemcpy(host.get(),view.persistent,sizeof(*host),cudaMemcpyDeviceToHost));
    std::vector<float> features(n*DenseFeatures),bias(DenseHeads*n*n),gradient(VoteCount*DenseWidth);
    for(std::size_t i=0;i<features.size();++i)features[i]=.3f*std::sin(.173f*(i+1));
    for(std::size_t i=0;i<bias.size();++i)bias[i]=.2f*std::cos(.7f*(i+1));
    for(std::size_t i=0;i<gradient.size();++i)gradient[i]=.03f*std::sin(.319f*(i+1));
    Device<float> input(features.size()),target(gradient.size());Device<int> active(1);int on=1;input.set(features.data());target.set(gradient.data());active.set(&on);
    check(cudaMemcpy(view.graphBias,bias.data(),bias.size()*4,cudaMemcpyHostToDevice));
    vabpnnDenseForwardFeatures(model,input.p,1,1'000'000'000,nullptr);check(cudaDeviceSynchronize());
    DenseBlockOutput actual{};check(cudaMemcpy(&actual,view.output,sizeof(actual),cudaMemcpyDeviceToHost));
    const auto ref=reference(host->parameters,bias,features,n);require(actual.valid,"valid full-graph output");
    double maxOutput=0;for(int i=0;i<VoteCount*DenseWidth;++i)maxOutput=std::max(maxOutput,std::abs(actual.hidden[i/DenseWidth][i%DenseWidth]-ref.hidden[i]));
    auto attention=bytes(const_cast<float*>(view.attention),bias.size()*4);const auto* prob=reinterpret_cast<const float*>(attention.data());
    for(int h=0;h<DenseHeads;++h)for(int i=0;i<n;++i) {
        float sum=0;for(int j=0;j<n;++j){const int at=(h*n+i)*n+j;require(prob[at]>0,"all ordered pairs participate");sum+=prob[at];require(std::abs(prob[at]-ref.attention[at])<3e-6,"exact softmax CPU reference");}
        require(std::abs(sum-1)<2e-6,"row normalized");
    }
    require(maxOutput<2e-5,"CPU reference output mismatch");
    vabpnnDenseBackwardHidden(model,target.p,active.p,nullptr);check(cudaDeviceSynchronize());
    std::vector<float> graphGradient(bias.size());check(cudaMemcpy(graphGradient.data(),view.graphGradient,bias.size()*4,cudaMemcpyDeviceToHost));
    auto smallGradient=std::make_unique<DenseBlockParameters>();check(cudaMemcpy(smallGradient.get(),view.gradient,sizeof(*smallGradient),cudaMemcpyDeviceToHost));
    constexpr float eps=.002f;double maxGradient=0;unsigned checked=0,nonzero=0;
    for(std::size_t at=0;at<bias.size();++at) {
        const float old=bias[at];bias[at]=old+eps;const double plus=loss(reference(host->parameters,bias,features,n),gradient);
        bias[at]=old-eps;const double minus=loss(reference(host->parameters,bias,features,n),gradient);bias[at]=old;
        const double expected=(plus-minus)/(2*eps),error=std::abs(expected-graphGradient[at]);maxGradient=std::max(maxGradient,error);
        require(error<2e-5+std::abs(expected)*.003,"ordered-pair bias gradient mismatch");++checked;if(std::abs(expected)>1e-9)++nonzero;
    }
    auto* p=reinterpret_cast<float*>(&host->parameters);const auto* g=reinterpret_cast<const float*>(smallGradient.get());
    for(std::size_t at=0;at<DenseSmallParameters;++at) {
        if(at>=n*DenseWidth && at<offsetof(DenseBlockParameters,input)/4){require(g[at]==0,"unused miniature node gradient");continue;}
        const float old=p[at];p[at]=old+eps;const double plus=loss(reference(host->parameters,bias,features,n),gradient);
        p[at]=old-eps;const double minus=loss(reference(host->parameters,bias,features,n),gradient);p[at]=old;
        const double expected=(plus-minus)/(2*eps),error=std::abs(expected-g[at]);maxGradient=std::max(maxGradient,error);
        if(error>=3e-5+std::abs(expected)*.004) {
            std::printf("BAD parameter=%zu expected=%.9g actual=%.9g\n",at,expected,g[at]);throw std::runtime_error("Q/K/V/FFN/normalization/readout gradient mismatch");
        }++checked;if(std::abs(expected)>1e-9)++nonzero;
    }
    std::printf("PASS: complete two-head attention, all pair probabilities, CPU outputs; max output error %.3g.\n",maxOutput);
    std::printf("PASS: %u pair/node/input/QKV/output/FFN/norm/readout gradients, %u nonzero; max error %.3g.\n",checked,nonzero,maxGradient);
    vabpnnDenseApplyAdamW(model,nullptr);check(cudaDeviceSynchronize());
    auto trained=std::make_unique<DenseBlockPersistent>();check(cudaMemcpy(trained.get(),view.persistent,sizeof(*trained),cudaMemcpyDeviceToHost));
    require(trained->steps==1,"one optimizer step");
    std::vector<float> updated(bias.size()),m(bias.size()),v(bias.size());
    check(cudaMemcpy(updated.data(),view.graphBias,bias.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(m.data(),view.first,bias.size()*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(v.data(),view.second,bias.size()*4,cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<bias.size();++i) {
        const double clipped=std::clamp(static_cast<double>(graphGradient[i]),-1.,1.);
        const double expected=bias[i]-.0003*clipped/(std::abs(clipped)+1e-8);
        require(std::abs(updated[i]-expected)<2e-7,"CPU AdamW pair update");
        require(std::abs(m[i]-.1*clipped)<1e-8 && std::abs(v[i]-.001*clipped*clipped)<1e-10,"CPU Adam moments");
    }
    auto frozen=bytes(view.persistent,sizeof(DenseBlockPersistent));const auto frozenPairs=bytes(view.graphBias,bias.size()*4);
    const auto frozenM=bytes(view.first,bias.size()*4),frozenV=bytes(view.second,bias.size()*4);
    on=0;active.set(&on);vabpnnDenseForwardFeatures(model,input.p,2,1'020'000'000,nullptr);
    vabpnnDenseBackwardHidden(model,target.p,active.p,nullptr);vabpnnDenseApplyAdamW(model,nullptr);check(cudaDeviceSynchronize());
    require(frozen==bytes(view.persistent,sizeof(DenseBlockPersistent)) && frozenPairs==bytes(view.graphBias,bias.size()*4)
        && frozenM==bytes(view.first,bias.size()*4) && frozenV==bytes(view.second,bias.size()*4),"evaluation freezes every dense weight/moment/step");
    ModelPart parts[4];require(vabpnnDenseBlockCheckpointParts(model,parts)==4,"four dense checkpoint parts");
    const std::wstring path=L"dense-compare-checkpoints/model.bin";std::vector<ModelPart> list(parts,parts+4);saveModelCheckpoint(path,list,9);
    check(cudaMemset(view.graphBias,0,bias.size()*4));require(loadModelCheckpoint(path,list,false,9),"restore dense v9");
    require(frozenPairs==bytes(view.graphBias,bias.size()*4),"restored exact ordered-pair weights");
    {
        std::fstream f(std::filesystem::path(path),std::ios::binary|std::ios::in|std::ios::out);f.seekp(270);char c=71;f.write(&c,1);
    }
    bool rejected=false;try{loadModelCheckpoint(path,list,false,9);}catch(const std::exception&){rejected=true;}
    require(rejected && frozenPairs==bytes(view.graphBias,bias.size()*4),"corrupt checkpoint rejected before writes");
    std::printf("PASS: CPU AdamW/moments, exact freeze, independent v9 roundtrip and corrupt-file protection.\n");
    verifyPolicyAndClocks(model,input,target,active,features,gradient);
    vabpnnDestroyDenseBlockModel(model);std::printf("DENSE_SPEC nodes=7680 heads=2 pairs=%zu trainable=%zu\n",DensePairParameters,DenseParameterCount);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
