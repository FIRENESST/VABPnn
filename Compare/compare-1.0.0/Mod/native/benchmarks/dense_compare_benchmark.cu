#include "DenseBlockTransformer.h"
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cmath>
using namespace vabpnn;
void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
__global__ void countGradients(const float* gradient,std::size_t pairs,unsigned long long* counts) {
    const auto at=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at>=pairs*2)return;if(isfinite(gradient[at]) && gradient[at]!=0)atomicAdd(counts+at/pairs,1ull);
}
int main()try {
    auto* model=vabpnnCreateDenseBlockModel(nullptr);const auto view=vabpnnDenseBlockModelView(model);
    float *features,*target;int* active;unsigned long long* nonzero;
    check(cudaMalloc(&features,DenseNodes*DenseFeatures*4));check(cudaMalloc(&target,VoteCount*DenseWidth*4));
    check(cudaMalloc(&active,4));check(cudaMalloc(&nonzero,16));check(cudaMemset(nonzero,0,16));
    std::vector<float> x(DenseNodes*DenseFeatures),g(VoteCount*DenseWidth);int one=1;
    for(std::size_t i=0;i<x.size();++i)x[i]=.1f*std::sin(.021f*(i+1));
    for(std::size_t i=0;i<g.size();++i)g[i]=.02f*std::cos(.111f*(i+1));
    check(cudaMemcpy(features,x.data(),x.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(target,g.data(),g.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(active,&one,4,cudaMemcpyHostToDevice));
    cudaEvent_t start,finish;check(cudaEventCreate(&start));check(cudaEventCreate(&finish));
    float maximum=0,total=0;
    for(int i=0;i<4;++i){vabpnnDenseForwardFeatures(model,features,i+1,1'000'000'000+i*20'000'000,nullptr);check(cudaDeviceSynchronize());}
    for(int i=0;i<16;++i) {
        check(cudaEventRecord(start));vabpnnDenseForwardFeatures(model,features,i+10,2'000'000'000+i*20'000'000,nullptr);
        check(cudaEventRecord(finish));check(cudaEventSynchronize(finish));float ms;check(cudaEventElapsedTime(&ms,start,finish));
        total+=ms;maximum=fmaxf(maximum,ms);
    }
    check(cudaEventRecord(start));vabpnnDenseBackwardHidden(model,target,active,nullptr);vabpnnDenseApplyAdamW(model,nullptr);
    check(cudaEventRecord(finish));check(cudaEventSynchronize(finish));float training;check(cudaEventElapsedTime(&training,start,finish));
    countGradients<<<(DensePairParameters+255)/256,256>>>(view.graphGradient,DenseNodes*static_cast<std::size_t>(DenseNodes),nonzero);
    unsigned long long counts[2];check(cudaMemcpy(counts,nonzero,16,cudaMemcpyDeviceToHost));
    DenseBlockStatus status{};check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
    if(status.nodes!=7680 || status.heads!=2 || status.parameters!=118092470 || !status.valid || status.steps!=1
        || !counts[0] || !counts[1] || status.gateSteps!=1 || status.memoryFrames!=20 || !status.memoryReady
        || !std::isfinite(status.memoryMeanAbs) || status.memoryMeanAbs<=0 || status.memoryMaximumAbs>1)throw std::runtime_error("full graph forward/backprop/AdamW verification failed");
    std::printf("COMPARE_FULL nodes=%u heads=%u pair_bias=%llu trainable=%llu allocated_bytes=%llu\n",
        status.nodes,status.heads,static_cast<unsigned long long>(status.pairs),static_cast<unsigned long long>(status.parameters),
        static_cast<unsigned long long>(status.allocatedBytes));
    std::printf("COMPARE_FORWARD measured=16 mean_ms=%.4f max_ms=%.4f\n",total/16,maximum);
    std::printf("COMPARE_BACKWARD_ADAMW ms=%.4f nonzero_pairs_head0=%llu nonzero_pairs_head1=%llu\n",training,counts[0],counts[1]);
    DenseMemoryGateParameters gates{};
    check(cudaMemcpy(&gates,&view.gradient->memoryGate,sizeof(gates),cudaMemcpyDeviceToHost));
    unsigned live=0;for(unsigned i=0;i<sizeof(gates)/4;++i) {
        const float v=reinterpret_cast<float*>(&gates)[i];if(!std::isfinite(v))throw std::runtime_error("Nonfinite full gate derivative");
        if(v!=0)++live;
    }
    if(!live)throw std::runtime_error("Full graph memory gate derivatives inactive");
    const auto initial=status.memoryFrames;int zero=0;check(cudaMemcpy(active,&zero,4,cudaMemcpyHostToDevice));
    for(int i=0;i<5;++i) {
        vabpnnDenseForwardFeatures(model,features,i+40,3'000'000'000+i*20'000'000,nullptr);
        vabpnnDenseBackwardHidden(model,target,active,nullptr);vabpnnDenseApplyAdamW(model,nullptr);
    }
    check(cudaDeviceSynchronize());check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
    if(status.memoryFrames!=initial+5 || status.gateSteps!=1 || status.steps!=1 || !status.memoryReady)
        throw std::runtime_error("Full matrix evaluation memory/freeze verification failed");
    std::printf("COMPARE_MEMORY frames=%llu gate_steps=%llu live_gate_derivatives=%u mean_abs=%.7f max_abs=%.7f\n",
        static_cast<unsigned long long>(status.memoryFrames),static_cast<unsigned long long>(status.gateSteps),
        live,status.memoryMeanAbs,status.memoryMaximumAbs);
    std::printf("PASS: real full 7680x7680 two-head weights, full forward/backprop/AdamW; no frame-budget gate.\n");
    cudaEventDestroy(start);cudaEventDestroy(finish);cudaFree(features);cudaFree(target);cudaFree(active);cudaFree(nonzero);
    vabpnnDestroyDenseBlockModel(model);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
