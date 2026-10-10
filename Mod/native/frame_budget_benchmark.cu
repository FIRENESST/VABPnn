#include "FrameOperator.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "NetworkScope.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <stdexcept>
#include <string>
bool PositionEnabled=true,BlocksEnabled=true,BlockStress=false;
void check(cudaError_t code) { if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
__global__ void pattern(unsigned char* image,std::size_t pitch,int width,int height) {
    const int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width || y>=height) return;
    auto* pixel=image+y*pitch+x*4;
    pixel[0]=(x*3+y*7+91)&255; pixel[1]=(x+y*11+43)&255;
    pixel[2]=(x*13+y+29)&255; pixel[3]=255;
}
// Stress capacity, independently of how long the human model has been trained.
__global__ void fullCapacity(vabpnn::SpikePixelStatistics* stats) {
    const auto pixel=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pixel>=vabpnn::NeuronCount) return;
    const int x=pixel%1920,y=pixel/1920;
    if(x%16==8 && y%16==8) stats[pixel]={.33f,.33f,.34f,1e10f,0};
}
// Synthetic maturity windows force ranking/mutation cost; this is a stress fixture, not a learned score.
__global__ void matureRankFixture(vabpnn::BlockPersistent* p) {
    const int tile=blockIdx.x*blockDim.x+threadIdx.x; if(tile>=vabpnn::BlockCount) return;
    auto& state=p->state[tile]; const float score=tile<1530 ? -.8f:.2f;
    state.examples=state.examples<1024 ? 1024:state.examples;
    state.windowExamples=512; state.windowSum=512*score; state.windowSquares=512*score*score;
    state.windowCoverage=512; state.previousRankScore=score+.01f; state.protectedAction=0; state.cooldownUntil=0;
}
struct Sample { double copy,op,total,wall; int jepa,pixel,topology,scope,actor,ranking,mutation; };
void report(const char* label,const std::vector<Sample>& values) {
    if(values.empty()) return;
    std::vector<double> times; double copy=0,wall=0,total=0; int over10=0; double maximumTotal=0;
    for(const auto& s:values) { times.push_back(s.op); copy+=s.copy; wall+=s.wall; total+=s.total; over10+=s.total>10; maximumTotal=std::max(maximumTotal,s.total); }
    std::sort(times.begin(),times.end());
    auto percentile=[&](double p) { return times[static_cast<std::size_t>((times.size()-1)*p)]; };
    const double average=std::accumulate(times.begin(),times.end(),0.0)/times.size();
    std::printf("%s n=%zu op_mean=%.4f op_p50=%.4f op_p95=%.4f op_p99=%.4f op_max=%.4f copy_mean=%.4f copy_op_control_mean=%.4f worker_wall_mean=%.4f ms\n",
        label,times.size(),average,percentile(.5),percentile(.95),percentile(.99),times.back(),
        copy/times.size(),total/times.size(),wall/times.size());
    std::printf(" budget_over10=%d total_max=%.4f ms\n",over10,maximumTotal);
}
void profile(int mode,int selfSupervised,const char* label,bool overlap=false,bool forceRanking=false) {
    constexpr int Width=1920,Height=1024,Warmup=64,Count=512;
    cudaStream_t stream=nullptr; unsigned char* source=nullptr; unsigned char* image=nullptr; float* rgb=nullptr;
    std::size_t sourcePitch=0,pitch=0; cudaEvent_t events[4]{};
    VabpnnNeuronModel* neuron=nullptr; VabpnnDecisionModel* decision=nullptr;
    VabpnnJepaModel* jepa=nullptr; VabpnnOnlineModel* online=nullptr; VabpnnSpikeModel* spike=nullptr;
    vabpnn::ScopeTelemetry* scopeDevice=nullptr;
    struct HostStatus { vabpnn::ControlCommand command; vabpnn::DecisionStatus decision; vabpnn::OnlineOutput online; vabpnn::JepaOutput jepa; vabpnn::SpikeCounters spike; vabpnn::BlockCounters blocks; vabpnn::ScopeTelemetry scope; };
    HostStatus* status=nullptr;
    try {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaMallocPitch(reinterpret_cast<void**>(&source),&sourcePitch,Width*4,Height));
        check(cudaMallocPitch(reinterpret_cast<void**>(&image),&pitch,Width*4,Height));
        check(cudaMalloc(reinterpret_cast<void**>(&rgb),static_cast<std::size_t>(Width)*Height*3*sizeof(float)));
        check(cudaMallocHost(reinterpret_cast<void**>(&status),sizeof(HostStatus))); *status=HostStatus{};
        check(cudaMalloc(reinterpret_cast<void**>(&scopeDevice),sizeof(vabpnn::ScopeTelemetry)));
        for(auto& event:events) check(cudaEventCreate(&event));
        pattern<<<dim3(120,64),dim3(16,16),0,stream>>>(source,sourcePitch,Width,Height);
        neuron=vabpnnCreateNeuronModel(stream); decision=vabpnnCreateDecisionModel(stream);
        jepa=vabpnnCreateJepaModel(stream); online=vabpnnCreateOnlineModel(stream); spike=vabpnnCreateSpikeModel(stream);
        auto blockConfig=vabpnn::BlockConfig{}; blockConfig.enabled=BlocksEnabled;
        if(forceRanking) { blockConfig.warmup=0; blockConfig.rankEvery=20; blockConfig.poorWindows=1; }
        auto* blockModel=vabpnnSpikeBlockModel(spike); vabpnnConfigureBlockModel(blockModel,blockConfig,stream);
        fullCapacity<<<(vabpnn::NeuronCount+255)/256,256,0,stream>>>(vabpnnMutableSpikeStatistics(spike,stream));
        auto config=vabpnn::JepaConfig{}; config.selfSupervisedEnabled=selfSupervised;
        if(overlap) config.trainEveryFrames=1;
        check(cudaMemcpyAsync(vabpnnMutableJepaConfig(jepa,stream),&config,sizeof(config),cudaMemcpyHostToDevice,stream));
        check(cudaStreamSynchronize(stream));
        std::vector<Sample> all,ordinary,jepaFrames,pixelFrames,both,topologyFrames,topologyPixel,scopeFrames,allOverlap;
        std::vector<Sample> actorFrames,rankingFrames,mutationFrames,completeOverlap;
        std::uint64_t previousActors=0,previousRanks=0,previousMutations=0;
        std::uint64_t previousTopology=0;
        std::uint64_t previousJepa=0,previousPixel=0;
        vabpnn::DemoSample demo; demo.active=1; demo.epoch=1; demo.modelControl=mode; demo.pointerMode=1; demo.heldKeys=1;
        for(int i=0;i<Warmup+Count;++i) {
            const auto id=static_cast<std::uint64_t>(i+1);
            const auto nanos=1'000'000'000ULL+id*16'666'667ULL;
            demo.mouseX+=1; demo.keyHeldNanos[0]+=16'666'667ULL;
            VabpnnGpuFrame frame{image,pitch,rgb,Width,Height,id,nanos,stream,neuron,decision,jepa,online,spike};
            const auto sampleId=(id-1)/2*2+1; // 30 Hz observations, 60 Hz predictions.
            if(PositionEnabled) frame.position={1'000'000'000ULL+sampleId*16'666'667ULL,1,
                125.0+sampleId/30.0,64.0,-82.0+sampleId/60.0};
            if(forceRanking && i>0 && i%20==0)
                matureRankFixture<<<30,256,0,stream>>>(vabpnnMutableBlockPersistent(blockModel,stream));
            const auto begin=std::chrono::steady_clock::now();
            check(cudaEventRecord(events[0],stream));
            check(cudaMemcpy2DAsync(image,pitch,source,sourcePitch,Width*4,Height,cudaMemcpyDeviceToDevice,stream));
            check(cudaEventRecord(events[1],stream));
            vabpnnSetDemoSample(online,&demo,stream); vabpnnLaunchOperator(frame);
            const bool sampleScope=forceRanking || i%6==0;
            if(sampleScope) vabpnnLaunchNetworkScope(frame,32*120+60,scopeDevice);
            check(cudaEventRecord(events[2],stream));
            check(cudaMemcpyAsync(&status->command,vabpnnDecisionModelView(decision).command,sizeof(status->command),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->decision,&vabpnnDecisionModelView(decision).output->status,sizeof(status->decision),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->online,vabpnnOnlineOutput(online),sizeof(status->online),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->jepa,vabpnnJepaModelView(jepa).output,sizeof(status->jepa),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->spike,vabpnnSpikeModelView(spike).counters,sizeof(status->spike),cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(&status->blocks,vabpnnBlockModelView(blockModel).counters,sizeof(status->blocks),cudaMemcpyDeviceToHost,stream));
            if(sampleScope) check(cudaMemcpyAsync(&status->scope,scopeDevice,sizeof(vabpnn::ScopeTelemetry),cudaMemcpyDeviceToHost,stream));
            check(cudaEventRecord(events[3],stream)); check(cudaEventSynchronize(events[3]));
            const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            float copy=0,op=0,total=0;
            check(cudaEventElapsedTime(&copy,events[0],events[1]));
            check(cudaEventElapsedTime(&op,events[1],events[2]));
            check(cudaEventElapsedTime(&total,events[0],events[3]));
            const bool learnedJepa=status->jepa.selfSupervisedSteps>previousJepa;
            const bool learnedPixel=status->online.pixelSteps>previousPixel;
            const bool topology=status->spike.topologyUpdates>previousTopology; previousTopology=status->spike.topologyUpdates;
            previousJepa=status->jepa.selfSupervisedSteps; previousPixel=status->online.pixelSteps;
            const bool actor=status->blocks.actorUpdates>previousActors,rank=status->blocks.rankings>previousRanks,
                mutation=status->blocks.mutations>previousMutations;
            previousActors=status->blocks.actorUpdates; previousRanks=status->blocks.rankings; previousMutations=status->blocks.mutations;
            if(i>=Warmup) {
                Sample sample{copy,op,total,wall,learnedJepa,learnedPixel,topology,sampleScope,actor,rank,mutation};
                if(actor) actorFrames.push_back(sample); if(rank) rankingFrames.push_back(sample); if(mutation) mutationFrames.push_back(sample);
                if(actor && rank && mutation && sampleScope && topology && learnedJepa && learnedPixel) completeOverlap.push_back(sample); all.push_back(sample);
                if(sampleScope) scopeFrames.push_back(sample);
                if(sampleScope && topology && learnedJepa && learnedPixel) allOverlap.push_back(sample);
                if(!learnedJepa && !learnedPixel && !topology) ordinary.push_back(sample);
                if(topology) topologyFrames.push_back(sample);
                if(topology && learnedPixel) topologyPixel.push_back(sample);
                if(learnedJepa && !learnedPixel) jepaFrames.push_back(sample);
                if(learnedPixel && !learnedJepa) pixelFrames.push_back(sample);
                if(learnedPixel && learnedJepa) both.push_back(sample);
            }
        }
        std::printf("PROFILE %s selfSteps=%llu demoExamples=%llu mouseSteps=%llu modelBytes=%llu\n",label,
            static_cast<unsigned long long>(status->jepa.selfSupervisedSteps),
            static_cast<unsigned long long>(status->online.examples),
            static_cast<unsigned long long>(status->jepa.motionSteps),
            static_cast<unsigned long long>(vabpnnNeuronModelBytes(neuron)+vabpnnDecisionModelBytes(decision)+vabpnnJepaModelBytes(jepa)+vabpnnOnlineModelBytes(online)+vabpnnSpikeModelBytes(spike)+sizeof(vabpnn::ScopeTelemetry)));
        report("all",all); report("ordinary",ordinary); report("JEPA_update",jepaFrames);
        report("pixel_update",pixelFrames); report("JEPA_and_pixel",both);
        report("topology_update",topologyFrames); report("topology_and_pixels",topologyPixel);
        report("block_actor_update",actorFrames); report("block_ranking",rankingFrames); report("block_mutation",mutationFrames);
        report("scope_topology_pixels_JEPA_actor_rank_mutation",completeOverlap);
        report("scope_sample",scopeFrames); report("scope_topology_pixels_JEPA",allOverlap);
        std::printf(" anchor_count=%u long_edges=%u topology_updates=%llu local_edges=%llu\n",status->spike.anchors,status->spike.longEdges,
            static_cast<unsigned long long>(status->spike.topologyUpdates),static_cast<unsigned long long>(status->spike.localEdges));
        std::printf(" block_samples=%llu actor_updates=%llu rankings=%llu mutations=%llu controls=%llu broadcasts=%llu stress_fixture=%d\n",
            (unsigned long long)status->blocks.scoredFrames,(unsigned long long)status->blocks.actorUpdates,
            (unsigned long long)status->blocks.rankings,(unsigned long long)status->blocks.mutations,
            (unsigned long long)status->blocks.controlUpdates,(unsigned long long)status->blocks.broadcasts,forceRanking);
        if(forceRanking && (rankingFrames.empty() || mutationFrames.empty() || completeOverlap.empty()))
            throw std::runtime_error("controller sorting/mutation/full overlap was not actually measured");
        if(status->spike.anchors!=256 || status->spike.longEdges<80000) throw std::runtime_error("full-capacity long graph was not measured");
    } catch(...) {
        if(stream) cudaStreamSynchronize(stream);
        if(scopeDevice) cudaFree(scopeDevice); vabpnnDestroySpikeModel(spike); vabpnnDestroyOnlineModel(online); vabpnnDestroyJepaModel(jepa);
        vabpnnDestroyDecisionModel(decision); vabpnnDestroyNeuronModel(neuron);
        if(source) cudaFree(source); if(image) cudaFree(image); if(rgb) cudaFree(rgb); if(status) cudaFreeHost(status);
        for(auto event:events) if(event) cudaEventDestroy(event); if(stream) cudaStreamDestroy(stream);
        throw;
    }
    check(cudaStreamSynchronize(stream));
    if(scopeDevice) cudaFree(scopeDevice); vabpnnDestroySpikeModel(spike); vabpnnDestroyOnlineModel(online); vabpnnDestroyJepaModel(jepa);
    vabpnnDestroyDecisionModel(decision); vabpnnDestroyNeuronModel(neuron);
    cudaFree(source); cudaFree(image); cudaFree(rgb); cudaFreeHost(status);
    for(auto event:events) cudaEventDestroy(event); cudaStreamDestroy(stream);
}
int main(int argc,char** argv) {
    for(int i=1;i<argc;i++) {
        const std::string arg=argv[i];
        if(arg=="--without-position") PositionEnabled=false;
        if(arg=="--without-blocks") BlocksEnabled=false;
        if(arg=="--block-stress") BlockStress=true;
    }
    std::printf("BLOCK_CONTROLLERS=%s\n",BlocksEnabled ? "enabled":"disabled");
    std::printf("POSITION_FEATURES=%s (same GPU pipeline, 30 Hz position metadata / 60 Hz predictions)\n",PositionEnabled ? "enabled":"disabled");
    try {
        cudaDeviceProp gpu{}; check(cudaGetDeviceProperties(&gpu,0));
        std::printf("GPU=%s frame=1920x1024 warmup=64 measured=512 mode=release/O3; no GL or Minecraft render included\n",gpu.name);
        profile(1,0,"inference_only");
        profile(1,1,"model_control_plus_JEPA");
        profile(0,1,"human_demo_plus_JEPA");
        profile(0,1,"stress_topology_pixels_JEPA_every_frame",true);
        if(BlockStress) profile(0,1,"stress_hierarchy_rank_mutation_fixture",true,true);
        return 0;
    } catch(const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}
