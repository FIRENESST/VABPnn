#include "FrameOperator.h"
#include "SpikeConnections.h"
#include <stdexcept>
#include <string>

namespace {
constexpr int BlockCount = vabpnn::TargetGridWidth * vabpnn::TargetGridHeight;
constexpr int WarpCount = vabpnn::ThreadsPerBlock / 32;
constexpr std::uint64_t ModelBytes = vabpnn::ParameterCount * sizeof(vabpnn::TernaryWeight)
    + static_cast<std::uint64_t>(vabpnn::VoteCount) * BlockCount * sizeof(std::int32_t)
    + vabpnn::VoteCount * sizeof(std::int32_t);
static_assert(vabpnn::ThreadsPerBlock == 256 && WarpCount == 8, "16x16 voting tiles.");

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

__device__ __forceinline__ int warpSum(int value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// Every model tile is complete, so every warp has 32 active lanes.
__global__ __launch_bounds__(vabpnn::ThreadsPerBlock)
void neuronVotes(VabpnnGpuFrame frame, const vabpnn::TernaryWeight* weights,
                 std::int32_t* blockVotes,const std::int8_t* spikes,std::uint32_t* packedVotes) {
    __shared__ int warpVotes[vabpnn::VoteCount][WarpCount];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const std::size_t pixel = static_cast<std::size_t>(y) * vabpnn::NeuronWidth + x;

    const int pulse=spikes ? spikes[pixel]:1;
    std::uint32_t encoded=0;
    // Every warp is complete. Silent lanes still participate in reduction;
    // a wholly silent warp can publish exact zero sums without reading RGB/weights.
    const bool activeWarp=__any_sync(0xffffffffu,pulse!=0);
    if(activeWarp) {
        int r=0,g=0,b=0;
        if(pulse!=0) {
            // Center-based nearest sampling; preserve fixed top-left-origin neurons.
            const int sourceX=static_cast<std::size_t>(2*x+1)*frame.width/(2*vabpnn::NeuronWidth);
            const int sourceY=static_cast<std::size_t>(2*y+1)*frame.height/(2*vabpnn::NeuronHeight);
            const unsigned char* rgba=frame.rgba
                +static_cast<std::size_t>(frame.height-1-sourceY)*frame.rgbaPitch
                +static_cast<std::size_t>(sourceX)*4;
            r=rgba[0];g=rgba[1];b=rgba[2];
        }
        #pragma unroll
        for(int k=0;k<vabpnn::VoteCount;++k) {
            int vote=0;
            if(pulse!=0) {
                const int score=r*static_cast<int>(weights[vabpnn::weightIndex(pixel,0,k)])
                    +g*static_cast<int>(weights[vabpnn::weightIndex(pixel,1,k)])
                    +b*static_cast<int>(weights[vabpnn::weightIndex(pixel,2,k)]);
                vote=((score>0)-(score<0))*pulse;
            }
            encoded|=static_cast<std::uint32_t>(vote+1)<<(k*2);
            const int sum=warpSum(vote);
            if(lane==0) warpVotes[k][warp]=sum;
        }
    } else {
        #pragma unroll
        for(int k=0;k<vabpnn::VoteCount;++k) {
            encoded|=1u<<(k*2); // Packed neutral is one, not zero (which means -1).
            if(lane==0) warpVotes[k][warp]=0;
        }
    }
    if(packedVotes) packedVotes[pixel]=encoded;
    __syncthreads();
    if (tid < vabpnn::VoteCount) {
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < WarpCount; ++w) sum += warpVotes[tid][w];
        const int block = blockIdx.y * gridDim.x + blockIdx.x;
        blockVotes[static_cast<std::size_t>(tid) * BlockCount + block] = sum;
    }
}

// One block per vote item, reducing 7680 tile sums to one signed int32 total.
__global__ __launch_bounds__(vabpnn::ThreadsPerBlock)
void finishVotes(const std::int32_t* blockVotes, std::int32_t* totals) {
    __shared__ int warpVotes[WarpCount];
    const int k = blockIdx.x;
    const int tid = threadIdx.x;
    int sum = 0;
    for (int block = tid; block < BlockCount; block += blockDim.x)
        sum += blockVotes[static_cast<std::size_t>(k) * BlockCount + block];
    sum = warpSum(sum);
    if ((tid & 31) == 0) warpVotes[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) {
        int total = 0;
        #pragma unroll
        for (int w = 0; w < WarpCount; ++w) total += warpVotes[w];
        totals[k] = total;
    }
}
}

struct VabpnnNeuronModel {
    vabpnn::TernaryWeight* weights = nullptr;
    std::int32_t* blockVotes = nullptr;
    std::int32_t* voteSums = nullptr;
    cudaEvent_t initialized = nullptr;
};

VabpnnNeuronModel* vabpnnCreateNeuronModel(cudaStream_t stream) {
    auto* model = new VabpnnNeuronModel();
    try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&model->weights),
                            vabpnn::ParameterCount * sizeof(vabpnn::TernaryWeight)), "allocate ternary weights");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&model->blockVotes),
                            vabpnn::VoteCount * BlockCount * sizeof(std::int32_t)), "allocate tile votes");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&model->voteSums),
                            vabpnn::VoteCount * sizeof(std::int32_t)), "allocate vote totals");
        checkCuda(cudaEventCreateWithFlags(&model->initialized, cudaEventDisableTiming), "create model ready event");
        checkCuda(cudaMemsetAsync(model->weights, 0, vabpnn::ParameterCount, stream), "zero initial weights");
        checkCuda(cudaMemsetAsync(model->voteSums, 0, vabpnn::VoteCount * sizeof(std::int32_t), stream), "zero vote totals");
        checkCuda(cudaEventRecord(model->initialized, stream), "record model ready");
        return model;
    } catch (...) {
        vabpnnDestroyNeuronModel(model);
        throw;
    }
}

void vabpnnDestroyNeuronModel(VabpnnNeuronModel* model) noexcept {
    if (!model) return;
    // Caller must drain frame/learning streams first. This wait only covers initialization.
    if (model->initialized) cudaEventSynchronize(model->initialized);
    if (model->weights) cudaFree(model->weights);
    if (model->blockVotes) cudaFree(model->blockVotes);
    if (model->voteSums) cudaFree(model->voteSums);
    if (model->initialized) cudaEventDestroy(model->initialized);
    delete model;
}
std::uint64_t vabpnnNeuronModelBytes(const VabpnnNeuronModel* model) noexcept {
    return model ? ModelBytes : 0;
}
vabpnn::DeviceNeuronView vabpnnNeuronModelView(const VabpnnNeuronModel* model) {
    if (!model) throw std::invalid_argument("Neuron model is not initialized");
    return {model->weights, model->voteSums, model->voteSums, vabpnn::ParameterCount,model->blockVotes};
}
vabpnn::TernaryWeight* vabpnnMutableNeuronWeights(VabpnnNeuronModel* model, cudaStream_t stream) {
    if (!model) throw std::invalid_argument("Neuron model is not initialized");
    checkCuda(cudaStreamWaitEvent(stream, model->initialized, 0), "wait before weight update");
    return model->weights;
}

void vabpnnLaunchTernaryVotes(const VabpnnGpuFrame& frame, VabpnnNeuronModel* model) {
    if (!model || !frame.rgba || frame.width <= 0 || frame.height <= 0
            || frame.rgbaPitch < static_cast<std::size_t>(frame.width) * 4)
        throw std::invalid_argument("Invalid neuron model or source frame");
    checkCuda(cudaStreamWaitEvent(frame.stream, model->initialized, 0), "wait for model initialization");
    const auto spike=frame.spikeModel ? vabpnnSpikeModelView(frame.spikeModel):vabpnn::DeviceSpikeView{};
    auto* cache=frame.spikeModel ? vabpnnMutableSpikeVoteCache(frame.spikeModel,frame.stream):nullptr;
    neuronVotes<<<vabpnn::gridShape(vabpnn::NeuronWidth, vabpnn::NeuronHeight),
                  vabpnn::blockShape(), 0, frame.stream>>>(frame, model->weights, model->blockVotes,spike.spikes,cache);
    checkCuda(cudaGetLastError(), "launch per-neuron ternary votes");
    finishVotes<<<vabpnn::VoteCount, vabpnn::ThreadsPerBlock, 0, frame.stream>>>(
                  model->blockVotes, model->voteSums);
    checkCuda(cudaGetLastError(), "reduce 13 raw vote totals");
}

