#include "JepaMotion.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <initializer_list>

namespace {
using namespace vabpnn;
struct AdamGroup {
    std::uint64_t steps, lastSerial;
    float bias1, bias2, scale;
    int active;
};
struct JepaState {
    std::uint64_t serial, lastNanos, originNanos, lastFrame;
    int historyReady, width, height, trainJepa, trainMotion;
};
struct JepaWorkspace {
    float currentFeatures[JepaTokens][JepaInput];
    float historyFeatures[JepaTokens][JepaInput];
    float previousFeatures[JepaTokens][JepaInput];
    float currentHidden[JepaTokens][JepaHidden], currentLatent[JepaTokens][JepaWidth];
    float previousHidden[JepaTokens][JepaHidden], previousLatent[JepaTokens][JepaWidth];
    float targetHidden[JepaTokens][JepaHidden], targetLatent[JepaTokens][JepaWidth];
    float currentMoments[JepaMoments], previousMoments[JepaMoments];
    float predictorInput[JepaTokens][JepaPredictInput];
    float predictorHidden[JepaTokens][JepaHidden], prediction[JepaTokens][JepaWidth];
    float predictionGradient[JepaTokens][JepaWidth];
    float predictorHiddenGradient[JepaTokens][JepaHidden];
    float predictorInputGradient[JepaTokens][JepaPredictInput];
    float previousLatentGradient[JepaTokens][JepaWidth];
    float previousHiddenGradient[JepaTokens][JepaHidden];
    float latentMean[JepaWidth], latentStd[JepaWidth], momentGradient[JepaMoments];
    float motionInput[JepaMotionInput], motionHidden[JepaHidden], motionRaw[2];
    float motionGradient[2], motionHiddenGradient[JepaHidden];
};
constexpr std::uint64_t JepaBytes = sizeof(JepaParameters) * 4 + sizeof(JepaEncoder)
    + sizeof(JepaConfig) + sizeof(JepaState) + sizeof(JepaWorkspace)
    + sizeof(JepaOutput) + sizeof(JepaMotionTarget) + sizeof(AdamGroup) * 2;
static_assert(offsetof(JepaParameters, motion) == JepaSelfSupervisedParameters * sizeof(float), "Optimizer groups.");
void check(cudaError_t code, const char* label) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(code));
}
__device__ float xPosition(int token) { return 2.0f * (token % JepaGridX + 0.5f) / JepaGridX - 1.0f; }
__device__ float yPosition(int token) { return 2.0f * (token / JepaGridX + 0.5f) / JepaGridY - 1.0f; }
__device__ bool selected(int token, const JepaOutput* out) {
    const int x = static_cast<int>(out->frameId % (JepaGridX - JepaMaskX + 1));
    const int y = static_cast<int>((out->frameId / 3) % (JepaGridY - JepaMaskY + 1));
    const int tx = token % JepaGridX, ty = token / JepaGridX;
    return tx >= x && tx < x + JepaMaskX && ty >= y && ty < y + JepaMaskY;
}
__device__ bool validConfig(const JepaConfig* c) {
    return isfinite(c->learningRate) && c->learningRate >= 0
        && isfinite(c->beta1) && c->beta1 >= 0 && c->beta1 < 1
        && isfinite(c->beta2) && c->beta2 >= 0 && c->beta2 < 1
        && isfinite(c->epsilon) && c->epsilon > 0
        && isfinite(c->weightDecay) && c->weightDecay >= 0
        && isfinite(c->gradientClip) && c->gradientClip > 0
        && isfinite(c->emaDecay) && c->emaDecay >= 0 && c->emaDecay <= 1
        && isfinite(c->varianceTarget) && c->varianceTarget > 0
        && isfinite(c->varianceWeight) && c->varianceWeight >= 0
        && isfinite(c->maxPairGapSeconds) && c->maxPairGapSeconds > 0
        && isfinite(c->maxMotionDtSeconds) && c->maxMotionDtSeconds > 0
        && isfinite(c->maxMouseUnitsPerSecond) && c->maxMouseUnitsPerSecond > 0
        && isfinite(c->maxMouseDelta) && c->maxMouseDelta > 0 && c->trainEveryFrames > 0;
}
template<int I, int H, int O>
__global__ void initializeMlp(JepaMlp<I,H,O>* p, unsigned int seed, bool neutralOutput) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int IW = I * H, HW = H * O, Count = IW + H + HW + O;
    if (index >= Count) return;
    unsigned int hash = static_cast<unsigned int>(index) + seed;
    hash ^= hash >> 16; hash *= 0x7feb352du;
    hash ^= hash >> 15; hash *= 0x846ca68bu; hash ^= hash >> 16;
    float value = 0;
    if (index < IW || (index >= IW + H && index < IW + H + HW && !neutralOutput)) {
        const float bound = sqrtf(6.0f / (index < IW ? I + H : H + O));
        value = (static_cast<float>(hash & 0xffffu) / 32767.5f - 1.0f) * bound;
    }
    reinterpret_cast<float*>(p)[index] = value;
}
__global__ void initializeConfig(JepaConfig* config) { *config = JepaConfig{}; }

__global__ void beginFrame(JepaState* state, JepaOutput* out, JepaMotionTarget* label,
                           const JepaConfig* config, const AdamGroup* groups,
                           std::uint64_t id, std::uint64_t nanos, int width, int height) {
    const bool consecutive = state->historyReady && id > state->lastFrame
        && nanos > state->lastNanos && width == state->width && height == state->height;
    if (!consecutive) state->originNanos = nanos;
    const float dt = consecutive ? static_cast<float>((nanos - state->lastNanos) * 1e-9) : 0;
    ++state->serial;
    state->trainJepa = state->trainMotion = 0;
    *out = JepaOutput{};
    out->frameId = id; out->captureNanos = nanos; out->deltaSeconds = dt;
    out->selfSupervisedSteps = groups[0].steps; out->motionSteps = groups[1].steps;
    out->valid = validConfig(config);
    out->pairReady = consecutive && out->valid && dt <= config->maxPairGapSeconds;
    *label = JepaMotionTarget{}; // A later trainer must supply a new, matching-frame target.
}
__global__ void poolPatches(VabpnnGpuFrame frame, JepaWorkspace* work) {
    // At 1920x1024 each block averages one 16x16 RGB cell; four-by-four cells form a patch.
    const int block = blockIdx.x, token = block / (JepaCells * JepaCells);
    const int cell = block % (JepaCells * JepaCells);
    const int cx = token % JepaGridX * JepaCells + cell % JepaCells;
    const int cy = token / JepaGridX * JepaCells + cell / JepaCells;
    const int x0 = cx * frame.width / (JepaGridX * JepaCells);
    const int y0 = cy * frame.height / (JepaGridY * JepaCells);
    const int x1 = min(frame.width, max(x0 + 1, (cx + 1) * frame.width / (JepaGridX * JepaCells)));
    const int y1 = min(frame.height, max(y0 + 1, (cy + 1) * frame.height / (JepaGridY * JepaCells)));
    const int count = (x1 - x0) * (y1 - y0), thread = threadIdx.y * blockDim.x + threadIdx.x;
    const std::size_t plane = static_cast<std::size_t>(frame.width) * frame.height;
    float r = 0, g = 0, b = 0;
    for (int at = thread; at < count; at += 256) {
        const std::size_t pixel = static_cast<std::size_t>(y0 + at / (x1 - x0)) * frame.width + x0 + at % (x1 - x0);
        r += frame.rgbNchw[pixel]; g += frame.rgbNchw[plane + pixel]; b += frame.rgbNchw[2 * plane + pixel];
    }
    for (int offset = 16; offset; offset >>= 1) {
        r += __shfl_down_sync(0xffffffffu, r, offset);
        g += __shfl_down_sync(0xffffffffu, g, offset);
        b += __shfl_down_sync(0xffffffffu, b, offset);
    }
    __shared__ float sums[3][8];
    const int lane = thread & 31, warp = thread >> 5;
    if (lane == 0) { sums[0][warp] = r; sums[1][warp] = g; sums[2][warp] = b; }
    __syncthreads();
    if (warp == 0) {
        r = lane < 8 ? sums[0][lane] : 0; g = lane < 8 ? sums[1][lane] : 0; b = lane < 8 ? sums[2][lane] : 0;
        for (int offset = 16; offset; offset >>= 1) {
            r += __shfl_down_sync(0xffffffffu, r, offset);
            g += __shfl_down_sync(0xffffffffu, g, offset);
            b += __shfl_down_sync(0xffffffffu, b, offset);
        }
        if (lane == 0) {
            work->currentFeatures[token][cell*3] = 2 * r / count - 1;
            work->currentFeatures[token][cell*3+1] = 2 * g / count - 1;
            work->currentFeatures[token][cell*3+2] = 2 * b / count - 1;
        }
    }
}
template<int I, int H, int O>
__global__ void forwardMlp(const JepaMlp<I,H,O>* p, const float* input, float* hidden, float* output,
                           const JepaOutput* status, bool masked) {
    const int token = blockIdx.x, d = threadIdx.x;
    if (masked && !selected(token, status)) {
        if (d < H) hidden[token*H+d] = 0;
        if (d < O) output[token*O+d] = 0;
        return; // The whole block agrees; no partial-block barrier exit.
    }
    __shared__ float x[I], h[H];
    for (int at = d; at < I; at += blockDim.x) x[at] = input[token*I+at];
    __syncthreads();
    if (d < H) {
        float value = p->hiddenBias[d];
        for (int at = 0; at < I; ++at) value += x[at] * p->inputWeight[at][d];
        h[d] = hidden[token*H+d] = tanhf(value);
    }
    __syncthreads();
    if (d < O) {
        float value = p->outputBias[d];
        for (int at = 0; at < H; ++at) value += h[at] * p->outputWeight[at][d];
        output[token*O+d] = value;
    }
}
__global__ void summarize(JepaWorkspace* work) {
    const int d = threadIdx.x;
    for (int moment = 0; moment < 3; ++moment) {
        float current = 0, previous = 0;
        for (int token = 0; token < JepaTokens; ++token) {
            const float weight = moment == 0 ? 1.0f : moment == 1 ? xPosition(token) : yPosition(token);
            current += work->currentLatent[token][d] * weight;
            previous += work->previousLatent[token][d] * weight;
        }
        work->currentMoments[moment*JepaWidth+d] = current / JepaTokens;
        work->previousMoments[moment*JepaWidth+d] = previous / JepaTokens;
    }
}
__global__ void buildPredictorInput(JepaWorkspace* work, const JepaOutput* out) {
    const int token = blockIdx.x;
    for (int d = threadIdx.x; d < JepaPredictInput; d += blockDim.x) {
        float value = 0;
        if (d < JepaWidth) value = work->previousLatent[token][d];
        else if (d < JepaWidth + JepaMoments) value = work->previousMoments[d-JepaWidth];
        else if (d == JepaWidth + JepaMoments) value = xPosition(token);
        else if (d == JepaWidth + JepaMoments+1) value = yPosition(token);
        else if (d == JepaWidth + JepaMoments+2) value = log1pf(out->deltaSeconds);
        else value = out->deltaSeconds;
        work->predictorInput[token][d] = value;
    }
}
__global__ void buildMotionInput(JepaWorkspace* work, const JepaOutput* out, const std::int32_t* votes) {
    for (int d = threadIdx.x; d < JepaMotionInput; d += blockDim.x) {
        float value;
        if (d < JepaMoments) value = work->currentMoments[d];
        else if (d < 2*JepaMoments) value = out->pairReady ? work->previousMoments[d-JepaMoments] : work->currentMoments[d-JepaMoments];
        else if (d < 3*JepaMoments) value = out->pairReady ? work->currentMoments[d-2*JepaMoments] - work->previousMoments[d-2*JepaMoments] : 0;
        else if (d < 3*JepaMoments+VoteCount) value = static_cast<float>(votes[d-3*JepaMoments]) / NeuronCount;
        else value = d == 3*JepaMoments+VoteCount ? log1pf(out->deltaSeconds) : out->deltaSeconds;
        work->motionInput[d] = value;
    }
}
__global__ void lossAndPredictionGradient(JepaWorkspace* work, JepaOutput* out, const JepaConfig* config) {
    const int d = threadIdx.x;
    float loss = 0, mean = 0, variance = 0;
    for (int token = 0; token < JepaTokens; ++token) mean += work->previousLatent[token][d];
    mean /= JepaTokens;
    for (int token = 0; token < JepaTokens; ++token) {
        const float difference = work->previousLatent[token][d] - mean;
        variance += difference * difference;
        float gradient = 0;
        if (out->pairReady && selected(token, out)) {
            const float error = work->prediction[token][d] - work->targetLatent[token][d];
            loss += 0.5f * error * error / (JepaTargets * JepaWidth);
            gradient = error / (JepaTargets * JepaWidth);
        }
        work->predictionGradient[token][d] = gradient;
    }
    const float std = sqrtf(variance / JepaTokens + 1e-4f);
    work->latentMean[d] = mean; work->latentStd[d] = std;
    __shared__ float losses[JepaWidth], penalties[JepaWidth];
    losses[d] = loss;
    penalties[d] = out->pairReady ? fmaxf(0, config->varianceTarget - std) / JepaWidth : 0;
    __syncthreads();
    for (int offset = JepaWidth/2; offset; offset >>= 1) {
        if (d < offset) { losses[d] += losses[d+offset]; penalties[d] += penalties[d+offset]; }
        __syncthreads();
    }
    if (d == 0) {
        out->predictionLoss = losses[0]; out->variancePenalty = penalties[0];
        out->jepaLoss = losses[0] + config->varianceWeight * penalties[0];
    }
}
__global__ void finishFrame(JepaWorkspace* work, JepaState* state, JepaOutput* out,
                            const JepaConfig* config, MotionProposal* proposal, int width, int height) {
    const float rx = tanhf(work->motionRaw[0]) * config->maxMouseUnitsPerSecond;
    const float ry = tanhf(work->motionRaw[1]) * config->maxMouseUnitsPerSecond;
    out->valid = out->valid && isfinite(rx) && isfinite(ry) && isfinite(out->jepaLoss);
    out->headRateX=out->valid && out->pairReady ? rx:0;
    out->headRateY=out->valid && out->pairReady ? ry:0;
    out->motionActive = out->valid && out->pairReady && config->motionEnabled;
    out->rateX = out->motionActive ? rx : 0; out->rateY = out->motionActive ? ry : 0;
    const float dt = fminf(out->deltaSeconds, config->maxMotionDtSeconds);
    out->mouseDx = out->motionActive ? fminf(config->maxMouseDelta, fmaxf(-config->maxMouseDelta, rx*dt)) : 0;
    out->mouseDy = out->motionActive ? fminf(config->maxMouseDelta, fmaxf(-config->maxMouseDelta, ry*dt)) : 0;
    if (proposal) { proposal->mouseDx = out->mouseDx; proposal->mouseDy = out->mouseDy; }
    state->historyReady = out->valid;
    state->lastNanos = out->captureNanos; state->lastFrame = out->frameId;
    state->width = width; state->height = height;
}

__global__ void prepareSelfSupervised(JepaState* state, JepaOutput* out,
                                     const JepaConfig* config, AdamGroup* groups) {
    state->trainJepa = out->valid && out->pairReady && config->selfSupervisedEnabled
        && state->serial % config->trainEveryFrames == 0 && groups[0].lastSerial != state->serial;
    groups[0].active = state->trainJepa;
}
__global__ void prepareMotionTraining(JepaWorkspace* work, JepaState* state, JepaOutput* out,
                                     const JepaConfig* config, const JepaMotionTarget* target, AdamGroup* groups) {
    state->trainMotion = out->valid && out->pairReady && target->valid == 1 && target->frameId == out->frameId
        && isfinite(target->rateX) && isfinite(target->rateY)
        && fabsf(target->rateX) <= config->maxMouseUnitsPerSecond
        && fabsf(target->rateY) <= config->maxMouseUnitsPerSecond
        && groups[1].lastSerial != state->serial;
    groups[1].active = state->trainMotion;
    out->motionLoss = 0;
    for (int d = 0; d < 2; ++d) {
        const float prediction = tanhf(work->motionRaw[d]);
        const float expected = (d == 0 ? target->rateX : target->rateY) / config->maxMouseUnitsPerSecond;
        const float difference = prediction - expected;
        work->motionGradient[d] = state->trainMotion ? 0.5f * difference * (1-prediction*prediction) : 0;
        if (state->trainMotion) out->motionLoss += 0.25f * difference * difference;
    }
}
template<int I, int H, int O>
__global__ void hiddenGradient(const JepaMlp<I,H,O>* p, const float* hidden, const float* outputGradient,
                               float* hiddenDelta, const int* active, const JepaOutput* out, bool masked) {
    if (!*active) return;
    const int token = blockIdx.x, d = threadIdx.x;
    if (masked && !selected(token, out)) { if (d < H) hiddenDelta[token*H+d] = 0; return; }
    __shared__ float delta[O];
    if (d < O) delta[d] = outputGradient[token*O+d];
    __syncthreads();
    if (d < H) {
        float value = 0;
        for (int e = 0; e < O; ++e) value += delta[e] * p->outputWeight[d][e];
        const float activation = hidden[token*H+d];
        hiddenDelta[token*H+d] = value * (1-activation*activation);
    }
}
template<int I, int H, int O>
__global__ void inputGradient(const JepaMlp<I,H,O>* p, const float* hiddenDelta, float* inputDelta,
                              const int* active, const JepaOutput* out, bool masked) {
    if (!*active) return;
    const int token = blockIdx.x;
    if (masked && !selected(token, out)) {
        for (int d = threadIdx.x; d < I; d += blockDim.x) inputDelta[token*I+d] = 0;
        return;
    }
    __shared__ float delta[H];
    if (threadIdx.x < H) delta[threadIdx.x] = hiddenDelta[token*H+threadIdx.x];
    __syncthreads();
    for (int d = threadIdx.x; d < I; d += blockDim.x) {
        float value = 0;
        for (int e = 0; e < H; ++e) value += delta[e] * p->inputWeight[d][e];
        inputDelta[token*I+d] = value;
    }
}
__device__ int targetToken(int index, const JepaOutput* out) {
    const int x = static_cast<int>(out->frameId % (JepaGridX - JepaMaskX + 1));
    const int y = static_cast<int>((out->frameId / 3) % (JepaGridY - JepaMaskY + 1));
    return (y + index/JepaMaskX) * JepaGridX + x + index%JepaMaskX;
}
template<int I, int H, int O>
__global__ void parameterGradient(const float* input, const float* hidden, const float* outputDelta,
                                  const float* hiddenDelta, float* gradient, int batch,
                                  const int* active, const JepaOutput* out, bool masked) {
    if (!*active) return;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int IW = I*H, HW = H*O, Count = IW+H+HW+O;
    if (index >= Count) return;
    float value = 0;
    const int count = masked ? JepaTargets : batch;
    for (int n = 0; n < count; ++n) {
        const int token = masked ? targetToken(n, out) : n;
        if (index < IW) value += input[token*I+index/H] * hiddenDelta[token*H+index%H];
        else if (index < IW+H) value += hiddenDelta[token*H+index-IW];
        else if (index < IW+H+HW) {
            const int at = index-IW-H;
            value += hidden[token*H+at/O] * outputDelta[token*O+at%O];
        } else value += outputDelta[token*O+index-IW-H-HW];
    }
    gradient[index] = value;
}
__global__ void momentGradient(JepaWorkspace* work, const JepaOutput* out, const JepaState* state) {
    if (!state->trainJepa) return;
    const int d = threadIdx.x;
    for (int moment = 0; moment < 3; ++moment) {
        float sum = 0;
        for (int n = 0; n < JepaTargets; ++n)
            sum += work->predictorInputGradient[targetToken(n,out)][JepaWidth+moment*JepaWidth+d];
        work->momentGradient[moment*JepaWidth+d] = sum;
    }
}
__global__ void latentGradient(JepaWorkspace* work, const JepaConfig* config,
                               const JepaState* state, const JepaOutput* out) {
    if (!state->trainJepa) return;
    const int token = blockIdx.x, d = threadIdx.x;
    float value = selected(token,out) ? work->predictorInputGradient[token][d] : 0;
    value += (work->momentGradient[d] + xPosition(token)*work->momentGradient[JepaWidth+d]
              + yPosition(token)*work->momentGradient[2*JepaWidth+d]) / JepaTokens;
    if (work->latentStd[d] < config->varianceTarget) {
        // d max(0,gamma-std(z)) / dz, population variance over patch tokens.
        value += config->varianceWeight * (work->latentMean[d]-work->previousLatent[token][d])
            / (JepaWidth * JepaTokens * work->latentStd[d]);
    }
    work->previousLatentGradient[token][d] = value;
}
__global__ void prepareAdam(const float* gradient, std::size_t offset, std::size_t count,
                            const JepaConfig* config, const JepaState* state,
                            JepaOutput* out, AdamGroup* group) {
    if (!group->active) return;
    float sum = 0;
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
        const float value = gradient[offset+index]; sum += value*value;
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (threadIdx.x < step) sums[threadIdx.x] += sums[threadIdx.x+step];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out->lastLearningValid = isfinite(sums[0]) && validConfig(config);
        group->active = out->lastLearningValid;
        out->gradientNorm = out->lastLearningValid ? sqrtf(sums[0]) : 0;
        if (group->active) {
            group->scale = out->gradientNorm > config->gradientClip ? config->gradientClip / out->gradientNorm : 1;
            ++group->steps; group->lastSerial = state->serial;
            group->bias1 = 1-powf(config->beta1, static_cast<float>(group->steps));
            group->bias2 = 1-powf(config->beta2, static_cast<float>(group->steps));
        }
    }
}
template<int I, int H, int O>
__device__ bool matrixParameter(std::size_t index) {
    return index < I*H || (index >= I*H+H && index < I*H+H+H*O);
}
__device__ bool decayParameter(std::size_t index) {
    if (index < JepaEncoderParameters) return matrixParameter<JepaInput,JepaHidden,JepaWidth>(index);
    if (index < JepaSelfSupervisedParameters)
        return matrixParameter<JepaPredictInput,JepaHidden,JepaWidth>(index-JepaEncoderParameters);
    return matrixParameter<JepaMotionInput,JepaHidden,2>(index-JepaSelfSupervisedParameters);
}
__global__ void adamw(float* parameters, const float* gradient, float* first, float* second,
                      std::size_t offset, std::size_t count, const JepaConfig* config, const AdamGroup* group) {
    if (!group->active) return;
    const std::size_t at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= count) return;
    const std::size_t index = offset+at;
    const float g = gradient[index]*group->scale;
    const float m = config->beta1*first[index] + (1-config->beta1)*g;
    const float v = config->beta2*second[index] + (1-config->beta2)*g*g;
    first[index] = m; second[index] = v;
    const float decay = decayParameter(index) ? config->weightDecay : 0;
    parameters[index] = parameters[index]*(1-config->learningRate*decay)
        - config->learningRate*(m/group->bias1)/(sqrtf(v/group->bias2)+config->epsilon);
}
__global__ void updateTarget(JepaEncoder* target, const JepaParameters* online,
                             const JepaConfig* config, const AdamGroup* group) {
    if (!group->active) return;
    const std::size_t index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < JepaEncoderParameters) {
        float* values = reinterpret_cast<float*>(target);
        const float* source = reinterpret_cast<const float*>(&online->encoder);
        values[index] = config->emaDecay*values[index] + (1-config->emaDecay)*source[index];
    }
}
__global__ void publishSteps(JepaOutput* out, const AdamGroup* groups) {
    out->selfSupervisedSteps = groups[0].steps; out->motionSteps = groups[1].steps;
}
__global__ void resetHistory(JepaState* state, JepaOutput* out, JepaMotionTarget* target) {
    const auto serial = state->serial;
    *state = JepaState{}; state->serial = serial;
    *out = JepaOutput{}; *target = JepaMotionTarget{};
}
} // namespace

struct VabpnnJepaModel {
    vabpnn::JepaParameters* parameters = nullptr;
    vabpnn::JepaEncoder* target = nullptr;
    vabpnn::JepaConfig* config = nullptr;
    vabpnn::JepaOutput* output = nullptr;
    vabpnn::JepaMotionTarget* motionTarget = nullptr;
    JepaState* state = nullptr;
    JepaWorkspace* work = nullptr;
    AdamGroup* groups = nullptr;
    float* gradient = nullptr;
    float* first = nullptr;
    float* second = nullptr;
    cudaEvent_t initialized = nullptr;
};

namespace {
template<class T> void allocate(T*& pointer, const char* name) {
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(T)), name);
}
void waitReady(VabpnnJepaModel* model, cudaStream_t stream) {
    if (!model) throw std::invalid_argument("JEPA model is not initialized");
    check(cudaStreamWaitEvent(stream, model->initialized, 0), "wait for JEPA initialization");
}
constexpr unsigned int grid(std::size_t count) { return static_cast<unsigned int>((count+255)/256); }
}
VabpnnJepaModel* vabpnnCreateJepaModel(cudaStream_t stream) {
    auto* model = new VabpnnJepaModel();
    try {
        allocate(model->parameters,"allocate JEPA parameters");
        allocate(model->target,"allocate EMA encoder");
        allocate(model->config,"allocate JEPA config");
        allocate(model->output,"allocate JEPA output");
        allocate(model->motionTarget,"allocate motion supervision");
        allocate(model->state,"allocate JEPA state");
        allocate(model->work,"allocate JEPA workspace");
        check(cudaMalloc(reinterpret_cast<void**>(&model->groups),2*sizeof(AdamGroup)),"allocate optimizer groups");
        for (auto pointer : {&model->gradient,&model->first,&model->second}) {
            check(cudaMalloc(reinterpret_cast<void**>(pointer),sizeof(JepaParameters)),"allocate AdamW buffers");
            check(cudaMemsetAsync(*pointer,0,sizeof(JepaParameters),stream),"zero AdamW buffers");
        }
        check(cudaEventCreateWithFlags(&model->initialized,cudaEventDisableTiming),"create JEPA ready event");
        check(cudaMemsetAsync(model->state,0,sizeof(JepaState),stream),"zero JEPA state");
        check(cudaMemsetAsync(model->work,0,sizeof(JepaWorkspace),stream),"zero JEPA workspace");
        check(cudaMemsetAsync(model->output,0,sizeof(JepaOutput),stream),"zero JEPA output");
        check(cudaMemsetAsync(model->motionTarget,0,sizeof(JepaMotionTarget),stream),"zero motion target");
        check(cudaMemsetAsync(model->groups,0,2*sizeof(AdamGroup),stream),"zero optimizer groups");
        initializeMlp<<<grid(JepaEncoderParameters),256,0,stream>>>(&model->parameters->encoder,101,false);
        initializeMlp<<<grid(JepaSelfSupervisedParameters-JepaEncoderParameters),256,0,stream>>>(&model->parameters->predictor,211,false);
        initializeMlp<<<grid(JepaParameterCount-JepaSelfSupervisedParameters),256,0,stream>>>(&model->parameters->motion,307,true);
        initializeConfig<<<1,1,0,stream>>>(model->config);
        check(cudaGetLastError(),"initialize JEPA");
        check(cudaMemcpyAsync(model->target,&model->parameters->encoder,sizeof(JepaEncoder),cudaMemcpyDeviceToDevice,stream),"initialize EMA copy");
        check(cudaEventRecord(model->initialized,stream),"record JEPA ready event");
        return model;
    } catch (...) { vabpnnDestroyJepaModel(model); throw; }
}
void vabpnnDestroyJepaModel(VabpnnJepaModel* model) noexcept {
    if (!model) return;
    // Caller drains all forward/training streams. This event covers initialization only.
    if (model->initialized) cudaEventSynchronize(model->initialized);
    if (model->parameters) cudaFree(model->parameters);
    if (model->target) cudaFree(model->target);
    if (model->config) cudaFree(model->config);
    if (model->output) cudaFree(model->output);
    if (model->motionTarget) cudaFree(model->motionTarget);
    if (model->state) cudaFree(model->state);
    if (model->work) cudaFree(model->work);
    if (model->groups) cudaFree(model->groups);
    if (model->gradient) cudaFree(model->gradient);
    if (model->first) cudaFree(model->first);
    if (model->second) cudaFree(model->second);
    if (model->initialized) cudaEventDestroy(model->initialized);
    delete model;
}
std::uint64_t vabpnnJepaModelBytes(const VabpnnJepaModel* model) noexcept { return model ? JepaBytes : 0; }
vabpnn::DeviceJepaView vabpnnJepaModelView(const VabpnnJepaModel* model) {
    if (!model) throw std::invalid_argument("JEPA model is not initialized");
    return {model->parameters, model->target, model->config, model->output,
        &model->work->currentFeatures[0][0], &model->work->previousFeatures[0][0],
        &model->work->currentLatent[0][0], &model->work->previousLatent[0][0],
        &model->work->targetLatent[0][0], &model->work->prediction[0][0],
        model->gradient, model->first, model->second};
}
vabpnn::JepaParameters* vabpnnMutableJepaParameters(VabpnnJepaModel* model, cudaStream_t stream) {
    waitReady(model,stream); return model->parameters;
}
vabpnn::JepaConfig* vabpnnMutableJepaConfig(VabpnnJepaModel* model, cudaStream_t stream) {
    waitReady(model,stream); return model->config;
}
vabpnn::JepaMotionTarget* vabpnnMutableJepaMotionTarget(VabpnnJepaModel* model, cudaStream_t stream) {
    waitReady(model,stream); return model->motionTarget;
}

void vabpnnLaunchJepaForward(const VabpnnGpuFrame& frame, const std::int32_t* votes,
                            VabpnnJepaModel* model, vabpnn::MotionProposal* proposal) {
    if (!frame.rgbNchw || !votes || frame.width <= 0 || frame.height <= 0)
        throw std::invalid_argument("Invalid JEPA device frame or votes");
    waitReady(model,frame.stream);
    auto* w = model->work;
    beginFrame<<<1,1,0,frame.stream>>>(model->state,model->output,model->motionTarget,
        model->config,model->groups,frame.frameId,frame.captureNanos,frame.width,frame.height);
    // Re-encode historical raw features with current weights; no stale trained projections.
    check(cudaMemcpyAsync(w->previousFeatures,w->historyFeatures,sizeof(w->previousFeatures),
                          cudaMemcpyDeviceToDevice,frame.stream),"stage JEPA past features");
    poolPatches<<<JepaTokens*JepaCells*JepaCells,dim3(16,16),0,frame.stream>>>(frame,w);
    forwardMlp<<<JepaTokens,256,0,frame.stream>>>(&model->parameters->encoder,&w->currentFeatures[0][0],
        &w->currentHidden[0][0],&w->currentLatent[0][0],model->output,false);
    forwardMlp<<<JepaTokens,256,0,frame.stream>>>(&model->parameters->encoder,&w->previousFeatures[0][0],
        &w->previousHidden[0][0],&w->previousLatent[0][0],model->output,false);
    forwardMlp<<<JepaTokens,256,0,frame.stream>>>(model->target,&w->currentFeatures[0][0],
        &w->targetHidden[0][0],&w->targetLatent[0][0],model->output,false);
    summarize<<<1,JepaWidth,0,frame.stream>>>(w);
    buildPredictorInput<<<JepaTokens,256,0,frame.stream>>>(w,model->output);
    forwardMlp<<<JepaTokens,256,0,frame.stream>>>(&model->parameters->predictor,&w->predictorInput[0][0],
        &w->predictorHidden[0][0],&w->prediction[0][0],model->output,true);
    buildMotionInput<<<1,256,0,frame.stream>>>(w,model->output,votes);
    forwardMlp<<<1,256,0,frame.stream>>>(&model->parameters->motion,w->motionInput,
        w->motionHidden,w->motionRaw,model->output,false);
    lossAndPredictionGradient<<<1,JepaWidth,0,frame.stream>>>(w,model->output,model->config);
    finishFrame<<<1,1,0,frame.stream>>>(w,model->state,model->output,model->config,proposal,frame.width,frame.height);
    check(cudaGetLastError(),"launch JEPA forward");
    check(cudaMemcpyAsync(w->historyFeatures,w->currentFeatures,sizeof(w->historyFeatures),
                          cudaMemcpyDeviceToDevice,frame.stream),"cache JEPA patch history");
}
void vabpnnTrainJepaSelfSupervised(VabpnnJepaModel* model, cudaStream_t stream) {
    waitReady(model,stream);
    auto* w = model->work;
    const int* active = &model->state->trainJepa;
    prepareSelfSupervised<<<1,1,0,stream>>>(model->state,model->output,model->config,model->groups);
    hiddenGradient<<<JepaTokens,256,0,stream>>>(&model->parameters->predictor,&w->predictorHidden[0][0],
        &w->predictionGradient[0][0],&w->predictorHiddenGradient[0][0],active,model->output,true);
    inputGradient<<<JepaTokens,256,0,stream>>>(&model->parameters->predictor,&w->predictorHiddenGradient[0][0],
        &w->predictorInputGradient[0][0],active,model->output,true);
    parameterGradient<JepaPredictInput,JepaHidden,JepaWidth><<<grid(JepaSelfSupervisedParameters-JepaEncoderParameters),256,0,stream>>>(
        &w->predictorInput[0][0],&w->predictorHidden[0][0],&w->predictionGradient[0][0],
        &w->predictorHiddenGradient[0][0],model->gradient+JepaEncoderParameters,JepaTokens,active,model->output,true);
    momentGradient<<<1,JepaWidth,0,stream>>>(w,model->output,model->state);
    latentGradient<<<JepaTokens,JepaWidth,0,stream>>>(w,model->config,model->state,model->output);
    hiddenGradient<<<JepaTokens,256,0,stream>>>(&model->parameters->encoder,&w->previousHidden[0][0],
        &w->previousLatentGradient[0][0],&w->previousHiddenGradient[0][0],active,model->output,false);
    parameterGradient<JepaInput,JepaHidden,JepaWidth><<<grid(JepaEncoderParameters),256,0,stream>>>(
        &w->previousFeatures[0][0],&w->previousHidden[0][0],&w->previousLatentGradient[0][0],
        &w->previousHiddenGradient[0][0],model->gradient,JepaTokens,active,model->output,false);
    prepareAdam<<<1,256,0,stream>>>(model->gradient,0,JepaSelfSupervisedParameters,
        model->config,model->state,model->output,&model->groups[0]);
    adamw<<<grid(JepaSelfSupervisedParameters),256,0,stream>>>(reinterpret_cast<float*>(model->parameters),
        model->gradient,model->first,model->second,0,JepaSelfSupervisedParameters,model->config,&model->groups[0]);
    updateTarget<<<grid(JepaEncoderParameters),256,0,stream>>>(model->target,model->parameters,model->config,&model->groups[0]);
    publishSteps<<<1,1,0,stream>>>(model->output,model->groups);
    check(cudaGetLastError(),"launch JEPA backprop/AdamW/EMA");
}
void vabpnnTrainJepaMotion(VabpnnJepaModel* model, cudaStream_t stream) {
    waitReady(model,stream);
    auto* w = model->work;
    const int* active = &model->state->trainMotion;
    prepareMotionTraining<<<1,1,0,stream>>>(w,model->state,model->output,model->config,model->motionTarget,model->groups);
    hiddenGradient<<<1,256,0,stream>>>(&model->parameters->motion,w->motionHidden,w->motionGradient,
        w->motionHiddenGradient,active,model->output,false);
    parameterGradient<JepaMotionInput,JepaHidden,2><<<grid(JepaParameterCount-JepaSelfSupervisedParameters),256,0,stream>>>(
        w->motionInput,w->motionHidden,w->motionGradient,w->motionHiddenGradient,
        model->gradient+JepaSelfSupervisedParameters,1,active,model->output,false);
    prepareAdam<<<1,256,0,stream>>>(model->gradient,JepaSelfSupervisedParameters,
        JepaParameterCount-JepaSelfSupervisedParameters,model->config,model->state,model->output,&model->groups[1]);
    adamw<<<grid(JepaParameterCount-JepaSelfSupervisedParameters),256,0,stream>>>(
        reinterpret_cast<float*>(model->parameters),model->gradient,model->first,model->second,
        JepaSelfSupervisedParameters,JepaParameterCount-JepaSelfSupervisedParameters,model->config,&model->groups[1]);
    publishSteps<<<1,1,0,stream>>>(model->output,model->groups);
    check(cudaGetLastError(),"launch optional motion-head training");
}
void vabpnnResetJepaHistory(VabpnnJepaModel* model, cudaStream_t stream) {
    waitReady(model,stream);
    resetHistory<<<1,1,0,stream>>>(model->state,model->output,model->motionTarget);
    check(cudaGetLastError(),"reset JEPA history");
    check(cudaMemsetAsync(model->work->historyFeatures,0,sizeof(model->work->historyFeatures),stream),"clear JEPA cached features");
}


std::size_t vabpnnJepaCheckpointParts(VabpnnJepaModel* m, vabpnn::ModelPart* p) {
    p[0] = {m->parameters, sizeof(vabpnn::JepaParameters)};
    p[1] = {m->target, sizeof(vabpnn::JepaEncoder)};
    p[2] = {m->first, sizeof(vabpnn::JepaParameters)};
    p[3] = {m->second, sizeof(vabpnn::JepaParameters)};
    p[4] = {m->groups, 2*sizeof(AdamGroup)};
    return 5;
}

namespace {
__global__ void resumeJepaGroups(AdamGroup* groups) {
    for (int i=0;i<2;++i) { groups[i].lastSerial=0; groups[i].active=0; }
}
}
void vabpnnResumeJepaOptimizer(VabpnnJepaModel* m,cudaStream_t stream) {
    vabpnnResetJepaHistory(m,stream);
    resumeJepaGroups<<<1,1,0,stream>>>(m->groups);
    check(cudaGetLastError(),"resume JEPA optimizer counters without stale frame serials");
}
