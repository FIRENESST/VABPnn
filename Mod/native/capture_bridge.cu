#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#include <jni.h>
#include <cuda_gl_interop.h>
#include "frame_operator.h"
#include "TemporalDecision.h"
#include "JepaMotion.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "NetworkScope.h"
#include "model_checkpoint.h"
#include <cstring>
#include <cmath>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using GlSync = void*;
using ClientWaitSync = GLenum(APIENTRY*)(GlSync, GLbitfield, unsigned long long);
using DeleteSync = void(APIENTRY*)(GlSync);
using CreateContext = HGLRC(WINAPI*)(HDC, HGLRC, const int*);
constexpr GLenum ALREADY_SIGNALED = 0x911A, CONDITION_SATISFIED = 0x911C, SYNC_WAIT_FAILED = 0x911D;
constexpr int INIT = 0, READY = 1, RETIRING = 2, RETIRED = 3;
constexpr int TEX_FREE = 0, TEX_WRITING = 1, TEX_PENDING = 2, TEX_COPYING = 3;
constexpr int FRAME_FREE = 0, FRAME_COPYING = 1, FRAME_READY = 2, FRAME_PROCESSING = 3;

void checkCuda(cudaError_t code, const char* operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
bool done(cudaEvent_t event) {
    cudaError_t code = cudaEventQuery(event);
    if (code == cudaErrorNotReady) return false;
    checkCuda(code, "cudaEventQuery");
    return true;
}
template<class T> T glProc(const char* name) {
    PROC value = wglGetProcAddress(name);
    if (!value || reinterpret_cast<std::uintptr_t>(value) <= 3
            || reinterpret_cast<std::uintptr_t>(value) == static_cast<std::uintptr_t>(-1))
        throw std::runtime_error(std::string("Missing GL function: ") + name);
    return reinterpret_cast<T>(value);
}
struct TextureSlot {
    GLuint texture = 0;
    std::atomic<int> state{TEX_FREE};
    GlSync fence = nullptr;
    std::uint64_t frameId = 0, captureNanos = 0;
    cudaGraphicsResource_t resource = nullptr;
    bool mapped = false;
    vabpnn::DemoSample demo;
    vabpnn::PlayerPositionSample position;
};
struct FrameSlot {
    int state = FRAME_FREE, textureIndex = -1;
    bool hasScope=false;
    vabpnn::DemoSample demo;
    vabpnn::PlayerPositionSample position;
    unsigned char* rgba = nullptr;
    std::size_t pitch = 0;
    float* rgbNchw = nullptr;
    cudaEvent_t copied = nullptr, processed = nullptr, started = nullptr;
    std::uint64_t frameId = 0, captureNanos = 0;
};
struct Pool {
    int width, height, capacity;
    std::atomic<int> phase{INIT};
    std::atomic<bool> retire{false};
    std::mutex publishMutex;
    std::array<TextureSlot, 5> textures;
    std::array<FrameSlot, 5> frames;
    cudaStream_t copyStream = nullptr, operatorStream = nullptr;
    int processing = -1;
    Pool(int w, int h, int n) : width(w), height(h), capacity(n) {}
};
class Pipeline {
public:
    const int capacity;
    std::atomic<bool> stop{false}, failed{false};
    std::atomic<std::uint64_t> submitted{0}, copied{0}, processed{0}, skipped{0}, dropped{0},
        lastProcessed{0}, memoryBytes{0}, registrations{0};
    std::shared_ptr<Pool> pool;
    VabpnnNeuronModel* neuronModel = nullptr; // CUDA worker ownership; survives frame-pool resize.
    VabpnnDecisionModel* decisionModel = nullptr;
    VabpnnJepaModel* jepaModel = nullptr;
    VabpnnOnlineModel* onlineModel = nullptr;
    VabpnnSpikeModel* spikeModel = nullptr;
    std::atomic<std::uint64_t> lastGpuUs{0}, maximumGpuUs{0}, budgetOverruns{0};
    vabpnn::SpikeCounters* pendingSpike = nullptr;
    vabpnn::SpikeCounters latestSpike{};
    vabpnn::BlockCounters* pendingBlocks=nullptr;
    vabpnn::BlockCounters latestBlocks{};
    std::uint64_t latestBlockFrame=0;
    int requestedBlocks=-1,requestedCommunication=-1;
    std::atomic<bool> blocksChanged{false};
    std::atomic<bool> scopeEnabled{false};
    std::atomic<int> scopeTile{32*120+60};
    vabpnn::ScopeTelemetry *scopeDevice=nullptr,*scopePending=nullptr;
    vabpnn::ScopeTelemetry scopeLatest{};
    std::array<jlong,24> scopeMeta{};
    std::mutex scopeMutex;
    std::chrono::steady_clock::time_point nextScope{};
    vabpnn::DemoSample latestDemo;
    vabpnn::PlayerPositionSample latestPosition;
    std::mutex demoMutex, checkpointMutex;
    vabpnn::OnlineOutput* pendingOnline = nullptr;
    vabpnn::JepaOutput* pendingJepa = nullptr;
    vabpnn::OnlineOutput latestOnline{};
    vabpnn::JepaOutput latestJepa{};
    std::wstring checkpointPath;
    std::string checkpointMessage = "No checkpoint configured";
    std::atomic<int> checkpointStatus{0};
    std::atomic<bool> saveRequested{false};
    std::atomic<std::uint64_t> saveRequestSerial{0};
    std::uint64_t saveFinishedSerial=0; // Published with checkpoint status under checkpointMutex.
    bool checkpointOpened = false;
    std::uint64_t lastSavedExamples = 0, lastSavedSelfSteps = 0;
    std::chrono::steady_clock::time_point nextSave = std::chrono::steady_clock::now();
    vabpnn::ControlCommand* pendingControl = nullptr; // One pinned host command, worker-owned.
    vabpnn::ControlCommand latestControl{};
    vabpnn::DecisionStatus* pendingDecision=nullptr;
    vabpnn::DecisionStatus latestDecision{};
    std::mutex controlMutex;
    std::atomic<bool> resetTemporal{false};
    std::atomic<std::uint64_t> minimumControlFrame{0};
    std::atomic<std::uint64_t> modelBytes{0};
    HDC dc = nullptr;
    HGLRC context = nullptr;
    ClientWaitSync clientWait = nullptr;
    DeleteSync deleteSync = nullptr;
    std::thread worker;
    std::mutex sleepMutex, textMutex;
    std::condition_variable wake;
    std::string error, deviceName = "CUDA initializing";

    explicit Pipeline(int count) : capacity(count) {
        if (count < 3 || count > 5) throw std::runtime_error("Pool capacity must be 3..5");
        HGLRC gameContext = wglGetCurrentContext();
        dc = wglGetCurrentDC();
        if (!gameContext || !dc) throw std::runtime_error("No current game OpenGL context");
        const char* vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
        if (!vendor || std::string(vendor).find("NVIDIA") == std::string::npos)
            throw std::runtime_error("Game OpenGL must use NVIDIA GPU; select high-performance GPU for this Java executable");
        clientWait = glProc<ClientWaitSync>("glClientWaitSync");
        deleteSync = glProc<DeleteSync>("glDeleteSync");
        CreateContext createContext = glProc<CreateContext>("wglCreateContextAttribsARB");
        const int attributes[] = {0x2091, 3, 0x2092, 3, 0x9126, 1, 0};
        context = createContext(dc, gameContext, attributes);
        if (!context) throw std::runtime_error("Could not create shared WGL 3.3 context");
        try { worker = std::thread(&Pipeline::run, this); }
        catch (...) { wglDeleteContext(context); context = nullptr; throw; }
    }
    ~Pipeline() {
        stop.store(true);
        wake.notify_one();
        if (worker.joinable()) worker.join();
        if (context) wglDeleteContext(context);
    }
    void fail(const std::string& message) {
        { std::lock_guard<std::mutex> lock(textMutex); error = message; }
        failed.store(true);
    }
    void configure(JNIEnv* env, jintArray ids, int width, int height) {
        auto previous = std::atomic_load(&pool);
        if (previous && previous->phase.load() != RETIRED)
            throw std::runtime_error("Previous pool is still owned by CUDA");
        if (width <= 0 || height <= 0 || width > 16384 || height > 16384
                || env->GetArrayLength(ids) != capacity)
            throw std::runtime_error("Invalid texture pool");
        auto next = std::make_shared<Pool>(width, height, capacity);
        jint values[5]{};
        env->GetIntArrayRegion(ids, 0, capacity, values);
        if (env->ExceptionCheck()) return;
        for (int i = 0; i < capacity; ++i) {
            if (values[i] <= 0) throw std::runtime_error("Invalid GL texture id");
            next->textures[i].texture = static_cast<GLuint>(values[i]);
        }
        // Producer GL texture creation must have been flushed before publishing the pool.
        std::atomic_store(&pool, next);
        wake.notify_one();
    }
    int acquire() {
        if (failed.load()) return -1;
        auto current = std::atomic_load(&pool);
        if (!current || current->phase.load() != READY || current->retire.load()) return -1;
        for (int i = 0; i < capacity; ++i) {
            int expected = TEX_FREE;
            if (current->textures[i].state.compare_exchange_strong(expected, TEX_WRITING)) return i;
        }
        ++skipped;
        return -1;
    }
    bool submit(int index, GlSync fence, std::uint64_t id, std::uint64_t nanos) {
        auto current = std::atomic_load(&pool);
        if (!current || failed.load() || current->phase.load() != READY
                || current->retire.load() || index < 0 || index >= capacity || !fence) return false;
        std::unique_lock<std::mutex> publication(current->publishMutex, std::try_to_lock);
        if (!publication.owns_lock() || current->phase.load() != READY || current->retire.load()) return false;
        auto& texture = current->textures[index];
        if (texture.state.load() != TEX_WRITING) return false;
        texture.fence = fence;
        texture.frameId = id;
        texture.captureNanos = nanos;
        { std::lock_guard<std::mutex> lock(demoMutex); texture.demo = latestDemo; texture.position=latestPosition; }
        texture.state.store(TEX_PENDING, std::memory_order_release);
        ++submitted;
        wake.notify_one();
        return true;
    }
    void cancel(int index) {
        auto current = std::atomic_load(&pool);
        if (current && index >= 0 && index < capacity) {
            int expected = TEX_WRITING;
            current->textures[index].state.compare_exchange_strong(expected, TEX_FREE);
        }
    }
    void retire() {
        auto current = std::atomic_load(&pool);
        if (current) current->retire.store(true);
        wake.notify_one();
    }
    int phase() {
        if (failed.load()) return -1;
        auto current = std::atomic_load(&pool);
        return current ? current->phase.load() : RETIRED;
    }
private:
    std::vector<vabpnn::ModelPart> checkpointParts(bool connections=true,bool position=true,bool blocks=true) {
        std::vector<vabpnn::ModelPart> parts;
        parts.push_back({vabpnnMutableNeuronWeights(neuronModel,nullptr),vabpnn::ParameterCount});
        parts.push_back({vabpnnMutableDecisionParameters(decisionModel,nullptr),sizeof(vabpnn::DecisionParameters)});
        vabpnn::ModelPart extra[8];
        auto count = vabpnnJepaCheckpointParts(jepaModel,extra);
        parts.insert(parts.end(),extra,extra+count);
        count = vabpnnOnlineCheckpointParts(onlineModel,extra);
        parts.insert(parts.end(),extra,extra+count);
        if(connections) {
            count=vabpnnSpikeCheckpointParts(spikeModel,extra);
            parts.insert(parts.end(),extra,extra+count);
        }
        if(position) {
            count=vabpnnDecisionPositionCheckpointParts(decisionModel,extra);
            parts.insert(parts.end(),extra,extra+count);
        }
        if(blocks) parts.push_back(vabpnnBlockCheckpointPart(vabpnnSpikeBlockModel(spikeModel)));
        return parts;
    }
    void serviceBlockMode(cudaStream_t stream) {
        if(!blocksChanged.load()) return;
        if(!checkpointOpened) {
            std::lock_guard<std::mutex> lock(checkpointMutex);
            if(!checkpointPath.empty()) return;
        }
        int enabled,mode;
        { std::lock_guard<std::mutex> lock(controlMutex);
            enabled=requestedBlocks; mode=requestedCommunication;
            requestedBlocks=requestedCommunication=-1; blocksChanged.store(false); }
        vabpnnConfigureBlockMode(vabpnnSpikeBlockModel(spikeModel),enabled,mode,stream);
    }
    void serviceCheckpoint() {
        // The path is immutable after configuration. Most worker iterations have
        // no checkpoint work, so avoid path allocation and metadata locks.
        if(checkpointOpened && !saveRequested.load()
                && std::chrono::steady_clock::now()<nextSave) return;
        std::wstring path;
        { std::lock_guard<std::mutex> lock(checkpointMutex); path = checkpointPath; }
        if (path.empty()) return;
        auto p = std::atomic_load(&pool);
        if (p && p->processing >= 0) return;
        std::uint64_t saveSerial=0;
        try {
            if (!checkpointOpened) {
                // No operator runs here. Initialized model events are drained once on the worker.
                checkCuda(cudaDeviceSynchronize(),"wait before checkpoint resume");
                int migrated=0;
                bool loaded=vabpnn::loadModelCheckpoint(path,checkpointParts(),false,4);
                if(!loaded) {
                    const auto previous=(std::filesystem::path(path).parent_path()/L"online-v3.bin").wstring();
                    loaded=vabpnn::loadModelCheckpoint(previous,checkpointParts(true,true,false),false,3);
                    if(loaded) migrated=3;
                }
                if(!loaded) {
                    const auto previous=(std::filesystem::path(path).parent_path()/L"online-v2.bin").wstring();
                    loaded=vabpnn::loadModelCheckpoint(previous,checkpointParts(true,false,false));
                    if(loaded) migrated=2;
                }
                if(!loaded) {
                    const auto legacy=(std::filesystem::path(path).parent_path()/L"online-v1.bin").wstring();
                    loaded=vabpnn::loadModelCheckpoint(legacy,checkpointParts(false,false,false),true);
                    if(loaded) migrated=1;
                }
                if(loaded) {
                    vabpnnResumeJepaOptimizer(jepaModel,nullptr);
                    vabpnnResumeSpikeModel(spikeModel,nullptr);
                }
                checkpointOpened = true; checkpointStatus.store(loaded ? 2 : 1);
                std::lock_guard<std::mutex> lock(checkpointMutex);
                checkpointMessage = migrated==3 ? "V3 restored read-only; neutral preview controllers initialized"
                    : migrated==2 ? "V2 model, optimizer and connections restored; neutral position adapter and preview controllers initialized"
                    : migrated==1 ? "V1 model and optimizer restored; new connections and neutral position adapter initialized"
                    : loaded ? "Model, optimizer, connections and position adapter restored" : "Fresh model; human demonstration mode";
                nextSave = std::chrono::steady_clock::now()+std::chrono::seconds(120);
            }
            const auto now = std::chrono::steady_clock::now();
            bool changed = false;
            { std::lock_guard<std::mutex> lock(controlMutex);
                changed = latestOnline.examples!=lastSavedExamples || latestJepa.selfSupervisedSteps!=lastSavedSelfSteps; }
            const bool requested = saveRequested.exchange(false);
            if (requested || (now>=nextSave && changed && checkpointStatus.load()>0)) {
                saveSerial=saveRequestSerial.load();
                checkCuda(cudaDeviceSynchronize(),"wait before model checkpoint");
                vabpnn::saveModelCheckpoint(path,checkpointParts(),4);
                { std::lock_guard<std::mutex> lock(controlMutex);
                    lastSavedExamples=latestOnline.examples; lastSavedSelfSteps=latestJepa.selfSupervisedSteps; }
                { std::lock_guard<std::mutex> lock(checkpointMutex);
                    checkpointStatus.store(3); checkpointMessage="Model and optimizer saved";
                    saveFinishedSerial=saveSerial; }
                nextSave=now+std::chrono::seconds(120);
            }
        } catch (const std::exception& e) {
            checkpointOpened=true;
            std::lock_guard<std::mutex> lock(checkpointMutex);
            checkpointStatus.store(-1); checkpointMessage=e.what();
            if(saveSerial) saveFinishedSerial=saveSerial;
        }
    }
    void initializePool(Pool& p) {
        checkCuda(cudaStreamCreateWithFlags(&p.copyStream, cudaStreamNonBlocking), "create copy stream");
        checkCuda(cudaStreamCreateWithFlags(&p.operatorStream, cudaStreamNonBlocking), "create operator stream");
        std::uint64_t total = static_cast<std::uint64_t>(p.width) * p.height * 4 * capacity;
        for (int i = 0; i < capacity; ++i) {
            auto& t = p.textures[i];
            // Rebinding observes shared texture creation after the producer's flush.
            glBindTexture(GL_TEXTURE_2D, t.texture);
            checkCuda(cudaGraphicsGLRegisterImage(&t.resource, t.texture, GL_TEXTURE_2D,
                       cudaGraphicsRegisterFlagsReadOnly), "cudaGraphicsGLRegisterImage");
            ++registrations;
            auto& f = p.frames[i];
            checkCuda(cudaMallocPitch(reinterpret_cast<void**>(&f.rgba), &f.pitch,
                       static_cast<std::size_t>(p.width) * 4, p.height), "allocate pitched RGBA");
            std::size_t tensorBytes = static_cast<std::size_t>(p.width) * p.height * 3 * sizeof(float);
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&f.rgbNchw), tensorBytes), "allocate RGB tensor");
            checkCuda(cudaEventCreateWithFlags(&f.copied, cudaEventDisableTiming), "create copy event");
            checkCuda(cudaEventCreate(&f.processed), "create operator event");
            checkCuda(cudaEventCreate(&f.started), "create GPU budget event");
            total += f.pitch * p.height + tensorBytes;
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        memoryBytes.store(total);
        p.phase.store(READY);
    }
    void pollCompletions(Pool& p) {
        for (int i = 0; i < capacity; ++i) {
            auto& f = p.frames[i];
            if (f.state == FRAME_COPYING && done(f.copied)) {
                p.textures[f.textureIndex].state.store(TEX_FREE, std::memory_order_release);
                f.textureIndex = -1;
                f.state = FRAME_READY;
                ++copied;
            }
        }
        if (p.processing >= 0) {
            auto& f = p.frames[p.processing];
            if (done(f.processed)) {
                { std::lock_guard<std::mutex> lock(controlMutex);
                    if (!resetTemporal.load() && f.frameId >= minimumControlFrame.load()) {
                        latestControl = *pendingControl; latestDecision = *pendingDecision;
                    }
                    latestOnline = *pendingOnline; latestJepa = *pendingJepa; latestSpike = *pendingSpike;
                    latestBlocks=*pendingBlocks; latestBlockFrame=f.frameId;
                }
                float milliseconds=0;
                checkCuda(cudaEventElapsedTime(&milliseconds,f.started,f.processed),"measure GPU operator budget");
                const auto us=static_cast<std::uint64_t>(milliseconds*1000);
                lastGpuUs.store(us); maximumGpuUs.store(std::max(maximumGpuUs.load(),us));
                if(us>10000) ++budgetOverruns;
                if(f.hasScope && scopeEnabled.load() && !resetTemporal.load() && f.frameId>=minimumControlFrame.load()) {
                    const std::array<jlong,24> metadata={
                        static_cast<jlong>(f.frameId),static_cast<jlong>(f.captureNanos),static_cast<jlong>(us),
                        static_cast<jlong>(maximumGpuUs.load()),static_cast<jlong>(budgetOverruns.load()),
                        static_cast<jlong>(submitted.load()),static_cast<jlong>(copied.load()),static_cast<jlong>(processed.load()+1),
                        static_cast<jlong>(dropped.load()),static_cast<jlong>(skipped.load()),
                        static_cast<jlong>(pendingSpike->localEdges),pendingSpike->anchors,pendingSpike->longEdges,
                        static_cast<jlong>(pendingSpike->topologyUpdates),static_cast<jlong>(pendingSpike->contributionUpdates),
                        static_cast<jlong>(pendingOnline->examples),static_cast<jlong>(pendingOnline->pixelSteps),
                        static_cast<jlong>(pendingOnline->decisionSteps),static_cast<jlong>(pendingJepa->selfSupervisedSteps),
                        static_cast<jlong>(pendingJepa->motionSteps),static_cast<jlong>(modelBytes.load()),
                        static_cast<jlong>(memoryBytes.load()),checkpointStatus.load(),static_cast<jlong>(pendingOnline->rejected)};
                    std::lock_guard<std::mutex> lock(scopeMutex);
                    scopeLatest=*scopePending; scopeMeta=metadata;
                }
                lastProcessed.store(f.frameId);
                ++processed;
                f.state = FRAME_FREE;
                p.processing = -1;
            }
        }
    }
    void consumeLatest(Pool& p) {
        if (p.processing >= 0) return;
        serviceBlockMode(p.operatorStream);
        if (resetTemporal.exchange(false)) {
            vabpnnResetDecisionState(decisionModel, p.operatorStream);
            vabpnnResetJepaHistory(jepaModel, p.operatorStream);
            vabpnnResetOnlineHistory(onlineModel, p.operatorStream);
            vabpnnResetSpikeHistory(spikeModel, p.operatorStream);
        }
        for (int i = 0; i < capacity; ++i)
            if (p.frames[i].state == FRAME_READY && p.frames[i].frameId < minimumControlFrame.load()) {
                p.frames[i].state = FRAME_FREE; ++dropped;
            }
        int latest = -1;
        for (int i = 0; i < capacity; ++i)
            if (p.frames[i].state == FRAME_READY && (latest < 0
                    || p.frames[i].frameId > p.frames[latest].frameId)) latest = i;
        if (latest < 0) return;
        for (int i = 0; i < capacity; ++i)
            if (i != latest && p.frames[i].state == FRAME_READY) {
                p.frames[i].state = FRAME_FREE;
                ++dropped;
            }
        auto& f = p.frames[latest];
        VabpnnGpuFrame frame{f.rgba, f.pitch, f.rgbNchw, p.width, p.height,
                            f.frameId, f.captureNanos, p.operatorStream, neuronModel, decisionModel, jepaModel, onlineModel, spikeModel, f.position};
        f.state = FRAME_PROCESSING;
        p.processing = latest;
        checkCuda(cudaEventRecord(f.started,p.operatorStream),"start GPU frame budget");
        vabpnnSetDemoSample(onlineModel, &f.demo, p.operatorStream);
        vabpnnLaunchOperator(frame);
        checkCuda(cudaGetLastError(), "launch frame operator");
        const auto decision = vabpnnDecisionModelView(decisionModel);
        checkCuda(cudaMemcpyAsync(pendingControl, decision.command, sizeof(vabpnn::ControlCommand),
                                 cudaMemcpyDeviceToHost, p.operatorStream), "copy final control command");
        checkCuda(cudaMemcpyAsync(pendingDecision, &decision.output->status, sizeof(vabpnn::DecisionStatus),
                                 cudaMemcpyDeviceToHost, p.operatorStream), "copy decision counters");
        checkCuda(cudaMemcpyAsync(pendingOnline, vabpnnOnlineOutput(onlineModel), sizeof(vabpnn::OnlineOutput),
                                 cudaMemcpyDeviceToHost, p.operatorStream), "copy training counters");
        checkCuda(cudaMemcpyAsync(pendingJepa, vabpnnJepaModelView(jepaModel).output, sizeof(vabpnn::JepaOutput),
                                 cudaMemcpyDeviceToHost, p.operatorStream), "copy JEPA scalar training metrics");
        checkCuda(cudaMemcpyAsync(pendingSpike,vabpnnSpikeModelView(spikeModel).counters,sizeof(vabpnn::SpikeCounters),
                                 cudaMemcpyDeviceToHost,p.operatorStream),"copy connection counters");
        checkCuda(cudaMemcpyAsync(pendingBlocks,vabpnnBlockModelView(vabpnnSpikeBlockModel(spikeModel)).counters,
                                 sizeof(vabpnn::BlockCounters),cudaMemcpyDeviceToHost,p.operatorStream),"copy bounded block counters");
        f.hasScope=false;
        const auto scopeNow=std::chrono::steady_clock::now();
        if(scopeEnabled.load() && scopeNow>=nextScope) {
            vabpnnLaunchNetworkScope(frame,scopeTile.load(),scopeDevice);
            checkCuda(cudaMemcpyAsync(scopePending,scopeDevice,sizeof(vabpnn::ScopeTelemetry),
                                     cudaMemcpyDeviceToHost,p.operatorStream),"copy sampled network state");
            f.hasScope=true; nextScope=scopeNow+std::chrono::milliseconds(100);
        }
        checkCuda(cudaEventRecord(f.processed, p.operatorStream), "record operator completion");
    }
    int availableFrame(Pool& p) {
        for (int i = 0; i < capacity; ++i) if (p.frames[i].state == FRAME_FREE) return i;
        // READY guarantees the copy event completed. Never recycle COPY/PROCESSING buffers.
        int oldest = -1;
        for (int i = 0; i < capacity; ++i)
            if (p.frames[i].state == FRAME_READY && (oldest < 0
                    || p.frames[i].frameId < p.frames[oldest].frameId)) oldest = i;
        if (oldest >= 0) { p.frames[oldest].state = FRAME_FREE; ++dropped; }
        return oldest;
    }
    bool fenceReady(TextureSlot& t) {
        GLenum result = clientWait(t.fence, 0, 0);
        if (result == SYNC_WAIT_FAILED) throw std::runtime_error("glClientWaitSync failed");
        return result == ALREADY_SIGNALED || result == CONDITION_SATISFIED;
    }
    void copyLatest(Pool& p) {
        int latest = -1;
        bool ready[5]{};
        for (int i = 0; i < capacity; ++i) {
            auto& t = p.textures[i];
            if (t.state.load(std::memory_order_acquire) == TEX_PENDING && fenceReady(t)) {
                ready[i] = true;
                if (latest < 0 || t.frameId > p.textures[latest].frameId) latest = i;
            }
        }
        if (latest < 0) return;
        for (int i = 0; i < capacity; ++i) {
            if (ready[i] && i != latest) {
                auto& t = p.textures[i];
                deleteSync(t.fence); t.fence = nullptr;
                t.state.store(TEX_FREE, std::memory_order_release);
                ++dropped;
            }
        }
        int frameIndex = availableFrame(p);
        if (frameIndex < 0) return; // Keep the newest texture while active GPU work finishes.
        auto& t = p.textures[latest];
        auto& f = p.frames[frameIndex];
        deleteSync(t.fence); t.fence = nullptr;
        t.state.store(TEX_COPYING);
        glBindTexture(GL_TEXTURE_2D, t.texture);
        checkCuda(cudaGraphicsMapResources(1, &t.resource, p.copyStream), "map GL texture");
        t.mapped = true;
        cudaArray_t array = nullptr;
        checkCuda(cudaGraphicsSubResourceGetMappedArray(&array, t.resource, 0, 0), "get cudaArray");
        checkCuda(cudaMemcpy2DFromArrayAsync(f.rgba, f.pitch, array, 0, 0,
                   static_cast<std::size_t>(p.width) * 4, p.height,
                   cudaMemcpyDeviceToDevice, p.copyStream), "copy texture to device frame");
        checkCuda(cudaGraphicsUnmapResources(1, &t.resource, p.copyStream), "unmap GL texture");
        t.mapped = false;
        f.textureIndex = latest;
        f.frameId = t.frameId;
        f.captureNanos = t.captureNanos;
        f.demo = t.demo;
        f.position = t.position;
        f.state = FRAME_COPYING;
        checkCuda(cudaEventRecord(f.copied, p.copyStream), "record copy completion");
    }
    void cleanup(Pool& p) {
        {
            std::lock_guard<std::mutex> publication(p.publishMutex);
            p.phase.store(RETIRING);
        }
        // Only the worker or shutdown waits. The producer skips frames while retiring.
        if (p.copyStream) cudaStreamSynchronize(p.copyStream);
        if (p.operatorStream) cudaStreamSynchronize(p.operatorStream);
        p.processing = -1;
        for (int i = 0; i < capacity; ++i) {
            auto& t = p.textures[i];
            if (t.fence) {
                while (true) {
                    GLenum result = clientWait(t.fence, 0, 1000000);
                    if (result == ALREADY_SIGNALED || result == CONDITION_SATISFIED || result == SYNC_WAIT_FAILED) break;
                }
                deleteSync(t.fence); t.fence = nullptr;
            }
            if (t.mapped && t.resource) {
                cudaGraphicsUnmapResources(1, &t.resource, p.copyStream);
                if (p.copyStream) cudaStreamSynchronize(p.copyStream);
                t.mapped = false;
            }
            if (t.resource) { cudaGraphicsUnregisterResource(t.resource); t.resource = nullptr; }
            auto& f = p.frames[i];
            if (f.copied) { cudaEventDestroy(f.copied); f.copied = nullptr; }
            if (f.processed) { cudaEventDestroy(f.processed); f.processed = nullptr; }
            if (f.started) { cudaEventDestroy(f.started); f.started = nullptr; }
            if (f.rgba) { cudaFree(f.rgba); f.rgba = nullptr; }
            if (f.rgbNchw) { cudaFree(f.rgbNchw); f.rgbNchw = nullptr; }
        }
        if (p.copyStream) { cudaStreamDestroy(p.copyStream); p.copyStream = nullptr; }
        if (p.operatorStream) { cudaStreamDestroy(p.operatorStream); p.operatorStream = nullptr; }
        memoryBytes.store(0);
        p.phase.store(RETIRED, std::memory_order_release);
    }
    void run() {
        bool current = false;
        try {
            if (!wglMakeCurrent(dc, context)) throw std::runtime_error("Worker could not make shared GL context current");
            current = true;
            unsigned int count = 0;
            int devices[8]{};
            checkCuda(cudaGLGetDevices(&count, devices, 8, cudaGLDeviceListAll), "cudaGLGetDevices");
            if (count == 0) throw std::runtime_error("Game GL context has no compatible CUDA device");
            checkCuda(cudaSetDevice(devices[0]), "select matching CUDA device");
            cudaDeviceProp properties{};
            checkCuda(cudaGetDeviceProperties(&properties, devices[0]), "query CUDA device");
            { std::lock_guard<std::mutex> lock(textMutex); deviceName = properties.name; }
            neuronModel = vabpnnCreateNeuronModel(nullptr);
            decisionModel = vabpnnCreateDecisionModel(nullptr);
            jepaModel = vabpnnCreateJepaModel(nullptr);
            onlineModel = vabpnnCreateOnlineModel(nullptr);
            spikeModel = vabpnnCreateSpikeModel(nullptr);
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&scopeDevice),sizeof(vabpnn::ScopeTelemetry)),"allocate network telemetry");
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&scopePending),sizeof(vabpnn::ScopeTelemetry)),"allocate pinned network telemetry");
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&pendingSpike),sizeof(vabpnn::SpikeCounters)),"allocate connection counters");
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&pendingBlocks),sizeof(vabpnn::BlockCounters)),"allocate pinned block counters");
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&pendingOnline), sizeof(vabpnn::OnlineOutput)),
                      "allocate pinned training counters");
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&pendingJepa), sizeof(vabpnn::JepaOutput)),
                      "allocate pinned JEPA scalar metrics");
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&pendingControl), sizeof(vabpnn::ControlCommand)),
                      "allocate pinned final control command");
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&pendingDecision), sizeof(vabpnn::DecisionStatus)),
                      "allocate pinned decision counters");
            std::memset(pendingDecision, 0, sizeof(vabpnn::DecisionStatus));
            std::memset(pendingControl, 0, sizeof(vabpnn::ControlCommand));
            modelBytes.store(vabpnnNeuronModelBytes(neuronModel) + vabpnnDecisionModelBytes(decisionModel)
                + vabpnnJepaModelBytes(jepaModel) + vabpnnOnlineModelBytes(onlineModel) + vabpnnSpikeModelBytes(spikeModel) + sizeof(vabpnn::ScopeTelemetry));
            while (!stop.load()) {
                serviceCheckpoint();
                auto p = std::atomic_load(&pool);
                if (p && p->phase.load() != RETIRED) {
                    if (p->retire.load()) {
                        cleanup(*p);
                    } else {
                        if (p->phase.load() == INIT) initializePool(*p);
                        if (!p->retire.load()) {
                            pollCompletions(*p);
                            serviceCheckpoint();
                            consumeLatest(*p);
                            copyLatest(*p);
                        }
                    }
                }
                std::unique_lock<std::mutex> lock(sleepMutex);
                wake.wait_for(lock, std::chrono::milliseconds(1));
            }
        } catch (const std::exception& e) { fail(e.what()); }
        auto p = std::atomic_load(&pool);
        if (current && p && p->phase.load() != RETIRED) cleanup(*p);
        if (current && !failed.load() && neuronModel && onlineModel) {
            saveRequested.store(true); serviceCheckpoint();
        }
        if(scopeDevice) { cudaFree(scopeDevice); scopeDevice=nullptr; }
        if(scopePending) { cudaFreeHost(scopePending); scopePending=nullptr; }
        if(pendingSpike) { cudaFreeHost(pendingSpike); pendingSpike=nullptr; }
        if(pendingBlocks) { cudaFreeHost(pendingBlocks); pendingBlocks=nullptr; }
        vabpnnDestroySpikeModel(spikeModel); spikeModel=nullptr;
        if (pendingOnline) { cudaFreeHost(pendingOnline); pendingOnline = nullptr; }
        if (pendingJepa) { cudaFreeHost(pendingJepa); pendingJepa = nullptr; }
        vabpnnDestroyOnlineModel(onlineModel); onlineModel = nullptr;
        if (pendingDecision) { cudaFreeHost(pendingDecision); pendingDecision = nullptr; }
        if (pendingControl) { cudaFreeHost(pendingControl); pendingControl = nullptr; }
        vabpnnDestroyJepaModel(jepaModel);
        jepaModel = nullptr;
        vabpnnDestroyDecisionModel(decisionModel);
        decisionModel = nullptr;
        vabpnnDestroyNeuronModel(neuronModel);
        neuronModel = nullptr;
        modelBytes.store(0);
        if (current) { glBindTexture(GL_TEXTURE_2D, 0); wglMakeCurrent(nullptr, nullptr); }
    }
};
Pipeline* from(jlong handle) {
    if (!handle) throw std::runtime_error("CUDA pipeline is closed");
    return reinterpret_cast<Pipeline*>(handle);
}
void throwJava(JNIEnv* env, const std::exception& error) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nStart(JNIEnv* env, jclass, jint capacity) {
    try { return reinterpret_cast<jlong>(new Pipeline(capacity)); }
    catch (const std::exception& e) { throwJava(env, e); return 0; }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nConfigure(JNIEnv* env, jclass, jlong handle,
                                                             jintArray textures, jint width, jint height) {
    try { from(handle)->configure(env, textures, width, height); }
    catch (const std::exception& e) { throwJava(env, e); }
}
extern "C" JNIEXPORT jint JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nAcquire(JNIEnv* env, jclass, jlong handle) {
    try { return from(handle)->acquire(); }
    catch (const std::exception& e) { throwJava(env, e); return -1; }
}
extern "C" JNIEXPORT jboolean JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nSubmit(JNIEnv* env, jclass, jlong handle, jint index,
                                                        jlong fence, jlong frameId, jlong nanos) {
    try { return from(handle)->submit(index, reinterpret_cast<GlSync>(fence), frameId, nanos) ? JNI_TRUE : JNI_FALSE; }
    catch (const std::exception& e) { throwJava(env, e); return JNI_FALSE; }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nCancel(JNIEnv* env, jclass, jlong handle, jint index) {
    try { from(handle)->cancel(index); } catch (const std::exception& e) { throwJava(env, e); }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nRetire(JNIEnv* env, jclass, jlong handle) {
    try { from(handle)->retire(); } catch (const std::exception& e) { throwJava(env, e); }
}
extern "C" JNIEXPORT jint JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nPhase(JNIEnv* env, jclass, jlong handle) {
    try { return from(handle)->phase(); } catch (const std::exception& e) { throwJava(env, e); return -1; }
}
extern "C" JNIEXPORT jlongArray JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nStats(JNIEnv* env, jclass, jlong handle) {
    try {
        auto* p = from(handle);
        jlong values[] = {static_cast<jlong>(p->submitted.load()), static_cast<jlong>(p->copied.load()),
            static_cast<jlong>(p->processed.load()), static_cast<jlong>(p->skipped.load()),
            static_cast<jlong>(p->dropped.load()), static_cast<jlong>(p->lastProcessed.load()),
            static_cast<jlong>(p->memoryBytes.load()), static_cast<jlong>(p->registrations.load()),
            static_cast<jlong>(p->capacity), static_cast<jlong>(p->phase()),
            static_cast<jlong>(p->modelBytes.load())};
        jlongArray result = env->NewLongArray(11);
        if (result) env->SetLongArrayRegion(result, 0, 11, values);
        return result;
    } catch (const std::exception& e) { throwJava(env, e); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nText(JNIEnv* env, jclass, jlong handle, jboolean error) {
    try {
        auto* p = from(handle);
        std::lock_guard<std::mutex> lock(p->textMutex);
        return env->NewStringUTF((error ? p->error : p->deviceName).c_str());
    } catch (const std::exception& e) { throwJava(env, e); return nullptr; }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nClose(JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<Pipeline*>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nControl(JNIEnv* env, jclass, jlong handle, jlongArray target) {
    try {
        if (!target || env->GetArrayLength(target) != 7) throw std::runtime_error("Control buffer must contain seven longs");
        auto* pipeline = from(handle);
        vabpnn::ControlCommand command;
        { std::lock_guard<std::mutex> lock(pipeline->controlMutex); command = pipeline->latestControl; }
        std::uint32_t dx = 0, dy = 0, wheel = 0;
        std::memcpy(&dx, &command.mouseDx, sizeof(dx));
        std::memcpy(&dy, &command.mouseDy, sizeof(dy));
        std::memcpy(&wheel, &command.wheelSteps, sizeof(wheel));
        jlong values[] = {static_cast<jlong>(command.frameId), static_cast<jlong>(command.captureNanos),
            static_cast<jlong>(command.selectedMask), static_cast<jlong>(command.valid),
            static_cast<jlong>(dx), static_cast<jlong>(dy), static_cast<jlong>(wheel)};
        env->SetLongArrayRegion(target, 0, 7, values);
    } catch (const std::exception& e) { throwJava(env, e); }
}

extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nDecision(JNIEnv* env,jclass,jlong handle,jlongArray target) {
    try {
        if(!target || env->GetArrayLength(target)!=7) throw std::runtime_error("Seven-long decision status required");
        auto* p=from(handle);
        vabpnn::DecisionStatus status; vabpnn::OnlineOutput training;
        { std::lock_guard<std::mutex> lock(p->controlMutex); status=p->latestDecision; training=p->latestOnline; }
        jlong values[7]={static_cast<jlong>(status.frameId),status.valid,status.eligibleMask,
            status.requestedK,status.selectedCount,training.lastLabel,training.valid};
        env->SetLongArrayRegion(target,0,7,values);
    } catch(const std::exception& e) { throwJava(env,e); }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nResetDecision(JNIEnv* env, jclass, jlong handle, jlong minimumFrame) {
    try {
        auto* pipeline = from(handle);
        pipeline->minimumControlFrame.store(static_cast<std::uint64_t>(minimumFrame));
        pipeline->resetTemporal.store(true);
        { std::lock_guard<std::mutex> lock(pipeline->controlMutex);
          const auto positionSteps=pipeline->latestDecision.positionSteps;
          pipeline->latestControl = {}; pipeline->latestDecision = {};
          pipeline->latestDecision.positionSteps=positionSteps; }
        pipeline->wake.notify_one();
    } catch (const std::exception& e) { throwJava(env, e); }
}


extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nPosition(JNIEnv* env,jclass,jlong handle,
        jlong nanos,jlong epoch,jdouble x,jdouble y,jdouble z) {
    try {
        if(nanos && (!epoch || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)
            || std::fabs(x)>30'000'000 || std::fabs(y)>30'000'000 || std::fabs(z)>30'000'000))
            throw std::runtime_error("Invalid local-player coordinate metadata");
        const vabpnn::PlayerPositionSample sample{static_cast<std::uint64_t>(nanos),
            static_cast<std::uint64_t>(epoch),x,y,z};
        auto* pipeline=from(handle);
        std::lock_guard<std::mutex> lock(pipeline->demoMutex);
        pipeline->latestPosition=sample;
    } catch(const std::exception& e) { throwJava(env,e); }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nPositionStatus(JNIEnv* env,jclass,jlong handle,jlongArray target) {
    try {
        if(!target || env->GetArrayLength(target)!=7) throw std::runtime_error("Invalid position status buffer");
        auto* pipeline=from(handle); vabpnn::DecisionStatus s;
        { std::lock_guard<std::mutex> lock(pipeline->controlMutex); s=pipeline->latestDecision; }
        jlong xyz[3]; const double point[3]={s.positionX,s.positionY,s.positionZ};
        std::memcpy(xyz,point,sizeof(xyz));
        const jlong data[7]={static_cast<jlong>(s.positionNanos),static_cast<jlong>(s.positionEpoch),
            static_cast<jlong>(s.positionSteps),static_cast<jlong>(s.positionValid),xyz[0],xyz[1],xyz[2]};
        env->SetLongArrayRegion(target,0,7,data);
    } catch(const std::exception& e) { throwJava(env,e); }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nDemo(JNIEnv* env,jclass,jlong handle,jlongArray values,jdoubleArray motion) {
    try {
        if(!values || !motion || env->GetArrayLength(values)!=15 || env->GetArrayLength(motion)!=3)
            throw std::runtime_error("Invalid real-input metadata");
        jlong v[15]; jdouble d[3];
        env->GetLongArrayRegion(values,0,15,v); env->GetDoubleArrayRegion(motion,0,3,d);
        if(env->ExceptionCheck()) return;
        vabpnn::DemoSample sample;
        sample.epoch=v[0]; sample.heldKeys=static_cast<std::uint32_t>(v[1]);
        sample.active=v[2]==1; sample.modelControl=v[3]==1; sample.pointerMode=static_cast<std::uint32_t>(v[14]);
        for(int k=0;k<4;++k) sample.edge[k]=v[4+k];
        for(int k=0;k<6;++k) sample.keyHeldNanos[k]=v[8+k];
        sample.mouseX=d[0]; sample.mouseY=d[1]; sample.wheel=d[2];
        auto* pipeline=from(handle);
        std::lock_guard<std::mutex> lock(pipeline->demoMutex); pipeline->latestDemo=sample;
    } catch(const std::exception& e) { throwJava(env,e); }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nCheckpoint(JNIEnv* env,jclass,jlong handle,jstring path,jboolean save) {
    try {
        auto* p=from(handle);
        if(path) {
            const auto* chars=env->GetStringChars(path,nullptr);
            if(!chars) return;
            std::wstring value(reinterpret_cast<const wchar_t*>(chars),env->GetStringLength(path));
            env->ReleaseStringChars(path,chars);
            std::lock_guard<std::mutex> lock(p->checkpointMutex);
            if(!p->checkpointPath.empty() && p->checkpointPath!=value)
                throw std::runtime_error("Checkpoint path cannot change during a run");
            p->checkpointPath=value;
        }
        if(save) {
            std::lock_guard<std::mutex> lock(p->checkpointMutex);
            p->saveRequestSerial.fetch_add(1);
            p->saveRequested.store(true);
        }
        p->wake.notify_one();
    } catch(const std::exception& e) { throwJava(env,e); }
}
/** CPU counters only. No CUDA work, synchronization, model transfer or filesystem access. */
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nSaveStatus(JNIEnv* env,jclass,jlong handle,jlongArray target) {
    try {
        if(!target || env->GetArrayLength(target)!=3) throw std::runtime_error("Three-long save status required");
        auto* p=from(handle);
        std::lock_guard<std::mutex> lock(p->checkpointMutex);
        const jlong values[3]={static_cast<jlong>(p->saveRequestSerial.load()),
            static_cast<jlong>(p->saveFinishedSerial),p->checkpointStatus.load()};
        env->SetLongArrayRegion(target,0,3,values);
    } catch(const std::exception& e) { throwJava(env,e); }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nTraining(JNIEnv* env,jclass,jlong handle,jlongArray target) {
    try {
        if(!target || env->GetArrayLength(target)!=9) throw std::runtime_error("Nine-long training status required");
        auto* p=from(handle);
        vabpnn::OnlineOutput out; vabpnn::JepaOutput jepa;
        { std::lock_guard<std::mutex> lock(p->controlMutex); out=p->latestOnline; jepa=p->latestJepa; }
        std::uint32_t loss=0,wheel=0;
        std::memcpy(&loss,&out.decisionLoss,4); std::memcpy(&wheel,&out.wheelLoss,4);
        jlong values[9]={static_cast<jlong>(out.examples),static_cast<jlong>(out.rejected),
            static_cast<jlong>(out.pixelSteps),static_cast<jlong>(out.decisionSteps),
            static_cast<jlong>(jepa.selfSupervisedSteps),static_cast<jlong>(jepa.motionSteps),
            static_cast<jlong>(loss),static_cast<jlong>(wheel),p->checkpointStatus.load()};
        env->SetLongArrayRegion(target,0,9,values);
    } catch(const std::exception& e) { throwJava(env,e); }
}
extern "C" JNIEXPORT jstring JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nCheckpointText(JNIEnv* env,jclass,jlong handle) {
    try {
        auto* p=from(handle); std::lock_guard<std::mutex> lock(p->checkpointMutex);
        return env->NewStringUTF(p->checkpointMessage.c_str());
    } catch(const std::exception& e) { throwJava(env,e); return nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nTiming(JNIEnv* env,jclass,jlong handle,jlongArray status) {
    try {
        auto* p=from(handle);
        if(env->GetArrayLength(status)!=8) throw std::runtime_error("Timing buffer must have 8 elements");
        vabpnn::SpikeCounters c;
        { std::lock_guard<std::mutex> lock(p->controlMutex); c=p->latestSpike; }
        const jlong values[8]={static_cast<jlong>(p->lastGpuUs.load()),static_cast<jlong>(p->maximumGpuUs.load()),
            static_cast<jlong>(p->budgetOverruns.load()),static_cast<jlong>(c.topologyUpdates),
            static_cast<jlong>(c.localEdges),c.anchors,c.longEdges,static_cast<jlong>(c.contributionUpdates)};
        env->SetLongArrayRegion(status,0,8,values);
    } catch(const std::exception& e) { throwJava(env,e); }
}

extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nScopeEnabled(JNIEnv* env,jclass,jlong handle,jboolean enabled,jint tile) {
    try {
        if(tile<0 || tile>=vabpnn::SpikeTiles) throw std::runtime_error("Network scope tile out of range");
        auto* p=from(handle); p->scopeTile.store(tile); p->scopeEnabled.store(enabled==JNI_TRUE); p->wake.notify_one();
    } catch(const std::exception& e) { throwJava(env,e); }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nScope(JNIEnv* env,jclass,jlong handle,
    jlongArray meta,jfloatArray values,jintArray discrete) {
    try {
        if(env->GetArrayLength(meta)!=24 || env->GetArrayLength(values)!=vabpnn::ScopeFloats
            || env->GetArrayLength(discrete)!=vabpnn::ScopeInts) throw std::runtime_error("Network scope buffer shape mismatch");
        auto* p=from(handle); std::lock_guard<std::mutex> lock(p->scopeMutex);
        env->SetLongArrayRegion(meta,0,24,p->scopeMeta.data());
        env->SetFloatArrayRegion(values,0,vabpnn::ScopeFloats,p->scopeLatest.values);
        env->SetIntArrayRegion(discrete,0,vabpnn::ScopeInts,p->scopeLatest.discrete);
    } catch(const std::exception& e) { throwJava(env,e); }
}

extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nBlocks(JNIEnv* env,jclass,jlong handle,jint enabled,jint mode) {
    try {
        if(enabled < -1 || enabled>1 || mode < -1 || mode>2) throw std::runtime_error("Invalid block mode");
        auto* p=from(handle);
        { std::lock_guard<std::mutex> lock(p->controlMutex);
            if(enabled>=0) p->requestedBlocks=enabled;
            if(mode>=0) p->requestedCommunication=mode;
            p->blocksChanged.store(true); }
        p->wake.notify_one();
    } catch(const std::exception& e) { throwJava(env,e); }
}
extern "C" JNIEXPORT void JNICALL
Java_cn_vabpnn_mod_client_capture_NativeCudaBridge_nBlockStatus(JNIEnv* env,jclass,jlong handle,jlongArray target) {
    try {
        if(!target || env->GetArrayLength(target)!=14) throw std::runtime_error("Fourteen-long block status required");
        auto* p=from(handle); vabpnn::BlockCounters c; std::uint64_t frame;
        { std::lock_guard<std::mutex> lock(p->controlMutex); c=p->latestBlocks; frame=p->latestBlockFrame; }
        const jlong data[14]={static_cast<jlong>(c.scoredFrames),static_cast<jlong>(c.actorUpdates),
            static_cast<jlong>(c.rankings),static_cast<jlong>(c.mutations),static_cast<jlong>(c.controlUpdates),
            static_cast<jlong>(c.broadcasts),c.enabled,c.communication,c.radius,c.eligible,c.candidates,
            c.punished,c.invalidConfig,static_cast<jlong>(frame)};
        env->SetLongArrayRegion(target,0,14,data);
    } catch(const std::exception& e) { throwJava(env,e); }
}
