#pragma once
#include "LongTermMemory.h"
#include "model_checkpoint.h"
#include <mutex>
#include <atomic>
#include <filesystem>
// Worker owns all CUDA/index I/O. Main thread receives only a small scalar status.
class LongMemoryBridge {
    VabpnnLongMemory* memory=nullptr;
    vabpnn::LongMemoryStatus *pending=nullptr,latest{};
    std::mutex mutex;
    std::wstring requestedPath,storePath;
    std::string message="External vector database not configured";
    std::atomic<int> requested{-1};
    bool healthy=true,opened=false;
    std::uint64_t savedRevision=0;
    int ioState=0;
    static void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
public:
    void initialize(){memory=vabpnnCreateLongMemory(nullptr);check(cudaMallocHost(reinterpret_cast<void**>(&pending),sizeof(*pending)));*pending={};}
    void close() noexcept {vabpnnDestroyLongMemory(memory);memory=nullptr;if(pending)cudaFreeHost(pending);pending=nullptr;}
    VabpnnLongMemory* model(){return memory;}
    std::uint64_t bytes(){return vabpnnLongMemoryBytes(memory);}
    vabpnn::ModelPart checkpoint(){return vabpnnLongMemoryCheckpointPart(memory);}
    void path(const std::wstring& path){std::lock_guard<std::mutex> lock(mutex);if(opened||!requestedPath.empty()&&requestedPath!=path)throw std::invalid_argument("Vector database path is immutable during a run");requestedPath=path;}
    void configure(bool enabled){requested.store(enabled ? 1:0);}
    void service(cudaStream_t stream){const int next=requested.exchange(-1);if(next<0)return;vabpnnConfigureLongMemory(memory,next!=0&&healthy,stream);}
    void reset(cudaStream_t stream){vabpnnResetLongMemoryTransient(memory,stream);}
    void load(const std::wstring& modelPath){
        {std::lock_guard<std::mutex> lock(mutex);storePath=requestedPath.empty() ? modelPath+L".vdb":requestedPath;opened=true;}
        try{
            vabpnn::ModelPart parts[2];vabpnnLongMemoryStoreParts(memory,parts);
            const bool loaded=vabpnn::loadModelCheckpoint(storePath,{parts,parts+2},false,14);
            if(loaded){vabpnn::LongMemoryCatalog catalog{};check(cudaMemcpy(&catalog,parts[0].device,sizeof(catalog),cudaMemcpyDeviceToHost));
                if(catalog.used>vabpnn::LongMemoryCapacity||!std::isfinite(catalog.observedSeconds)||catalog.observedSeconds<0)throw std::runtime_error("Invalid vector catalog metadata");savedRevision=catalog.revision;}
            std::lock_guard<std::mutex> lock(mutex);ioState=loaded ? 2:1;message=loaded ? "External vector database restored":"Fresh external vector database";
        }catch(const std::exception& error){
            healthy=false;vabpnnConfigureLongMemory(memory,false,nullptr);
            std::lock_guard<std::mutex> lock(mutex);ioState=-1;message=std::string("Vector file retained; memory disabled: ")+error.what();
        }
    }
    void save(){
        if(!healthy||!opened||storePath.empty())return;
        vabpnn::ModelPart parts[2];vabpnnLongMemoryStoreParts(memory,parts);
        vabpnn::saveModelCheckpoint(storePath,{parts,parts+2},14);
        vabpnn::LongMemoryCatalog catalog{};check(cudaMemcpy(&catalog,parts[0].device,sizeof(catalog),cudaMemcpyDeviceToHost));savedRevision=catalog.revision;
        std::lock_guard<std::mutex> lock(mutex);ioState=3;message="External vector database atomically saved";
    }
    void enqueue(cudaStream_t stream){check(cudaMemcpyAsync(pending,&vabpnnLongMemoryView(memory).output->status,sizeof(*pending),cudaMemcpyDeviceToHost,stream));}
    void complete(){std::lock_guard<std::mutex> lock(mutex);latest=*pending;}
    vabpnn::LongMemoryStatus status(){std::lock_guard<std::mutex> lock(mutex);return latest;}
    int state(){std::lock_guard<std::mutex> lock(mutex);return ioState;}
    std::string text(){std::lock_guard<std::mutex> lock(mutex);return message;}
};
