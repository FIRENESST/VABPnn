#pragma once
#include "model_checkpoint.h"
#include "DenseBlockTransformer.h"
#include <cstddef>
namespace vabpnn {
// v9 has the same 23 parts, except three beta-small tensors and one optimizer counter.
constexpr std::size_t BetaDensePrefix=DenseBetaSmallParameters*sizeof(float);
constexpr std::size_t BetaDensePersistentBytes=BetaDensePrefix*3+sizeof(std::uint64_t);
static_assert(BetaDensePersistentBytes==1530656,"Original beta layout.");
inline bool migrateComparisonBeta(const std::wstring& path,const std::vector<ModelPart>& parts) {
    if(!std::filesystem::exists(path))return false;
    if(parts.size()!=23 || parts[19].bytes!=sizeof(DenseBlockPersistent))
        throw std::runtime_error("Invalid comparison migration destination");
    auto betaParts=parts;betaParts[19].bytes=BetaDensePersistentBytes;
    std::ifstream file(std::filesystem::path(path),std::ios::binary);
    const auto header=validateModelCheckpointFormat<CheckpointHeaderV5>(file,path,betaParts,false,9);
    // All layout and SHA checks finish before the first GPU write; the source stays read-only.
    file.clear();file.seekg(sizeof(header));
    std::vector<unsigned char> staging(1024*1024);
    auto copy=[&](unsigned char* destination,std::size_t length) {
        for(std::size_t offset=0;offset<length;) {
            const auto count=std::min(length-offset,staging.size());
            if(!file.read(reinterpret_cast<char*>(staging.data()),count))
                throw std::runtime_error("Beta checkpoint read failed");
            const auto code=cudaMemcpy(destination+offset,staging.data(),count,cudaMemcpyHostToDevice);
            if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
            offset+=count;
        }
    };
    for(std::size_t i=0;i<parts.size();++i) {
        auto* destination=static_cast<unsigned char*>(parts[i].device);
        if(i!=19){copy(destination,parts[i].bytes);continue;}
        copy(destination+offsetof(DenseBlockPersistent,parameters),BetaDensePrefix);
        copy(destination+offsetof(DenseBlockPersistent,first),BetaDensePrefix);
        copy(destination+offsetof(DenseBlockPersistent,second),BetaDensePrefix);
        copy(destination+offsetof(DenseBlockPersistent,steps),sizeof(std::uint64_t));
        // Preserve initialized gates. New moments/counter start fresh, independently of the beta counter.
        for(auto offset:{offsetof(DenseBlockPersistent,first),offsetof(DenseBlockPersistent,second)}) {
            const auto code=cudaMemset(destination+offset+offsetof(DenseBlockParameters,memoryGate),0,sizeof(DenseMemoryGateParameters));
            if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
        }
        const auto code=cudaMemset(destination+offsetof(DenseBlockPersistent,memorySteps),0,sizeof(std::uint64_t));
        if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
    }
    return true;
}
}
