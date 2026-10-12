#pragma once
#include "ModelSnapshot.h"
#include <bcrypt.h>
#include <cstring>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
namespace vabpnn {
struct CheckpointHeader {
    char magic[8]={'V','A','B','P','N','N','7',0};
    std::uint64_t version=2, count=0, bytes=0, sizes[16]{};
    unsigned char hash[32]{};
};
static_assert(sizeof(CheckpointHeader)==192,"Legacy checkpoint header stays byte-identical.");
struct CheckpointHeaderV5 {
    char magic[8]={'V','A','B','P','N','N','A',0};
    std::uint64_t version=5,count=0,bytes=0,sizes[24]{};
    unsigned char hash[32]{};
};
static_assert(sizeof(CheckpointHeaderV5)==256,"Version five has twenty-four size slots.");
class Sha256 {
    BCRYPT_ALG_HANDLE algorithm=nullptr;
    BCRYPT_HASH_HANDLE hash=nullptr;
public:
    Sha256() {
        if(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0
            || BCryptCreateHash(algorithm,&hash,nullptr,0,nullptr,0,0)<0)
            throw std::runtime_error("Could not create checkpoint SHA256");
    }
    ~Sha256() { if(hash) BCryptDestroyHash(hash); if(algorithm) BCryptCloseAlgorithmProvider(algorithm,0); }
    void add(unsigned char* bytes,std::size_t count) {
        if(BCryptHashData(hash,bytes,static_cast<ULONG>(count),0)<0)
            throw std::runtime_error("Could not hash checkpoint");
    }
    void finish(unsigned char* result) {
        if(BCryptFinishHash(hash,result,32,0)<0) throw std::runtime_error("Could not finalize checkpoint hash");
    }
};
template<class Header> inline Header checkpointLayoutFor(const std::vector<ModelPart>& parts) {
    Header header;
    if(parts.empty() || parts.size()>sizeof(header.sizes)/sizeof(header.sizes[0])) throw std::runtime_error("Invalid checkpoint layout");
    header.count=parts.size();
    for(std::size_t i=0;i<parts.size();++i) {
        header.sizes[i]=parts[i].bytes; header.bytes+=parts[i].bytes;
    }
    return header;
}
inline CheckpointHeader checkpointLayout(const std::vector<ModelPart>& parts) {return checkpointLayoutFor<CheckpointHeader>(parts);}
template<class Header> inline Header validateModelCheckpointFormat(std::ifstream& file,const std::wstring& path,const std::vector<ModelPart>& parts,bool legacy,int version) {
    auto expected=checkpointLayoutFor<Header>(parts);
    if(legacy) { expected.magic[6]='6'; expected.version=1; }
    else if(version==3) { expected.magic[6]='8'; expected.version=3; }
    else if(version==4) { expected.magic[6]='9'; expected.version=4; }
    else if(version==6) { expected.magic[6]='B'; expected.version=6; }
    else if(version==7) { expected.magic[6]='C'; expected.version=7; }
    else if(version==8) { expected.magic[6]='D'; expected.version=8; }
    Header header{};
    if(!file.read(reinterpret_cast<char*>(&header),sizeof(header))
        || std::memcmp(header.magic,expected.magic,8)!=0 || header.version!=expected.version
        || header.count!=expected.count || header.bytes!=expected.bytes
        || std::memcmp(header.sizes,expected.sizes,sizeof(header.sizes))!=0
        || std::filesystem::file_size(path)!=sizeof(header)+expected.bytes)
        throw std::runtime_error("Checkpoint layout does not match this model; original file retained");
    // Validate the ENTIRE file before changing even one GPU parameter.
    std::vector<unsigned char> buffer(1024*1024);
    Sha256 hash;
    std::uint64_t remaining=header.bytes;
    while(remaining) {
        const auto bytes=static_cast<std::size_t>(std::min<std::uint64_t>(remaining,buffer.size()));
        if(!file.read(reinterpret_cast<char*>(buffer.data()),bytes)) throw std::runtime_error("Incomplete checkpoint");
        hash.add(buffer.data(),bytes); remaining-=bytes;
    }
    unsigned char digest[32]; hash.finish(digest);
    if(std::memcmp(digest,header.hash,32)!=0) throw std::runtime_error("Checkpoint SHA256 mismatch; original file retained");
    return header;
}
template<class Header> inline bool loadModelCheckpointFormat(const std::wstring& path,const std::vector<ModelPart>& parts,bool legacy,int version) {
    if(!std::filesystem::exists(path)) return false;
    std::ifstream file(std::filesystem::path(path),std::ios::binary);
    const auto header=validateModelCheckpointFormat<Header>(file,path,parts,legacy,version);
    std::vector<unsigned char> buffer(1024*1024);
    file.clear(); file.seekg(sizeof(header));
    for(const auto& part:parts) for(std::size_t offset=0;offset<part.bytes;) {
        const auto bytes=std::min(part.bytes-offset,buffer.size());
        if(!file.read(reinterpret_cast<char*>(buffer.data()),bytes)) throw std::runtime_error("Checkpoint read failed");
        auto code=cudaMemcpy(static_cast<unsigned char*>(part.device)+offset,buffer.data(),bytes,cudaMemcpyHostToDevice);
        if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
        offset+=bytes;
    }
    return true;
}
template<class Header> inline void saveModelCheckpointFormat(const std::wstring& path,const std::vector<ModelPart>& parts,int version) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    const auto temporary=path+L".tmp";
    auto header=checkpointLayoutFor<Header>(parts);
    if(version==3) { header.magic[6]='8'; header.version=3; }
    else if(version==4) { header.magic[6]='9'; header.version=4; }
    else if(version==6) { header.magic[6]='B'; header.version=6; }
    else if(version==7) { header.magic[6]='C'; header.version=7; }
    else if(version==8) { header.magic[6]='D'; header.version=8; }
    std::ofstream file(std::filesystem::path(temporary),std::ios::binary|std::ios::trunc);
    if(!file.write(reinterpret_cast<char*>(&header),sizeof(header))) throw std::runtime_error("Could not open checkpoint");
    Sha256 hash;
    std::vector<unsigned char> buffer(1024*1024); // Bounded staging of weights only, never observations.
    for(const auto& part:parts) for(std::size_t offset=0;offset<part.bytes;) {
        const auto bytes=std::min(part.bytes-offset,buffer.size());
        const auto code=cudaMemcpy(buffer.data(),static_cast<unsigned char*>(part.device)+offset,bytes,cudaMemcpyDeviceToHost);
        if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
        hash.add(buffer.data(),bytes);
        if(!file.write(reinterpret_cast<char*>(buffer.data()),bytes)) throw std::runtime_error("Checkpoint write failed");
        offset+=bytes;
    }
    hash.finish(header.hash); file.seekp(0);
    file.write(reinterpret_cast<char*>(&header),sizeof(header)); file.flush();
    if(!file) throw std::runtime_error("Could not flush checkpoint");
    file.close();
    if(!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Could not atomically publish checkpoint; prior checkpoint retained");
}
inline bool loadModelCheckpoint(const std::wstring& path,const std::vector<ModelPart>& parts,bool legacy=false,int version=2) {
    if(!legacy && version>=5) return loadModelCheckpointFormat<CheckpointHeaderV5>(path,parts,false,version);
    return loadModelCheckpointFormat<CheckpointHeader>(path,parts,legacy,version);
}
inline void saveModelCheckpoint(const std::wstring& path,const std::vector<ModelPart>& parts,int version=2) {
    if(version>=5) saveModelCheckpointFormat<CheckpointHeaderV5>(path,parts,version);
    else saveModelCheckpointFormat<CheckpointHeader>(path,parts,version);
}

}

