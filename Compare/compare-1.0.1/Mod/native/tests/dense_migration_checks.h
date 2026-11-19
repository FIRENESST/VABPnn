#pragma once
#include "compare_checkpoint_migration.h"
void verifyBetaMigration() {
    auto* model=vabpnnCreateDenseBlockModel(nullptr,4);const auto view=vabpnnDenseBlockModelView(model);
    auto fresh=bytes(view.persistent,sizeof(DenseBlockPersistent));
    ModelPart dense[4];vabpnnDenseBlockCheckpointParts(model,dense);
    Device<unsigned char> dummy(19*16),oldDense(BetaDensePersistentBytes);
    std::vector<unsigned char> prefix(dummy.n);for(std::size_t i=0;i<prefix.size();++i)prefix[i]=static_cast<unsigned char>(i*7);dummy.set(prefix.data());
    std::vector<unsigned char> beta(BetaDensePersistentBytes,0);
    for(int bank=0;bank<3;++bank)
        std::memcpy(beta.data()+bank*BetaDensePrefix,fresh.data()+bank*sizeof(DenseBlockParameters),BetaDensePrefix);
    const std::uint64_t oldSteps=97'321;std::memcpy(beta.data()+3*BetaDensePrefix,&oldSteps,8);
    oldDense.set(beta.data());
    std::vector<ModelPart> destination,source;
    for(int i=0;i<19;++i)destination.push_back({dummy.p+i*16,16});
    destination.insert(destination.end(),dense,dense+4);source=destination;source[19]={oldDense.p,BetaDensePersistentBytes};
    const std::wstring path=L"dense-compare-checkpoints/compare-1.0.0-beta.bin";saveModelCheckpoint(path,source,9);
    require(migrateComparisonBeta(path,destination),"beta migration loads exact old layout");
    auto migrated=bytes(view.persistent,sizeof(DenseBlockPersistent));auto* persistent=reinterpret_cast<const DenseBlockPersistent*>(migrated.data());
    for(int bank=0;bank<3;++bank)
        require(!std::memcmp(beta.data()+bank*BetaDensePrefix,migrated.data()+bank*sizeof(DenseBlockParameters),BetaDensePrefix),"all beta parameter and moment prefixes inherited");
    require(persistent->steps==oldSteps && persistent->memorySteps==0,"old counter inherited and new gate counter fresh");
    require(!std::memcmp(&persistent->parameters.memoryGate,fresh.data()+offsetof(DenseBlockPersistent,parameters)+offsetof(DenseBlockParameters,memoryGate),
        sizeof(DenseMemoryGateParameters)),"fresh gate initialization preserved");
    DenseMemoryGateParameters zero{};
    require(!std::memcmp(&persistent->first.memoryGate,&zero,sizeof(zero)) && !std::memcmp(&persistent->second.memoryGate,&zero,sizeof(zero)),"gate moments start zero");
    {
        std::fstream file(std::filesystem::path(path),std::ios::binary|std::ios::in|std::ios::out);
        file.seekp(270);char c=1;file.write(&c,1);
    }
    bool rejected=false;try{migrateComparisonBeta(path,destination);}catch(const std::exception&){rejected=true;}
    require(rejected && migrated==bytes(view.persistent,sizeof(DenseBlockPersistent)),"corrupt beta rejected before any parameter writes");
    require(bytes(dummy.p,prefix.size())==prefix,"corrupt beta rejected before writes to preceding parts");
    vabpnnDestroyDenseBlockModel(model);
    std::printf("PASS: read-only v9 migration, all three parameter/moment prefixes and old counter exact; new gates/moments/counter fresh; corruption atomic.\n");
}
