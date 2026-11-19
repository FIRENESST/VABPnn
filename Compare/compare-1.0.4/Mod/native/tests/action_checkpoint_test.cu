#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "FrameOperator.h"
#include "OnlineLearning.h"
#include "SpikeConnections.h"
#include "MemeState.h"
#include "action_checkpoint_migration.h"
#include <cstdio>
#include <vector>
#include <array>
#include <filesystem>
#include <stdexcept>
using namespace vabpnn;
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
void check(cudaError_t value) { checkpointCuda(value); }
template<class T> T get(const T* device) { T value;check(cudaMemcpy(&value,device,sizeof(T),cudaMemcpyDeviceToHost));return value; }
template<class T> void equal(const T& a,const T& b) {require(std::memcmp(&a,&b,sizeof(T))==0,"Old parameter or optimizer bytes changed");}
template<std::size_t A,std::size_t B> void common(const float (&a)[A],const float (&b)[B]) {
    require(std::memcmp(a,b,sizeof(float)*std::min(A,B))==0,"Action-array values changed");
}
template<std::size_t A,std::size_t B,std::size_t C,std::size_t D>
void common(const float (&a)[A][B],const float (&b)[C][D]) {for(std::size_t i=0;i<std::min(A,C);++i)common(a[i],b[i]);}
void verifyDecision(const DecisionParameters& p,const LegacyDecision& old) {
#define CHECK(field) common(p.field,old.field)
    CHECK(nodeEmbedding);CHECK(voteEmbedding);CHECK(timeEmbedding);CHECK(query);CHECK(key);CHECK(value);CHECK(attentionOutput);
    for(int h=0;h<DecisionHeads;++h)common(p.graphBias[h],old.graphBias[h]);
    CHECK(logAttentionDecay);equal(p.logClockFrequency,old.logClockFrequency);CHECK(attentionGain);CHECK(attentionBias);
    CHECK(feedForwardUp);CHECK(feedForwardUpBias);CHECK(feedForwardDown);CHECK(feedForwardDownBias);
    CHECK(outputGain);CHECK(outputBias);CHECK(logLikelihoodWeight);CHECK(logLikelihoodBias);CHECK(rawVoteWeight);
    CHECK(thresholdWeight);CHECK(thresholdBias);CHECK(kBias);CHECK(initialPriorLogit);CHECK(logOnRate);CHECK(logOffRate);
#undef CHECK
    equal(p.logGateTemperature,old.logGateTemperature);
    for(int k=0;k<=13;++k) {
        for(int d=0;d<16+13;++d)equal(p.kWeight[k][d],old.kWeight[k][d]);
        for(int d=0;d<13;++d)equal(p.kWeight[k][16+15+d],old.kWeight[k][16+13+d]);
        for(int d:{29,30,44,45}) require(p.kWeight[k][d]==0,"New k features initialized neutral");
    }
    for(int k:{13,14}) {
        require(p.rawVoteWeight[k]==0 && p.logLikelihoodBias[k]==0 && p.thresholdBias[k]==0,"New action head neutral");
        for(int d=0;d<16;++d)require(p.logLikelihoodWeight[k][d]==0 && p.thresholdWeight[k][d]==0,"New evidence/threshold weights neutral");
    }
}
void verifyJepa(const JepaParameters& p,const LegacyJepa& old) {
    equal(p.encoder,old.encoder);equal(p.predictor,old.predictor);
    for(int i=0;i<1152+13;++i)equal(p.motion.inputWeight[i],old.motion.inputWeight[i]);
    for(int i=0;i<2;++i)equal(p.motion.inputWeight[1152+15+i],old.motion.inputWeight[1152+13+i]);
    equal(p.motion.hiddenBias,old.motion.hiddenBias);equal(p.motion.outputWeight,old.motion.outputWeight);equal(p.motion.outputBias,old.motion.outputBias);
    for(int i:{1165,1166})for(float value:p.motion.inputWeight[i])require(value==0,"New JEPA vote feature has neutral weight/moments");
}
struct Fixture {
    cudaStream_t stream=nullptr;VabpnnNeuronModel* neuron=nullptr;VabpnnDecisionModel* decision=nullptr;
    VabpnnJepaModel* jepa=nullptr;VabpnnOnlineModel* online=nullptr;VabpnnSpikeModel* spike=nullptr;
    VabpnnGpuFrame frame{};DemoSample demo{};
    Fixture() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        neuron=vabpnnCreateNeuronModel(stream);decision=vabpnnCreateDecisionModel(stream);
        jepa=vabpnnCreateJepaModel(stream);online=vabpnnCreateOnlineModel(stream);spike=vabpnnCreateSpikeModel(stream);
        frame.width=37;frame.height=23;frame.frameId=0;frame.captureNanos=1'000'000'000;
        frame.stream=stream;frame.neuronModel=neuron;frame.decisionModel=decision;frame.jepaModel=jepa;frame.onlineModel=online;frame.spikeModel=spike;
        unsigned char* image=nullptr;float* rgb=nullptr;
        check(cudaMallocPitch(reinterpret_cast<void**>(&image),&frame.rgbaPitch,37*4,23));frame.rgba=image;
        check(cudaMemset2DAsync(image,frame.rgbaPitch,204,37*4,23,stream));
        check(cudaMalloc(reinterpret_cast<void**>(&rgb),37*23*3*sizeof(float)));frame.rgbNchw=rgb;
        demo.active=1;demo.epoch=1;demo.pointerMode=1;
        check(cudaStreamSynchronize(stream));
    }
    ~Fixture() {
        cudaStreamSynchronize(stream);cudaFree(const_cast<unsigned char*>(frame.rgba));cudaFree(frame.rgbNchw);
        vabpnnDestroySpikeModel(spike);vabpnnDestroyOnlineModel(online);vabpnnDestroyJepaModel(jepa);
        vabpnnDestroyDecisionModel(decision);vabpnnDestroyNeuronModel(neuron);cudaStreamDestroy(stream);
    }
    void next() {frame.evaluationMode=demo.modelControl!=0;frame.frameId++;frame.captureNanos+=20'000'000;vabpnnSetDemoSample(online,&demo,stream);vabpnnLaunchOperator(frame);check(cudaStreamSynchronize(stream));}
    std::vector<ModelPart> parts() {
        std::vector<ModelPart> result{{vabpnnMutableNeuronWeights(neuron,stream),ParameterCount},
            {vabpnnMutableDecisionParameters(decision,stream),sizeof(DecisionParameters)}};
        ModelPart extra[8];auto add=[&](std::size_t n){result.insert(result.end(),extra,extra+n);};
        add(vabpnnJepaCheckpointParts(jepa,extra));add(vabpnnOnlineCheckpointParts(online,extra));
        add(vabpnnSpikeCheckpointParts(spike,extra));add(vabpnnDecisionPositionCheckpointParts(decision,extra));
        auto* block=vabpnnSpikeBlockModel(spike);result.push_back(vabpnnBlockCheckpointPart(block));
        result.push_back(vabpnnMemeCheckpointPart(vabpnnBlockMemeModel(block)));check(cudaStreamSynchronize(stream));return result;
    }
};
void migrationGating() {
    Fixture f;f.frame.frameId=1;auto* device=vabpnnMutableDecisionParameters(f.decision,f.stream);
    auto parameters=std::make_unique<DecisionParameters>();check(cudaMemcpy(parameters.get(),device,sizeof(*parameters),cudaMemcpyDeviceToHost));
    for(int k=0;k<16;++k)parameters->kBias[k]=k==1 ? -1:k<14 ? -2:0;
    for(int k=0;k<15;++k)parameters->logLikelihoodBias[k]=k<2 ? 2:0;
    parameters->enabledKChoices=14;
    check(cudaMemcpy(device,parameters.get(),sizeof(*parameters),cudaMemcpyHostToDevice));
    vabpnnLaunchTemporalDecision(f.frame,vabpnnNeuronModelView(f.neuron).output,f.decision);check(cudaStreamSynchronize(f.stream));
    auto output=get(vabpnnDecisionModelView(f.decision).output);
    require(output.requestedK==1 && output.selectedCount==1 && output.kProbability[14]==0 && output.kProbability[15]==0,
            "Negative old logits must not be overridden by dormant zero new k slots");
    parameters->enabledKChoices=16;check(cudaMemcpy(device,parameters.get(),sizeof(*parameters),cudaMemcpyHostToDevice));
    f.frame.frameId++;f.frame.captureNanos+=20'000'000;
    vabpnnLaunchTemporalDecision(f.frame,vabpnnNeuronModelView(f.neuron).output,f.decision);check(cudaStreamSynchronize(f.stream));
    output=get(vabpnnDecisionModelView(f.decision).output);
    require(output.requestedK==14 && output.selectedCount==2,"Trainer activation enables the new k branch intentionally");
    require(MemeInputScale==.15f/13 && std::abs(MemeMaximumMeanShift-.2596154f)<1e-6f,
            "Legacy meme projection scale retained, expanded bound still small");
    std::puts("PASS: negative-logit migration counterexample, dormant/new k activation and preserved 13-channel projection scale.");
}
void modifierLearning() {
    Fixture f;f.demo.heldKeys=(1u<<13)|(1u<<14);
    for(int i=0;i<36;++i){f.demo.keyHeldNanos[6]+=20'000'000;f.demo.keyHeldNanos[7]+=20'000'000;f.next();}
    const auto learned=get(vabpnnOnlineOutput(f.online));
    require(learned.examples==35 && learned.lastLabel==((1u<<13)|(1u<<14)),"Ctrl/Shift labels reach GPU online learning");
    const auto out=get(vabpnnDecisionModelView(f.decision).output);
    require(out.command.valid && out.selectedMask==((1u<<13)|(1u<<14)) && out.requestedK==2,"Bayes, learned threshold and top-k select both new actions");
    auto parts=f.parts();auto* weights=static_cast<TernaryWeight*>(parts[0].device);auto* master=static_cast<float*>(parts[7].device);
    for(int k:{13,14})require(get(weights+weightIndex(0,0,k))==TernaryWeight::Positive && get(master+weightIndex(0,0,k))>0.05f,
            "Each modifier has its own trained RGB/pixel weights");
    f.demo.heldKeys=0;f.demo.keyHeldNanos[6]+=10'000'000;f.demo.keyHeldNanos[7]+=10'000'000;f.next();
    require(get(vabpnnOnlineOutput(f.online)).lastLabel==((1u<<13)|(1u<<14)),"Short modifier pulses survive cumulative timing");
    const auto before=get(vabpnnOnlineOutput(f.online));auto* block=vabpnnSpikeBlockModel(f.spike);
    const auto memes=get(&vabpnnMutableMemePersistent(vabpnnBlockMemeModel(block),f.stream)->counters);check(cudaStreamSynchronize(f.stream));
    f.demo.modelControl=1;f.demo.epoch++;
    for(int i=0;i<8;++i)f.next();
    const auto after=get(vabpnnOnlineOutput(f.online));
    require(after.examples==before.examples && after.pixelSteps==before.pixelSteps && after.decisionSteps==before.decisionSteps,"F8 excludes modifier imitation");
    require(get(&vabpnnMutableMemePersistent(vabpnnBlockMemeModel(block),f.stream)->counters).supervisedFrames==memes.supervisedFrames,"F8 excludes new-action meme teaching");
    std::puts("PASS: independent Ctrl/Shift RGB neurons, causal duration/pulse labels, Bayesian eligibility/top-k and F8 learning freeze.");
}
void compareRegion(LegacyCheckpointReader& reader,int part,std::uint64_t offset,const void* device,std::size_t bytes,bool zero=false) {
    std::vector<unsigned char> source(1024*1024),actual(source.size());
    for(std::size_t at=0;at<bytes;) {
        const auto count=std::min(source.size(),bytes-at);
        if(zero)std::fill_n(source.data(),count,0);else reader.read(part,offset+at,source.data(),count);
        check(cudaMemcpy(actual.data(),static_cast<const unsigned char*>(device)+at,count,cudaMemcpyDeviceToHost));
        require(std::memcmp(actual.data(),source.data(),count)==0,"Legacy GPU tensor was not copied bit-for-bit or new plane nonzero");
        at+=count;
    }
}
void actualMigration(const std::wstring& path,int version) {
    Fixture f;auto parts=f.parts();require(loadLegacyActionCheckpoint(path,parts,version),"Actual legacy file found");
    LegacyCheckpointReader reader(path,parts,version);
    for(int index:{0,7}) {
        const auto element=index==0 ? 1:sizeof(float),plane=NeuronCount*element;
        for(int c=0;c<3;++c)for(int k=0;k<15;++k)
            compareRegion(reader,index,(c*13+k)*plane,static_cast<unsigned char*>(parts[index].device)+(c*15+k)*plane,plane,k>=13);
    }
    auto decision=std::make_unique<DecisionParameters>();check(cudaMemcpy(decision.get(),parts[1].device,sizeof(*decision),cudaMemcpyDeviceToHost));
    verifyDecision(*decision,*reader.readObject<LegacyDecision>(1));
    require(decision->enabledKChoices==14,"Unused k=14/15 remain dormant after legacy migration");
    for(int index:{2,4,5}) {
        auto p=std::make_unique<JepaParameters>();check(cudaMemcpy(p.get(),parts[index].device,sizeof(*p),cudaMemcpyDeviceToHost));
        verifyJepa(*p,*reader.readObject<LegacyJepa>(index));
    }
    auto current=std::make_unique<OnlinePersistent>();check(cudaMemcpy(current.get(),parts[8].device,sizeof(*current),cudaMemcpyDeviceToHost));
    auto old=reader.readObject<LegacyOnline>(8);verifyDecision(current->first,old->first);verifyDecision(current->second,old->second);
    equal(current->output,old->output);
    for(int d=0;d<29;++d){equal(current->wheel[d],old->wheel[d]);equal(current->wheelFirst[d],old->wheelFirst[d]);equal(current->wheelSecond[d],old->wheelSecond[d]);}
    for(int d:{29,30})require(current->wheel[d]==0 && current->wheelFirst[d]==0 && current->wheelSecond[d]==0,"New wheel features neutral");
    equal(current->wheel[31],old->wheel[29]);equal(current->wheelFirst[31],old->wheelFirst[29]);equal(current->wheelSecond[31],old->wheelSecond[29]);
    for(int index=3;index<static_cast<int>(reader.count());++index) {
        if(index==4 || index==5 || index==7 || index==8 || index==14 || index==16)continue;
        compareRegion(reader,index,0,parts[index].device,parts[index].bytes);
    }
    if(version>=3) {
        auto position=std::make_unique<DecisionPositionAdapter>();check(cudaMemcpy(position.get(),parts[14].device,sizeof(*position),cudaMemcpyDeviceToHost));
        auto saved=reader.readObject<LegacyPosition>(14);
        common(position->evidence,saved->evidence);common(position->evidenceFirst,saved->evidenceFirst);common(position->evidenceSecond,saved->evidenceSecond);
        common(position->k,saved->k);common(position->kFirst,saved->kFirst);common(position->kSecond,saved->kSecond);equal(position->steps,saved->steps);
    }
    if(version>=5) {
        auto memes=std::make_unique<MemePersistent>();check(cudaMemcpy(memes.get(),parts[16].device,sizeof(*memes),cudaMemcpyDeviceToHost));
        auto saved=reader.readObject<LegacyMeme>(16);equal(memes->config,saved->config);equal(memes->counters,saved->counters);
        for(int i=0;i<MemeBlocks;++i) {
            auto& a=memes->prototypes[i];const auto& b=saved->prototypes[i];
            common(a.evidence,b.evidence);equal(a.context,b.context);
            equal(a.utility,b.utility);equal(a.agreement,b.agreement);equal(a.coverage,b.coverage);equal(a.confidence,b.confidence);
            equal(a.age,b.age);equal(a.support,b.support);equal(a.revision,b.revision);equal(a.sequence,b.sequence);equal(a.origin,b.origin);equal(a.hops,b.hops);
            require(a.evidence[13]==0 && a.evidence[14]==0,"New prototype evidence neutral");
            const auto& policy=memes->policy[i];const auto& oldPolicy=saved->policy[i];
            for(int out=0;out<4;++out)for(int k=0;k<15;++k) {
                const auto at=out*15+k,oldAt=out*13+k;
                if(k<13){equal(policy.weights[at],oldPolicy.weights[oldAt]);equal(policy.first[at],oldPolicy.first[oldAt]);equal(policy.second[at],oldPolicy.second[oldAt]);equal(policy.bestWeights[at],oldPolicy.bestWeights[oldAt]);}
                else require(policy.weights[at]==0 && policy.first[at]==0 && policy.second[at]==0 && policy.bestWeights[at]==0,"New projection column neutral");
            }
            equal(policy.optimizerSteps,oldPolicy.optimizerSteps);
        }
    }
    const auto destination=std::filesystem::absolute("action-migration-test/release-v6.bin").wstring();
    saveModelCheckpoint(destination,parts,6);
    const auto prior=get(static_cast<TernaryWeight*>(parts[0].device));
    check(cudaMemset(parts[0].device,1,1));require(loadModelCheckpoint(destination,parts,false,6),"Version six roundtrip");
    require(get(static_cast<TernaryWeight*>(parts[0].device))==prior,"Version six restores trained state");
    std::printf("PASS: actual v%d checkpoint expanded 13->15; every old pixel plane, JEPA vote/time feature, decision/wheel/position/meme weight and AdamW state preserved; v6 cold restore.\n",version);
    const auto bad=std::filesystem::absolute("action-migration-test/corrupt-legacy.bin");
    std::filesystem::copy_file(path,bad,std::filesystem::copy_options::overwrite_existing);
    {std::fstream file(bad,std::ios::binary|std::ios::in|std::ios::out);file.seekg(-1,std::ios::end);char value;file.read(&value,1);value^=1;file.seekp(-1,std::ios::end);file.write(&value,1);}
    TernaryWeight sentinel=TernaryWeight::Positive;check(cudaMemcpy(parts[0].device,&sentinel,1,cudaMemcpyHostToDevice));
    const int lastIndex=version==5 ? 16:version==4 ? 15:version==3 ? 14:version==2 ? 13:8;
    auto* last=reinterpret_cast<float*>(static_cast<unsigned char*>(parts[lastIndex].device)+parts[lastIndex].bytes-sizeof(float));float sentinelLast=.123456f;check(cudaMemcpy(last,&sentinelLast,sizeof(float),cudaMemcpyHostToDevice));
    bool rejected=false;try{loadLegacyActionCheckpoint(bad.wstring(),parts,version);}catch(const std::runtime_error&){rejected=true;}
    require(rejected && get(static_cast<TernaryWeight*>(parts[0].device))==sentinel && get(last)==sentinelLast,"Corrupt legacy must not overwrite any GPU parameters");
    std::filesystem::remove(bad);
    std::puts("PASS: corrupt original rejected before any first/last GPU write; original checkpoint opened read-only.");
}
int main(int argc,char** argv) {
    try {
        migrationGating();modifierLearning();
        for(int i=1;i+1<argc;i+=2) actualMigration(std::filesystem::u8path(argv[i+1]).wstring(),std::stoi(argv[i]));
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
