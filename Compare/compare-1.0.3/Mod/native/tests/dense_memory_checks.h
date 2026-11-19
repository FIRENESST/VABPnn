#pragma once
void verifyMatrixMemory(VabpnnDenseBlockModel* model,Device<float>& input,Device<float>& target,Device<int>& active,
                        const std::vector<float>& features,const std::vector<float>& gradient) {
    const auto view=vabpnnDenseBlockModelView(model);const int n=view.nodes;const std::size_t pairs=DenseHeads*n*n;
    vabpnnResetDenseBlocks(model,nullptr);check(cudaDeviceSynchronize());
    auto host=std::make_unique<DenseBlockPersistent>();
    check(cudaMemcpy(host.get(),view.persistent,sizeof(*host),cudaMemcpyDeviceToHost));
    host->steps=10'000;host->memorySteps=0;host->first.memoryGate={};host->second.memoryGate={};
    for(int h=0;h<DenseHeads;++h)for(int g=0;g<DenseMemoryGates;++g)for(int f=0;f<DenseMemoryFeatures;++f)
        host->parameters.memoryGate.weight[h][g][f]=.06f*std::sin(.31f*(1+h*57+g*19+f));
    for(int h=0;h<DenseHeads;++h) {
        host->parameters.memoryGate.weight[h][0][18]=-.8f;
        host->parameters.memoryGate.weight[h][1][18]=.2f;
        host->parameters.memoryGate.weight[h][2][18]=-.3f;
        host->parameters.memoryGate.gain[h]=.7f+.2f*h;
    }
    check(cudaMemcpy(view.persistent,host.get(),sizeof(*host),cudaMemcpyHostToDevice));
    std::vector<float> bias(pairs),previous(pairs);
    check(cudaMemcpy(bias.data(),view.graphBias,pairs*4,cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<pairs;++i)previous[i]=.4f*std::cos(.57f*(i+1));
    vabpnnDenseForwardFeatures(model,input.p,100,10'000'000'000,nullptr);check(cudaDeviceSynchronize());
    check(cudaMemcpy(view.memory,previous.data(),pairs*4,cudaMemcpyHostToDevice));
    vabpnnDenseForwardFeatures(model,input.p,101,10'300'000'000,nullptr);check(cudaDeviceSynchronize());
    const auto ref=reference(host->parameters,bias,features,n,previous,.3);
    auto memory=bytes(view.memory,pairs*4),probability=bytes(const_cast<float*>(view.attention),pairs*4);
    const auto* m=reinterpret_cast<const float*>(memory.data());const auto* probabilityValues=reinterpret_cast<const float*>(probability.data());
    DenseBlockOutput output{};check(cudaMemcpy(&output,view.output,sizeof(output),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<pairs;++i) {
        require(std::abs(m[i]-ref.memory[i])<3e-6 && std::abs(probabilityValues[i]-ref.attention[i])<3e-6,"warm gated matrix CPU reference");
        require(std::abs(m[i])<=1,"bounded matrix memory");
    }
    for(int i=0;i<VoteCount*DenseWidth;++i)require(std::abs(output.hidden[i/DenseWidth][i%DenseWidth]-ref.hidden[i])<3e-5,"warm memory output reference");
    int on=1;active.set(&on);vabpnnDenseBackwardHidden(model,target.p,active.p,nullptr);check(cudaDeviceSynchronize());
    auto actual=std::make_unique<DenseBlockParameters>();check(cudaMemcpy(actual.get(),view.gradient,sizeof(*actual),cudaMemcpyDeviceToHost));
    std::vector<float> gb(pairs);check(cudaMemcpy(gb.data(),view.graphGradient,pairs*4,cudaMemcpyDeviceToHost));
    auto* parameter=reinterpret_cast<float*>(&host->parameters);const auto* derivative=reinterpret_cast<const float*>(actual.get());
    constexpr float epsilon=.002f;double maximum=0;int nonzero=0,checked=0;
    auto verify=[&](float& value,float gpu) {
        const float old=value;value=old+epsilon;const double plus=loss(reference(host->parameters,bias,features,n,previous,.3),gradient);
        value=old-epsilon;const double minus=loss(reference(host->parameters,bias,features,n,previous,.3),gradient);value=old;
        const double expected=(plus-minus)/(2*epsilon),error=std::abs(expected-gpu);
        maximum=std::max(maximum,error);++checked;if(std::abs(expected)>1e-9)++nonzero;
        if(error>3e-5+std::abs(expected)*.005) {
            std::printf("BAD warm derivative expected=%.9g actual=%.9g\n",expected,gpu);
            throw std::runtime_error("Memory gate/Q/K/pair finite difference mismatch");
        }
    };
    for(std::size_t i=0;i<pairs;++i)verify(bias[i],gb[i]);
    for(std::size_t i=offsetof(DenseBlockParameters,query)/4;i<offsetof(DenseBlockParameters,value)/4;++i)verify(parameter[i],derivative[i]);
    for(std::size_t i=offsetof(DenseBlockParameters,memoryGate)/4;i<DenseSmallParameters;++i)verify(parameter[i],derivative[i]);
    require(nonzero>500,"memory gradients actually active");
    vabpnnDenseApplyAdamW(model,nullptr);check(cudaDeviceSynchronize());
    auto trained=std::make_unique<DenseBlockPersistent>();check(cudaMemcpy(trained.get(),view.persistent,sizeof(*trained),cudaMemcpyDeviceToHost));
    require(trained->steps==10'001 && trained->memorySteps==1,"new gates have independent Adam bias correction after beta migration");
    const auto* after=reinterpret_cast<const float*>(&trained->parameters);
    const auto* first=reinterpret_cast<const float*>(&trained->first);const auto* second=reinterpret_cast<const float*>(&trained->second);
    for(std::size_t i=offsetof(DenseBlockParameters,memoryGate)/4;i<DenseSmallParameters;++i) {
        const auto at=i-offsetof(DenseBlockParameters,memoryGate)/4;
        const bool decay=at<DenseHeads*DenseGateWeightsPerHead && at%DenseMemoryFeatures<DenseMemoryFeatures-1;
        const double g=std::clamp(static_cast<double>(derivative[i]),-1.,1.);
        const double expected=parameter[i]*(decay ? .999997:1)-.0003*g/(std::abs(g)+1e-8);
        require(std::abs(after[i]-expected)<3e-7 && std::abs(first[i]-.1*g)<2e-8 && std::abs(second[i]-.001*g*g)<2e-10,"new gate AdamW/moment reference");
    }
    const auto frozen=bytes(view.persistent,sizeof(*trained)),frozenBias=bytes(view.graphBias,pairs*4);
    const auto frozenFirst=bytes(view.first,pairs*4),frozenSecond=bytes(view.second,pairs*4);
    on=0;active.set(&on);vabpnnDenseForwardFeatures(model,input.p,102,10'600'000'000,nullptr);
    vabpnnDenseBackwardHidden(model,target.p,active.p,nullptr);vabpnnDenseApplyAdamW(model,nullptr);check(cudaDeviceSynchronize());
    require(frozen==bytes(view.persistent,sizeof(*trained)) && frozenBias==bytes(view.graphBias,pairs*4)
        && frozenFirst==bytes(view.first,pairs*4) && frozenSecond==bytes(view.second,pairs*4),"frozen neural tensors and moments");
    const auto evolved=bytes(view.memory,pairs*4),status=bytes(const_cast<DenseBlockStatus*>(view.status),sizeof(DenseBlockStatus));
    require(memory!=evolved,"read-only predictions update matrix memory");
    vabpnnDenseForwardFeatures(model,input.p,102,10'600'000'000,nullptr);check(cudaDeviceSynchronize());
    require(evolved==bytes(view.memory,pairs*4) && status==bytes(const_cast<DenseBlockStatus*>(view.status),sizeof(DenseBlockStatus)),"duplicate frame does not advance memory twice");
    vabpnnDenseForwardFeatures(model,input.p,103,1'000'000'000,nullptr);check(cudaDeviceSynchronize());
    auto cleared=bytes(view.memory,pairs*4);for(unsigned char c:cleared)require(c==0,"reversed clock clears matrix memory");
    vabpnnResetDenseBlocks(model,nullptr);check(cudaDeviceSynchronize());
    require(frozen==bytes(view.persistent,sizeof(*trained)),"episode reset keeps weights and optimizer");
    DenseBlockStatus state{};check(cudaMemcpy(&state,view.status,sizeof(state),cudaMemcpyDeviceToHost));
    require(state.memoryFrames==0 && !state.memoryReady,"episode reset clears memory telemetry");
    auto decay=std::make_unique<DenseBlockPersistent>(*trained);decay->parameters.memoryGate={};
    for(int h=0;h<DenseHeads;++h) {
        decay->parameters.memoryGate.weight[h][0][18]=-2.252168461f;
        decay->parameters.memoryGate.weight[h][1][18]=-80;
        decay->parameters.memoryGate.gain[h]=1;
    }
    check(cudaMemcpy(view.persistent,decay.get(),sizeof(*decay),cudaMemcpyHostToDevice));
    auto seed=[&] {
        vabpnnResetDenseBlocks(model,nullptr);vabpnnDenseForwardFeatures(model,input.p,200,20'000'000'000,nullptr);check(cudaDeviceSynchronize());
        check(cudaMemcpy(view.memory,previous.data(),pairs*4,cudaMemcpyHostToDevice));
    };
    seed();vabpnnDenseForwardFeatures(model,input.p,201,20'250'000'000,nullptr);
    vabpnnDenseForwardFeatures(model,input.p,202,20'500'000'000,nullptr);check(cudaDeviceSynchronize());const auto two=bytes(view.memory,pairs*4);
    seed();vabpnnDenseForwardFeatures(model,input.p,203,20'500'000'000,nullptr);check(cudaDeviceSynchronize());const auto one=bytes(view.memory,pairs*4);
    const auto* a=reinterpret_cast<const float*>(one.data());const auto* b=reinterpret_cast<const float*>(two.data());
    for(std::size_t i=0;i<pairs;++i)require(std::abs(a[i]-b[i])<2e-7 && std::abs(a[i]-previous[i]*std::exp(-.05))<2e-7,"forgetting follows elapsed seconds, independent of FPS");
    vabpnnDenseForwardFeatures(model,input.p,204,10'020'500'000'000,nullptr);check(cudaDeviceSynchronize());
    auto longGap=bytes(view.memory,pairs*4);const auto* longValues=reinterpret_cast<const float*>(longGap.data());
    for(std::size_t i=0;i<pairs;++i)require(std::isfinite(longValues[i]) && std::abs(longValues[i])<=1,"long gap memory stable");
    std::printf("PASS: %d warm memory/pair/Q/K finite differences (%d nonzero), max error %.3g; independent 116-parameter AdamW.\n",checked,nonzero,maximum);
    std::printf("PASS: frozen weights with evolving memory, duplicate guards, seconds/FPS-independent forgetting, long gap and episode/clock reset.\n");
}
