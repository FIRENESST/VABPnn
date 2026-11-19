#pragma once
// These checks reuse the independent CPU Transformer reference above.
double probability(double x){return 1/(1+std::exp(-x));}
double positive(double x){return x>20 ? x:std::log1p(std::exp(x));}
double policyLoss(const Reference& ref,const DecisionParameters& p,std::uint32_t label,
                  DecisionOutput* result=nullptr) {
    double post[VoteCount],threshold[VoteCount],gate[VoteCount],feature[DecisionKFeatures]{};
    double loss=0;const double temperature=positive(p.logGateTemperature);
    for(int a=0;a<VoteCount;++a) {
        double evidence=p.logLikelihoodBias[a],t=p.thresholdBias[a];
        for(int d=0;d<DenseWidth;++d) {
            evidence+=ref.hidden[a*DenseWidth+d]*p.logLikelihoodWeight[a][d];
            t+=ref.hidden[a*DenseWidth+d]*p.thresholdWeight[a][d];
            feature[d]+=ref.hidden[a*DenseWidth+d]/VoteCount;
        }
        post[a]=probability(evidence);threshold[a]=probability(t);
        gate[a]=probability((post[a]-threshold[a])/temperature);
        const double y=(label>>a)&1u,target=y ? .25:.75;
        loss+=(-y*std::log(post[a])-(1-y)*std::log1p(-post[a])
               -target*std::log(threshold[a])-(1-target)*std::log1p(-threshold[a]))/VoteCount;
        feature[DenseWidth+a]=post[a];feature[DenseWidth+VoteCount+a]=gate[a];
        if(result) {
            result->posterior[a]=static_cast<float>(post[a]);result->threshold[a]=static_cast<float>(threshold[a]);
            result->softEligibility[a]=static_cast<float>(gate[a]);
        }
    }
    double logits[DecisionKChoices],maximum=-INFINITY,sum=0;
    for(int k=0;k<DecisionKChoices;++k) {
        double v=p.kBias[k];for(int d=0;d<DecisionKFeatures;++d)v+=feature[d]*p.kWeight[k][d];
        logits[k]=v;maximum=std::max(maximum,v);
    }
    for(int k=0;k<DecisionKChoices;++k)sum+=std::exp(logits[k]-maximum);
    const int count=static_cast<int>(__popcnt(label));
    loss+=-logits[count]+maximum+std::log(sum);
    if(result)for(int k=0;k<DecisionKChoices;++k)result->kProbability[k]=static_cast<float>(std::exp(logits[k]-maximum)/sum);
    return loss;
}
void verifyPolicyAndClocks(VabpnnDenseBlockModel* model,Device<float>& input,Device<float>& target,
                           Device<int>& active,const std::vector<float>& features,const std::vector<float>& linearGradient) {
    const auto view=vabpnnDenseBlockModelView(model);const int n=view.nodes,on=1;active.set(&on);
    vabpnnResetDenseBlocks(model,nullptr);vabpnnDenseForwardFeatures(model,input.p,10,3'000'000'000,nullptr);
    check(cudaDeviceSynchronize());auto state=std::make_unique<DenseBlockPersistent>();
    check(cudaMemcpy(state.get(),view.persistent,sizeof(*state),cudaMemcpyDeviceToHost));
    std::vector<float> bias(DenseHeads*n*n);check(cudaMemcpy(bias.data(),view.graphBias,bias.size()*4,cudaMemcpyDeviceToHost));
    auto base=std::make_unique<DecisionParameters>();*base=DecisionParameters{};
    for(int a=0;a<VoteCount;++a)for(int d=0;d<DenseWidth;++d) {
        base->logLikelihoodWeight[a][d]=.2f*std::sin(.71f*(a*DenseWidth+d+1));
        base->thresholdWeight[a][d]=.1f*std::cos(.31f*(a*DenseWidth+d+1));
    }
    for(int k=0;k<DecisionKChoices;++k)for(int d=0;d<DecisionKFeatures;++d)
        base->kWeight[k][d]=.1f*std::sin(.113f*(k*DecisionKFeatures+d+1));
    base->logGateTemperature=-1.2f;
    const std::uint32_t label=(1u<<0)|(1u<<1)|(1u<<5);
    const auto ref=reference(state->parameters,bias,features,n);
    DecisionOutput out{};out.valid=1;out.frameId=10;policyLoss(ref,*base,label,&out);
    Device<DecisionParameters> gpuBase(1);Device<DecisionOutput> gpuOutput(1);Device<std::uint32_t> gpuLabel(1);
    gpuBase.set(base.get());gpuOutput.set(&out);gpuLabel.set(&label);
    vabpnnTrainDenseBlocks(model,gpuBase.p,gpuOutput.p,active.p,gpuLabel.p,nullptr);check(cudaDeviceSynchronize());
    std::vector<float> graphGradient(bias.size());check(cudaMemcpy(graphGradient.data(),view.graphGradient,bias.size()*4,cudaMemcpyDeviceToHost));
    auto gradient=std::make_unique<DenseBlockParameters>();check(cudaMemcpy(gradient.get(),view.gradient,sizeof(*gradient),cudaMemcpyDeviceToHost));
    constexpr float eps=.002f;double maximum=0;
    for(std::size_t i=0;i<bias.size();++i) {
        const float old=bias[i];bias[i]=old+eps;const double plus=policyLoss(reference(state->parameters,bias,features,n),*base,label);
        bias[i]=old-eps;const double minus=policyLoss(reference(state->parameters,bias,features,n),*base,label);bias[i]=old;
        const double expected=(plus-minus)/(2*eps),error=std::abs(expected-graphGradient[i]);
        maximum=std::max(maximum,error);require(error<4e-5+std::abs(expected)*.004,"Bayes/threshold/learned-k pair gradient");
    }
    const std::size_t indices[]={3,offsetof(DenseBlockParameters,input)/4+17,offsetof(DenseBlockParameters,query)/4+19,
        offsetof(DenseBlockParameters,key)/4+11,offsetof(DenseBlockParameters,value)/4+12,
        offsetof(DenseBlockParameters,attentionOutput)/4+9,offsetof(DenseBlockParameters,attentionGain)/4+3,
        offsetof(DenseBlockParameters,attentionBias)/4+7,offsetof(DenseBlockParameters,up)/4+8,
        offsetof(DenseBlockParameters,upBias)/4+5,offsetof(DenseBlockParameters,down)/4+11,
        offsetof(DenseBlockParameters,downBias)/4+9,offsetof(DenseBlockParameters,outputGain)/4+2,
        offsetof(DenseBlockParameters,outputBias)/4+11,offsetof(DenseBlockParameters,readout)/4+13};
    auto* parameters=reinterpret_cast<float*>(&state->parameters);
    for(auto i:indices) {
        const float old=parameters[i];parameters[i]=old+eps;const double plus=policyLoss(reference(state->parameters,bias,features,n),*base,label);
        parameters[i]=old-eps;const double minus=policyLoss(reference(state->parameters,bias,features,n),*base,label);parameters[i]=old;
        const double expected=(plus-minus)/(2*eps),error=std::abs(expected-reinterpret_cast<float*>(gradient.get())[i]);
        maximum=std::max(maximum,error);require(error<5e-5+std::abs(expected)*.004,"Bayesian action loss backprop reaches full Transformer");
    }
    const auto frozen=bytes(view.persistent,sizeof(DenseBlockPersistent)),frozenPairs=bytes(view.graphBias,bias.size()*4);
    const auto frozenM=bytes(view.first,bias.size()*4),frozenV=bytes(view.second,bias.size()*4);
    out.frameId=999;gpuOutput.set(&out);vabpnnTrainDenseBlocks(model,gpuBase.p,gpuOutput.p,active.p,gpuLabel.p,nullptr);
    check(cudaDeviceSynchronize());
    require(frozen==bytes(view.persistent,sizeof(DenseBlockPersistent)) && frozenPairs==bytes(view.graphBias,bias.size()*4)
        && frozenM==bytes(view.first,bias.size()*4) && frozenV==bytes(view.second,bias.size()*4),"stale policy frame cannot update weights or AdamW");
    std::printf("PASS: actual Bayes + auxiliary threshold + posterior/gate-dependent learned-k backprop; stale-frame guard; max error %.3g.\n",maximum);
    vabpnnResetDenseBlocks(model,nullptr);Device<std::int32_t> votes(n*VoteCount);
    std::vector<std::int32_t> first(n*VoteCount),second(first.size());
    for(std::size_t i=0;i<first.size();++i){first[i]=-128+static_cast<int>(i*7%128);second[i]=static_cast<int>(i*11%200);}
    VabpnnGpuFrame frame{};frame.frameId=201;frame.captureNanos=5'000'000'000;votes.set(first.data());
    vabpnnLaunchDenseBlocks(model,frame,votes.p,nullptr);
    frame.frameId=202;frame.captureNanos=7'000'000'000;votes.set(second.data());vabpnnLaunchDenseBlocks(model,frame,votes.p,nullptr);
    check(cudaDeviceSynchronize());std::vector<float> encoded(n*DenseFeatures);
    check(cudaMemcpy(encoded.data(),view.features,encoded.size()*4,cudaMemcpyDeviceToHost));
    for(int node=0;node<n;++node)for(int k=0;k<VoteCount;++k) {
        require(encoded[node*DenseFeatures+k]==second[k*n+node]/256.f,"current per-block signed vote input");
        require(encoded[node*DenseFeatures+VoteCount+k]==first[k*n+node]/256.f,"causal previous block vote input");
        for(int age=2;age<DenseVoteHistory;++age)require(encoded[node*DenseFeatures+age*VoteCount+k]==0,"missing history is neutral");
    }
    vabpnnDenseBackwardHidden(model,target.p,active.p,nullptr);check(cudaDeviceSynchronize());
    check(cudaMemcpy(state.get(),view.persistent,sizeof(*state),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(gradient.get(),view.gradient,sizeof(*gradient),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(bias.data(),view.graphBias,bias.size()*4,cudaMemcpyDeviceToHost));
    for(int minute=0;minute<2;++minute) {
        float& parameter=minute ? state->parameters.logMinutesFrequency:state->parameters.logSecondsFrequency;
        const float old=parameter;double side[2];
        for(int dir=0;dir<2;++dir) {
            parameter=old+(dir ? -eps:eps);auto perturbed=encoded;const double elapsed=minute ? 2./60:2.;
            const int column=VoteCount*DenseVoteHistory+(minute ? 5:1);
            for(int node=0;node<n;++node) {
                perturbed[node*DenseFeatures+column]=static_cast<float>(std::sin(elapsed*positive(parameter)));
                perturbed[node*DenseFeatures+column+1]=static_cast<float>(std::cos(elapsed*positive(parameter)));
            }
            side[dir]=loss(reference(state->parameters,bias,perturbed,n,{},2),linearGradient);
        }
        parameter=old;const double expected=(side[0]-side[1])/(2*eps);
        const float actual=minute ? gradient->logMinutesFrequency:gradient->logSecondsFrequency;
        require(std::abs(expected-actual)<3e-5+std::abs(expected)*.005,"production seconds/minutes clock gradient");
        require(std::abs(actual)>1e-8,"both clock frequencies have real gradients");
        std::printf("PASS: production %s clock gradient expected=%.7g actual=%.7g.\n",minute ? "minute":"second",expected,actual);
    }
    std::printf("PASS: real per-block 8-frame signed vote history, missing-slot padding, seconds/minutes feature generation and trainable clock gradients.\n");
}
