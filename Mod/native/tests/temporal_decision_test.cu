#include "TemporalDecision.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace vabpnn;
using Votes = std::array<std::int32_t, VoteCount>;
static void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
static void require(bool value, const char* label) {
    if (!value) throw std::runtime_error(label);
}
static void near(double actual, double expected, const char* label, double tolerance = 4e-6) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr, "%s: got %.9g, expected %.9g\n", label, actual, expected);
        throw std::runtime_error(label);
    }
}
static float logit(double probability) { return static_cast<float>(std::log(probability / (1 - probability))); }
static float inverseSoftplus(double rate) { return static_cast<float>(std::log(std::expm1(rate))); }
static double bayes(double prior, double logRatio) {
    // Independent probability-space reference: L1/L0 = exp(logRatio).
    const double ratio = std::exp(logRatio);
    return prior * ratio / (prior * ratio + 1 - prior);
}
static DecisionParameters fixture() {
    DecisionParameters p{};
    for (int d = 0; d < DecisionWidth; ++d) p.attentionGain[d] = p.outputGain[d] = 1;
    for (int i = 0; i < VoteCount; ++i) {
        p.logOnRate[i] = inverseSoftplus(0.3);
        p.logOffRate[i] = inverseSoftplus(0.7);
        p.thresholdBias[i] = logit(0.99);
    }
    p.logGateTemperature = inverseSoftplus(0.1);
    return p;
}
static void chooseK(DecisionParameters& p, int k) {
    for (int j = 0; j < DecisionKChoices; ++j) p.kBias[j] = 0;
    p.kBias[k] = 2;
}
static void upload(DecisionParameters* target, const DecisionParameters& parameters, cudaStream_t stream) {
    check(cudaMemcpyAsync(target, &parameters, sizeof(parameters), cudaMemcpyHostToDevice, stream));
}
static DecisionOutput run(VabpnnDecisionModel* model, std::int32_t* deviceVotes,
                          cudaStream_t stream, const Votes& votes, std::uint64_t nanos, std::uint64_t id, bool continuous=false) {
    check(cudaMemcpyAsync(deviceVotes, votes.data(), sizeof(votes), cudaMemcpyHostToDevice, stream));
    VabpnnGpuFrame frame{};
    frame.stream = stream; frame.captureNanos = nanos; frame.frameId = id; frame.evaluationMode=continuous;
    vabpnnLaunchTemporalDecision(frame, deviceVotes, model);
    DecisionOutput result{};
    check(cudaMemcpyAsync(&result, vabpnnDecisionModelView(model).output, sizeof(result), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    require(result.command.frameId == id && result.command.captureNanos == nanos, "Control metadata belongs to current frame");
    require(result.command.selectedMask == result.selectedMask, "Control mask matches GPU decision");
    require(result.status.frameId==id && result.status.eligibleMask==result.eligibleMask
        && result.status.requestedK==result.requestedK && result.status.selectedCount==result.selectedCount
        && result.status.valid==result.valid,"Diagnostic counters match the completed GPU decision");
    if (result.valid) {
        double total = 0;
        for (float value : result.kProbability) total += value;
        near(total, 1, "Learned k distribution is normalized");
        for (int i = 0; i < VoteCount; ++i) near(result.confidence[i], result.posterior[i], "Confidence is posterior");
    }
    return result;
}
static std::array<double, DecisionWidth> normalize(std::array<double, DecisionWidth> input) {
    double mean = 0, variance = 0;
    for (double x : input) mean += x;
    mean /= input.size();
    for (double x : input) variance += (x - mean) * (x - mean);
    const double scale = 1 / std::sqrt(variance / input.size() + 1e-5);
    for (double& x : input) x = (x - mean) * scale;
    return input;
}

int main() {
    cudaStream_t stream = nullptr;
    VabpnnDecisionModel* model = nullptr;
    std::int32_t* deviceVotes = nullptr;
    int exitCode = 0;
    try {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        model = vabpnnCreateDecisionModel(stream);
        check(cudaMalloc(reinterpret_cast<void**>(&deviceVotes), sizeof(Votes)));
        auto* parameters = vabpnnMutableDecisionParameters(model, stream);
        const auto originalView = vabpnnDecisionModelView(model);
        const auto bytes = vabpnnDecisionModelBytes(model);
        require(bytes > sizeof(DecisionParameters) && bytes < 1'000'000, "Bounded decision memory");
        Votes votes{}; votes[0] = -5; votes[1] = 655360; votes[2] = 1966080;
        auto result = run(model, deviceVotes, stream, votes, 1'000'000'000ull, 1);
        require(result.valid && !result.eligibleMask && !result.selectedMask && result.requestedK == 0, "Neutral untrained model stays idle");
        for (int i = 0; i < VoteCount; ++i) {
            near(result.prior[i], .5, "Neutral prior"); near(result.posterior[i], .5, "Neutral posterior");
            near(result.threshold[i], .5, "Neutral threshold");
        }
        std::puts("PASS: neutral graph/time model, confidence=posterior, strict equality rejected, k=0.");

        auto p = fixture();
        p.initialPriorLogit[0] = logit(.2); p.logLikelihoodBias[0] = std::log(4.0f);
        p.initialPriorLogit[1] = logit(.8); p.logLikelihoodBias[1] = std::log(.25f);
        p.thresholdBias[0] = p.thresholdBias[1] = logit(.6);
        p.logLikelihoodBias[2] = std::log(9.0f); p.thresholdBias[2] = logit(.8);
        p.logLikelihoodBias[3] = std::log(3.0f); p.thresholdBias[3] = logit(.7);
        p.thresholdBias[4] = 0; // Exact .5 posterior == .5 threshold must fail.
        p.logLikelihoodBias[5] = std::log(4.0f); p.thresholdBias[5] = logit(.7);
        chooseK(p, 2);
        upload(parameters, p, stream);
        vabpnnResetDecisionState(model, stream);
        const MotionProposal motion{3.25f, -1.5f, 2};
        check(cudaMemcpyAsync(vabpnnMutableMotionProposal(model, stream), &motion, sizeof(motion), cudaMemcpyHostToDevice, stream));
        result = run(model, deviceVotes, stream, votes, 1'000'000'000ull, 2);
        const auto previous = result;
        for (int i = 0; i < VoteCount; ++i) {
            const double prior = i == 0 ? .2 : i == 1 ? .8 : .5;
            near(result.posterior[i], bayes(prior, p.logLikelihoodBias[i]), "Bayes matches independent reference");
        }
        require(result.selectedMask == ((1u << 2) | (1u << 5)) && result.selectedIndices[0] == 2
                && result.selectedIndices[1] == 5 && result.selectedCount == 2, "Top-k ranks only eligible nodes");
        require(!(result.eligibleMask & (1u << 4)), "Strict threshold equality");
        near(result.command.mouseDx, 3.25, "Selected X permission");
        near(result.command.mouseDy, 0, "Y permission off"); near(result.command.wheelSteps, 0, "Wheel permission off");
        result = run(model, deviceVotes, stream, votes, 1'250'000'000ull, 3);
        near(result.deltaSeconds, .25, "Seconds from actual timestamp gap");
        for (int i = 0; i < VoteCount; ++i) {
            const double prior = .3 + (previous.posterior[i] - .3) * std::exp(-.25);
            near(result.prior[i], prior, "Two-state CTMC prior");
            near(result.posterior[i], bayes(prior, p.logLikelihoodBias[i]), "CTMC prior plus new evidence");
        }
        std::puts("PASS: Bayes likelihood ratios/nonuniform priors, seconds-based Markov transition and probability ranking.");

        p = fixture(); p.logLikelihoodBias[0] = std::log(9.0f); p.thresholdBias[0] = logit(.95);
        p.logLikelihoodBias[2] = std::log(4.0f); p.thresholdBias[2] = logit(.7);
        p.logLikelihoodBias[3] = std::log(3.0f); p.thresholdBias[3] = logit(.7);
        chooseK(p, VoteCount); upload(parameters, p, stream); vabpnnResetDecisionState(model, stream);
        result = run(model, deviceVotes, stream, votes, 1'000'000'000ull, 4);
        require(result.requestedK == VoteCount && result.selectedCount == 2
                && result.selectedMask == ((1u << 2) | (1u << 3)), "High posterior can be excluded; k is bounded by eligibility");
        p.thresholdBias[0] = logit(.8); chooseK(p, 1);
        upload(parameters, p, stream); vabpnnResetDecisionState(model, stream);
        result = run(model, deviceVotes, stream, votes, 1'000'000'000ull, 5);
        require(result.selectedMask == 1 && result.requestedK == 1, "Updated threshold and k heads affect next decision");
        std::puts("PASS: dynamic threshold parameters, learned-k head changes, k=15 capped by candidate count.");

        const std::array<std::array<int, 2>, 4> pairs{{{{7,8}},{{9,10}},{{7,11}},{{8,11}}}};
        for (const auto& pair : pairs) {
            p = fixture(); chooseK(p, 2);
            p.logLikelihoodBias[pair[0]] = std::log(9.0f); p.thresholdBias[pair[0]] = logit(.6);
            p.logLikelihoodBias[pair[1]] = std::log(4.0f); p.thresholdBias[pair[1]] = logit(.6);
            upload(parameters, p, stream); vabpnnResetDecisionState(model, stream);
            result = run(model, deviceVotes, stream, votes, 1'000'000'000ull, 6);
            require(result.selectedCount == 1 && result.selectedMask == (1u << pair[0]), "Declared mutual exclusion");
        }
        for (const auto& pair : std::array<std::array<int,2>,2>{{{{0,2}},{{7,9}}}}) {
            p = fixture(); chooseK(p, 2);
            for (int i : pair) { p.logLikelihoodBias[i] = std::log(9.0f); p.thresholdBias[i] = logit(.6); }
            upload(parameters, p, stream); vabpnnResetDecisionState(model, stream);
            result = run(model, deviceVotes, stream, votes, 1'000'000'000ull, 7);
            require(result.selectedCount == 2, "Other declared pairs are allowed together");
        }
        p = fixture(); chooseK(p, 3);
        for (int i : {5,6,11}) { p.logLikelihoodBias[i] = std::log(9.0f); p.thresholdBias[i] = logit(.6); }
        upload(parameters, p, stream); vabpnnResetDecisionState(model, stream);
        check(cudaMemcpyAsync(vabpnnMutableMotionProposal(model, stream), &motion, sizeof(motion), cudaMemcpyHostToDevice, stream));
        result = run(model, deviceVotes, stream, votes, 1'000'000'000ull, 8);
        near(result.command.mouseDx, motion.mouseDx, "External X motion"); near(result.command.mouseDy, motion.mouseDy, "External Y motion");
        near(result.command.wheelSteps, motion.wheelSteps, "External wheel motion");
        vabpnnClearMotionProposal(model, stream);
        result = run(model, deviceVotes, stream, votes, 1'010'000'000ull, 9);
        near(result.command.mouseDx, 0, "Missing motion is cleared"); near(result.command.wheelSteps, 0, "Old wheel command cannot repeat");
        std::puts("PASS: all four and only four conflict pairs, legal W+S and press+release, independent external motion gates.");

        p=fixture();p.enabledKChoices=DecisionKChoices;chooseK(p,2);
        upload(parameters,p,stream);vabpnnResetDecisionState(model,stream);
        for(int step=0;step<12;++step) {
            result=run(model,deviceVotes,stream,votes,2'000'000'000ull+step*20'000'000ull,50+step,true);
            require(result.valid && result.eligibleMask==((1u<<VoteCount)-1) && result.selectedCount==2,
                "Continuous frozen control emits fresh top-k even with every posterior below threshold");
            for(int k=0;k<VoteCount;++k) require(result.posterior[k]<result.threshold[k],"Fixture is below every confidence threshold");
        }
        chooseK(p,0);upload(parameters,p,stream);
        result=run(model,deviceVotes,stream,votes,2'260'000'000ull,63,true);
        require(result.valid && !result.selectedMask && !result.requestedK,"Learned k=0 is a current rest packet, never forced random input");
        std::puts("PASS: continuous no-threshold frozen selection emits fresh packets; learned rest and conflict rules remain.");

        // Controlled attention: Q=K=0, V/output=identity, no FFN. CPU uses explicit weighted
        // history averages and two layer normalizations, rather than the production kernel.
        p = fixture();
        for (int d = 0; d < DecisionWidth; ++d) p.value[d][d] = p.attentionOutput[d][d] = 1;
        for (int i = 0; i < VoteCount; ++i) { p.nodeEmbedding[i][1] = 1; p.logLikelihoodWeight[i][0] = 1; }
        p.voteEmbedding[0] = 1;
        for (int h = 0; h < DecisionHeads; ++h) p.logAttentionDecay[h] = inverseSoftplus(1);
        chooseK(p, 0); upload(parameters, p, stream); vabpnnResetDecisionState(model, stream);
        std::vector<Votes> history;
        std::vector<std::uint64_t> historyTime;
        double oldPosterior = .5;
        for (int step = 0; step < 12; ++step) {
            if (step == 5) p.value[0][0] = 2; // Old raw history must use new projection weights.
            if (step == 6) for (int h = 0; h < DecisionHeads; ++h) p.graphBias[h][0][0] = std::log(2.0f);
            upload(parameters, p, stream);
            Votes input{}; input[0] = step % 2 ? -983040 : 983040; input[1] = step % 3 ? 491520 : -491520;
            const std::uint64_t nanos = 1'000'000'000ull + step * 500'000'000ull;
            history.push_back(input); historyTime.push_back(nanos);
            if (history.size() > DecisionHistory) { history.erase(history.begin()); historyTime.erase(historyTime.begin()); }
            result = run(model, deviceVotes, stream, input, nanos, 100 + step);
            double context = 0, denominator = 0;
            const double ownWeight = step >= 6 ? 2 : 1;
            for (std::size_t pos = 0; pos < history.size(); ++pos) {
                const double timeWeight = std::exp(-(nanos - historyTime[pos]) * 1e-9);
                for (int j = 0; j < VoteCount; ++j) {
                    const double weight = timeWeight * (j == 0 ? ownWeight : 1);
                    context += weight * static_cast<double>(history[pos][j]) / NeuronCount;
                    denominator += weight;
                }
            }
            std::array<double, DecisionWidth> encoded{};
            encoded[0] = static_cast<double>(input[0]) / NeuronCount + p.value[0][0] * context / denominator;
            encoded[1] = 2;
            const double evidence = normalize(normalize(encoded))[0];
            near(result.logLikelihoodRatio[0], evidence, "Causal graph/time attention reference", 6e-5);
            const double prior = step == 0 ? .5 : .3 + (oldPosterior - .3) * std::exp(-.5);
            const double posterior = bayes(prior, evidence);
            near(result.posterior[0], posterior, "Temporal attention plus Bayesian filtering", 6e-5);
            oldPosterior = posterior;
        }
        result = run(model, deviceVotes, stream, Votes{}, 500'000'000ull, 200);
        near(result.deltaSeconds, 0, "Reversed timestamp resets time origin"); near(result.prior[0], .5, "Episode reset retains weights but clears belief");
        require(vabpnnDecisionModelView(model).parameters == originalView.parameters
                && vabpnnDecisionModelBytes(model) == bytes, "History and parameter updates do not reallocate model");
        Votes unchanged{};
        check(cudaMemcpyAsync(unchanged.data(), deviceVotes, sizeof(unchanged), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        require(unchanged == Votes{}, "Decision never mutates the original int32 votes");
        p.logLikelihoodBias[0] = std::numeric_limits<float>::quiet_NaN();
        upload(parameters, p, stream); vabpnnResetDecisionState(model, stream);
        result = run(model, deviceVotes, stream, votes, 1'000'000'000ull, 201);
        require(!result.valid && !result.selectedMask && !result.command.valid, "Nonfinite model output cannot issue controls");
        std::puts("PASS: causal 8-frame history, seconds decay, learned graph bias, fresh projections after weight updates, episode reset and invalid-output handling.");
        std::printf("PASS: persistent temporal decision allocation %llu bytes; final control command %zu bytes.\n",
                    static_cast<unsigned long long>(bytes), sizeof(ControlCommand));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what()); exitCode = 1;
    }
    if (stream) cudaStreamSynchronize(stream);
    vabpnnDestroyDecisionModel(model);
    if (deviceVotes) cudaFree(deviceVotes);
    if (stream) cudaStreamDestroy(stream);
    return exitCode;
}

