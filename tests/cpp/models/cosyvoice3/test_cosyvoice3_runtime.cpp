// SPDX-License-Identifier: Apache-2.0
#include "runtime/models/cosyvoice3/pipeline.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace trtmc;
using namespace trtmc::cosyvoice3;
namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
template<class F> void rejects(F f, const char* message) {
    try { f(); check(false, message); } catch (const std::runtime_error&) {}
}
class Tokenizer final : public ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& s) const override {
        return s.find("<|endofprompt|>") == std::string::npos ? std::vector<int32_t>{42}
            : std::vector<int32_t>{13,151646};
    }
    std::string decode(const std::vector<int32_t>&) const override { return ""; }
    int32_t id_for_token(std::string_view) const override { return 151646; }
    std::string token_for_id(int32_t) const override { return ""; }
};
struct Trace { int live{0}, peak{0}, llm{0}, flow{0}; std::vector<float> noise; };
class FakeModule final : public ITrtModule {
    std::string stage_;
    Trace& trace_;
    std::unordered_map<std::string, std::vector<float>> storage_;
    std::unordered_map<std::string, std::vector<int64_t>> shapes_;
  public:
    FakeModule(std::string stage, Trace& trace) : stage_(std::move(stage)), trace_(trace) {
        ++trace_.live; trace_.peak = std::max(trace_.peak, trace_.live);
        if (stage_ == "llm") shapes_ = {{"ids",{1,512}}, {"positions",{512}}, {"mask",{1,1,512,1024}},
                                        {"keys",{24,2,511,64}}, {"values",{24,2,511,64}}};
        if (stage_ == "conditioning") shapes_ = {{"tokens",{1,128}}, {"speaker",{1,192}}};
        if (stage_ == "flow") shapes_ = {{"x",{2,80,256}}, {"mask",{2,1,256}}, {"mu",{2,80,256}},
                                         {"t",{2}}, {"spks",{2,80}}, {"cond",{2,80,256}}, {"positions",{256}}};
        if (stage_ == "hift") shapes_ = {{"mel",{1,80,256}}, {"noise",{1,9,122880}}};
    }
    ~FakeModule() override { --trace_.live; }
    TensorMap forward(const TensorMap& in) override {
        TensorMap result;
        auto out = [&](const std::string& name, std::vector<int64_t> shape, float value) {
            size_t count = 1; for (int64_t d : shape) count *= d;
            auto& data = storage_[name]; data.assign(count, value);
            result[name] = {data.data(), shape, DType::kFloat32};
        };
        if (stage_ == "llm") {
            int step = trace_.llm++ % 3;
            int past = in.at("keys").shape[2], n = in.at("ids").shape[1], total = past + n;
            check(step ? past > 0 : past == 0, "request-local compact cache");
            auto mask = static_cast<uint8_t*>(in.at("mask").data);
            for (int q = 0; q < n; ++q) for (int k = 0; k < total; ++k)
                check(mask[q * total + k] == (k <= past + q), "causal BOOL mask");
            out("logits", {1,1,6761}, -1000);
            storage_["logits"][step == 2 ? 6562 : 7 + step] = 100;
            out("present_keys", {24,2,total,64}, float(step));
            out("present_values", {24,2,total,64}, float(step));
        } else if (stage_ == "conditioning") {
            check(in.at("tokens").shape[1] == 3, "prompt and generated tokens concatenated");
            out("mu", {1,80,6}, 2); out("spks", {1,80}, 3);
        } else if (stage_ == "flow") {
            int step = trace_.flow++ % 10;
            auto x = static_cast<float*>(in.at("x").data);
            if (!step) trace_.noise.assign(x, x + 480);
            for (int i = 0; i < 480; ++i) {
                check(x[i] == x[480+i], "CFG rows share state");
                check(static_cast<float*>(in.at("mu").data)[480+i] == 0, "unconditional mu zero");
                check(static_cast<float*>(in.at("cond").data)[480+i] == 0, "unconditional prompt zero");
            }
            auto mask = static_cast<float*>(in.at("mask").data);
            check(std::all_of(mask, mask + 12, [](float v) { return v == 1; }), "both CFG masks retained");
            out("velocity", {2,80,6}, 0);
            std::fill_n(storage_["velocity"].begin(), 480, 1.f);
        } else {
            check(in.at("mel").shape == std::vector<int64_t>({1,80,4}), "prompt frames removed");
            auto mel = static_cast<float*>(in.at("mel").data);
            for (int c = 0; c < 80; ++c) for (int f = 0; f < 4; ++f)
                check(std::abs(mel[c*4+f] - (trace_.noise[c*6+2+f] + 1.7f)) < 2e-6,
                      "ten-step guided Euler analytical oracle");
            auto noise = static_cast<float*>(in.at("noise").data);
            check(std::all_of(noise, noise + in.at("noise").numel(), [](float v) { return v >= 0 && v < 1; }),
                  "HiFT uniform noise contract");
            out("audio", {1,1920}, .1f);
        }
        return result;
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { throw std::runtime_error("unused"); }
    void forward_device_async(const DeviceTensorMap&) override { throw std::runtime_error("unused"); }
    void forward_async(const TensorMap&) override { throw std::runtime_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return std::vector<TensorInfo>(shapes_.size()); }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override { return shapes_.count(name); }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string& name) const override {
        if (name == "ids" || name == "positions" || name == "tokens") return DType::kInt32;
        return stage_ == "llm" && name == "mask" ? DType::kBool : DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override { return shapes_.at(name); }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t, ProfileShapeSelector selector) const override {
        auto shape = shapes_.at(name);
        if (selector == ProfileShapeSelector::kMin) std::fill(shape.begin(), shape.end(), 0);
        return shape;
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};
}
int main(int argc, char** argv) {
    if (argc == 2) {
        std::ifstream input(argv[1]);
        nlohmann::json fixture; input >> fixture;
        std::string serialized = fixture.at("tokenizer");
        auto tokenizer = CreateBpeTokenizer(serialized.data(), serialized.size(), false);
        for (const auto& item : fixture.at("cases")) {
            Settings settings; settings.transcript = item.at("transcript");
            Voice voice; voice.tokens = item.at("speech").get<std::vector<int32_t>>();
            check(pack(*tokenizer, settings, voice, item.at("text")) == item.at("packed").get<std::vector<int32_t>>(),
                  "native Qwen BPE and Python packed prompt agree");
        }
    }
    check(dtype_size(DType::kBool) == 1, "BOOL occupies one byte");
    check(static_cast<int>(DType::kInt8) == 4, "existing dtype enum ABI preserved");
    std::vector<float> logits(6761, -1000.f);
    std::function<double()> draw = [] { return 0.0; };
    logits[20] = 1; logits[10] = 1;
    check(cosyvoice3::sample(logits, {}, 0, true, draw) == 10, "greedy stable tie");
    check(cosyvoice3::sample(logits, {}, 0, false, draw) == 10, "RAS stable tie");
    check(cosyvoice3::sample(logits, {10}, 0, false, draw) == 20, "repeat redraw excludes picked token");
    logits[6561] = 100; logits[6562] = 90;
    check(cosyvoice3::sample(logits, {}, 1, true, draw) == 6562, "min length masks only SOS, not other stops");
    check(cosyvoice3::sample(logits, {}, 0, true, draw) == 6561, "SOS allowed after minimum");
    logits[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { cosyvoice3::sample(logits, {}, 0, true, draw); }, "reject NaN");
    rejects([&] { cosyvoice3::sample({}, {}, 0, true, draw); }, "reject wrong vocabulary");
    logits[0] = 0;
    rejects([&] { cosyvoice3::sample(logits, {}, 0, false, [] { return 1.; }); }, "reject invalid draw");
    Tokenizer tokenizer; Settings settings; Voice voice{{8}, std::vector<float>(160), std::vector<float>(192,1)};
    check(pack(tokenizer, settings, voice, "Hello") == std::vector<int32_t>({158497,13,151646,42,158499}),
          "instruction-mode prompt does not include speech prefix");
    settings.transcript = "Reference";
    check(pack(tokenizer, settings, voice, "Hello").back() == 151944, "zero-shot speech offset");
    rejects([&] { pack(tokenizer, settings, voice, "<tag>"); }, "control tags rejected");
    rejects([&] { pack(tokenizer, settings, voice, "  "); }, "empty text rejected");
    Trace trace;
    Pipeline pipeline(settings, voice, std::make_unique<Tokenizer>(),
                      [&](const std::string& name) { return std::make_unique<FakeModule>(name, trace); });
    GenerateConfig cfg; cfg.max_new_tokens = 5;
    auto audio = pipeline.generate_audio("Hello", cfg);
    check(audio.sample_rate == 24000 && audio.num_samples == 1920, "audio output contract");
    auto second = pipeline.generate_audio("Hello", cfg);
    check(audio.samples == second.samples && trace.llm == 6 && trace.flow == 20, "repeat requests reset state");
    check(trace.peak == 1 && trace.live == 0, "only one stage engine alive, all released");
    cfg.max_new_tokens = 1;
    rejects([&] { pipeline.generate_audio("Hello", cfg); }, "truncated generation rejected");
    check(trace.live == 0, "engine released after exception");
    voice.speaker.assign(192, 0);
    rejects([&] { Pipeline p(settings, voice, std::make_unique<Tokenizer>(),
                             [](const std::string&) -> std::unique_ptr<ITrtModule> { return {}; }); },
            "placeholder voice rejected");
    std::cout << "CosyVoice3 runtime checks: " << failures << " failures\n";
    return failures ? 1 : 0;
}
