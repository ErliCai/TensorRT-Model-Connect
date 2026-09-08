// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"
#include <functional>
#include <random>

namespace trtmc::cosyvoice3 {

// Fixed prepared voice; reference WAV feature extraction is a separate build step.
struct Voice {
    std::vector<int32_t> tokens;
    std::vector<float> features; // [2 * tokens, 80]
    std::vector<float> speaker;  // [192]
};
struct Settings {
    std::string model_id, instruction{"You are a helpful assistant."}, transcript;
    int max_context{512}, max_tokens{100};
    bool greedy{false};
};
using ModuleFactory = std::function<std::unique_ptr<ITrtModule>(const std::string&)>;

// Explicit draws make sampling testable independently of a particular RNG library.
int sample(const std::vector<float>& logits, const std::vector<int32_t>& history,
           int minimum, bool greedy, const std::function<double()>& draw);
std::vector<int32_t> pack(const ITokenizer&, const Settings&, const Voice&, const std::string&);

class Pipeline final : public IPipeline {
  public:
    Pipeline(Settings settings, Voice voice, std::unique_ptr<ITokenizer> tokenizer,
             ModuleFactory factory);
    AudioResult generate_audio(const std::string&, const GenerateConfig& = {}) override;
    const char* model_id() const override { return settings_.model_id.c_str(); }
    const char* pipeline_type() const override { return "text_to_audio_cosyvoice3"; }
  private:
    Settings settings_;
    Voice voice_;
    std::unique_ptr<ITokenizer> tokenizer_;
    ModuleFactory factory_;
};
} // namespace trtmc::cosyvoice3
