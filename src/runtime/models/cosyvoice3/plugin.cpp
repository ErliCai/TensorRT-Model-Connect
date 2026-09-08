// SPDX-License-Identifier: Apache-2.0
#include "pipeline.h"
#include "bundle/bundle_format.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "utils/sha256.h"
#include <nlohmann/json.hpp>

namespace trtmc {
namespace {
class CosyVoice3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto config = nlohmann::json::parse(ctx.config_json);
        if (config.at("cosyvoice3_schema") != 1 || config.at("precision") != "fp32" || !ctx.backend)
            throw std::runtime_error("Unsupported CosyVoice3 bundle");
        auto c = config.at("cosyvoice3");
        cosyvoice3::Settings settings;
        settings.model_id = ctx.bundle.info.model_id;
        settings.instruction = c.at("instruction"); settings.transcript = c.at("transcript");
        settings.max_context = c.at("max_context"); settings.max_tokens = c.at("max_tokens");
        settings.greedy = c.at("greedy");
        cosyvoice3::Voice voice{c.at("prompt_tokens").get<std::vector<int32_t>>(),
            c.at("prompt_features").get<std::vector<float>>(), c.at("speaker").get<std::vector<float>>()};
        auto header = ReadBundleHeader(ctx.bundle_path);
        auto section = [&](const std::string& name) {
            for (const auto& entry : header.sections)
                if (entry.name == name) return entry;
            throw std::runtime_error("Missing CosyVoice3 bundle section: " + name);
        };
        auto tokenizer_bytes = ReadBundleSection(ctx.bundle_path, section("tokenizer.json"));
        auto tokenizer = CreateBpeTokenizer(tokenizer_bytes.data(), tokenizer_bytes.size(), false);
        // BackendLoader retains backends until process shutdown. Never capture ctx or its references.
        auto factory = [path = ctx.bundle_path, header, hashes = c.at("plan_sha256"), backend = ctx.backend]
                       (const std::string& name) {
            for (const auto& entry : header.sections) {
                if (entry.name != name + ".plan") continue;
                auto bytes = ReadBundleSection(path, entry);
                internal::Sha256 digest; digest.update(bytes.data(), bytes.size());
                if (digest.hex_digest() != hashes.at(name).get<std::string>())
                    throw std::runtime_error("CosyVoice3 plan checksum mismatch: " + name);
                auto module = backend->create_module(bytes.data(), bytes.size(), ModuleCreateOptions{});
                if (!module || !module->ok()) throw std::runtime_error("Cannot load CosyVoice3 " + name);
                return module;
            }
            throw std::runtime_error("Missing CosyVoice3 plan: " + name);
        };
        return std::make_unique<cosyvoice3::Pipeline>(std::move(settings), std::move(voice),
                                                     std::move(tokenizer), std::move(factory));
    }
};
} // namespace
REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_cosyvoice3_plugin, CosyVoice3Plugin, "text_to_audio_cosyvoice3");
} // namespace trtmc
