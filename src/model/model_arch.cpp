/// Architecture abstraction implementations (P-A foundation).
///
/// Qwen2Arch is the reference implementation.  Its weight-name mapping MUST
/// reproduce the exact strings the engine hardcoded before this refactor, so
/// weight loading stays byte-identical for Qwen2.5.
///
/// Factory: a small local static map keyed by canonical HF architecture name.
/// A runtime-constructed spec rules out the no-arg NANOINFER_REGISTER_BACKEND
/// macro (which fits the plugin backends only), so this map is the equivalent
/// registry for architectures.

#include "nanoinfer/model/model_arch.h"

#include <functional>
#include <map>
#include <stdexcept>
#include <utility>

namespace nanoinfer {
namespace model {

// ============================================================================
// make_arch_spec — map a parsed HF ModelConfig into the ArchSpec the layer
// loop reads.  Every value must equal what the engine previously hardcoded.
// ============================================================================

ArchSpec make_arch_spec(const ModelConfig& cfg) {
    ArchSpec s;
    // Numeric shape: mirrors the EngineServer constructor's D_/Hq_/Hkv_/hd_/
    // n_layers_/vocab_ assignments.
    s.hidden_size = cfg.hidden_size;
    s.num_heads = cfg.num_attention_heads;
    s.num_kv_heads = cfg.num_key_value_heads;
    // head_dim = hidden_size / num_heads (== cfg.head_dim as parsed).
    s.head_dim = (cfg.num_attention_heads > 0)
                     ? (cfg.hidden_size / cfg.num_attention_heads)
                     : cfg.head_dim;
    s.num_layers = cfg.num_hidden_layers;
    s.vocab_size = cfg.vocab_size;
    // Numeric constants: direct passthrough from config.
    s.rope_theta = cfg.rope_theta;
    s.rms_norm_eps = cfg.rms_norm_eps;
    // Qwen2 keeps the q/k/v projection biases (reference semantics).
    s.qkv_bias = true;
    // pre-LN block layout (norm first, then attention/MLP with residuals).
    s.pre_norm = true;
    // Untied models (Qwen2.5-7B, tie_word_embeddings=false) carry a separate
    // lm_head.weight.
    s.untied_lm_head = !cfg.tie_word_embeddings;
    return s;
}

// ============================================================================
// Qwen2Arch
// ============================================================================

Qwen2Arch::Qwen2Arch(const ArchSpec& spec) : spec_(spec) {}

const std::string& Qwen2Arch::name() const {
    static const std::string kName = "Qwen2ForCausalLM";
    return kName;
}

std::string Qwen2Arch::layer_weight_name(int layer, const std::string& kind,
                                         const std::string& suffix) const {
    // Reproduces the exact strings the engine previously hardcoded:
    //   model.layers.{layer}.self_attn.{q,k,v,o}_proj.{weight,bias}
    //   model.layers.{layer}.mlp.{gate,up,down}_proj.{weight,bias}
    const std::string prefix = "model.layers." + std::to_string(layer);
    if (kind == "q" || kind == "k" || kind == "v" || kind == "o")
        return prefix + ".self_attn." + kind + "_proj." + suffix;
    if (kind == "gate" || kind == "up" || kind == "down")
        return prefix + ".mlp." + kind + "_proj." + suffix;
    throw std::runtime_error(
        "Qwen2Arch: unknown layer weight kind '" + kind + "'");
}

std::string Qwen2Arch::layer_norm_name(int layer,
                                       const std::string& kind) const {
    // Reproduces the exact strings the engine previously hardcoded:
    //   model.layers.{layer}.input_layernorm.weight
    //   model.layers.{layer}.post_attention_layernorm.weight
    const char* name = (kind == "input")
                           ? "input_layernorm"
                           : (kind == "post")
                                 ? "post_attention_layernorm"
                                 : "";
    if (!name[0])
        throw std::runtime_error(
            "Qwen2Arch: unknown layer norm kind '" + kind + "'");
    return "model.layers." + std::to_string(layer) + "." + name + ".weight";
}

std::string Qwen2Arch::embed_name() const { return "model.embed_tokens.weight"; }

std::string Qwen2Arch::final_norm_name() const { return "model.norm.weight"; }

std::string Qwen2Arch::lm_head_name() const { return "lm_head.weight"; }

bool Qwen2Arch::has_qkv_bias() const { return true; }

// ============================================================================
// Architecture registry + factory
// ============================================================================

std::unique_ptr<IModelArchitecture> create_model_architecture(
    const std::string& arch_name, const ArchSpec& spec) {
    // Empty config architecture defaults to the reference Qwen2 path.
    const std::string name = arch_name.empty() ? "Qwen2ForCausalLM" : arch_name;

    // Small local static map keyed by canonical HF architecture name.  The
    // factory takes the parsed ArchSpec, which the no-arg plugin-registry
    // macro cannot express.
    using Factory = std::function<std::unique_ptr<IModelArchitecture>(
        const ArchSpec&)>;
    static const std::map<std::string, Factory> factories = {
        {"Qwen2ForCausalLM",
         [](const ArchSpec& s) { return std::make_unique<Qwen2Arch>(s); }},
    };

    auto it = factories.find(name);
    if (it == factories.end()) {
        std::string known;
        for (const auto& kv : factories) {
            if (!known.empty()) known += ", ";
            known += kv.first;
        }
        throw std::runtime_error(
            "EngineServer: unknown architecture '" + arch_name
            + "' (registered: " + (known.empty() ? "<none>" : known) + ")");
    }
    return it->second(spec);
}

}  // namespace model
}  // namespace nanoinfer
