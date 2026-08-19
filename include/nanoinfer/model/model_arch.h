#pragma once
/// Architecture abstraction (P-A foundation).
///
/// IModelArchitecture isolates everything about a model family that differs
/// between architectures: the HF weight tensor-name mapping (so the weight
/// loader needs no hardcoded "model.layers.i.self_attn.q_proj.weight" strings)
/// and the ArchSpec describing the transformer block's numerics.
///
/// Qwen2Arch is the reference implementation and MUST reproduce the engine's
/// previous hardcoded Qwen2 weight names byte-for-byte, so weight loading is
/// unchanged for Qwen2.5.
///
/// Factory note: unlike the plugin backends, an architecture is constructible
/// only at runtime with the parsed ArchSpec, so the no-arg
/// NANOINFER_REGISTER_BACKEND macro does not fit.  create_model_architecture
/// uses a small local static factory map keyed by arch name
/// (std::function<unique_ptr<IModelArchitecture>(const ArchSpec&)>).

#include <memory>
#include <string>

#include "nanoinfer/model/arch_spec.h"

namespace nanoinfer {
namespace model {

class IModelArchitecture {
public:
    virtual ~IModelArchitecture() = default;

    /// The architecture's numeric spec (norm/PE/MLP/heads/eps/theta/...).
    virtual const ArchSpec& spec() const = 0;

    /// Canonical HF architecture name (e.g. "Qwen2ForCausalLM").
    virtual const std::string& name() const = 0;

    /// HF weight tensor name for a per-layer weight.
    ///   kind ∈ {q,k,v,o,gate,up,down}, suffix ∈ {weight,bias}
    /// Qwen2: "model.layers.{layer}.self_attn.{q,k,v,o}_proj.{suffix}"
    ///        "model.layers.{layer}.mlp.{gate,up,down}_proj.{suffix}"
    virtual std::string layer_weight_name(int layer,
                                          const std::string& kind,
                                          const std::string& suffix) const = 0;

    /// HF weight tensor name for a per-layer normalization weight.
    ///   kind ∈ {input, post}  (input_layernorm / post_attention_layernorm)
    virtual std::string layer_norm_name(int layer,
                                        const std::string& kind) const = 0;

    /// HF weight tensor name of the token embedding ("model.embed_tokens.weight").
    virtual std::string embed_name() const = 0;

    /// HF weight tensor name of the final norm ("model.norm.weight").
    virtual std::string final_norm_name() const = 0;

    /// HF weight tensor name of the LM head ("lm_head.weight"; "" when tied).
    virtual std::string lm_head_name() const = 0;

    /// Whether the q/k/v projections carry bias tensors (Qwen2: yes).
    virtual bool has_qkv_bias() const = 0;
};

/// Reference architecture — structurally identical HF weight names for Qwen2.
class Qwen2Arch : public IModelArchitecture {
public:
    explicit Qwen2Arch(const ArchSpec& spec);

    const ArchSpec& spec() const override { return spec_; }
    const std::string& name() const override;  // "Qwen2ForCausalLM"

    std::string layer_weight_name(int layer, const std::string& kind,
                                  const std::string& suffix) const override;
    std::string layer_norm_name(int layer, const std::string& kind) const override;
    std::string embed_name() const override;      // "model.embed_tokens.weight"
    std::string final_norm_name() const override; // "model.norm.weight"
    std::string lm_head_name() const override;    // "lm_head.weight"
    bool has_qkv_bias() const override;           // true

private:
    ArchSpec spec_;
};

/// Create an IModelArchitecture by canonical HF architecture name.
///   - empty name or "Qwen2ForCausalLM" -> Qwen2Arch (the reference).
///   - unknown name -> throws std::runtime_error listing the registered names.
std::unique_ptr<IModelArchitecture> create_model_architecture(
    const std::string& arch_name, const ArchSpec& spec);

}  // namespace model
}  // namespace nanoinfer
