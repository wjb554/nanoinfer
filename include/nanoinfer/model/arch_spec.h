#pragma once
/// Architecture spec (P-A foundation) — architecture-agnostic numeric
/// configuration for a transformer block.
///
/// A parsed HF ModelConfig is mapped into this struct by make_arch_spec().
/// The engine's layer loop (forward_layers_impl) reads ONLY these values where
/// it previously read hardcoded Qwen2 constants, so adding a new architecture
/// (Llama / Mistral / Qwen3) is a config mapping, not an engine edit.
///
/// BEHAVIOR-PRESERVING CONSTRAINT: for Qwen2.5 (the reference model) every
/// field make_arch_spec() produces must equal the value the engine already
/// used, so the default path computes byte-identical numbers.

#include "nanoinfer/model/model_config.h"

namespace nanoinfer {
namespace model {

/// Pre-attention / block normalization type.
enum class NormType { RMSNorm, LayerNorm };

/// Positional-encoding type applied to Q/K after projection.
enum class PosEncType { RoPE, ALiBi, Learned, None };

/// Feed-forward (MLP) activation structure.
enum class MLPType { SwiGLU, GELU, GeGLU, ReLU };

/// Architecture-agnostic description of one transformer block family.
/// Numeric fields mirror the ModelConfig values the engine actually uses.
struct ArchSpec {
    NormType norm = NormType::RMSNorm;
    PosEncType pe = PosEncType::RoPE;
    MLPType mlp = MLPType::SwiGLU;

    bool qkv_bias = true;             // Qwen2 keeps q/k/v projection biases
    bool pre_norm = true;             // norm-then-attention (pre-LN) layout
    bool parallel_attn_mlp = false;   // attention + MLP computed in parallel

    int hidden_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int num_layers = 0;
    int vocab_size = 0;

    float rope_theta = 10000.f;
    float rms_norm_eps = 1e-6f;
    float layer_norm_eps = 1e-5f;

    bool untied_lm_head = false;      // separate lm_head.weight (not tied to embed)

    int num_experts = 0;              // MoE (unused by Qwen2 reference path)
    int moe_top_k = 0;
};

/// Map a parsed HF ModelConfig into the architecture-agnostic ArchSpec.
///
/// For Qwen2.5 (the only validated model) every produced value matches the
/// hardcoded constants the engine previously used:
///   head_dim   = hidden_size / num_attention_heads  (== cfg.head_dim)
///   rope_theta / rms_norm_eps / num_layers / vocab_size from cfg
///   qkv_bias   = true       (Qwen2 keeps q/k/v biases)
///   untied_lm_head = !cfg.tie_word_embeddings
ArchSpec make_arch_spec(const ModelConfig& cfg);

}  // namespace model
}  // namespace nanoinfer
