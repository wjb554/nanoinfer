#pragma once
/// Paged-attention backend plugin interface + default implementation.
///
/// The default backend forwards verbatim to kv_cache::paged_attention and
/// kv_cache::prefill_paged_attention with the exact arguments the engine's
/// call sites pass today.

#include "nanoinfer/tensor.h"

namespace nanoinfer {
namespace kv_cache {
class BlockAllocator;
}  // namespace kv_cache

namespace backend {

/// Interface for the two paged-attention entry points used by the engine.
class IAttentionBackend {
public:
    virtual ~IAttentionBackend() = default;

    /// Exact contract of the current batched decode paged_attention call
    /// sites (run_layers(), step()).
    ///   q               [num_tokens, num_q_heads, head_dim]
    ///   block_table     flat device array of physical block IDs
    ///                   [num_seqs * max_blocks_per_seq]
    ///   max_blocks_per_seq  stride of block_table
    ///   seq_lens        [num_seqs] device array of sequence lengths
    ///   allocator       BlockAllocator providing K/V data pointers
    ///   max_seq_hint    upper bound on the largest seq_len (kernel hint)
    virtual Tensor decode_paged_attention(
        const Tensor& q, const int* block_table, int max_blocks_per_seq,
        const int* seq_lens, kv_cache::BlockAllocator& allocator,
        int max_seq_hint) const = 0;

    /// Exact contract of the current chunked-prefill call site (step()).
    ///   q            [P, Hq, D] — the prefill chunk Qs
    ///   block_table  device pointer to this request's physical block IDs
    ///   max_blocks   stride of block_table
    ///   seq_len_ptr  device pointer to a single int (total_seq)
    ///   allocator    BlockAllocator for this layer
    ///   causal       apply causal mask (legacy prefill passes false)
    virtual Tensor prefill_paged_attention(
        const Tensor& q, const int* block_table, int max_blocks,
        const int* seq_len_ptr, kv_cache::BlockAllocator& allocator,
        bool causal) const = 0;
};

/// Default backend: forwards to kv_cache::paged_attention /
/// kv_cache::prefill_paged_attention with identical arguments.
class AttentionBackendPaged : public IAttentionBackend {
public:
    Tensor decode_paged_attention(
        const Tensor& q, const int* block_table, int max_blocks_per_seq,
        const int* seq_lens, kv_cache::BlockAllocator& allocator,
        int max_seq_hint) const override;
    Tensor prefill_paged_attention(
        const Tensor& q, const int* block_table, int max_blocks,
        const int* seq_len_ptr, kv_cache::BlockAllocator& allocator,
        bool causal) const override;
};

}  // namespace backend
}  // namespace nanoinfer
