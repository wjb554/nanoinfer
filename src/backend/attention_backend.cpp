/// Paged-attention backend plugin — default implementation.
///
/// AttentionBackendPaged forwards verbatim to kv_cache::paged_attention and
/// kv_cache::prefill_paged_attention with the exact arguments the engine's
/// call sites pass today.  Registered under BOTH "paged" and "paged_v2"
/// (same factory) so plugin selection can be proven without altering numerics.

#include "nanoinfer/backend/attention_backend.h"
#include "nanoinfer/backend/registry.h"
#include "nanoinfer/kv_cache/block_allocator.h"
#include "nanoinfer/kv_cache/paged_attention.h"

namespace nanoinfer {
namespace backend {

// Declared in registry.h; referencing this symbol from engine_server.o forces
// the static-archive linker to pull this TU (and therefore the registrar
// objects that register "paged" / "paged_v2").
void force_link_attention_backend() {}

Tensor AttentionBackendPaged::decode_paged_attention(
    const Tensor& q, const int* block_table, int max_blocks_per_seq,
    const int* seq_lens, kv_cache::BlockAllocator& allocator,
    int max_seq_hint) const {
    return kv_cache::paged_attention(q, block_table, max_blocks_per_seq,
                                     seq_lens, allocator, max_seq_hint);
}

Tensor AttentionBackendPaged::prefill_paged_attention(
    const Tensor& q, const int* block_table, int max_blocks,
    const int* seq_len_ptr, kv_cache::BlockAllocator& allocator,
    bool causal) const {
    return kv_cache::prefill_paged_attention(q, block_table, max_blocks,
                                             seq_len_ptr, allocator, causal);
}

}  // namespace backend
}  // namespace nanoinfer

// Registered inside the namespace so the interface/impl names resolve without
// qualification.
namespace nanoinfer {
namespace backend {
NANOINFER_REGISTER_BACKEND(IAttentionBackend, paged, AttentionBackendPaged);
NANOINFER_REGISTER_BACKEND(IAttentionBackend, paged_v2, AttentionBackendPaged);
}  // namespace backend
}  // namespace nanoinfer
