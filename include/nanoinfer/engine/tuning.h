#pragma once
/// Engine tuning defaults — VRAM-derived batch budget heuristics.
///
/// These are library-side so the scheduler factory, ServerEngine and the
/// benchmark tools all agree on the same derivation, instead of each layer
/// hardcoding its own (the scheduler previously hardcoded 256, which silently
/// capped server/bench_server batches far below what the GPU could hold).

namespace nanoinfer {
namespace engine {

/// Per-step token budget ceiling (max_batch_tokens).
/// Heuristic: keep activation budget around ~8% of total VRAM,
/// clamped to [256, 4096].  Same formula as the old tools/bench_common.h.
int default_batch_tokens();

/// Prefill chunk size (max prompt tokens processed per entry per step).
/// Currently returns 16 (the legacy value).  Raising it is gated on the
/// first-prefill attention kernel becoming block-tiled — today it is O(T^2),
/// so a large chunk makes long single-request prompts slower.
int default_chunk_size();

}  // namespace engine
}  // namespace nanoinfer
