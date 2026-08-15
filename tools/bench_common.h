#pragma once
/// Shared helpers for the NanoInfer benchmark tools (tools/*.cpp).

#include <cuda_runtime.h>

/// Auto-derive a per-step token budget from the GPU's TOTAL memory, so the
/// same binary adapts to different cards (RTX 2060 6GB / 4090 24GB /
/// A100 80GB / H800 ...) without recompiling or editing constants.
///
/// Heuristic: keep the activation budget around ~8% of VRAM,
/// clamped to [256, 4096].  KV cache is sized separately by EngineServer
/// from actual free memory at load time.
inline int default_batch_tokens() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && total_bytes > 0) {
        int t = static_cast<int>((total_bytes >> 20) / 12);  // total MB / 12
        if (t < 256) t = 256;
        if (t > 4096) t = 4096;
        return t;
    }
    return 512;
}
