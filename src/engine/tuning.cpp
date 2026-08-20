/// VRAM-derived tuning defaults (see tuning.h).

#include "nanoinfer/engine/tuning.h"

#include <cuda_runtime.h>

namespace nanoinfer {
namespace engine {

int default_batch_tokens() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && total_bytes > 0) {
        int t = static_cast<int>((total_bytes >> 20) / 12);  // total MB / 12
        if (t < 256) t = 256;
        if (t > 4096) t = 4096;
        return t;
    }
    return 512;
}

int default_chunk_size() {
    // ~1/4 of the batch budget, clamped to [64, 1024].  Now that prefill
    // attention is a single tiled batched kernel (no O(T^2) first-prefill
    // path), large chunks no longer hurt long prompts — measured on 7B:
    // chunk=501 vs 16 gives ~equal or better TTFT (500 tok: 1008 vs 1299 ms).
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && total_bytes > 0) {
        int t = static_cast<int>((total_bytes >> 20) / 48);  // total MB / 48
        if (t < 64) t = 64;
        if (t > 1024) t = 1024;
        return t;
    }
    return 128;
}

}  // namespace engine
}  // namespace nanoinfer
