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
    // KEEP 16.  Raising the default is gated on first-prefill attention being
    // made block-tiled: the current first_prefill_attn_batched kernel is O(T^2)
    // per entry, so a large chunk makes a long single-request prompt SLOWER
    // (measured: chunk=501 vs 16 roughly doubles TTFT at 300-800 prompt tokens).
    // Until that kernel is tiled, a small chunk keeps each prefill step's T
    // small and lets the fast tiled chunked-prefill path handle the rest.
    (void)0;
    return 16;
}

}  // namespace engine
}  // namespace nanoinfer
