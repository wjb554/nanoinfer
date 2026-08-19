/// GEMM backend plugin — reference implementation.
///
/// GemmBackendReference::layer_gemm is the ORIGINAL EngineServer::layer_gemm
/// body moved here verbatim.  No numeric op, constant, or env read changed.
/// Registered under BOTH "reference" and "reference_v2" (same factory) so
/// plugin selection can be proven without altering any numerics.

#include "nanoinfer/backend/gemm_backend.h"
#include "nanoinfer/backend/registry.h"
#include "nanoinfer/ops/gemm.h"

#include <cuda_fp16.h>
#include <cstdlib>

namespace nanoinfer {
namespace backend {

// ============================================================================
// F32 -> F16 conversion helper.
//
// The original layer_gemm called the file-static cast_f32_to_f16() helper that
// lives in engine_server.cpp (still used there by the engine's other paths).
// This TU carries its own identical copy so the moved body compiles unchanged.
// ============================================================================

__global__ void cast_f32_to_f16_kernel(half* __restrict__ dst,
                                       const float* __restrict__ src,
                                       int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}

static Tensor cast_f32_to_f16(const Tensor& x) {
    int N = static_cast<int>(x.numel());
    Tensor out(x.shape(), DType::F16, Device::CUDA);
    int block = 256, grid = (N + 255) / 256;
    cast_f32_to_f16_kernel<<<grid, block>>>(out.data<half>(), x.data<float>(), N);
    return out;
}

// Declared in registry.h; referencing this symbol from engine_server.o forces
// the static-archive linker to pull this TU (and therefore the registrar
// objects that register "reference" / "reference_v2").
void force_link_gemm_backend() {}

Tensor GemmBackendReference::layer_gemm(const Tensor& h, const Tensor& w_f16,
                                        const Tensor& w_f8, const Tensor& w_s,
                                        const GemmLayerConfig& cfg,
                                        bool fp32_out) const {
    // MOVED VERBATIM from the original EngineServer::layer_gemm.  The only
    // substitutions are the two state reads: use_fp8_ -> cfg.use_fp8 and
    // weight_dtype_ -> cfg.weight_dtype.  The env read of NANOINFER_FP8_W8A16
    // is the same static-cached expression the original used.
    if (cfg.use_fp8) {
        // Route B (W8A8 fp8 tensor cores) by default; set NANOINFER_FP8_W8A16=1
        // to fall back to Route A (dequant fp8->fp16, cuBLAS fp16 tensor cores).
        static const bool fp8_w8a16 = [] {
            const char* e = std::getenv("NANOINFER_FP8_W8A16");
            return e && e[0] && e[0] != '0';
        }();
        // Effective W8A16: forced, or W8A8 unsupported (cuBLAS fp8 is Hopper-only;
        // Ada RTX 4090 falls back to W8A16).
        const bool use_w8a16 = fp8_w8a16 || !ops::fp8_w8a8_available();
        // Decode (M==1): the dedicated split-K GEMV is 2.4-13x faster than the
        // dequant+cuBLAS GEMM and numerically the same W8A16 dequant.
        if (h.size(0) == 1 && use_w8a16) {
            Tensor o = ops::gemv_f16_fp8(h, w_f8, w_s);
            if (fp32_out) return o;
            return cast_f32_to_f16(o);   // residual path wants weight_dtype_ (F16)
        }
        Tensor o = use_w8a16
            ? ops::gemm_f16_fp8_cublas(h, w_f8, w_s, /*transpose_b=*/true)
            : ops::gemm_f16_fp8_w8a8(h, w_f8, w_s, /*transpose_b=*/true);
        if (fp32_out) return o;
        return cast_f32_to_f16(o);   // residual path wants weight_dtype_ (F16)
    }
    if (cfg.weight_dtype == DType::F16) {
        if (fp32_out) return ops::gemm_f16f32(h, w_f16, /*transpose_b=*/true);  // F32
        return ops::gemm(h, w_f16, /*transpose_b=*/true);                       // F16
    }
    return ops::gemm(h, w_f16, /*transpose_b=*/true);
}

}  // namespace backend
}  // namespace nanoinfer

// Registered inside the namespace so the interface/impl names resolve without
// qualification.
namespace nanoinfer {
namespace backend {
NANOINFER_REGISTER_BACKEND(IGemmBackend, reference, GemmBackendReference);
NANOINFER_REGISTER_BACKEND(IGemmBackend, reference_v2, GemmBackendReference);
}  // namespace backend
}  // namespace nanoinfer
