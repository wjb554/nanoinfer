#pragma once
/// Matrix multiplication — cuBLAS, naive, and tiled shared-memory backends.

#include "nanoinfer/tensor.h"

namespace nanoinfer {
namespace ops {

enum class GemmBackend { CuBLAS, Naive, Tiled, Vectorized, DoubleBuf };

void set_gemm_backend(GemmBackend b);
GemmBackend get_gemm_backend();

/// C = A @ B  (all fp32, row-major)
Tensor gemm(const Tensor& a, const Tensor& b, bool transpose_b = false);

/// Mixed precision: A(fp16) @ B(fp16) → C(fp32)
/// Uses Tensor Cores for compute, accumulates in fp32.
Tensor gemm_f16f32(const Tensor& a, const Tensor& b, bool transpose_b = false);

/// Quantize FP16 weights to FP8 E4M3 with a per-row scale.
/// w [R, C] fp16 -> q [R, C] uint8 (e4m3) + scale [R] f32 (scale[r] = max|w[r,:]|/448).
void quantize_fp8(const Tensor& w, Tensor& q, Tensor& scale);

/// Mixed precision: A(fp16) @ B(fp8 E4M3, per-N scale) → C(fp32).
/// B is [N, K] used transposed (transpose_b=true), scale [N] per output column.
/// Weights stay FP8 in memory (2x savings); dequant is fused into the GEMM.
Tensor gemm_f16_fp8(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale,
                    bool transpose_b = true);

/// cuBLAS-backed fp8 weight-only (Route A): dequant fp8 weights → fp16 scratch,
/// then cublasGemmEx fp16×fp16→fp32 (tensor cores).  Same W8A16 dequant numerics
/// as gemm_f16_fp8, but replaces the naive custom GEMM with cuBLAS.
Tensor gemm_f16_fp8_cublas(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale,
                           bool transpose_b = true);

/// W8A8 (Route B): A(fp16) is quantized to fp8 (per-token scale) on the fly,
/// then cublasGemmEx fp8×fp8→fp32 (FP8 tensor cores, sm_89+).  Output scale
/// sA[i]*sB[j] applied in a post kernel.  Weights stay fp8 in memory; this is
/// the path that actually beats fp16 (1 byte/operand, fp8 tensor cores).
Tensor gemm_f16_fp8_w8a8(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale,
                         bool transpose_b = true);

/// Optimized hand-written fp8 weight-only GEMM (same W8A16 dequant numerics as
/// gemm_f16_fp8, but register-blocked 4x4, vectorized float4/uint32 loads and
/// double-buffered shared memory — for comparing against the naive kernel and
/// cuBLAS).  Weights stay fp8 in memory.
Tensor gemm_f16_fp8_opt(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale,
                        bool transpose_b = true);

/// Dedicated fp8 weight-only GEMV for decode (M==1): C[1,N] = A[1,K] @ B[N,K]^T.
/// All parallelism on N (thread-per-column), B read once with coalesced uint32
/// loads, dequant fp8→float into a shared-memory transposed tile, A shared.
/// Significantly better than a GEMM kernel at M=1 (no M-dimension waste).
/// b_fp8 is [N,K], b_scale [N] per-row.  Throws if a.size(0) != 1.
Tensor gemv_f16_fp8(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale);

/// Whether cuBLAS fp8 GEMM (W8A8) is available on this device.  cuBLAS 11.x/12.x
/// expose fp8 GEMM only on Hopper (sm_90); Ada (sm_89, RTX 4090) needs a custom
/// kernel or RTX 50-series (sm_120) + CUDA 13.  Probing is one-time (cached).
bool fp8_w8a8_available();

}  // namespace ops
}  // namespace nanoinfer
