#pragma once
/// GEMM backend plugin interface + reference implementation declaration.
///
/// The reference backend is the CURRENT EngineServer::layer_gemm body moved
/// here verbatim; selecting it must reproduce the engine's numerics exactly.

#include <memory>

#include "nanoinfer/tensor.h"

namespace nanoinfer {
namespace backend {

/// Precision-mode flags that select the numeric path inside a GEMM backend.
/// Mirrors the (use_fp8_, weight_dtype_, NANOINFER_FP8_W8A16) state the
/// original EngineServer::layer_gemm dispatched on.
struct GemmLayerConfig {
    bool use_fp8 = false;              // fp8 weight-only weights are active
    DType weight_dtype = DType::F32;   // activation / residual dtype
    bool fp8_w8a16 = false;            // effective W8A16 (env-forced W8A16)
};

/// Interface for one layer's linear projections (q/k/v/o/gate/up/down).
/// The default backend reproduces EngineServer::layer_gemm exactly.
class IGemmBackend {
public:
    virtual ~IGemmBackend() = default;

    /// Exact contract of EngineServer::layer_gemm today.
    ///   h          [M, K] input activation (weight_dtype_)
    ///   w_f16      [K, N] fp16/fp32 weight (empty in fp8 mode)
    ///   w_f8       [N, K] fp8 E4M3 weight (fp8 mode only)
    ///   w_s        [N]    per-row fp8 scale (fp8 mode only)
    ///   cfg        precision-mode flags (use_fp8, weight_dtype, fp8_w8a16)
    ///   fp32_out   true for q/k/v (attention wants F32); false for
    ///              o/gate/up/down (residual wants weight_dtype_).
    virtual Tensor layer_gemm(const Tensor& h, const Tensor& w_f16,
                              const Tensor& w_f8, const Tensor& w_s,
                              const GemmLayerConfig& cfg, bool fp32_out) const = 0;
};

/// Default backend: the CURRENT EngineServer::layer_gemm body moved here
/// VERBATIM (same dispatch, same env read of NANOINFER_FP8_W8A16).
class GemmBackendReference : public IGemmBackend {
public:
    Tensor layer_gemm(const Tensor& h, const Tensor& w_f16,
                      const Tensor& w_f8, const Tensor& w_s,
                      const GemmLayerConfig& cfg, bool fp32_out) const override;
};

}  // namespace backend
}  // namespace nanoinfer
