#pragma once
/// Element-wise CUDA kernels: add, mul, scale, silu, etc.
///
/// All kernels follow the pattern: one thread per element.

#include "nanoinfer/tensor.h"

namespace nanoinfer {
namespace ops {

// --- arithmetic ---
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& x, float factor);

/// In-place broadcast-add of a 1-D bias to every row: x[t, j] += bias[j].
/// x: [T, out], bias: [out].
void add_bias_inplace(Tensor& x, const Tensor& bias);

// --- activation ---
Tensor silu(const Tensor& x);          // SiLU(x) = x * sigmoid(x)
void silu_inplace(Tensor& x);          // in-place SiLU

}  // namespace ops
}  // namespace nanoinfer
