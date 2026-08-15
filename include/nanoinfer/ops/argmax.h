#pragma once
/// GPU argmax — find the maximum-value index entirely on GPU.
/// Eliminates the ~600KB GPU→CPU logits copy per greedy sample.

#include "nanoinfer/tensor.h"

namespace nanoinfer {
namespace ops {

/// Returns the index of the maximum value in a 1-D tensor.
/// Equivalent to std::max_element on GPU.
int argmax(const Tensor& logits);

}  // namespace ops
}  // namespace nanoinfer
