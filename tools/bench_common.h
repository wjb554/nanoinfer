#pragma once
/// Shared helpers for the NanoInfer benchmark tools (tools/*.cpp).
///
/// The VRAM-derived defaults now live in the library (engine/tuning.h) so the
/// scheduler factory, ServerEngine and benchmarks all agree.  This header just
/// pulls default_batch_tokens() into the global scope so the tools can keep
/// calling it unqualified (they are inside namespace nanoinfer::engine via
/// `using namespace`, and tuning.h defines it in namespace nanoinfer::engine).

#include "nanoinfer/engine/tuning.h"

// Re-export under the name the tools already use.  `using` (not an inline
// wrapper) keeps a single definition, so it never clashes with the library
// symbol when a tool also includes engine.h.
using nanoinfer::engine::default_batch_tokens;
using nanoinfer::engine::default_chunk_size;
