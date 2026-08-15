#pragma once
/// Safetensors file loader — reads HuggingFace model weights into GPU Tensors.
///
/// Format: [8-byte header_len (uint64)] [JSON header] [raw tensor data]
/// Each tensor in the header has: dtype, shape, data_offsets [begin, end].

#include <string>
#include <unordered_map>
#include <vector>

#include "nanoinfer/tensor.h"

namespace nanoinfer {
namespace model {

struct TensorInfo {
    DType dtype;
    std::vector<int> shape;
    size_t offset_begin;
    size_t offset_end;
};

class SafetensorsLoader {
public:
    /// Open a single safetensors file and parse the header.
    explicit SafetensorsLoader(const std::string& path);

    /// Open a model directory: auto-detects a single model.safetensors,
    /// otherwise reads the sharded layout via model.safetensors.index.json
    /// (Qwen2.5-3B/7B+ are sharded on HF).
    static SafetensorsLoader from_dir(const std::string& model_dir);

    /// Get metadata for a tensor by name.
    const TensorInfo* get_info(const std::string& name) const;

    /// Load one tensor into GPU memory (allocates and copies).
    Tensor load_tensor(const std::string& name) const;

    /// Load one tensor into a pre-allocated Tensor (must match shape/dtype).
    void load_into(const std::string& name, Tensor& dst) const;

    /// List all tensor names in the file(s).
    std::vector<std::string> tensor_names() const;

    /// Number of tensors.
    size_t num_tensors() const { return tensors_.size(); }

private:
    SafetensorsLoader() = default;  // sharded mode via from_dir
    void parse_single(const std::string& path);         // one .safetensors file
    void parse_sharded(const std::string& model_dir);   // index.json + shards

    std::string path_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    // tensor name -> safetensors file that holds it; file -> data start offset
    std::unordered_map<std::string, std::string> tensor_file_;
    std::unordered_map<std::string, uint64_t> file_data_offset_;
};

}  // namespace model
}  // namespace nanoinfer
