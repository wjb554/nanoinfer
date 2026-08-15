/// Safetensors loader — manual JSON parsing to avoid external dependencies.
///
/// The safetensors header is a flat JSON object:
///   {"tensor_name": {"dtype": "BF16", "shape": [a,b], "data_offsets": [x,y]}, ...}
///
/// We parse it with simple string operations (no JSON library needed).

#include "nanoinfer/model/model_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace nanoinfer {
namespace model {

// --- helpers ---------------------------------------------------------------

static DType parse_dtype(const std::string& s) {
    if (s == "\"F32\"") return DType::F32;
    if (s == "\"F16\"") return DType::F16;
    if (s == "\"BF16\"") return DType::BF16;
    throw std::runtime_error("Unknown safetensors dtype: " + s);
}

static std::vector<int> parse_shape(const std::string& s) {
    // s is like "[151936,896]" or "[896]"
    std::vector<int> shape;
    size_t i = s.find('[');
    if (i == std::string::npos) return shape;
    i++;
    while (i < s.size() && s[i] != ']') {
        while (i < s.size() && (s[i] == ' ' || s[i] == ',')) i++;
        if (i >= s.size() || s[i] == ']') break;
        size_t end = i;
        while (end < s.size() && s[end] >= '0' && s[end] <= '9') end++;
        shape.push_back(std::stoi(s.substr(i, end - i)));
        i = end;
    }
    return shape;
}

// --- SafetensorsLoader -----------------------------------------------------

SafetensorsLoader::SafetensorsLoader(const std::string& path) : path_(path) {
    parse_single(path);
}

SafetensorsLoader SafetensorsLoader::from_dir(const std::string& model_dir) {
    // Single-file layout: models/<name>/model.safetensors
    std::string single = model_dir + "/model.safetensors";
    std::ifstream probe(single, std::ios::binary);
    if (probe.good()) {
        probe.close();
        return SafetensorsLoader(single);
    }
    probe.close();
    // Sharded layout: models/<name>/model.safetensors.index.json + model-*.safetensors
    SafetensorsLoader l;
    l.parse_sharded(model_dir);
    return l;
}

void SafetensorsLoader::parse_single(const std::string& path) {
    // Read the entire file to get the header
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);

    // Read header length (8 bytes, little-endian uint64)
    char len_buf[8];
    file.read(len_buf, 8);
    uint64_t header_len = *reinterpret_cast<uint64_t*>(len_buf);

    // Read header JSON
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    file_data_offset_[path] = 8 + header_len;

    // Parse: find each "tensor_name": { ... }
    size_t pos = header.find('{');  // first '{' after __metadata__
    if (pos == std::string::npos) throw std::runtime_error("Invalid safetensors header");

    // Move past the opening '{' and any __metadata__ entry
    if (header.find("\"__metadata__\"") != std::string::npos) {
        // Skip the __metadata__ object
        pos = header.find("},\"", pos);
        if (pos == std::string::npos) pos = header.find("}\"", pos);
        if (pos != std::string::npos) pos += 2;  // skip "},"
    } else {
        pos = 1;
    }

    // Parse each tensor entry: "name": {"dtype":...,"shape":...,"data_offsets":...}
    while (pos < header.size()) {
        if (header[pos] == '}' || header[pos] == '\0') break;

        // Find tensor name
        size_t name_start = header.find('"', pos);
        if (name_start == std::string::npos) break;
        size_t name_end = header.find('"', name_start + 1);
        std::string name = header.substr(name_start + 1, name_end - name_start - 1);
        if (name.empty() || name == "__metadata__") { pos = name_end + 1; continue; }

        // Find the value object: { ... }
        size_t obj_start = header.find('{', name_end);
        size_t obj_end = header.find('}', obj_start);
        std::string obj = header.substr(obj_start, obj_end - obj_start + 1);

        TensorInfo info;

        // Parse dtype
        size_t dt_pos = obj.find("\"dtype\"");
        size_t dt_colon = obj.find(':', dt_pos);
        size_t dt_start = obj.find('"', dt_colon);
        size_t dt_end = obj.find('"', dt_start + 1);
        info.dtype = parse_dtype(obj.substr(dt_start, dt_end - dt_start + 1));

        // Parse shape
        size_t sh_pos = obj.find("\"shape\"");
        size_t sh_colon = obj.find(':', sh_pos);
        size_t sh_start = obj.find('[', sh_colon);
        size_t sh_end = obj.find(']', sh_start);
        info.shape = parse_shape(obj.substr(sh_start, sh_end - sh_start + 1));

        // Parse data_offsets
        size_t do_pos = obj.find("\"data_offsets\"");
        size_t do_colon = obj.find(':', do_pos);
        size_t do_start = obj.find('[', do_colon);
        size_t do_comma = obj.find(',', do_start);
        size_t do_end = obj.find(']', do_comma);
        info.offset_begin = std::stoull(obj.substr(do_start + 1, do_comma - do_start - 1));
        info.offset_end = std::stoull(obj.substr(do_comma + 1, do_end - do_comma - 1));

        tensors_[name] = info;
        tensor_file_[name] = path;
        pos = obj_end + 2;  // skip "},"
    }
}

void SafetensorsLoader::parse_sharded(const std::string& model_dir) {
    // index.json: {"metadata": {...}, "weight_map": {"tensor": "model-XX-of-NN.safetensors", ...}}
    std::ifstream idx(model_dir + "/model.safetensors.index.json", std::ios::binary);
    if (!idx)
        throw std::runtime_error("Cannot open index: " + model_dir +
                                 "/model.safetensors.index.json");
    std::stringstream ss;
    ss << idx.rdbuf();
    std::string json = ss.str();

    // Collect unique shard files from the weight_map.
    std::vector<std::string> shards;
    size_t pos = json.find("\"weight_map\"");
    if (pos == std::string::npos) throw std::runtime_error("index.json: no weight_map");
    pos = json.find('{', pos);
    if (pos == std::string::npos) throw std::runtime_error("index.json: bad weight_map");
    size_t i = pos + 1;
    while (i < json.size()) {
        while (i < json.size() &&
               (json[i]==' '||json[i]=='\t'||json[i]=='\n'||json[i]=='\r'||json[i]==',')) i++;
        if (i >= json.size() || json[i] == '}') break;
        size_t ns = json.find('"', i);
        size_t ne = json.find('"', ns + 1);
        if (ns == std::string::npos || ne == std::string::npos) break;
        size_t fc = json.find(':', ne);
        size_t fs = json.find('"', fc);
        size_t fe = json.find('"', fs + 1);
        if (fc == std::string::npos || fs == std::string::npos || fe == std::string::npos) break;
        std::string file = json.substr(fs + 1, fe - fs - 1);
        if (!file.empty() &&
            std::find(shards.begin(), shards.end(), file) == shards.end())
            shards.push_back(file);
        i = fe + 1;
    }
    if (shards.empty()) throw std::runtime_error("index.json: empty weight_map");

    for (const auto& shard : shards)
        parse_single(model_dir + "/" + shard);
}

const TensorInfo* SafetensorsLoader::get_info(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

Tensor SafetensorsLoader::load_tensor(const std::string& name) const {
    auto* info = get_info(name);
    if (!info) throw std::runtime_error("Tensor not found: " + name);

    Tensor t(info->shape, info->dtype, Device::CPU);
    load_into(name, t);
    return t;
}

void SafetensorsLoader::load_into(const std::string& name, Tensor& dst) const {
    auto* info = get_info(name);
    if (!info) throw std::runtime_error("Tensor not found: " + name);

    if (dst.shape() != info->shape || dst.dtype() != info->dtype)
        throw std::runtime_error("Shape/dtype mismatch for: " + name);

    size_t tensor_bytes = info->offset_end - info->offset_begin;
    if (tensor_bytes != dst.nbytes())
        throw std::runtime_error("Byte size mismatch for: " + name);

    // Which file holds this tensor (single-file mode: the one file; sharded:
    // the specific shard), and its data start offset.
    auto fit = tensor_file_.find(name);
    if (fit == tensor_file_.end())
        throw std::runtime_error("No source file for tensor: " + name);
    const std::string& file = fit->second;
    auto oit = file_data_offset_.find(file);
    if (oit == file_data_offset_.end())
        throw std::runtime_error("No data offset recorded for: " + file);
    uint64_t data_offset = oit->second;

    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + file);

    // For large tensors, read in chunks
    f.seekg(data_offset + info->offset_begin);
    std::vector<char> buf(tensor_bytes);
    f.read(buf.data(), tensor_bytes);

    // Copy from CPU buffer to Tensor (whether CPU or GPU)
    if (dst.is_cuda()) {
        // If destination is GPU, copy via pinned memory for speed
        dst.copy_from(buf.data(), tensor_bytes);
    } else {
        std::memcpy(dst.raw(), buf.data(), tensor_bytes);
    }
}

std::vector<std::string> SafetensorsLoader::tensor_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : tensors_) names.push_back(name);
    return names;
}

}  // namespace model
}  // namespace nanoinfer
