/// dump_weight — verify the C++ SafetensorsLoader against a Python (numpy)
/// reference.  Prints embed[0][0:5], embed[1001][0:5] and q_proj stats.
///
/// Usage: dump_weight [model_dir]

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "nanoinfer/model/model_loader.h"

using namespace nanoinfer;
using namespace nanoinfer::model;

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "models/qwen2.5-0.5b";
    SafetensorsLoader loader = SafetensorsLoader::from_dir(dir);

    auto dump = [&](const char* name) {
        printf("[%s] get_info...\n", name); fflush(stdout);
        auto* info = loader.get_info(name);
        if (!info) { printf("%s: not found\n", name); return; }
        printf("[%s] info: dtype=%d shape=[%d,%d] offsets=[%zu,%zu]\n",
               name, (int)info->dtype, info->shape[0],
               info->shape.size() > 1 ? info->shape[1] : 0,
               info->offset_begin, info->offset_end); fflush(stdout);
        printf("[%s] load_tensor...\n", name); fflush(stdout);
        Tensor t = loader.load_tensor(name);
        printf("[%s] loaded shape=[%d,%d] nbytes=%zu, copy_to...\n",
               name, t.size(0), t.size(1), t.nbytes()); fflush(stdout);
        std::vector<char> raw(t.nbytes());
        t.copy_to(raw.data(), raw.size());
        size_t nel = (size_t)t.size(0) * t.size(1);
        std::vector<float> f(nel);
        if (info->dtype == DType::BF16) {
            for (size_t i = 0; i < nel; i++) {
                uint32_t u = static_cast<uint32_t>(((const uint16_t*)raw.data())[i]);
                uint32_t bits = u << 16;
                f[i] = *reinterpret_cast<float*>(&bits);
            }
        } else if (info->dtype == DType::F32) {
            for (size_t i = 0; i < nel; i++) f[i] = ((const float*)raw.data())[i];
        } else {
            printf("%s: unsupported dtype %d\n", name, (int)info->dtype);
            return;
        }
        double mean = 0, sum2 = 0;
        for (float v : f) { mean += v; sum2 += (double)v * v; }
        mean /= nel;
        double std = (nel > 1) ? sum2 / nel - mean * mean : 0;
        if (std < 0) std = 0;
        printf("%s: dtype=%d shape=[%d,%d] mean=%.5f std=%.5f\n",
               name, (int)info->dtype, t.size(0), t.size(1), mean, sqrt(std));
        if (info->shape.size() == 2) {
            size_t D = info->shape[1];
            printf("  [0][0:5]:"); for (int j = 0; j < 5; j++) printf(" %.6f", f[j]);
            printf("\n  [1001][0:5]:"); for (int j = 0; j < 5; j++) printf(" %.6f", f[1001 * D + j]);
            printf("\n");
        }
    };

    // dump("model.embed_tokens.weight");
    dump("model.layers.0.input_layernorm.weight");
    dump("model.layers.0.self_attn.q_proj.weight");
    return 0;
}
