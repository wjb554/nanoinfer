/// bench_fp8_gemm — compare the fp8 weight-only GEMM implementations:
///   naive (gemm_f16_fp8) vs optimized hand-written (gemm_f16_fp8_opt)
///   vs cuBLAS dequant (gemm_f16_fp8_cublas) vs cuBLAS fp16 reference.
/// For each shape: maxdiff vs the fp16 reference, then average time/GFLOPS.
///
/// Shapes cover real Qwen2.5 GEMMs (decode M=1 and prefill M>1).

#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <cuda_fp16.h>

#include "nanoinfer/tensor.h"
#include "nanoinfer/ops/gemm.h"

using namespace nanoinfer;
using namespace nanoinfer::ops;

static double ms_now() {
    static auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

static float maxdiff(const Tensor& a, const Tensor& b) {
    std::vector<float> va(a.numel()), vb(b.numel());
    a.copy_to(va.data(), a.nbytes());
    b.copy_to(vb.data(), b.nbytes());
    float mx = 0;
    for (size_t i = 0; i < va.size(); i++)
        mx = fmaxf(mx, fabsf(va[i] - vb[i]));
    return mx;
}

struct Case { int M, K, N; };

int main() {
    const Case cases[] = {
        {1, 896, 896},       // 0.5B q/k/v/o decode
        {1, 896, 4864},      // 0.5B gate/up decode
        {64, 896, 896},      // 0.5B q/k/v prefill
        {64, 896, 4864},     // 0.5B mlp prefill
        {1, 3584, 18944},    // 7B mlp decode
        {20, 896, 896},      // mid-size (RM=2 branch)
        {128, 3584, 18944},  // 7B mlp prefill
        {256, 4096, 4096},   // generic large
    };
    const int NITER = 20;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 0.15f);

    for (auto [M, K, N] : cases) {
        std::vector<__half> ah(M * K), wh(N * K);
        for (auto& v : ah) v = __float2half(nd(rng));
        for (auto& v : wh) v = __float2half(nd(rng));
        Tensor A({M, K}, DType::F16, Device::CUDA);
        Tensor W({N, K}, DType::F16, Device::CUDA);
        A.copy_from(ah.data(), ah.size() * sizeof(__half));
        W.copy_from(wh.data(), wh.size() * sizeof(__half));
        Tensor Wq, Ws;
        quantize_fp8(W, Wq, Ws);

        Tensor ref = gemm_f16f32(A, W, true);   // cuBLAS fp16 reference

        auto run = [&](auto&& fn, const char* name) {
            Tensor o = fn();
            float md = maxdiff(ref, o);
            cudaDeviceSynchronize();
            double t0 = ms_now();
            for (int it = 0; it < NITER; it++) { Tensor tmp = fn(); }
            cudaDeviceSynchronize();
            double avg = (ms_now() - t0) / NITER;
            double gf = 2.0 * M * K * N / (avg * 1e-3) / 1e9;
            printf("    %-28s maxdiff=%.4f  %8.3f ms  %8.1f GFLOPS\n",
                   name, md, avg, gf);
            fflush(stdout);
        };

        printf("M=%d K=%d N=%d (%.0f MFLOP):\n", M, K, N, 2.0 * M * K * N / 1e6);
        fflush(stdout);
        run([&]{ return gemm_f16_fp8(A, Wq, Ws, true); },       "naive  gemm_f16_fp8");
        run([&]{ return gemm_f16_fp8_opt(A, Wq, Ws, true); },    "opt    gemm_f16_fp8_opt");
        run([&]{ return gemm_f16_fp8_cublas(A, Wq, Ws, true); }, "cuBLAS gemm_f16_fp8_cublas");
        if (M == 1)
            run([&]{ return gemv_f16_fp8(A, Wq, Ws); },          "gemv   gemv_f16_fp8");
        printf("\n");
    }
    return 0;
}
