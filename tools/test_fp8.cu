/// test_fp8 — validate FP8 weight-only GEMM vs cuBLAS fp16 GEMM.
/// Builds random A(fp16) and W(fp16), quantizes W to E4M3, then compares
/// gemm_f16f32(A, W) vs gemm_f16_fp8(A, W_q, scale).
#include <cstdio>
#include <cuda_fp16.h>
#include <cmath>
#include <random>
#include <vector>
#include "nanoinfer/tensor.h"
#include "nanoinfer/ops/gemm.h"

using namespace nanoinfer;
using namespace nanoinfer::ops;

static float maxdiff(const Tensor& a, const Tensor& b) {
    std::vector<float> va(a.numel()), vb(b.numel());
    a.copy_to(va.data(), a.nbytes());
    b.copy_to(vb.data(), b.nbytes());
    float mx = 0;
    for (size_t i = 0; i < va.size(); i++) mx = fmaxf(mx, fabsf(va[i] - vb[i]));
    return mx;
}

// Small hand-checkable case: C = A @ W^T, M=2, K=4, N=4.
static void small_case() {
    __half A_h[2*4] = { __float2half(1),__float2half(2),__float2half(3),__float2half(4),
                        __float2half(0.5f),__float2half(1.5f),__float2half(2.5f),__float2half(3.5f)};
    __half W_h[4*4] = { __float2half(1),__float2half(0),__float2half(0),__float2half(0),
                        __float2half(0),__float2half(1),__float2half(0),__float2half(0),
                        __float2half(0),__float2half(0),__float2half(1),__float2half(0),
                        __float2half(0),__float2half(0),__float2half(0),__float2half(1)};
    Tensor A({2,4}, DType::F16, Device::CUDA);
    Tensor W({4,4}, DType::F16, Device::CUDA);
    A.copy_from(A_h, sizeof(A_h)); W.copy_from(W_h, sizeof(W_h));
    Tensor W_q, W_s; quantize_fp8(W, W_q, W_s);
    Tensor C1 = gemm_f16f32(A, W, true);   // F32 output [2,4]
    Tensor C2 = gemm_f16_fp8(A, W_q, W_s, true);
    // expected: C1 = A @ W^T = A (W is identity); row0 = [1,2,3,4]
    std::vector<float> c1(8), c2(8);
    C1.copy_to(c1.data(), C1.nbytes());  // F32
    C2.copy_to(c2.data(), C2.nbytes());
    printf("small case fp16: ");
    for (int i = 0; i < 8; i++) printf("%.2f/%.2f ", c1[i], c2[i]);
    printf("\n");
}

// Round-trip check: quantize random W, dequant on host, compare to original.
static float roundtrip_err(int N, int K, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0, 0.2f);
    std::vector<__half> wh(N * K);
    for (auto& v : wh) v = __float2half(nd(rng));
    Tensor W({N, K}, DType::F16, Device::CUDA);
    W.copy_from(wh.data(), wh.size() * sizeof(__half));
    Tensor W_q, W_s; quantize_fp8(W, W_q, W_s);
    std::vector<uint8_t> qh(N * K); W_q.copy_to(qh.data(), qh.size());
    std::vector<float> sh(N); W_s.copy_to(sh.data(), N * sizeof(float));
    auto dh = [](uint8_t b) {
        uint32_t s=(b>>7)&1, e=(b>>3)&0xF, m=b&7;
        float v;
        if(e==0) v=(float)m*0.001953125f;
        else if(e==0xF) v=(m==7)?NAN:(1.0f+(float)m/8.0f)*256.0f;
        else v=(1.0f+(float)m/8.0f)*exp2f((float)e-7.0f);
        return s?-v:v;
    };
    double mx = 0;
    for (int r = 0; r < N; r++)
        for (int c = 0; c < K; c++) {
            float orig = __half2float(wh[r*K+c]);
            float deq = dh(qh[r*K+c]) * sh[r];
            mx = fmax(mx, fabs((double)orig - deq));
        }
    printf("roundtrip N=%d K=%d: maxerr=%.6f\n", N, K, mx);
    return (float)mx;
}

static float run_case(int M, int K, int N, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0, 0.2f);
    std::vector<__half> ah(M * K), wh(N * K);
    for (auto& v : ah) v = __float2half(nd(rng));
    for (auto& v : wh) v = __float2half(nd(rng));
    Tensor A({M, K}, DType::F16, Device::CUDA);
    Tensor W({N, K}, DType::F16, Device::CUDA);
    A.copy_from(ah.data(), ah.size() * sizeof(__half));
    W.copy_from(wh.data(), wh.size() * sizeof(__half));
    Tensor W_q, W_s; quantize_fp8(W, W_q, W_s);
    Tensor C1 = gemm_f16f32(A, W, true);
    Tensor C2 = gemm_f16_fp8(A, W_q, W_s, true);
    float md = maxdiff(C1, C2);
    printf("case M=%d K=%d N=%d: maxdiff=%.5f\n", M, K, N, md);
    return md;
}

int main() {
    small_case();
    roundtrip_err(4, 8, 11);
    run_case(4, 8, 4, 1);
    run_case(4, 16, 4, 2);
    run_case(4, 32, 4, 3);
    run_case(4, 16, 16, 4);
    int M = 8, K = 256, N = 512;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 0.2f);

    std::vector<__half> ah(M * K), wh(N * K);
    for (auto& v : ah) v = __float2half(nd(rng));
    for (auto& v : wh) v = __float2half(nd(rng));

    Tensor A({M, K}, DType::F16, Device::CUDA);
    Tensor W({N, K}, DType::F16, Device::CUDA);
    A.copy_from(ah.data(), ah.size() * sizeof(__half));
    W.copy_from(wh.data(), wh.size() * sizeof(__half));

    Tensor W_q, W_s;
    quantize_fp8(W, W_q, W_s);

    Tensor C1 = gemm_f16f32(A, W, true);       // reference
    Tensor C2 = gemm_f16_fp8(A, W_q, W_s, true); // fp8 path
    printf("fp8 GEMM maxdiff: %.6f\n", maxdiff(C1, C2));

    // diagnostics
    std::vector<float> a1(M * K), w1(N * K), c1(M * N), c2(M * N), s1(N);
    A.copy_to(a1.data(), A.nbytes());
    W.copy_to(w1.data(), W.nbytes());
    C1.copy_to(c1.data(), C1.nbytes());
    C2.copy_to(c2.data(), C2.nbytes());
    W_s.copy_to(s1.data(), W_s.nbytes());
    std::vector<uint8_t> q1(N * K);
    W_q.copy_to(q1.data(), q1.size());
    auto deq_host = [](uint8_t b) {
        uint32_t s=(b>>7)&1, e=(b>>3)&0xF, m=b&7;
        float v;
        if(e==0) v=(float)m*0.001953125f;
        else if(e==0xF) v=(m==7)?NAN:(1.0f+(float)m/8.0f)*256.0f;
        else v=(1.0f+(float)m/8.0f)*exp2f((float)e-7.0f);
        return s?-v:v;
    };
    printf("A[0][0:3]=%.3f %.3f %.3f  W[0][0:3]=%.4f %.4f %.4f scale[0]=%.5f\n",
           a1[0], a1[1], a1[2], w1[0], w1[1], w1[2], s1[0]);
    printf("W_q[0][0:3]=%d %d %d deq=%.4f %.4f %.4f (orig %.4f %.4f %.4f)\n",
           q1[0], q1[1], q1[2], deq_host(q1[0])*s1[0], deq_host(q1[1])*s1[0], deq_host(q1[2])*s1[0],
           w1[0], w1[1], w1[2]);
    printf("C1[0][0:3]=%.4f %.4f %.4f\nC2[0][0:3]=%.4f %.4f %.4f\n",
           c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]);

    // check the fp16 path is unchanged
    Tensor C3 = gemm_f16f32(A, W, true);
    printf("fp16 GEMM deterministic: %s\n", maxdiff(C1, C3) < 1e-6 ? "OK" : "FAIL");
    return 0;
}
