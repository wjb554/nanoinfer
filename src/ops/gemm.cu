/// GEMM — progressive fp32 backends + cuBLAS fp16 (Tensor Core)
#include "nanoinfer/ops/gemm.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace nanoinfer {
namespace ops {

GemmBackend g_gemm_backend = GemmBackend::CuBLAS;
void set_gemm_backend(GemmBackend b) { g_gemm_backend = b; }
GemmBackend get_gemm_backend() { return g_gemm_backend; }
static cublasHandle_t g_handle = nullptr;

// ====== FP32 Backends ======

template <typename T>
__global__ void gemm_naive_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    int r=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=M||c>=N)return; float acc=0;
    for(int k=0;k<K;k++)acc+=float(A[r*K+k])*float(B[k*N+c]);
    C[r*N+c]=T(acc);
}

template <typename T>
__global__ void gemm_tiled_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    constexpr int TM=16,TN=16,TK=8;
    __shared__ float As[TM][TK],Bs[TK][TN];
    int br=blockIdx.y*TM,bc=blockIdx.x*TN,ty=threadIdx.y,tx=threadIdx.x;float acc=0;
    for(int kb=0;kb<(K+TK-1)/TK;kb++){
        for(int i=ty*TN+tx;i<TM*TK;i+=256){int m=i/TK,k=i%TK;As[m][k]=(br+m<M&&kb*TK+k<K)?float(A[(br+m)*K+kb*TK+k]):0.f;}
        for(int i=ty*TN+tx;i<TK*TN;i+=256){int k=i/TN,n=i%TN;Bs[k][n]=(kb*TK+k<K&&bc+n<N)?float(B[(kb*TK+k)*N+bc+n]):0.f;}
        __syncthreads();
        for(int k=0;k<TK;k++)acc+=As[ty][k]*Bs[k][tx];
        __syncthreads();
    }
    if(br+ty<M&&bc+tx<N)C[(br+ty)*N+bc+tx]=T(acc);
}

static constexpr int TM=64,TN=64,TK=8,BM=16,BN=16,PAD=1,RM=TM/BM,RN=TN/BN;

template <typename T>
__global__ void gemm_vectorized_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ float As[TM][TK+PAD],Bs[TK][TN+PAD];
    int br=blockIdx.y*TM,bc=blockIdx.x*TN,ty=threadIdx.y,tx=threadIdx.x;
    float acc[RM][RN]={};
    for(int kb=0;kb<(K+TK-1)/TK;kb++){
        for(int i=ty*BN+tx;i<(TM*TK)/4;i+=BM*BN){int m=i/(TK/4),k4=(i%(TK/4))*4,gr=br+m,gk=kb*TK+k4;if(gr<M&&gk+3<K){float4 v=reinterpret_cast<const float4*>(A+gr*K+gk)[0];As[m][k4]=v.x;As[m][k4+1]=v.y;As[m][k4+2]=v.z;As[m][k4+3]=v.w;}else for(int d=0;d<4;d++)As[m][k4+d]=(gr<M&&gk+d<K)?float(A[gr*K+gk+d]):0.f;}
        for(int i=ty*BN+tx;i<(TK*TN)/4;i+=BM*BN){int k=i/(TN/4),n4=(i%(TN/4))*4,gk=kb*TK+k,gn=bc+n4;if(gk<K&&gn+3<N){float4 v=reinterpret_cast<const float4*>(B+gk*N+gn)[0];Bs[k][n4]=v.x;Bs[k][n4+1]=v.y;Bs[k][n4+2]=v.z;Bs[k][n4+3]=v.w;}else for(int d=0;d<4;d++)Bs[k][n4+d]=(gk<K&&gn+d<N)?float(B[gk*N+gn+d]):0.f;}
        __syncthreads();
        for(int rn=0;rn<RN;rn++)for(int k=0;k<TK;k++){float bv=Bs[k][tx*RN+rn];for(int rm=0;rm<RM;rm++)acc[rm][rn]+=As[ty*RM+rm][k]*bv;}
        __syncthreads();
    }
    for(int rm=0;rm<RM;rm++)for(int rn=0;rn<RN;rn++){int gr=br+ty*RM+rm,gn=bc+tx*RN+rn;if(gr<M&&gn<N)C[gr*N+gn]=T(acc[rm][rn]);}
}

template <typename T>
__global__ void gemm_doublebuf_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    __shared__ float As0[TM][TK+PAD],As1[TM][TK+PAD],Bs0[TK][TN+PAD],Bs1[TK][TN+PAD];
    int br=blockIdx.y*TM,bc=blockIdx.x*TN,ty=threadIdx.y,tx=threadIdx.x;
    float acc[RM][RN]={};
    // Prefetch tile 0
    for(int i=ty*BN+tx;i<(TM*TK)/4;i+=BM*BN){int m=i/(TK/4),k4=(i%(TK/4))*4,gr=br+m;if(gr<M&&k4+3<K){float4 v=reinterpret_cast<const float4*>(A+gr*K+k4)[0];As0[m][k4]=v.x;As0[m][k4+1]=v.y;As0[m][k4+2]=v.z;As0[m][k4+3]=v.w;}else for(int d=0;d<4;d++)As0[m][k4+d]=(gr<M&&k4+d<K)?float(A[gr*K+k4+d]):0.f;}
    for(int i=ty*BN+tx;i<(TK*TN)/4;i+=BM*BN){int k=i/(TN/4),n4=(i%(TN/4))*4,gn=bc+n4;if(k<K&&gn+3<N){float4 v=reinterpret_cast<const float4*>(B+k*N+gn)[0];Bs0[k][n4]=v.x;Bs0[k][n4+1]=v.y;Bs0[k][n4+2]=v.z;Bs0[k][n4+3]=v.w;}else for(int d=0;d<4;d++)Bs0[k][n4+d]=(k<K&&gn+d<N)?float(B[k*N+gn+d]):0.f;}
    __syncthreads();
    for(int kb=1;kb<(K+TK-1)/TK;kb++){
        int gk_off=kb*TK;
        for(int i=ty*BN+tx;i<(TM*TK)/4;i+=BM*BN){int m=i/(TK/4),k4=(i%(TK/4))*4,gr=br+m,gk=gk_off+k4;if(gr<M&&gk+3<K){float4 v=reinterpret_cast<const float4*>(A+gr*K+gk)[0];As1[m][k4]=v.x;As1[m][k4+1]=v.y;As1[m][k4+2]=v.z;As1[m][k4+3]=v.w;}else for(int d=0;d<4;d++)As1[m][k4+d]=(gr<M&&gk+d<K)?float(A[gr*K+gk+d]):0.f;}
        for(int i=ty*BN+tx;i<(TK*TN)/4;i+=BM*BN){int k=i/(TN/4),n4=(i%(TN/4))*4,gk=gk_off+k,gn=bc+n4;if(gk<K&&gn+3<N){float4 v=reinterpret_cast<const float4*>(B+gk*N+gn)[0];Bs1[k][n4]=v.x;Bs1[k][n4+1]=v.y;Bs1[k][n4+2]=v.z;Bs1[k][n4+3]=v.w;}else for(int d=0;d<4;d++)Bs1[k][n4+d]=(gk<K&&gn+d<N)?float(B[gk*N+gn+d]):0.f;}
        // Compute from set 0, next tile loads into set 1 in parallel
        for(int rn=0;rn<RN;rn++)for(int k=0;k<TK;k++){float bv=Bs0[k][tx*RN+rn];for(int rm=0;rm<RM;rm++)acc[rm][rn]+=As0[ty*RM+rm][k]*bv;}
        __syncthreads();
        // Swap
        for(int i=ty*BN+tx;i<(TM*TK)/4;i+=BM*BN){int m=i/(TK/4),k4=(i%(TK/4))*4;As0[m][k4]=As1[m][k4];As0[m][k4+1]=As1[m][k4+1];As0[m][k4+2]=As1[m][k4+2];As0[m][k4+3]=As1[m][k4+3];}
        for(int i=ty*BN+tx;i<(TK*TN)/4;i+=BM*BN){int k=i/(TN/4),n4=(i%(TN/4))*4;Bs0[k][n4]=Bs1[k][n4];Bs0[k][n4+1]=Bs1[k][n4+1];Bs0[k][n4+2]=Bs1[k][n4+2];Bs0[k][n4+3]=Bs1[k][n4+3];}
        __syncthreads();
    }
    for(int rn=0;rn<RN;rn++)for(int k=0;k<TK;k++){float bv=Bs0[k][tx*RN+rn];for(int rm=0;rm<RM;rm++)acc[rm][rn]+=As0[ty*RM+rm][k]*bv;}
    for(int rm=0;rm<RM;rm++)for(int rn=0;rn<RN;rn++){int gr=br+ty*RM+rm,gn=bc+tx*RN+rn;if(gr<M&&gn<N)C[gr*N+gn]=T(acc[rm][rn]);}
}

// ====== Dispatch ======

Tensor gemm(const Tensor& a, const Tensor& b, bool transpose_b) {
    int M=a.size(0),K=a.size(1),N=transpose_b?b.size(0):b.size(1);
    if(a.size(1)!=(transpose_b?b.size(1):b.size(0)))throw std::runtime_error("gemm: inner dim mismatch");
    bool fp16=(a.dtype()==DType::F16);
    Tensor c({M,N},fp16?DType::F16:DType::F32,Device::CUDA);

    if(g_gemm_backend==GemmBackend::CuBLAS||fp16){
        float al=1,be=0;if(!g_handle)cublasCreate(&g_handle);
        if(fp16)cublasGemmEx(g_handle,transpose_b?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,
            N,M,K,&al,b.raw(),CUDA_R_16F,transpose_b?K:N,a.raw(),CUDA_R_16F,K,
            &be,c.raw(),CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
        else cublasSgemm_v2(g_handle,transpose_b?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,
            N,M,K,&al,b.data<float>(),transpose_b?K:N,a.data<float>(),K,&be,c.data<float>(),N);
    }else{
        if(g_gemm_backend==GemmBackend::Naive){dim3 b16(16,16),g16((N+15)/16,(M+15)/16);gemm_naive_kernel<float><<<g16,b16>>>(a.data<float>(),b.data<float>(),c.data<float>(),M,N,K);}
        else if(g_gemm_backend==GemmBackend::Tiled){dim3 b16(16,16),g16((N+15)/16,(M+15)/16);gemm_tiled_kernel<float><<<g16,b16>>>(a.data<float>(),b.data<float>(),c.data<float>(),M,N,K);}
        else if(g_gemm_backend==GemmBackend::Vectorized){dim3 bv(BN,BM),gv((N+TN-1)/TN,(M+TM-1)/TM);gemm_vectorized_kernel<float><<<gv,bv>>>(a.data<float>(),b.data<float>(),c.data<float>(),M,N,K);}
        else{dim3 bv(BN,BM),gv((N+TN-1)/TN,(M+TM-1)/TM);gemm_doublebuf_kernel<float><<<gv,bv>>>(a.data<float>(),b.data<float>(),c.data<float>(),M,N,K);}
    }
    return std::move(c);
}

// fp16 inputs → fp32 output (mixed precision via Tensor Cores)
Tensor gemm_f16f32(const Tensor& a, const Tensor& b, bool transpose_b) {
    int M=a.size(0), K=a.size(1), N=transpose_b?b.size(0):b.size(1);
    if(a.size(1)!=(transpose_b?b.size(1):b.size(0)))
        throw std::runtime_error("gemm_f16f32: inner dim mismatch");
    if(a.dtype()!=DType::F16||b.dtype()!=DType::F16)
        throw std::runtime_error("gemm_f16f32: both inputs must be fp16");

    Tensor c({M,N}, DType::F32, Device::CUDA);

    float al=1, be=0;
    if(!g_handle) cublasCreate(&g_handle);
    cublasGemmEx(g_handle,
        transpose_b?CUBLAS_OP_T:CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &al,
        b.raw(), CUDA_R_16F, transpose_b?K:N,
        a.raw(), CUDA_R_16F, K,
        &be,
        c.raw(), CUDA_R_32F, N,          // ← fp32 output!
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    return std::move(c);
}



// ============================================================================
// FP8 E4M3 weight-only quantization (per-row scale) + fused-dequant GEMM
// ============================================================================

// Manual E4M3 conversions (1 sign + 4 exp bias 7 + 3 mant).  Max finite 448.
__device__ inline uint8_t float_to_fp8_e4m3(float x) {
    uint32_t u = __float_as_uint(x);
    uint32_t s = (u >> 31) & 1;
    uint32_t e = (u >> 23) & 0xFF;
    uint32_t m = u & 0x7FFFFF;
    if ((u & 0x7FFFFFFF) == 0) return (uint8_t)(s << 7);   // zero
    if (e == 0xFF) return (uint8_t)((s << 7) | 0x7F);      // nan -> nan
    int32_t ex = (int32_t)e - 127 + 7;                     // e4m3 exp field
    if (ex <= 0) return (uint8_t)(s << 7);                 // |x| < 2^-6 -> 0
    if (ex >= 16) return (uint8_t)((s << 7) | 0x7E);       // overflow beyond 448
    if (ex == 15) {
        // x in [256, 512): value = (1 + m/8)*256, so mant = round(m / 2^20).
        uint32_t me = (m + (1u << 19)) >> 20;
        if (me > 6) me = 6;                                // 448 = exp15, mant6
        return (uint8_t)((s << 7) | (0xF << 3) | me);
    }
    uint32_t mant = (m + (1u << 19)) >> 20;                // 23->3 bit, round
    if (mant == 8) { mant = 0; ex++; if (ex >= 16) return (uint8_t)((s << 7) | 0x7E); }
    return (uint8_t)((s << 7) | ((uint32_t)ex << 3) | mant);
}
__device__ inline float fp8_to_float(uint8_t b) {
    uint32_t s = (b >> 7) & 1;
    uint32_t e = (b >> 3) & 0xF;
    uint32_t m = b & 7;
    float v;
    if (e == 0) v = (float)m * 0.001953125f;               // denormal: m * 2^-9
    else if (e == 0xF) v = (m == 7) ? NAN : (1.0f + (float)m / 8.0f) * 256.0f;
    else v = (1.0f + (float)m / 8.0f) * exp2f((float)e - 7.0f);
    return s ? -v : v;
}

// Quantize FP16 weights [R,C] -> FP8 E4M3 [R,C] with per-row scale.
// scale[r] = max|w[r,:]| / 448  (E4M3 max finite value).
__global__ void quantize_fp8_kernel(const half* w, uint8_t* q, float* scale,
                                    int R, int C) {
    int r = blockIdx.x;
    float mx = 0;
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        mx = fmaxf(mx, fabsf(__half2float(w[r * C + i])));
    __shared__ float sm[32];
    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (lid == 0) sm[wid] = mx;
    __syncthreads();
    if (tid == 0) {
        mx = sm[0];
        for (int w = 1; w < (int)(blockDim.x / 32); w++) mx = fmaxf(mx, sm[w]);
        sm[0] = mx;
    }
    __syncthreads();
    float sc = sm[0] / 448.0f;
    if (sc == 0) sc = 1.0f;
    if (threadIdx.x == 0) scale[r] = sc;
    __syncthreads();
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = __half2float(w[r * C + i]);
        q[r * C + i] = float_to_fp8_e4m3(v / sc);
    }
}

void quantize_fp8(const Tensor& w, Tensor& q, Tensor& scale) {
    int R = w.size(0), C = w.size(1);
    q = Tensor({R, C}, DType::FP8_E4M3, Device::CUDA);
    scale = Tensor({R}, DType::F32, Device::CUDA);
    quantize_fp8_kernel<<<R, 256>>>(w.data<half>(), q.data<uint8_t>(),
                                    scale.data<float>(), R, C);
}

// C[M,N] = A[M,K] @ dequant(B[N,K])^T  — B fp8 with per-N scale, fused dequant.
constexpr int kFP8TM = 16, kFP8TN = 16, kFP8TK = 8;
__global__ void gemm_f16_fp8_kernel(
    const half* __restrict__ A, const uint8_t* __restrict__ B,
    const float* __restrict__ Bscale, float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half As[kFP8TM][kFP8TK];
    __shared__ half Bs[kFP8TN][kFP8TK];
    int br = blockIdx.y * kFP8TM, bc = blockIdx.x * kFP8TN;
    int ty = threadIdx.y, tx = threadIdx.x;
    float acc = 0;
    for (int kb = 0; kb < (K + kFP8TK - 1) / kFP8TK; kb++) {
        for (int i = ty * kFP8TN + tx; i < kFP8TM * kFP8TK; i += 256) {
            int m = i / kFP8TK, k = i % kFP8TK;
            As[m][k] = (br + m < M && kb * kFP8TK + k < K)
                           ? A[(br + m) * K + kb * kFP8TK + k] : __float2half(0);
        }
        for (int i = ty * kFP8TN + tx; i < kFP8TN * kFP8TK; i += 256) {
            int n = i / kFP8TK, k = i % kFP8TK;
            float v = (bc + n < N && kb * kFP8TK + k < K)
                          ? fp8_to_float(B[(bc + n) * K + kb * kFP8TK + k]) * Bscale[bc + n]
                          : 0.f;
            Bs[n][k] = __float2half(v);
        }
        __syncthreads();
        for (int k = 0; k < kFP8TK; k++)
            acc += __half2float(As[ty][k]) * __half2float(Bs[tx][k]);
        __syncthreads();
    }
    if (br + ty < M && bc + tx < N) C[(br + ty) * N + bc + tx] = acc;
}

Tensor gemm_f16_fp8(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale,
                    bool transpose_b) {
    int M = a.size(0), K = a.size(1);
    int N = transpose_b ? b_fp8.size(0) : b_fp8.size(1);
    if (a.dtype() != DType::F16 || b_fp8.dtype() != DType::FP8_E4M3)
        throw std::runtime_error("gemm_f16_fp8: need fp16 activation + fp8 weight");
    Tensor c({M, N}, DType::F32, Device::CUDA);
    dim3 block(kFP8TN, kFP8TM), grid((N + kFP8TN - 1) / kFP8TN, (M + kFP8TM - 1) / kFP8TM);
    gemm_f16_fp8_kernel<<<grid, block>>>(
        a.data<half>(), b_fp8.data<uint8_t>(), b_scale.data<float>(),
        c.data<float>(), M, N, K);
    return std::move(c);
}

// ============================================================================
// FP8 weight-only via cuBLAS (Route A): dequant fp8→fp16 scratch, then
// cublasGemmEx fp16×fp16→fp32 (tensor cores).  Same W8A16 dequant numerics
// as gemm_f16_fp8_kernel, but the GEMM itself is cuBLAS instead of the naive
// custom 16x16 kernel.  The fp8 weights stay 1 byte in memory (2x savings);
// the fp16 scratch is per-GEMM transient.
// ============================================================================
__global__ void dequant_fp8_to_f16_kernel(const uint8_t* __restrict__ w,
                                          const float* __restrict__ scale,
                                          __half* __restrict__ wf,
                                          int N, int K) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * K;
    if (i >= total) return;
    int n = (int)(i / K);
    wf[i] = __float2half(fp8_to_float(w[i]) * scale[n]);
}

Tensor gemm_f16_fp8_cublas(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale,
                           bool transpose_b) {
    int N = b_fp8.size(0), K = b_fp8.size(1);
    if (a.dtype() != DType::F16 || b_fp8.dtype() != DType::FP8_E4M3)
        throw std::runtime_error("gemm_f16_fp8_cublas: need fp16 activation + fp8 weight");
    int64_t total = (int64_t)N * K;
    Tensor w_f16({N, K}, DType::F16, Device::CUDA);
    dequant_fp8_to_f16_kernel<<<(int)((total + 255) / 256), 256>>>(
        b_fp8.data<uint8_t>(), b_scale.data<float>(), w_f16.data<__half>(), N, K);
    return gemm_f16f32(a, w_f16, transpose_b);
}

// ============================================================================
// W8A8 (Route B): A(fp16) quantized on-the-fly to fp8 (per-token scale), then
// cublasGemmEx fp8×fp8→fp32 (FP8 tensor cores, sm_89+).  The raw GEMM output
// is scaled by sA[i]*sB[j] (per-token activation × per-row weight) afterwards.
// C_true[i,j] = sum_k (f8A[i,k]*sA[i]) * (f8B[j,k]*sB[j])
//             = sA[i]*sB[j] * (f8A @ f8B^T)[i,j]
// ============================================================================
__global__ void apply_fp8_scale_kernel(float* __restrict__ C,
                                       const float* __restrict__ sA,
                                       const float* __restrict__ sB,
                                       int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) return;
    C[i * N + j] *= sA[i] * sB[j];
}

// Probe once whether cuBLAS fp8 GEMM is available.  cuBLAS 11.x supports fp8
// only on Hopper (sm_90); Ada (sm_89, e.g. RTX 4090) needs CUDA 12.x.  On an
// unsupported platform cublasGemmEx returns CUBLAS_STATUS_NOT_SUPPORTED and
// leaves C uninitialized — so we must not feed garbage into the engine.
static bool fp8_w8a8_supported() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    const int M = 8, K = 8, N = 8;
    uint8_t *dA = nullptr, *dB = nullptr; float* dC = nullptr;
    cudaMalloc(&dA, M * K); cudaMalloc(&dB, N * K); cudaMalloc(&dC, M * N * sizeof(float));
    cudaMemset(dA, 0, M * K); cudaMemset(dB, 0, N * K); cudaMemset(dC, 0, M * N * sizeof(float));
    if (!g_handle) cublasCreate(&g_handle);
    float al = 1, be = 0;
    cublasStatus_t st = cublasGemmEx(g_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K, &al, dB, CUDA_R_8F_E4M3, K, dA, CUDA_R_8F_E4M3, K,
        &be, dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cached = (st == CUBLAS_STATUS_SUCCESS);
    return cached != 0;
}

bool fp8_w8a8_available() { return fp8_w8a8_supported(); }

Tensor gemm_f16_fp8_w8a8(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale,
                         bool transpose_b) {
    int M = a.size(0), K = a.size(1);
    int N = transpose_b ? b_fp8.size(0) : b_fp8.size(1);
    if (a.dtype() != DType::F16 || b_fp8.dtype() != DType::FP8_E4M3)
        throw std::runtime_error("gemm_f16_fp8_w8a8: need fp16 activation + fp8 weight");

    // cuBLAS fp8 GEMM unavailable (e.g. cuBLAS 11.x on sm_89) → fall back to the
    // W8A16 dequant path (Route A).  Keeps fp8 output correct everywhere.
    if (!fp8_w8a8_supported())
        return gemm_f16_fp8_cublas(a, b_fp8, b_scale, transpose_b);

    // 1. Quantize activations to fp8 with a per-token (per-row) scale.
    Tensor a_f8, a_scale;
    quantize_fp8(a, a_f8, a_scale);          // a[M,K] fp16 → a_f8[M,K] fp8 + a_scale[M]

    // 2. Raw fp8×fp8 → fp32 via cuBLAS (FP8 tensor cores).
    Tensor c({M, N}, DType::F32, Device::CUDA);
    float al = 1, be = 0;
    if (!g_handle) cublasCreate(&g_handle);
    cublasGemmEx(g_handle,
        transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &al,
        b_fp8.raw(), CUDA_R_8F_E4M3, transpose_b ? K : N,
        a_f8.raw(),  CUDA_R_8F_E4M3, K,
        &be,
        c.raw(), CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 3. Apply per-token × per-row-weight scales.
    dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
    apply_fp8_scale_kernel<<<grd, blk>>>(c.data<float>(),
        a_scale.data<float>(), b_scale.data<float>(), M, N);
    return std::move(c);
}

// ============================================================================
// Optimized fp8 weight-only GEMM (Route A numerics, hand-written):
//   C[M,N] = A[M,K](fp16) @ dequant(B[N,K](fp8)·scale[N])^T
// Same fused-dequant W8A16 semantics as gemm_f16_fp8_kernel, but with the
// classic GEMM optimizations already used for the fp32 backends in this file:
//   - 64×64 output tile, 16×16 threads → each thread computes a 4×4 register
//     block (16 outputs vs the naive kernel's 1)
//   - vectorized loads: A as float4 (8 halves), B as uint32 (4 fp8 bytes)
//   - double-buffered shared memory (load next K-tile while computing)
// fp8 weights stay 1 byte in memory; the dequant happens on shared-mem load.
// ============================================================================
constexpr int kFP8O_TK = 8;
constexpr int kFP8O_BM = 16, kFP8O_BN = 16, kFP8O_PAD = 1;
constexpr int kFP8O_RN = 4;

// Templated on the register-block height RM: TM = 16*RM.  RM=1 for decode
// (M<16) avoids the 4x wasted MACs of RM=4; RM=2 for mid-size; RM=4 for
// prefill (large M) where the full 4x4 block is used.
template<int RM>
__global__ void gemm_f16_fp8_opt_kernel(
    const half* __restrict__ A, const uint8_t* __restrict__ B,
    const float* __restrict__ Bscale, float* __restrict__ C,
    int M, int N, int K)
{
    constexpr int BM = kFP8O_BM, BN = kFP8O_BN, PAD = kFP8O_PAD;
    constexpr int RN = kFP8O_RN, TK = kFP8O_TK;
    constexpr int TM = BM * RM;   // 16 / 32 / 64
    constexpr int TN = BN * RN;   // 64
    __shared__ half As0[TM][TK+PAD], As1[TM][TK+PAD];
    __shared__ half Bs0[TK][TN+PAD], Bs1[TK][TN+PAD];
    int br = blockIdx.y * TM, bc = blockIdx.x * TN;
    int ty = threadIdx.y, tx = threadIdx.x;
    float acc[RM][RN] = {};

    // ---- load A tile [TM,TK] fp16 (float4 = 8 halves) into shared ----
    auto loadA = [&](half (*As)[TK+PAD], int kb) {
        int gk0 = kb * TK;
        for (int i = ty * BN + tx; i < (TM * TK) / 8; i += BM * BN) {
            int m = i / (TK / 8), k8 = (i % (TK / 8)) * 8;
            int gr = br + m, gk = gk0 + k8;
            if (gr < M && gk + 7 < K) {
                float4 v = *(const float4*)(A + gr * K + gk);
                const __half* hp = reinterpret_cast<const __half*>(&v);
                #pragma unroll
                for (int d = 0; d < 8; d++) As[m][k8 + d] = hp[d];
            } else {
                #pragma unroll
                for (int d = 0; d < 8; d++)
                    As[m][k8 + d] = (gr < M && gk + d < K) ? A[gr * K + gk + d] : __half(0);
            }
        }
    };
    // ---- load B tile [TK,TN] fp8, dequant with per-row scale into shared ----
    // B is [N, K] row-major: 4 consecutive bytes = 4 consecutive K elements of
    // one row (one output column), all scaled by the same per-row scale.  Store
    // them as Bs[k4..k4+3][n] (K-contiguous in shared, matching the compute loop).
    auto loadB = [&](half (*Bs)[TN+PAD], int kb) {
        int gk0 = kb * TK;
        for (int i = ty * BN + tx; i < (TK * TN) / 4; i += BM * BN) {
            int n = i % TN;                // 0..63 (output column)
            int k4 = (i / TN) * 4;         // 0 or 4 (4 consecutive K, 4-byte aligned)
            int gk = gk0 + k4, gn = bc + n;
            if (gk + 3 < K && gn < N) {
                uint32_t w = *(const uint32_t*)(B + gn * K + gk);
                float sc = Bscale[gn];
                Bs[k4 + 0][n] = __float2half(fp8_to_float(w & 0xFF)       * sc);
                Bs[k4 + 1][n] = __float2half(fp8_to_float((w >> 8) & 0xFF)  * sc);
                Bs[k4 + 2][n] = __float2half(fp8_to_float((w >> 16) & 0xFF) * sc);
                Bs[k4 + 3][n] = __float2half(fp8_to_float((w >> 24) & 0xFF) * sc);
            } else {
                #pragma unroll
                for (int d = 0; d < 4; d++)
                    Bs[k4 + d][n] = (gk + d < K && gn < N)
                        ? __float2half(fp8_to_float(B[gn * K + gk + d]) * Bscale[gn])
                        : __half(0);
            }
        }
    };
    // ---- 4x4 register-blocked MAC over a shared tile ----
    auto compute = [&](half (*As)[TK+PAD], half (*Bs)[TN+PAD]) {
        for (int rn = 0; rn < RN; rn++)
            for (int k = 0; k < TK; k++) {
                float bv = __half2float(Bs[k][tx * RN + rn]);
                for (int rm = 0; rm < RM; rm++)
                    acc[rm][rn] += __half2float(As[ty * RM + rm][k]) * bv;
            }
    };
    auto swapTiles = [&]() {
        for (int i = ty * BN + tx; i < (TM * TK) / 8; i += BM * BN) {
            int m = i / (TK / 8), k8 = (i % (TK / 8)) * 8;
            #pragma unroll
            for (int d = 0; d < 8; d++) As0[m][k8 + d] = As1[m][k8 + d];
        }
        for (int i = ty * BN + tx; i < (TK * TN) / 4; i += BM * BN) {
            int k = i / (TN / 4), n4 = (i % (TN / 4)) * 4;
            #pragma unroll
            for (int d = 0; d < 4; d++) Bs0[k][n4 + d] = Bs1[k][n4 + d];
        }
    };

    loadA(As0, 0); loadB(Bs0, 0);
    __syncthreads();
    int nkb = (K + TK - 1) / TK;
    for (int kb = 1; kb < nkb; kb++) {
        loadA(As1, kb); loadB(Bs1, kb);   // prefetch next K-tile
        compute(As0, Bs0);                // compute current tile
        __syncthreads();
        swapTiles();                      // move set1 → set0
        __syncthreads();
    }
    compute(As0, Bs0);

    for (int rm = 0; rm < RM; rm++)
        for (int rn = 0; rn < RN; rn++) {
            int gr = br + ty * RM + rm, gn = bc + tx * RN + rn;
            if (gr < M && gn < N) C[gr * N + gn] = acc[rm][rn];
        }
}

Tensor gemm_f16_fp8_opt(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale,
                        bool transpose_b) {
    int M = a.size(0), K = a.size(1);
    int N = transpose_b ? b_fp8.size(0) : b_fp8.size(1);
    if (a.dtype() != DType::F16 || b_fp8.dtype() != DType::FP8_E4M3)
        throw std::runtime_error("gemm_f16_fp8_opt: need fp16 activation + fp8 weight");
    Tensor c({M, N}, DType::F32, Device::CUDA);
    dim3 block(kFP8O_BN, kFP8O_BM);   // 16x16 threads
    const half* A = a.data<half>();
    const uint8_t* B = b_fp8.data<uint8_t>();
    const float* S = b_scale.data<float>();
    float* C = c.data<float>();
    // Adaptive register-block height: RM=1 for decode (M<16) to avoid 4x wasted
    // MACs, RM=2 for mid, RM=4 for prefill (large M).  TN=64 fixed.
    if (M < 16) {
        gemm_f16_fp8_opt_kernel<1><<<dim3((N + 63) / 64, (M + 15) / 16), block>>>(A, B, S, C, M, N, K);
    } else if (M < 32) {
        gemm_f16_fp8_opt_kernel<2><<<dim3((N + 63) / 64, (M + 31) / 32), block>>>(A, B, S, C, M, N, K);
    } else {
        gemm_f16_fp8_opt_kernel<4><<<dim3((N + 63) / 64, (M + 63) / 64), block>>>(A, B, S, C, M, N, K);
    }
    return std::move(c);
}

// ============================================================================
// Dedicated fp8 weight-only GEMV for decode (M==1):
//   C[1,N] = A[1,K](fp16) @ dequant(B[N,K](fp8)·scale[N])^T
// Why a GEMV instead of a GEMM kernel:
//   - M=1 means the only parallelism is over N; a GEMM's M-tiling/register
//     blocking wastes compute.  Here each thread owns one output column.
//   - The workload is bandwidth-bound: B[N,K] fp8 (1 byte/element) must be
//     read exactly once.  No dequant-to-fp16 materialization (Route A writes
//     2 bytes then cuBLAS reads 2 more), so per-weight traffic is 1B not 5B.
// Layout trick: B is [N,K] row-major, so 4 consecutive bytes are 4 consecutive
// K of one row (one output column).  Threads read them coalesced as uint32 and
// store into a TRANSPOSED shared tile Bs[k][n], so the compute loop reads
// consecutive columns per thread (bank-conflict-free) and A is broadcast.
// ============================================================================
constexpr int kGEMV_NT   = 128;   // output columns per block
constexpr int kGEMV_KSEG = 16;    // K per segment (per-part shared tile)
constexpr int kGEMV_SPLIT = 4;    // split-K parts per column (block = NT*SPLIT threads)

// Split-K GEMV: each output column's K-dot is split across SPLIT threads (each
// owns its own shared B tile → they compute concurrently).  This raises the
// thread count to N*SPLIT, which is what actually saturates memory bandwidth
// for the M=1 case (thread-per-column alone leaves the GPU mostly idle).
__global__ void gemv_f16_fp8_kernel(
    const half* __restrict__ A, const uint8_t* __restrict__ B,
    const float* __restrict__ Bscale, float* __restrict__ C,
    int N, int K)
{
    constexpr int NT = kGEMV_NT, KSEG = kGEMV_KSEG, SPLIT = kGEMV_SPLIT;
    constexpr int NPAD = NT + 1;             // pad: stride (NT+1)%32==1
    __shared__ float As[SPLIT][KSEG];
    __shared__ float Bs[SPLIT][KSEG][NPAD];  // per-part transposed tile
    __shared__ float Red[SPLIT][NPAD];
    int n0 = blockIdx.x * NT;
    int tid = threadIdx.x;                   // 0..NT*SPLIT-1
    int part = tid / NT;
    int n_local = tid % NT;
    int KperPart = (K + SPLIT - 1) / SPLIT;
    int k0_begin = part * KperPart;
    int k0_end = (k0_begin + KperPart < K) ? k0_begin + KperPart : K;
    float acc = 0;

    for (int k0 = k0_begin; k0 < k0_end; k0 += KSEG) {
        // A segment for this part.
        if (n_local < KSEG && k0 + n_local < K) As[part][n_local] = __half2float(A[k0 + n_local]);

        // Per-part B tile [NT cols x KSEG] → transposed Bs[part][k][n].
        for (int p = n_local; p < NT * (KSEG / 4); p += NT) {
            int n_l = p / (KSEG / 4);
            int k4 = (p % (KSEG / 4)) * 4;
            int n = n0 + n_l;
            if (n < N && k0 + k4 + 3 < K) {
                uint32_t w = *(const uint32_t*)(B + (size_t)n * K + k0 + k4);
                Bs[part][k4 + 0][n_l] = fp8_to_float(w & 0xFF);
                Bs[part][k4 + 1][n_l] = fp8_to_float((w >> 8) & 0xFF);
                Bs[part][k4 + 2][n_l] = fp8_to_float((w >> 16) & 0xFF);
                Bs[part][k4 + 3][n_l] = fp8_to_float((w >> 24) & 0xFF);
            } else {
                for (int d = 0; d < 4; d++)
                    Bs[part][k4 + d][n_l] = (n < N && k0 + k4 + d < K)
                        ? fp8_to_float(B[(size_t)n * K + k0 + k4 + d]) : 0.f;
            }
        }
        __syncthreads();

        // 4 independent accumulators (ILP) over this part's segment.
        if (k0 + KSEG <= k0_end) {
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            #pragma unroll
            for (int k = 0; k < KSEG; k += 4) {
                a0 += As[part][k + 0] * Bs[part][k + 0][n_local];
                a1 += As[part][k + 1] * Bs[part][k + 1][n_local];
                a2 += As[part][k + 2] * Bs[part][k + 2][n_local];
                a3 += As[part][k + 3] * Bs[part][k + 3][n_local];
            }
            acc += (a0 + a1) + (a2 + a3);
        } else {
            for (int k = 0; k < KSEG && k0 + k < K; k++)
                acc += As[part][k] * Bs[part][k][n_local];
        }
        __syncthreads();   // compute done before next segment overwrites this part's tile
    }

    // Cross-part reduction (SPLIT partials per column), part 0 stores.
    Red[part][n_local] = acc;
    __syncthreads();
    if (part == 0 && n0 + n_local < N) {
        float t = 0;
        for (int p = 0; p < SPLIT; p++) t += Red[p][n_local];
        C[n0 + n_local] = t * Bscale[n0 + n_local];
    }
}

Tensor gemv_f16_fp8(const Tensor& a, const Tensor& b_fp8, const Tensor& b_scale) {
    int K = a.size(1), N = b_fp8.size(0);
    if (a.dtype() != DType::F16 || b_fp8.dtype() != DType::FP8_E4M3)
        throw std::runtime_error("gemv_f16_fp8: need fp16 activation + fp8 weight");
    if (a.size(0) != 1)
        throw std::runtime_error("gemv_f16_fp8: M must be 1 (decode only; use gemm_f16_fp8_opt for M>1)");
    Tensor c({1, N}, DType::F32, Device::CUDA);
    int threads = kGEMV_NT * kGEMV_SPLIT;   // 512
    gemv_f16_fp8_kernel<<<(N + kGEMV_NT - 1) / kGEMV_NT, threads>>>(
        a.data<half>(), b_fp8.data<uint8_t>(), b_scale.data<float>(),
        c.data<float>(), N, K);
    return std::move(c);
}

}  // namespace ops
}  // namespace nanoinfer
