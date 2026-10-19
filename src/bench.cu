// GEMM benchmark harness: every kernel vs cuBLAS on the same inputs.
//
//   ./bench [kernel ...]          default: all kernels
//
// For each shape: cuBLAS (cublasGemmEx, fp16 in, fp32 compute, fp16 out) is
// the reference for both correctness and speed. Each kernel is checked
// element-wise against it, then timed with CUDA events around the kernel
// only, L2 flushed before every repetition, median of the repetitions.
// One JSON line per (kernel, shape) on stdout.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "kernels.cuh"

#define CK(x) do { cudaError_t ck_err_ = (x); if (ck_err_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(ck_err_), __FILE__, __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while (0)

struct Shape { int M, N, K; const char* name; };
struct Kernel { const char* name; int tile_m, tile_n, tile_k; std::function<void(const half*, const half*, half*, int, int, int)> launch; };

static cublasHandle_t g_blas;
static void cublas_gemm(const half* A, const half* W, half* C, int M, int N, int K) {
    // Row-major C[M,N] = A[M,K] W[N,K]^T  ==  column-major C^T[N,M] = W^T... : op(W)=T on [K,N] col-major view.
    float alpha = 1.f, beta = 0.f;
    CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K,
                    &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

static std::vector<Kernel> kernels() {
    return {
        {"v0_naive", 16, 16, 1, [](const half* A, const half* W, half* C, int M, int N, int K) {
             k0_naive<<<dim3(N / 16, M / 16), dim3(16, 16)>>>(A, W, C, M, N, K); }},
        {"v1_smem", T1, T1, T1, [](const half* A, const half* W, half* C, int M, int N, int K) {
             k1_smem<<<dim3(N / T1, M / T1), dim3(T1, T1)>>>(A, W, C, M, N, K); }},
        {"v2_regblock", BM2, BN2, BK2, [](const half* A, const half* W, half* C, int M, int N, int K) {
             k2_regblock<<<dim3(N / BN2, M / BM2), 256>>>(A, W, C, M, N, K); }},
        {"v3_mma", BM3, BN3, BK3, [](const half* A, const half* W, half* C, int M, int N, int K) {
             k3_mma<<<dim3(N / BN3, M / BM3), 256>>>(A, W, C, M, N, K); }},
        {"v4_cpasync", BM3, BN3, BK3, [](const half* A, const half* W, half* C, int M, int N, int K) {
             k45_pipe<false><<<dim3(N / BN3, M / BM3), 256>>>(A, W, C, M, N, K); }},
        {"v5_swizzle", BM3, BN3, BK3, [](const half* A, const half* W, half* C, int M, int N, int K) {
             k45_pipe<true><<<dim3(N / BN3, M / BM3), 256>>>(A, W, C, M, N, K); }},
    };
}

static float time_ms(const std::function<void()>& f, void* flush, size_t flush_bytes, int reps) {
    cudaEvent_t s, e;
    CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        CK(cudaMemsetAsync(flush, i, flush_bytes));   // evict A, W, C from L2
        CK(cudaEventRecord(s));
        f();
        CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e));
        float ms; CK(cudaEventElapsedTime(&ms, s, e)); t.push_back(ms);
    }
    CK(cudaGetLastError());
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char** argv) {
    std::vector<std::string> want(argv + 1, argv + argc);
    // Square plus LLM shapes: Qwen3-8B / Llama-3-8B attention-out and MLP projections at prefill M.
    std::vector<Shape> shapes = {
        {4096, 4096, 4096, "square_4096"},
        {2048, 4096, 4096, "llama8b_o_proj_m2048"},
        {2048, 14336, 4096, "llama8b_up_proj_m2048"},
        {2048, 4096, 14336, "llama8b_down_proj_m2048"},
    };
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    size_t flush_bytes = (size_t)prop.l2CacheSize * 4;
    void* flush; CK(cudaMalloc(&flush, flush_bytes));
    CB(cublasCreate(&g_blas));
    fprintf(stderr, "GPU %s, SMs %d, L2 %d KB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1024);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (const Shape& sh : shapes) {
        int M = sh.M, N = sh.N, K = sh.K;
        size_t nA = (size_t)M * K, nW = (size_t)N * K, nC = (size_t)M * N;
        std::vector<half> hA(nA), hW(nW);
        for (auto& v : hA) v = __float2half(U(rng));
        for (auto& v : hW) v = __float2half(U(rng));
        half *A, *W, *C, *Cref;
        CK(cudaMalloc(&A, nA * 2)); CK(cudaMalloc(&W, nW * 2)); CK(cudaMalloc(&C, nC * 2)); CK(cudaMalloc(&Cref, nC * 2));
        CK(cudaMemcpy(A, hA.data(), nA * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(W, hW.data(), nW * 2, cudaMemcpyHostToDevice));
        double flops = 2.0 * M * N * K;

        cublas_gemm(A, W, Cref, M, N, K);
        CK(cudaDeviceSynchronize());
        std::vector<half> hRef(nC), hC(nC);
        CK(cudaMemcpy(hRef.data(), Cref, nC * 2, cudaMemcpyDeviceToHost));
        for (int w = 0; w < 3; ++w) cublas_gemm(A, W, Cref, M, N, K);
        float blas_ms = time_ms([&] { cublas_gemm(A, W, Cref, M, N, K); }, flush, flush_bytes, 20);
        printf("{\"kernel\":\"cublas\",\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ms\":%.4f,\"tflops\":%.3f}\n",
               sh.name, M, N, K, blas_ms, flops / blas_ms / 1e9);
        fflush(stdout);

        for (const Kernel& kr : kernels()) {
            if (!want.empty() && std::find(want.begin(), want.end(), kr.name) == want.end()) continue;
            if (M % kr.tile_m || N % kr.tile_n || K % kr.tile_k) continue;
            CK(cudaMemset(C, 0, nC * 2));
            kr.launch(A, W, C, M, N, K);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hC.data(), C, nC * 2, cudaMemcpyDeviceToHost));
            // |c - ref| <= 0.05 + 1e-2 |ref|: fp16 output rounding plus a different fp32 summation order.
            double max_err = 0; size_t bad = 0;
            for (size_t i = 0; i < nC; ++i) {
                double r = __half2float(hRef[i]), c = __half2float(hC[i]), d = std::fabs(c - r);
                if (!(d <= 0.05 + 1e-2 * std::fabs(r))) ++bad;   // also counts NaN
                if (d > max_err || std::isnan(d)) max_err = std::isnan(d) ? INFINITY : d;
            }
            // Slow kernels: fewer repetitions, but at least 3.
            float probe = time_ms([&] { kr.launch(A, W, C, M, N, K); }, flush, flush_bytes, 1);
            int reps = std::max(3, std::min(20, (int)(2000.f / std::max(probe, 0.01f))));
            float ms = time_ms([&] { kr.launch(A, W, C, M, N, K); }, flush, flush_bytes, reps);
            printf("{\"kernel\":\"%s\",\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ms\":%.4f,\"tflops\":%.3f,"
                   "\"pct_cublas\":%.1f,\"max_abs_err\":%.4g,\"bad\":%zu,\"correct\":%s,\"reps\":%d}\n",
                   kr.name, sh.name, M, N, K, ms, flops / ms / 1e9, 100.0 * blas_ms / ms, max_err, bad,
                   bad == 0 ? "true" : "false", reps);
            fflush(stdout);
        }
        CK(cudaFree(A)); CK(cudaFree(W)); CK(cudaFree(C)); CK(cudaFree(Cref));
    }
    return 0;
}
