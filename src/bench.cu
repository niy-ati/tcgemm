// GEMM benchmark harness: every kernel vs cuBLAS on the same inputs.
//
//   ./bench [kernel ...]          default: all kernels
//
// For each shape: cuBLAS (cublasGemmEx, fp16 in, fp32 compute, fp16 out) is
// the reference for both correctness and speed. Each kernel is checked
// element-wise against it, then timed with CUDA events around the kernel
// only, L2 flushed before every repetition, median of the repetitions.
// One JSON line per (kernel, shape) on stdout.
#include <cublasLt.h>
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

// v8: per-shape autotuning. The first call for a shape times each candidate
// (v6 128x128, v7 256x128) and caches the fastest; later calls dispatch to it.
// No single tile wins every shape (see results), which is what cuBLAS's
// heuristics handle too.
#include <map>
#include <tuple>
static void launch_v6(const half* A, const half* W, half* C, int M, int N, int K) {
    k45_pipe<true, true><<<(N / BN3) * (M / BM3), 256>>>(A, W, C, M, N, K);
}
template <int BM, int BN>
static void launch_big(const half* A, const half* W, half* C, int M, int N, int K) {
    constexpr int smem = smem_big<BM, BN>();
    static bool attr = (cudaFuncSetAttribute(k7_big<BM, BN>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), true);
    (void)attr;
    k7_big<BM, BN><<<(N / BN) * (M / BM), 256, smem>>>(A, W, C, M, N, K);
}
static void launch_v7(const half* A, const half* W, half* C, int M, int N, int K) { launch_big<256, 128>(A, W, C, M, N, K); }
static void launch_v9(const half* A, const half* W, half* C, int M, int N, int K) { launch_big<128, 256>(A, W, C, M, N, K); }
static std::map<std::tuple<int, int, int>, int> g_choice;
static void launch_auto(const half* A, const half* W, half* C, int M, int N, int K) {
    auto key = std::make_tuple(M, N, K);
    auto it = g_choice.find(key);
    using Fn = void (*)(const half*, const half*, half*, int, int, int);
    static const Fn cand[3] = {launch_v6, launch_v7, launch_v9};
    static const int tm[3] = {BM3, 256, 128}, tn[3] = {BN3, 128, 256};
    static const char* names[3] = {"v6_raster", "v7_256x128", "v9_128x256"};
    if (it == g_choice.end()) {
        float best = 1e30f;
        int pick = 0;
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);
        for (int c = 0; c < 3; ++c) {
            if (M % tm[c] || N % tn[c]) continue;
            for (int w = 0; w < 2; ++w) cand[c](A, W, C, M, N, K);  // warm-up
            std::vector<float> t;
            for (int r = 0; r < 15; ++r) {  // median of 15: 5 runs picked the wrong tile under noise
                cudaEventRecord(s);
                cand[c](A, W, C, M, N, K);
                cudaEventRecord(e);
                cudaEventSynchronize(e);
                float ms;
                cudaEventElapsedTime(&ms, s, e);
                t.push_back(ms);
            }
            std::sort(t.begin(), t.end());
            if (t[7] < best) { best = t[7]; pick = c; }
        }
        cudaEventDestroy(s);
        cudaEventDestroy(e);
        it = g_choice.emplace(key, pick).first;
        fprintf(stderr, "autotune %dx%dx%d -> %s\n", M, N, K, names[pick]);
    }
    cand[it->second](A, W, C, M, N, K);
}

// Stronger baseline: cuBLASLt asked for up to 16 algorithms for the shape,
// each timed, fastest kept ("did you autotune cuBLAS too?"). Same column-major
// trick as cublas_gemm. Workspace 32 MB so split-K algorithms are eligible.
static cublasLtHandle_t g_lt;
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; int n_tried; };
static std::map<std::tuple<int, int, int>, LtPlan> g_lt_plans;
static void* g_lt_ws = nullptr;
static const size_t LT_WS = 32u << 20;
static void lt_run(const LtPlan& p, const half* A, const half* W, half* C) {
    float alpha = 1.f, beta = 0.f;
    CB(cublasLtMatmul(g_lt, p.op, &alpha, W, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.algo, g_lt_ws, LT_WS, 0));
}
static void cublaslt_best(const half* A, const half* W, half* C, int M, int N, int K) {
    auto key = std::make_tuple(M, N, K);
    auto it = g_lt_plans.find(key);
    if (it == g_lt_plans.end()) {
        if (!g_lt_ws) CK(cudaMalloc(&g_lt_ws, LT_WS));
        LtPlan p{};
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        CB(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
        CB(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
        CB(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, K, N, K));   // W viewed as K x N col-major, op T
        CB(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, K, M, K));   // A viewed as K x M col-major
        CB(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16F, N, M, N));   // C^T, N x M col-major
        cublasLtMatmulPreference_t pref;
        CB(cublasLtMatmulPreferenceCreate(&pref));
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &LT_WS, sizeof(LT_WS)));
        cublasLtMatmulHeuristicResult_t res[16];
        int found = 0;
        CB(cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.c, p.c, pref, 16, res, &found));
        cudaEvent_t s0, e0;
        CK(cudaEventCreate(&s0));
        CK(cudaEventCreate(&e0));
        float best = 1e30f;
        for (int i = 0; i < found; ++i) {
            p.algo = res[i].algo;
            float alpha = 1.f, beta = 0.f;
            if (cublasLtMatmul(g_lt, p.op, &alpha, W, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.algo, g_lt_ws, LT_WS, 0) != CUBLAS_STATUS_SUCCESS) continue;
            std::vector<float> t;
            for (int r = 0; r < 9; ++r) {
                CK(cudaEventRecord(s0));
                lt_run(p, A, W, C);
                CK(cudaEventRecord(e0));
                CK(cudaEventSynchronize(e0));
                float ms;
                CK(cudaEventElapsedTime(&ms, s0, e0));
                t.push_back(ms);
            }
            std::sort(t.begin(), t.end());
            if (t[4] < best) { best = t[4]; res[0].algo = res[i].algo; }
        }
        p.algo = res[0].algo;
        p.n_tried = found;
        fprintf(stderr, "cublaslt %dx%dx%d: best of %d heuristic algorithms\n", M, N, K, found);
        it = g_lt_plans.emplace(key, p).first;
    }
    lt_run(it->second, A, W, C);
}

// v10 launcher: zero the fp32 workspace, split-K kernel, convert to fp16. All
// three are inside the timed region. The split factor S is autotuned per shape.
static float* g_ws = nullptr;
static size_t g_ws_elems = 0;
static std::map<std::tuple<int, int, int, int>, int> g_split;
static void splitk_run(const half* A, const half* W, half* C, int M, int N, int K, int S, int ver = 10) {
    size_t n = (size_t)M * N;
    if (n > g_ws_elems) {
        if (g_ws) CK(cudaFree(g_ws));
        CK(cudaMalloc(&g_ws, n * 4));
        g_ws_elems = n;
    }
    CK(cudaMemsetAsync(g_ws, 0, n * 4));
    if (ver == 11) {
        static bool attr = (cudaFuncSetAttribute(k11_splitk, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM11), true);
        (void)attr;
        k11_splitk<<<dim3(N / BN10, S), 128, SMEM11>>>(A, W, g_ws, M, N, K, K / S);
    } else {
        k10_splitk<<<dim3(N / BN10, S), 128>>>(A, W, g_ws, M, N, K, K / S);
    }
    k10_to_half<<<(int)((n + 255) / 256), 256>>>(g_ws, C, (int)n);
}
static void launch_splitk_v(const half* A, const half* W, half* C, int M, int N, int K, int ver) {
    int bk = ver == 11 ? BK11 : BK10;
    auto key = std::make_tuple(M, N, K, ver);
    auto it = g_split.find(key);
    if (it == g_split.end()) {
        float best = 1e30f;
        int pick = 1;
        cudaEvent_t s0, e0;
        CK(cudaEventCreate(&s0));
        CK(cudaEventCreate(&e0));
        for (int S : {1, 2, 4, 8, 16, 32}) {
            if (K % (S * bk)) continue;
            for (int w = 0; w < 2; ++w) splitk_run(A, W, C, M, N, K, S, ver);
            std::vector<float> t;
            for (int r = 0; r < 15; ++r) {
                CK(cudaEventRecord(s0));
                splitk_run(A, W, C, M, N, K, S, ver);
                CK(cudaEventRecord(e0));
                CK(cudaEventSynchronize(e0));
                float ms;
                CK(cudaEventElapsedTime(&ms, s0, e0));
                t.push_back(ms);
            }
            std::sort(t.begin(), t.end());
            if (t[7] < best) { best = t[7]; pick = S; }
        }
        it = g_split.emplace(key, pick).first;
        fprintf(stderr, "splitk v%d %dx%dx%d -> S=%d\n", ver, M, N, K, pick);
    }
    splitk_run(A, W, C, M, N, K, it->second, ver);
}
static void launch_splitk(const half* A, const half* W, half* C, int M, int N, int K) { launch_splitk_v(A, W, C, M, N, K, 10); }
static void launch_splitk11(const half* A, const half* W, half* C, int M, int N, int K) { launch_splitk_v(A, W, C, M, N, K, 11); }

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
        {"v6_raster", BM3, BN3, BK3, [](const half* A, const half* W, half* C, int M, int N, int K) {
             k45_pipe<true, true><<<(N / BN3) * (M / BM3), 256>>>(A, W, C, M, N, K); }},
        {"v7_bigtile", 256, 128, BK7, launch_v7},
        {"v9_widen", 128, 256, BK7, launch_v9},
        {"v8_autotune", BM3, BN3, BK3, launch_auto},
        {"cublaslt_best", 1, 1, 1, cublaslt_best},
        {"v10_splitk", 1, BN10, BK10, launch_splitk},
        {"v11_splitk_bk64", 1, BN10, BK11, launch_splitk11},
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

// EPI=1 ./bench : y = SiLU(A W^T + bias) + residual, three ways, on the prefill shapes.
// Reference: fp32 host computation from the fp16 GEMM result of cuBLAS (the GEMM
// itself was validated above); the fused kernel keeps the pre-epilogue value in
// fp32, so it differs from the unfused paths by at most one fp16 rounding.
static void epilogue_bench(void* flush, size_t flush_bytes) {
    struct S { int M, N, K; const char* name; };
    std::vector<S> shapes = {{2048, 14336, 4096, "llama8b_up_proj_m2048"}, {2048, 4096, 14336, "llama8b_down_proj_m2048"},
                             {2048, 4096, 4096, "llama8b_o_proj_m2048"}};
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (const S& sh : shapes) {
        int M = sh.M, N = sh.N, K = sh.K;
        size_t nC = (size_t)M * N;
        std::vector<half> hA((size_t)M * K), hW((size_t)N * K), hb(N), hR(nC);
        for (auto& v : hA) v = __float2half(U(rng));
        for (auto& v : hW) v = __float2half(U(rng));
        for (auto& v : hb) v = __float2half(U(rng));
        for (auto& v : hR) v = __float2half(U(rng));
        half *A, *W, *b, *R, *C;
        CK(cudaMalloc(&A, hA.size() * 2)); CK(cudaMalloc(&W, hW.size() * 2)); CK(cudaMalloc(&b, N * 2));
        CK(cudaMalloc(&R, nC * 2)); CK(cudaMalloc(&C, nC * 2));
        CK(cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(W, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b, hb.data(), N * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(R, hR.data(), nC * 2, cudaMemcpyHostToDevice));
        int g1 = (int)((nC + 255) / 256), g2 = (int)((nC / 2 + 255) / 256);
        auto eager = [&] { cublas_gemm(A, W, C, M, N, K); ew_bias<<<g1, 256>>>(C, b, M, N); ew_silu<<<g1, 256>>>(C, nC); ew_residual<<<g1, 256>>>(C, R, nC); };
        auto unfused = [&] { cublas_gemm(A, W, C, M, N, K); ew_fused<<<g2, 256>>>(C, b, R, M, N); };
        auto fused = [&] {
            constexpr int smem = smem_big<128, 256>();
            static bool attr = (cudaFuncSetAttribute(k7_big<128, 256, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), true);
            (void)attr;
            k7_big<128, 256, true><<<(N / 256) * (M / 128), 256, smem>>>(A, W, C, M, N, K, b, R);
        };
        // reference from cuBLAS GEMM output
        cublas_gemm(A, W, C, M, N, K);
        CK(cudaDeviceSynchronize());
        std::vector<half> hG(nC), hC(nC);
        CK(cudaMemcpy(hG.data(), C, nC * 2, cudaMemcpyDeviceToHost));
        std::vector<float> ref(nC);
        for (size_t i = 0; i < nC; ++i) {
            float x = __half2float(hG[i]) + __half2float(hb[i % N]);
            ref[i] = x / (1.f + std::exp(-x)) + __half2float(hR[i]);
        }
        // v9 GEMM followed by the separate elementwise kernel: same GEMM as the fused
        // version, so fused vs this isolates what fusion itself saves.
        auto v9_unfused = [&] { launch_v9(A, W, C, M, N, K); ew_fused<<<g2, 256>>>(C, b, R, M, N); };
        const char* names[4] = {"cublas+3_kernels", "cublas+1_fused_kernel", "v9+1_fused_kernel", "v9_fused_epilogue"};
        std::function<void()> fns[4] = {eager, unfused, v9_unfused, fused};
        float base_ms = 0;
        for (int k = 0; k < 4; ++k) {
            fns[k]();
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hC.data(), C, nC * 2, cudaMemcpyDeviceToHost));
            size_t bad = 0;
            double max_err = 0;
            for (size_t i = 0; i < nC; ++i) {
                double d = std::fabs(__half2float(hC[i]) - ref[i]);
                if (!(d <= 0.05 + 1e-2 * std::fabs(ref[i]))) ++bad;
                max_err = std::max(max_err, d);
            }
            for (int w = 0; w < 3; ++w) fns[k]();
            float ms = time_ms(fns[k], flush, flush_bytes, 20);
            if (k == 0) base_ms = ms;
            printf("{\"mode\":\"epilogue\",\"kernel\":\"%s\",\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ms\":%.4f,"
                   "\"speedup_vs_eager\":%.3f,\"max_abs_err\":%.4g,\"bad\":%zu,\"correct\":%s}\n",
                   names[k], sh.name, M, N, K, ms, base_ms / ms, max_err, bad, bad == 0 ? "true" : "false");
            fflush(stdout);
        }
        CK(cudaFree(A)); CK(cudaFree(W)); CK(cudaFree(b)); CK(cudaFree(R)); CK(cudaFree(C));
    }
}

// BWD=1 ./bench : the two backward GEMMs of a Linear layer y = x W^T, vs cuBLAS.
//   dX[M,Kf] = dY[M,N] . W[N,Kf]        dW[N,Kf] = dY^T[N,M] . X[M,Kf]
// cuBLAS (column-major view of row-major buffers):
//   dX: gemm(N, N, m=Kf, n=M, k=N, W lda=Kf, dY ldb=N, dX ldc=Kf)
//   dW: gemm(N, T, m=Kf, n=N, k=M, X lda=Kf, dY ldb=N, dW ldc=Kf)
static void backward_bench(void* flush, size_t flush_bytes) {
    struct L { int M, N, Kf; const char* name; };
    std::vector<L> layers = {{2048, 4096, 4096, "llama8b_o_proj_t2048"}, {2048, 14336, 4096, "llama8b_up_proj_t2048"},
                             {2048, 4096, 14336, "llama8b_down_proj_t2048"}};
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (const L& L_ : layers) {
        int M = L_.M, N = L_.N, Kf = L_.Kf;
        std::vector<half> hX((size_t)M * Kf), hW((size_t)N * Kf), hdY((size_t)M * N);
        for (auto& v : hX) v = __float2half(U(rng));
        for (auto& v : hW) v = __float2half(U(rng));
        for (auto& v : hdY) v = __float2half(U(rng));
        half *X, *W, *dY, *out, *ref;
        size_t big = std::max((size_t)M * Kf, (size_t)N * Kf);
        CK(cudaMalloc(&X, hX.size() * 2)); CK(cudaMalloc(&W, hW.size() * 2)); CK(cudaMalloc(&dY, hdY.size() * 2));
        CK(cudaMalloc(&out, big * 2)); CK(cudaMalloc(&ref, big * 2));
        CK(cudaMemcpy(X, hX.data(), hX.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(W, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dY, hdY.data(), hdY.size() * 2, cudaMemcpyHostToDevice));
        float alpha = 1.f, beta = 0.f;
        struct G { const char* name; int m, n, k; std::function<void(half*)> blas, ours; };
        std::vector<G> gs = {
            {"dX", M, Kf, N,
             [&](half* o) { CB(cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, Kf, M, N, &alpha, W, CUDA_R_16F, Kf, dY, CUDA_R_16F, N, &beta, o, CUDA_R_16F, Kf, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); },
             [&](half* o) { k12_any<false, true><<<(M / 128) * (Kf / 128), 256>>>(dY, W, o, M, Kf, N); }},
            {"dW", N, Kf, M,
             [&](half* o) { CB(cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, Kf, N, M, &alpha, X, CUDA_R_16F, Kf, dY, CUDA_R_16F, N, &beta, o, CUDA_R_16F, Kf, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); },
             [&](half* o) { k12_any<true, true><<<(N / 128) * (Kf / 128), 256>>>(dY, X, o, N, Kf, M); }},
        };
        for (G& g : gs) {
            size_t n = (size_t)g.m * g.n;
            g.blas(ref);
            g.ours(out);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<half> hr(n), ho(n);
            CK(cudaMemcpy(hr.data(), ref, n * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ho.data(), out, n * 2, cudaMemcpyDeviceToHost));
            size_t bad = 0;
            double max_err = 0;
            for (size_t i = 0; i < n; ++i) {
                double r = __half2float(hr[i]), c = __half2float(ho[i]), d = std::fabs(c - r);
                if (!(d <= 0.05 + 1e-2 * std::fabs(r))) ++bad;
                max_err = std::max(max_err, std::isnan(d) ? INFINITY : d);
            }
            for (int w = 0; w < 3; ++w) { g.blas(ref); g.ours(out); }
            float tb = time_ms([&] { g.blas(ref); }, flush, flush_bytes, 15);
            float to = time_ms([&] { g.ours(out); }, flush, flush_bytes, 15);
            double fl = 2.0 * g.m * g.n * g.k;
            printf("{\"mode\":\"backward\",\"gemm\":\"%s\",\"layer\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"cublas_tflops\":%.3f,"
                   "\"ours_tflops\":%.3f,\"pct_cublas\":%.1f,\"max_abs_err\":%.4g,\"bad\":%zu,\"correct\":%s}\n",
                   g.name, L_.name, g.m, g.n, g.k, fl / tb / 1e9, fl / to / 1e9, 100.0 * tb / to, max_err, bad,
                   bad == 0 ? "true" : "false");
            fflush(stdout);
        }
        CK(cudaFree(X)); CK(cudaFree(W)); CK(cudaFree(dY)); CK(cudaFree(out)); CK(cudaFree(ref));
    }
}

// FP4=1 ./bench : decode GEMM with FP4 (E2M1, group-32 fp16 scale) weights.
// Weights are quantized on the host (scale = absmax / 6, round to nearest code).
// Correctness: against cuBLAS run on the dequantized fp16 weights, so the check
// isolates the kernel from quantization error (reported separately vs the fp16 GEMM).
// Speed: against cuBLAS and v11 on the original fp16 weights, i.e. what FP4 replaces.
static void fp4_bench(void* flush, size_t flush_bytes) {
    struct S { int M, N, K; const char* name; };
    std::vector<S> shapes = {{1, 4096, 4096, "decode_o_proj_m1"}, {16, 4096, 4096, "decode_o_proj_m16"},
                             {1, 14336, 4096, "decode_up_proj_m1"}, {16, 14336, 4096, "decode_up_proj_m16"},
                             {1, 4096, 14336, "decode_down_proj_m1"}, {16, 4096, 14336, "decode_down_proj_m16"}};
    const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    std::mt19937 rng(3);
    std::normal_distribution<float> Nd(0.f, 1.f);
    static bool attr = (cudaFuncSetAttribute(k13_fp4, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM13), true);
    (void)attr;
    for (const S& sh : shapes) {
        int M = sh.M, N = sh.N, K = sh.K, G = K / 32;
        std::vector<half> hA((size_t)M * K), hW((size_t)N * K), hWd((size_t)N * K), hS((size_t)N * G);
        std::vector<uint8_t> hQ((size_t)N * K / 2, 0);
        for (auto& v : hA) v = __float2half(Nd(rng));
        for (auto& v : hW) v = __float2half(Nd(rng) * 0.02f);
        for (int n = 0; n < N; ++n)
            for (int g = 0; g < G; ++g) {
                float amax = 0;
                for (int k = g * 32; k < g * 32 + 32; ++k) amax = std::max(amax, std::fabs(__half2float(hW[(size_t)n * K + k])));
                half sc = __float2half(amax > 0 ? amax / 6.f : 1.f);
                hS[(size_t)n * G + g] = sc;
                float fs = __half2float(sc);
                for (int k = g * 32; k < g * 32 + 32; ++k) {
                    float x = __half2float(hW[(size_t)n * K + k]) / fs;
                    int best = 0;
                    for (int c = 1; c < 8; ++c) if (std::fabs(std::fabs(x) - lut[c]) < std::fabs(std::fabs(x) - lut[best])) best = c;
                    int code = best | (x < 0 && best ? 8 : 0);
                    hQ[((size_t)n * K + k) / 2] |= (uint8_t)(code << ((k & 1) * 4));
                    hWd[(size_t)n * K + k] = __float2half((code & 8 ? -1.f : 1.f) * lut[code & 7] * fs);
                }
            }
        half *A, *W, *Wd, *Sc, *C, *Cref, *C16;
        uint8_t* Q;
        CK(cudaMalloc(&A, hA.size() * 2)); CK(cudaMalloc(&W, hW.size() * 2)); CK(cudaMalloc(&Wd, hWd.size() * 2));
        CK(cudaMalloc(&Sc, hS.size() * 2)); CK(cudaMalloc(&Q, hQ.size()));
        CK(cudaMalloc(&C, (size_t)M * N * 2)); CK(cudaMalloc(&Cref, (size_t)M * N * 2)); CK(cudaMalloc(&C16, (size_t)M * N * 2));
        CK(cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(W, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(Wd, hWd.data(), hWd.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(Sc, hS.data(), hS.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(Q, hQ.data(), hQ.size(), cudaMemcpyHostToDevice));
        size_t n = (size_t)M * N;
        if (n > g_ws_elems) {
            if (g_ws) CK(cudaFree(g_ws));
            CK(cudaMalloc(&g_ws, n * 4));
            g_ws_elems = n;
        }
        int split = 1;
        auto run = [&] {
            CK(cudaMemsetAsync(g_ws, 0, n * 4));
            k13_fp4<<<dim3(N / BN10, split), 128, SMEM13>>>(A, Q, Sc, g_ws, M, N, K, K / split);
            k10_to_half<<<(int)((n + 255) / 256), 256>>>(g_ws, C, (int)n);
        };
        float best = 1e30f;
        int pick = 1;
        for (int sp : {1, 2, 4, 8, 16}) {
            if (K % (sp * BK13)) continue;
            split = sp;
            for (int w = 0; w < 2; ++w) run();
            float ms = time_ms(run, flush, flush_bytes, 15);
            if (ms < best) { best = ms; pick = sp; }
        }
        split = pick;
        cublas_gemm(A, Wd, Cref, M, N, K);
        cublas_gemm(A, W, C16, M, N, K);
        run();
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<half> hC(n), hR(n), h16(n);
        CK(cudaMemcpy(hC.data(), C, n * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hR.data(), Cref, n * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h16.data(), C16, n * 2, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        double max_err = 0, qnum = 0, qden = 0;
        for (size_t i = 0; i < n; ++i) {
            double r = __half2float(hR[i]), c = __half2float(hC[i]), f = __half2float(h16[i]), d = std::fabs(c - r);
            if (!(d <= 0.02 + 1e-2 * std::fabs(r))) ++bad;
            max_err = std::max(max_err, std::isnan(d) ? INFINITY : d);
            qnum += (c - f) * (c - f);
            qden += f * f;
        }
        for (int w = 0; w < 3; ++w) { run(); cublas_gemm(A, W, C16, M, N, K); launch_splitk11(A, W, C16, M, N, K); }
        float t4 = time_ms(run, flush, flush_bytes, 30);
        float tb = time_ms([&] { cublas_gemm(A, W, C16, M, N, K); }, flush, flush_bytes, 30);
        float t11 = time_ms([&] { launch_splitk11(A, W, C16, M, N, K); }, flush, flush_bytes, 30);
        double bytes4 = (double)N * K / 2 + (double)N * G * 2 + (double)M * K * 2 + (double)M * N * 2;
        printf("{\"mode\":\"fp4\",\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"split\":%d,\"fp4_ms\":%.4f,\"fp4_gbps\":%.1f,"
               "\"cublas_fp16_ms\":%.4f,\"v11_fp16_ms\":%.4f,\"speedup_vs_cublas_fp16\":%.3f,\"max_abs_err\":%.4g,\"bad\":%zu,"
               "\"correct\":%s,\"quant_rel_err\":%.4f}\n",
               sh.name, M, N, K, pick, t4, bytes4 / t4 / 1e6, tb, t11, tb / t4, max_err, bad, bad == 0 ? "true" : "false",
               std::sqrt(qnum / qden));
        fflush(stdout);
        CK(cudaFree(A)); CK(cudaFree(W)); CK(cudaFree(Wd)); CK(cudaFree(Sc)); CK(cudaFree(Q));
        CK(cudaFree(C)); CK(cudaFree(Cref)); CK(cudaFree(C16));
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> want(argv + 1, argv + argc);
    // Square plus LLM shapes: Qwen3-8B / Llama-3-8B attention-out and MLP projections at prefill M.
    std::vector<Shape> shapes = {
        {4096, 4096, 4096, "square_4096"},
        {2048, 4096, 4096, "llama8b_o_proj_m2048"},
        {2048, 14336, 4096, "llama8b_up_proj_m2048"},
        {2048, 4096, 14336, "llama8b_down_proj_m2048"},
        // decode: M = tokens in flight
        {1, 4096, 4096, "decode_o_proj_m1"}, {4, 4096, 4096, "decode_o_proj_m4"}, {16, 4096, 4096, "decode_o_proj_m16"},
        {1, 14336, 4096, "decode_up_proj_m1"}, {16, 14336, 4096, "decode_up_proj_m16"},
        {1, 4096, 14336, "decode_down_proj_m1"}, {16, 4096, 14336, "decode_down_proj_m16"},
    };
    if (const char* only = getenv("SHAPE"))   // profile one shape: SHAPE=square_4096 ./bench ...
        shapes.erase(std::remove_if(shapes.begin(), shapes.end(), [&](const Shape& s) { return std::string(s.name) != only; }), shapes.end());
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    size_t flush_bytes = (size_t)prop.l2CacheSize * 4;
    void* flush; CK(cudaMalloc(&flush, flush_bytes));
    CB(cublasCreate(&g_blas));
    CB(cublasLtCreate(&g_lt));
    fprintf(stderr, "GPU %s, SMs %d, L2 %d KB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1024);
    if (getenv("EPI")) { epilogue_bench(flush, flush_bytes); return 0; }
    if (getenv("BWD")) { backward_bench(flush, flush_bytes); return 0; }
    if (getenv("FP4")) { fp4_bench(flush, flush_bytes); return 0; }

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
        double bytes = 2.0 * ((double)M * K + (double)N * K + (double)M * N);   // each operand moved once

        cublas_gemm(A, W, Cref, M, N, K);
        CK(cudaDeviceSynchronize());
        std::vector<half> hRef(nC), hC(nC);
        CK(cudaMemcpy(hRef.data(), Cref, nC * 2, cudaMemcpyDeviceToHost));
        for (int w = 0; w < 3; ++w) cublas_gemm(A, W, Cref, M, N, K);
        float blas_ms = time_ms([&] { cublas_gemm(A, W, Cref, M, N, K); }, flush, flush_bytes, 20);
        printf("{\"kernel\":\"cublas\",\"shape\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ms\":%.4f,\"tflops\":%.3f,\"gbps\":%.1f}\n",
               sh.name, M, N, K, blas_ms, flops / blas_ms / 1e9, bytes / blas_ms / 1e6);
        fflush(stdout);

        for (const Kernel& kr : kernels()) {
            if (!want.empty() && std::find(want.begin(), want.end(), kr.name) == want.end()) continue;
            if (M % kr.tile_m || N % kr.tile_n || K % kr.tile_k) continue;
            if (std::string(kr.name).find("splitk") != std::string::npos && M > 16) continue;   // decode kernel: one 16-row mma tile
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
                   "\"gbps\":%.1f,\"pct_cublas\":%.1f,\"max_abs_err\":%.4g,\"bad\":%zu,\"correct\":%s,\"reps\":%d}\n",
                   kr.name, sh.name, M, N, K, ms, flops / ms / 1e9, bytes / ms / 1e6, 100.0 * blas_ms / ms, max_err, bad,
                   bad == 0 ? "true" : "false", reps);
            fflush(stdout);
        }
        CK(cudaFree(A)); CK(cudaFree(W)); CK(cudaFree(C)); CK(cudaFree(Cref));
    }
    return 0;
}
