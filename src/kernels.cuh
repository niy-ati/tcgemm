// C[M,N] = A[M,K] * W[N,K]^T   (fp16 in, fp32 accumulate, fp16 out)
// A: activations, row-major. W: weights, row-major [N,K] -- the nn.Linear
// layout, so these kernels drop straight into LLM inference.
// Shapes must be multiples of the kernel's tile sizes (checked on the host).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// ------------------------------------------------------- tensor-core helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// Four 8x8 b16 matrices from shared memory into the mma fragment layout.
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// D(16x8, fp32) += A(16x16, fp16, row) * B(16x8, fp16, col)
__device__ __forceinline__ void mma_16816(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------- v0: naive
// One thread per output element. Consecutive threads walk consecutive n, so
// W reads stride by K: uncoalesced, every operand fetched from DRAM/L2.
__global__ void k0_naive(const half* A, const half* W, half* C, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(A[(size_t)m * K + k]) * __half2float(W[(size_t)n * K + k]);
    C[(size_t)m * N + n] = __float2half(acc);
}

// ------------------------------------------------------- v1: smem tiling 32x32
// Each block stages a 32x32 tile of A and W in shared memory, so every element
// loaded from global memory is reused 32 times. Loads are coalesced along K.
// +1 padding on the inner dimension avoids bank conflicts on the column reads.
constexpr int T1 = 32;
__global__ void k1_smem(const half* A, const half* W, half* C, int M, int N, int K) {
    __shared__ float As[T1][T1 + 1], Ws[T1][T1 + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int m0 = blockIdx.y * T1, n0 = blockIdx.x * T1;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += T1) {
        As[ty][tx] = __half2float(A[(size_t)(m0 + ty) * K + k0 + tx]);
        Ws[ty][tx] = __half2float(W[(size_t)(n0 + ty) * K + k0 + tx]);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < T1; ++kk) acc += As[ty][kk] * Ws[tx][kk];
        __syncthreads();
    }
    C[(size_t)(m0 + ty) * N + n0 + tx] = __float2half(acc);
}

// --------------------------------------------- v2: 2-D register blocking (CUDA cores)
// 128x128 block tile, BK = 8, 256 threads, each computing an 8x8 sub-tile held
// in registers: per smem load, 8 FMAs instead of 1. Tiles are stored transposed
// (k-major) so a thread's 8 values per k are contiguous.
constexpr int BM2 = 128, BN2 = 128, BK2 = 8, TM2 = 8, TN2 = 8;
__global__ void __launch_bounds__(256) k2_regblock(const half* A, const half* W, half* C, int M, int N, int K) {
    __shared__ float As[BK2][BM2], Ws[BK2][BN2];
    int tid = threadIdx.x;
    int m0 = blockIdx.y * BM2, n0 = blockIdx.x * BN2;
    int lrow = tid / 2, lcol = (tid % 2) * 4;        // each thread loads 4 consecutive k of one row
    int ty = tid / 16, tx = tid % 16;                // 16x16 threads, 8x8 outputs each
    float acc[TM2][TN2] = {};
    float ra[TM2], rb[TN2];
    for (int k0 = 0; k0 < K; k0 += BK2) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            As[lcol + i][lrow] = __half2float(A[(size_t)(m0 + lrow) * K + k0 + lcol + i]);
            Ws[lcol + i][lrow] = __half2float(W[(size_t)(n0 + lrow) * K + k0 + lcol + i]);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK2; ++kk) {
#pragma unroll
            for (int i = 0; i < TM2; ++i) ra[i] = As[kk][ty * TM2 + i];
#pragma unroll
            for (int j = 0; j < TN2; ++j) rb[j] = Ws[kk][tx * TN2 + j];
#pragma unroll
            for (int i = 0; i < TM2; ++i)
#pragma unroll
                for (int j = 0; j < TN2; ++j) acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM2; ++i)
#pragma unroll
        for (int j = 0; j < TN2; ++j)
            C[(size_t)(m0 + ty * TM2 + i) * N + n0 + tx * TN2 + j] = __float2half(acc[i][j]);
}

// ---------------------------------------------------- v3: tensor cores (mma.sync)
// 128x128x32 block tile, 8 warps as 2 (M) x 4 (N), each warp a 64x32 tile =
// 4 x 4 mma.m16n8k16 per 16-wide k step. A and W both have K contiguous, which
// is exactly the row.col operand layout mma wants, so W needs no transpose.
// Global -> smem with 16-byte vector loads; smem -> registers with ldmatrix.
// Smem rows are 64 B (32 halves) with no swizzle: ldmatrix's 8 row addresses
// land on only two 16-bank groups, a 4-way bank conflict that v5 removes.
constexpr int BM3 = 128, BN3 = 128, BK3 = 32;
__global__ void __launch_bounds__(256) k3_mma(const half* A, const half* W, half* C, int M, int N, int K) {
    __shared__ __align__(16) half As[BM3][BK3], Ws[BN3][BK3];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wm = warp / 4, wn = warp % 4;                       // warp tile origin: (wm*64, wn*32)
    int m0 = blockIdx.y * BM3, n0 = blockIdx.x * BN3;
    float acc[4][4][4] = {};
    for (int k0 = 0; k0 < K; k0 += BK3) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {                       // 128 rows x 4 chunks of 8 halves, 2 per thread
            int idx = tid + i * 256, row = idx >> 2, ch = (idx & 3) * 8;
            *reinterpret_cast<uint4*>(&As[row][ch]) = *reinterpret_cast<const uint4*>(&A[(size_t)(m0 + row) * K + k0 + ch]);
            *reinterpret_cast<uint4*>(&Ws[row][ch]) = *reinterpret_cast<const uint4*>(&W[(size_t)(n0 + row) * K + k0 + ch]);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK3; kk += 16) {
            uint32_t a[4][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)                  // lanes 0-15: rows 0-15 at k; lanes 16-31: same rows at k+8
                ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                            smem_u32(&As[wm * 64 + mi * 16 + (lane & 15)][kk + (lane >> 4) * 8]));
#pragma unroll
            for (int p = 0; p < 2; ++p)                     // one x4 load = two n8 tiles (k and k+8 halves each)
                ldmatrix_x4(b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1],
                            smem_u32(&Ws[wn * 32 + p * 16 + (lane & 7) + (lane >> 4) * 8][kk + ((lane >> 3) & 1) * 8]));
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_16816(acc[mi][ni], a[mi], b[ni]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int r = m0 + wm * 64 + mi * 16 + (lane >> 2), c = n0 + wn * 32 + ni * 8 + (lane & 3) * 2;
            *reinterpret_cast<half2*>(&C[(size_t)r * N + c]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(&C[(size_t)(r + 8) * N + c]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
}

// ------------------------------------------- v4/v5: cp.async pipeline (+ swizzle)
// v4: global -> smem copies go through cp.async (bypassing registers) into a
// 3-stage ring, so tile k+2 loads while tile k computes. 3 x 16 KB = 48 KB.
// v5: same, with an XOR swizzle on the 16-byte chunk index: chunk ^ ((row>>1)&3).
// The 8 rows an ldmatrix reads then cover all 32 banks instead of two 16-bank
// groups, removing v3/v4's 4-way bank conflict. Writes use the same mapping.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <bool SWZ> __device__ __forceinline__ int swz(int row, int chunk) { return SWZ ? (chunk ^ ((row >> 1) & 3)) : chunk; }

constexpr int STAGES = 3;
// v6 adds threadblock rasterization (RASTER): blocks are numbered so that each
// group of GROUP_M block-rows sweeps N together. Blocks resident at the same
// time then share A row-tiles and W column-tiles, which stay in L2 instead of
// being re-read from DRAM. Needs a 1-D grid.
constexpr int GROUP_M = 8;
template <bool SWZ, bool RASTER = false>
__global__ void __launch_bounds__(256) k45_pipe(const half* A, const half* W, half* C, int M, int N, int K) {
    __shared__ __align__(128) half As[STAGES][BM3][BK3], Ws[STAGES][BN3][BK3];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wm = warp / 4, wn = warp % 4;
    int m0, n0;
    if (RASTER) {
        int pid = blockIdx.x, grid_m = M / BM3, grid_n = N / BN3;
        int group = pid / (GROUP_M * grid_n), first_m = group * GROUP_M;
        int gm = min(grid_m - first_m, GROUP_M), in_group = pid % (GROUP_M * grid_n);
        m0 = (first_m + in_group % gm) * BM3;
        n0 = (in_group / gm) * BN3;
    } else {
        m0 = blockIdx.y * BM3;
        n0 = blockIdx.x * BN3;
    }
    int KT = K / BK3;
    auto load = [&](int stage, int kt) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 256, row = idx >> 2, ch = idx & 3;
            int pc = swz<SWZ>(row, ch) * 8;
            cp_async16(smem_u32(&As[stage][row][pc]), &A[(size_t)(m0 + row) * K + kt * BK3 + ch * 8]);
            cp_async16(smem_u32(&Ws[stage][row][pc]), &W[(size_t)(n0 + row) * K + kt * BK3 + ch * 8]);
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    float acc[4][4][4] = {};
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();   // tile kt has landed
        __syncthreads();               // ...for every thread; and the slot refilled below is no longer being read
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();             // always commit, so group counting stays aligned at the tail
        int st = kt % STAGES;
#pragma unroll
        for (int kk = 0; kk < BK3; kk += 16) {
            uint32_t a[4][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int row = wm * 64 + mi * 16 + (lane & 15), ch = (kk >> 3) + (lane >> 4);
                ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], smem_u32(&As[st][row][swz<SWZ>(row, ch) * 8]));
            }
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                int row = wn * 32 + p * 16 + (lane & 7) + (lane >> 4) * 8, ch = (kk >> 3) + ((lane >> 3) & 1);
                ldmatrix_x4(b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1], smem_u32(&Ws[st][row][swz<SWZ>(row, ch) * 8]));
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_16816(acc[mi][ni], a[mi], b[ni]);
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int r = m0 + wm * 64 + mi * 16 + (lane >> 2), c = n0 + wn * 32 + ni * 8 + (lane & 3) * 2;
            *reinterpret_cast<half2*>(&C[(size_t)r * N + c]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(&C[(size_t)(r + 8) * N + c]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
}

// ------------------------------------- v7/v9: big block tiles, 64x64 warp tiles
// v7 = 256x128, v9 = 128x256. A 64x64 warp tile does 32 mma per 12 ldmatrix
// (v6: 16 per 8), so more math per shared-memory byte. 8 warps, laid out as
// (BM/64) x (BN/64). 3 stages x (BM + BN) x 32 x 2 B = 72 KB: dynamic smem.
// Which orientation wins depends on the shape: DRAM traffic is about
// |A| * N/BN + |W| * M/BM (the operand that doesn't fit in L2 is re-read once
// per tile column), so a wide-N GEMM wants BN = 256. Swizzle + rasterization kept.
constexpr int BK7 = 32;
template <int BM, int BN> constexpr int smem_big() { return STAGES * (BM + BN) * BK7 * 2; }
// EPI = true fuses y = SiLU(acc + bias[n]) + residual[m, n] into the store, in fp32,
// so C is written once instead of being re-read by follow-up elementwise kernels.
template <int BM, int BN, bool EPI = false>
__global__ void __launch_bounds__(256) k7_big(const half* A, const half* W, half* C, int M, int N, int K,
                                              const half* bias = nullptr, const half* R = nullptr) {
    constexpr int WN_WARPS = BN / 64;                        // warps along N
    extern __shared__ __align__(128) half sm[];
    auto As = [&](int st, int row, int col) -> half* { return &sm[(st * BM + row) * BK7 + col]; };
    auto Ws = [&](int st, int row, int col) -> half* { return &sm[STAGES * BM * BK7 + (st * BN + row) * BK7 + col]; };
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wm = warp / WN_WARPS, wn = warp % WN_WARPS;          // warp tile origin: (wm*64, wn*64)
    int pid = blockIdx.x, grid_m = M / BM, grid_n = N / BN;
    int group = pid / (GROUP_M * grid_n), first_m = group * GROUP_M;
    int gm = min(grid_m - first_m, GROUP_M), in_group = pid % (GROUP_M * grid_n);
    int m0 = (first_m + in_group % gm) * BM, n0 = (in_group / gm) * BN;
    int KT = K / BK7;
    auto load = [&](int st, int kt) {
#pragma unroll
        for (int i = 0; i < BM / 64; ++i) {                  // BM rows x 4 chunks over 256 threads
            int idx = tid + i * 256, row = idx >> 2, ch = idx & 3;
            cp_async16(smem_u32(As(st, row, swz<true>(row, ch) * 8)), &A[(size_t)(m0 + row) * K + kt * BK7 + ch * 8]);
        }
#pragma unroll
        for (int i = 0; i < BN / 64; ++i) {
            int idx = tid + i * 256, row = idx >> 2, ch = idx & 3;
            cp_async16(smem_u32(Ws(st, row, swz<true>(row, ch) * 8)), &W[(size_t)(n0 + row) * K + kt * BK7 + ch * 8]);
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    float acc[4][8][4] = {};
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();
        int st = kt % STAGES;
#pragma unroll
        for (int kk = 0; kk < BK7; kk += 16) {
            uint32_t a[4][4], b[8][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int row = wm * 64 + mi * 16 + (lane & 15), ch = (kk >> 3) + (lane >> 4);
                ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], smem_u32(As(st, row, swz<true>(row, ch) * 8)));
            }
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int row = wn * 64 + p * 16 + (lane & 7) + (lane >> 4) * 8, ch = (kk >> 3) + ((lane >> 3) & 1);
                ldmatrix_x4(b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1], smem_u32(Ws(st, row, swz<true>(row, ch) * 8)));
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < 8; ++ni) mma_16816(acc[mi][ni], a[mi], b[ni]);
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 8; ++ni) {
            int r = m0 + wm * 64 + mi * 16 + (lane >> 2), c = n0 + wn * 64 + ni * 8 + (lane & 3) * 2;
            float* v = acc[mi][ni];
            if (EPI) {
                float2 b = __half22float2(*reinterpret_cast<const half2*>(&bias[c]));
                float2 r0 = __half22float2(*reinterpret_cast<const half2*>(&R[(size_t)r * N + c]));
                float2 r1 = __half22float2(*reinterpret_cast<const half2*>(&R[(size_t)(r + 8) * N + c]));
                float t[4] = {v[0] + b.x, v[1] + b.y, v[2] + b.x, v[3] + b.y};
#pragma unroll
                for (int q = 0; q < 4; ++q) t[q] = t[q] / (1.f + __expf(-t[q]));   // SiLU
                v[0] = t[0] + r0.x; v[1] = t[1] + r0.y; v[2] = t[2] + r1.x; v[3] = t[3] + r1.y;
            }
            *reinterpret_cast<half2*>(&C[(size_t)r * N + c]) = __floats2half2_rn(v[0], v[1]);
            *reinterpret_cast<half2*>(&C[(size_t)(r + 8) * N + c]) = __floats2half2_rn(v[2], v[3]);
        }
}

// ------------------------------------------------ v10: split-K decode GEMM (M <= 16)
// Decode is bound by streaming W (N x K) from DRAM; the math is tiny. Two problems
// with the prefill kernels: they need M % 128 == 0, and at M = 1 a 4096-wide layer
// is only 32 blocks of 128 columns for 20 SMs. Here M is padded to 16 inside the
// mma tile (rows >= M are zero-filled by cp.async with src-size 0, so they cost
// no DRAM traffic), each block takes 128 columns and a 1/S slice of K, and the S
// partial sums meet in an fp32 workspace via atomicAdd, then convert to fp16.
// 4 warps, each a 16 x 32 output slice. Same 3-stage cp.async + swizzle as v5.
constexpr int BN10 = 128, BK10 = 32;
__device__ __forceinline__ void cp_async16_zfill(uint32_t dst, const void* src, bool valid) {
    int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__global__ void __launch_bounds__(128) k10_splitk(const half* A, const half* W, float* Cf, int M, int N, int K, int k_per_split) {
    __shared__ __align__(128) half As[STAGES][16][BK10], Ws[STAGES][BN10][BK10];
    int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    int n0 = blockIdx.x * BN10, kbeg = blockIdx.y * k_per_split;
    int KT = k_per_split / BK10;
    auto load = [&](int st, int kt) {
        int k = kbeg + kt * BK10;
        if (tid < 64) {                                      // A: 16 rows x 4 chunks, rows >= M zero-filled
            int row = tid >> 2, ch = tid & 3;
            bool ok = row < M;
            cp_async16_zfill(smem_u32(&As[st][row][swz<true>(row, ch) * 8]), &A[(size_t)(ok ? row : 0) * K + k + ch * 8], ok);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {                        // W: 128 rows x 4 chunks over 128 threads
            int idx = tid + i * 128, row = idx >> 2, ch = idx & 3;
            cp_async16(smem_u32(&Ws[st][row][swz<true>(row, ch) * 8]), &W[(size_t)(n0 + row) * K + k + ch * 8]);
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    float acc[4][4] = {};
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();
        int st = kt % STAGES;
#pragma unroll
        for (int kk = 0; kk < BK10; kk += 16) {
            uint32_t a[4], b[4][2];
            int arow = lane & 15, ach = (kk >> 3) + (lane >> 4);
            ldmatrix_x4(a[0], a[1], a[2], a[3], smem_u32(&As[st][arow][swz<true>(arow, ach) * 8]));
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                int row = wn * 32 + p * 16 + (lane & 7) + (lane >> 4) * 8, ch = (kk >> 3) + ((lane >> 3) & 1);
                ldmatrix_x4(b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1], smem_u32(&Ws[st][row][swz<true>(row, ch) * 8]));
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) mma_16816(acc[ni], a, b[ni]);
        }
    }
    cp_async_wait<0>();
    int r = lane >> 2;
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        int c = n0 + wn * 32 + ni * 8 + (lane & 3) * 2;
        if (r < M) { atomicAdd(&Cf[(size_t)r * N + c], acc[ni][0]); atomicAdd(&Cf[(size_t)r * N + c + 1], acc[ni][1]); }
        if (r + 8 < M) { atomicAdd(&Cf[(size_t)(r + 8) * N + c], acc[ni][2]); atomicAdd(&Cf[(size_t)(r + 8) * N + c + 1], acc[ni][3]); }
    }
}
__global__ void k10_to_half(const float* Cf, half* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = __float2half(Cf[i]);
}

// ------------------------------------- v11: split-K decode with 128-byte row reads
// v10 reads each W row 64 bytes at a time (BK = 32 halves) across 128 rows that sit
// K * 2 bytes apart: many short, scattered DRAM bursts. v11 takes BK = 64, so each
// row is read as one contiguous 128-byte segment per step. Rows are 8 chunks of 16 B,
// so the swizzle becomes chunk ^ (row & 7). 3 stages x (16 + 128) x 64 x 2 B = 54 KB:
// dynamic shared memory.
constexpr int BK11 = 64, SMEM11 = STAGES * (16 + BN10) * BK11 * 2;
__device__ __forceinline__ int swz8(int row, int chunk) { return chunk ^ (row & 7); }
__global__ void __launch_bounds__(128) k11_splitk(const half* A, const half* W, float* Cf, int M, int N, int K, int k_per_split) {
    extern __shared__ __align__(128) half sm11[];
    auto As = [&](int st, int row, int col) -> half* { return &sm11[(st * 16 + row) * BK11 + col]; };
    auto Ws = [&](int st, int row, int col) -> half* { return &sm11[STAGES * 16 * BK11 + (st * BN10 + row) * BK11 + col]; };
    int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    int n0 = blockIdx.x * BN10, kbeg = blockIdx.y * k_per_split;
    int KT = k_per_split / BK11;
    auto load = [&](int st, int kt) {
        int k = kbeg + kt * BK11;
        {                                                    // A: 16 rows x 8 chunks = 128, one per thread
            int row = tid >> 3, ch = tid & 7;
            bool ok = row < M;
            cp_async16_zfill(smem_u32(As(st, row, swz8(row, ch) * 8)), &A[(size_t)(ok ? row : 0) * K + k + ch * 8], ok);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {                        // W: 128 rows x 8 chunks over 128 threads
            int idx = tid + i * 128, row = idx >> 3, ch = idx & 7;
            cp_async16(smem_u32(Ws(st, row, swz8(row, ch) * 8)), &W[(size_t)(n0 + row) * K + k + ch * 8]);
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    float acc[4][4] = {};
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();
        int st = kt % STAGES;
#pragma unroll
        for (int kk = 0; kk < BK11; kk += 16) {
            uint32_t a[4], b[4][2];
            int arow = lane & 15, ach = (kk >> 3) + (lane >> 4);
            ldmatrix_x4(a[0], a[1], a[2], a[3], smem_u32(As(st, arow, swz8(arow, ach) * 8)));
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                int row = wn * 32 + p * 16 + (lane & 7) + (lane >> 4) * 8, ch = (kk >> 3) + ((lane >> 3) & 1);
                ldmatrix_x4(b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1], smem_u32(Ws(st, row, swz8(row, ch) * 8)));
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) mma_16816(acc[ni], a, b[ni]);
        }
    }
    cp_async_wait<0>();
    int r = lane >> 2;
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        int c = n0 + wn * 32 + ni * 8 + (lane & 3) * 2;
        if (r < M) { atomicAdd(&Cf[(size_t)r * N + c], acc[ni][0]); atomicAdd(&Cf[(size_t)r * N + c + 1], acc[ni][1]); }
        if (r + 8 < M) { atomicAdd(&Cf[(size_t)(r + 8) * N + c], acc[ni][2]); atomicAdd(&Cf[(size_t)(r + 8) * N + c + 1], acc[ni][3]); }
    }
}

// ------------------------------------------------ unfused epilogue baselines
// eager: three kernels, each a full pass over C (bias add, SiLU, residual add).
__global__ void ew_bias(half* C, const half* bias, int M, int N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)M * N) C[i] = __float2half(__half2float(C[i]) + __half2float(bias[i % N]));
}
__global__ void ew_silu(half* C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = __half2float(C[i]); C[i] = __float2half(x / (1.f + __expf(-x))); }
}
__global__ void ew_residual(half* C, const half* R, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = __float2half(__half2float(C[i]) + __half2float(R[i]));
}
// best unfused: one elementwise kernel doing all three, half2-vectorized
__global__ void ew_fused(half* C, const half* bias, const half* R, int M, int N) {
    size_t i = 2 * ((size_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= (size_t)M * N) return;
    float2 c = __half22float2(*reinterpret_cast<half2*>(&C[i]));
    float2 b = __half22float2(*reinterpret_cast<const half2*>(&bias[i % N]));
    float2 r = __half22float2(*reinterpret_cast<const half2*>(&R[i]));
    float x0 = c.x + b.x, x1 = c.y + b.y;
    x0 = x0 / (1.f + __expf(-x0)) + r.x;
    x1 = x1 / (1.f + __expf(-x1)) + r.y;
    *reinterpret_cast<half2*>(&C[i]) = __floats2half2_rn(x0, x1);
}

// ------------------------------------- v12: any-layout GEMM for the backward pass
// C[M,N] = sum_k opA[m,k] * opB[n,k], with each operand stored either reduction-
// contiguous (as in the forward pass) or output-contiguous:
//   TA = false: A[m*K + k]      TA = true: A[k*M + m]
//   TB = false: B[n*K + k]      TB = true: B[k*N + n]
// Linear backward:  dX = dY . W     -> TA = false, TB = true
//                   dW = dY^T . X   -> TA = true,  TB = true
// Output-contiguous tiles are staged as they lie in memory ([BK][128], 256-byte
// rows) and transposed on the way into registers by ldmatrix.trans, so no operand
// is ever transposed in memory. Those rows hold 16 chunks; swizzle chunk ^ (k & 7)
// keeps the 8 rows an ldmatrix reads on distinct banks.
// 128x128x32 tile, 8 warps as 2 x 4 (64x32 warp tiles), cp.async 3-stage, rasterized.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
template <bool TA, bool TB>
__global__ void __launch_bounds__(256) k12_any(const half* A, const half* B, half* C, int M, int N, int K) {
    __shared__ __align__(128) half sA[STAGES][128 * 32], sB[STAGES][128 * 32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int wm = warp / 4, wn = warp % 4;
    int pid = blockIdx.x, grid_m = M / 128, grid_n = N / 128;
    int group = pid / (GROUP_M * grid_n), first_m = group * GROUP_M;
    int gm = min(grid_m - first_m, GROUP_M), in_group = pid % (GROUP_M * grid_n);
    int m0 = (first_m + in_group % gm) * 128, n0 = (in_group / gm) * 128;
    int KT = K / 32;
    auto load = [&](int st, int kt) {
        int k0 = kt * 32;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 256;
            if (!TA) {   // 128 rows (m) x 4 chunks (k)
                int row = idx >> 2, ch = idx & 3;
                cp_async16(smem_u32(&sA[st][row * 32 + swz<true>(row, ch) * 8]), &A[(size_t)(m0 + row) * K + k0 + ch * 8]);
            } else {     // 32 rows (k) x 16 chunks (m)
                int row = idx >> 4, ch = idx & 15;
                cp_async16(smem_u32(&sA[st][row * 128 + (ch ^ (row & 7)) * 8]), &A[(size_t)(k0 + row) * M + m0 + ch * 8]);
            }
            if (!TB) {
                int row = idx >> 2, ch = idx & 3;
                cp_async16(smem_u32(&sB[st][row * 32 + swz<true>(row, ch) * 8]), &B[(size_t)(n0 + row) * K + k0 + ch * 8]);
            } else {
                int row = idx >> 4, ch = idx & 15;
                cp_async16(smem_u32(&sB[st][row * 128 + (ch ^ (row & 7)) * 8]), &B[(size_t)(k0 + row) * N + n0 + ch * 8]);
            }
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    float acc[4][4][4] = {};
    int j = lane >> 3, i8 = lane & 7;   // which of the four 8x8 matrices this lane addresses, and its row
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();
        int st = kt % STAGES;
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            uint32_t a[4][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                if (!TA) {
                    int row = wm * 64 + mi * 16 + (lane & 15), ch = (kk >> 3) + (lane >> 4);
                    ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], smem_u32(&sA[st][row * 32 + swz<true>(row, ch) * 8]));
                } else {   // matrices: (m 0-7,k 0-7) (m 8-15,k 0-7) (m 0-7,k 8-15) (m 8-15,k 8-15)
                    int k = kk + (j >> 1) * 8 + i8, m = wm * 64 + mi * 16 + (j & 1) * 8;
                    ldmatrix_x4_trans(a[mi][0], a[mi][1], a[mi][2], a[mi][3], smem_u32(&sA[st][k * 128 + ((m >> 3) ^ (k & 7)) * 8]));
                }
            }
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                if (!TB) {
                    int row = wn * 32 + p * 16 + (lane & 7) + (lane >> 4) * 8, ch = (kk >> 3) + ((lane >> 3) & 1);
                    ldmatrix_x4(b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1], smem_u32(&sB[st][row * 32 + swz<true>(row, ch) * 8]));
                } else {   // matrices: (n 0-7,k 0-7) (n 0-7,k 8-15) (n 8-15,k 0-7) (n 8-15,k 8-15)
                    int k = kk + (j & 1) * 8 + i8, n = wn * 32 + p * 16 + (j >> 1) * 8;
                    ldmatrix_x4_trans(b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1], smem_u32(&sB[st][k * 128 + ((n >> 3) ^ (k & 7)) * 8]));
                }
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_16816(acc[mi][ni], a[mi], b[ni]);
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int r = m0 + wm * 64 + mi * 16 + (lane >> 2), c = n0 + wn * 32 + ni * 8 + (lane & 3) * 2;
            *reinterpret_cast<half2*>(&C[(size_t)r * N + c]) = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *reinterpret_cast<half2*>(&C[(size_t)(r + 8) * N + c]) = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
}

// ------------------------------------- v13: FP4 (E2M1) weights, fp16 activations, decode
// sm_86 has no FP4 tensor-core instruction, so this is the weight-only path: W is
// stored as packed E2M1 (two codes per byte, low nibble = even k) with one fp16 scale
// per 32 consecutive k, and is dequantized to fp16 in registers right before the
// usual m16n8k16 fp16 mma. Decode is DRAM-bound, so the point is 4x fewer weight bytes.
//
// E2M1 -> fp16 without a table: put the 3 magnitude bits at fp16 bits 11..9 and the
// sign at bit 15. That fp16 equals the E2M1 value times 2^-14 exactly (code 1 lands
// on the fp16 subnormal 2^-15), so one multiply by (scale * 2^14) finishes it.
//
// Tile: 16 x 128 x 256 per stage. 256 k of FP4 is 128 bytes per W row, the same
// 128-byte row reads that made v11 fast. W rows are 8 chunks of 16 B, swizzled
// chunk ^ (row & 7) so the 8 rows a B fragment touches sit on distinct banks.
constexpr int BK13 = 256;
constexpr int A13 = 16 * BK13 * 2, W13 = BN10 * BK13 / 2, S13 = BN10 * (BK13 / 32) * 2;
constexpr int STAGE13 = A13 + W13 + S13, SMEM13 = STAGES * STAGE13;
__device__ __forceinline__ uint32_t e2m1x2_to_half2(uint32_t v) {
    return ((v & 0x7u) << 9) | ((v & 0x8u) << 12) | ((v & 0x70u) << 21) | ((v & 0x80u) << 24);
}
__global__ void __launch_bounds__(128) k13_fp4(const half* A, const uint8_t* Wq, const half* Ws, float* Cf, int M, int N, int K,
                                               int k_per_split) {
    extern __shared__ __align__(128) uint8_t sm13[];
    auto As = [&](int st) { return reinterpret_cast<half*>(sm13 + st * STAGE13); };
    auto Wb = [&](int st) { return sm13 + st * STAGE13 + A13; };
    auto Sc = [&](int st) { return reinterpret_cast<half*>(sm13 + st * STAGE13 + A13 + W13); };
    int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    int n0 = blockIdx.x * BN10, kbeg = blockIdx.y * k_per_split;
    int KT = k_per_split / BK13, G = K / 32;
    auto load = [&](int st, int kt) {
        int k = kbeg + kt * BK13;
#pragma unroll
        for (int i = 0; i < 4; ++i) {                        // A: 16 rows x 32 chunks
            int idx = tid + i * 128, row = idx >> 5, ch = idx & 31;
            bool ok = row < M;
            cp_async16_zfill(smem_u32(&As(st)[row * BK13 + (ch ^ (row & 7)) * 8]), &A[(size_t)(ok ? row : 0) * K + k + ch * 8], ok);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {                        // W: 128 rows x 8 chunks of packed FP4
            int idx = tid + i * 128, row = idx >> 3, ch = idx & 7;
            cp_async16(smem_u32(&Wb(st)[row * 128 + (ch ^ (row & 7)) * 16]), &Wq[(size_t)(n0 + row) * (K / 2) + k / 2 + ch * 16]);
        }
        cp_async16(smem_u32(&Sc(st)[tid * 8]), &Ws[(size_t)(n0 + tid) * G + k / 32]);   // 8 scales per row
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    float acc[4][4] = {};
    const half2 two14 = __float2half2_rn(16384.f);
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_async_commit();
        int st = kt % STAGES;
        const half* as = As(st);
        const uint8_t* wb = Wb(st);
        const half* sc = Sc(st);
#pragma unroll
        for (int kk = 0; kk < BK13; kk += 16) {
            uint32_t a[4], b[4][2];
            int arow = lane & 15, ach = (kk >> 3) + (lane >> 4);
            ldmatrix_x4(a[0], a[1], a[2], a[3], smem_u32(&as[arow * BK13 + (ach ^ (arow & 7)) * 8]));
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                // B fragment of m16n8k16: n = lane / 4, k pairs at 2*(lane%4) and 8 + 2*(lane%4)
                int n = wn * 32 + ni * 8 + (lane >> 2);
                half2 s2 = __hmul2(__half2half2(sc[n * 8 + kk / 32]), two14);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int kb = kk / 2 + h * 4 + (lane & 3);
                    uint32_t v = wb[n * 128 + (((kb >> 4) ^ (n & 7)) << 4) + (kb & 15)];
                    uint32_t raw = e2m1x2_to_half2(v);
                    half2 w = __hmul2(*reinterpret_cast<half2*>(&raw), s2);
                    b[ni][h] = *reinterpret_cast<uint32_t*>(&w);
                }
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) mma_16816(acc[ni], a, b[ni]);
        }
    }
    cp_async_wait<0>();
    int r = lane >> 2;
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        int c = n0 + wn * 32 + ni * 8 + (lane & 3) * 2;
        if (r < M) { atomicAdd(&Cf[(size_t)r * N + c], acc[ni][0]); atomicAdd(&Cf[(size_t)r * N + c + 1], acc[ni][1]); }
        if (r + 8 < M) { atomicAdd(&Cf[(size_t)(r + 8) * N + c], acc[ni][2]); atomicAdd(&Cf[(size_t)(r + 8) * N + c + 1], acc[ni][3]); }
    }
}
