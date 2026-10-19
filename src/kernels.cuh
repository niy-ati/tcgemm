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
