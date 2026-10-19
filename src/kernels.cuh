// C[M,N] = A[M,K] * W[N,K]^T   (fp16 in, fp32 accumulate, fp16 out)
// A: activations, row-major. W: weights, row-major [N,K] -- the nn.Linear
// layout, so these kernels drop straight into LLM inference.
// Shapes must be multiples of the kernel's tile sizes (checked on the host).
#pragma once
#include <cuda_fp16.h>

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
