# tcgemm

fp16 tensor-core GEMM for LLM inference, written from scratch in CUDA C++ with inline PTX
(`mma.sync`, `ldmatrix`, `cp.async`) and taken step by step from a naive kernel to within
8-14% of cuBLAS's best algorithm. Every step is profiled in Nsight Compute, and every kernel
is checked element by element against cuBLAS before it is timed.

Hardware: RTX 3050 6 GB Laptop GPU (Ampere sm_86, 20 SMs, 1.5 MB L2, ~120 GB/s DRAM).
Layout is `nn.Linear`: `C[M,N] = A[M,K] · W[N,K]^T`, fp16 in, fp32 accumulate, fp16 out.

## Results

| Workload | Result | Baseline |
|---|---|---|
| Prefill, Llama-3-8B projections at 2048 tokens + 4096^3 | **14.2-14.6 TFLOPS, 86-92%** | cuBLASLt, fastest of up to 16 algorithms, each timed |
| Decode, M = 1-16 | **90-102%** of the better of cuBLAS / cuBLASLt (108-126 GB/s) | same |
| Decode with FP4 (E2M1) weights | **3.15-3.24x faster** than cuBLAS fp16 for 3.3-3.5x fewer DRAM bytes; 2.4-3.25x timed end to end | cuBLAS fp16 |
| Backward pass dX, dW | **65-111%**, one untuned kernel; checked against an fp64 host reference | cuBLAS |
| Fused bias + SiLU + residual epilogue | 1.03-1.11x as one elementwise kernel; **slower** fused into the GEMM on large N (see below) | cuBLAS + 3 elementwise kernels |

Full tables: [`results/summary.md`](results/summary.md). Profiles:
[`results/ncu_square_4096.md`](results/ncu_square_4096.md),
[`results/ncu_fp4_decode.md`](results/ncu_fp4_decode.md),
[`results/ncu_epilogue.md`](results/ncu_epilogue.md).

## The kernels, and what each measurement showed

| | Change | Measured effect |
|---|---|---|
| v0-v2 | naive, shared-memory tiles, 8x8 register blocking | CUDA-core baseline |
| v3 | `mma.sync.m16n8k16` + `ldmatrix.x4` | ~10 TFLOPS; 4-way `ldmatrix` bank conflicts (151M) |
| v4 | `cp.async` 3-stage pipeline | stalls move off global loads; **no speed-up** |
| v5 | XOR swizzle `chunk ^ ((row >> 1) & 3)` | bank conflicts 151M -> 9.5K; **still no speed-up**: DRAM-bound at 80% |
| v6 | threadblock rasterization (GROUP_M = 8) | DRAM reads 1.44 -> 1.08 GB, L2 hit 35 -> 51%, 82 -> 94% of cuBLAS (locked clocks) |
| v7 | 256 x 128 tile, 64 x 64 warp tiles | better on tall GEMMs, worse on wide ones |
| v9 | 128 x 256 tile | up_proj 60 -> 86%. DRAM traffic ~ \|A\|·N/BN + \|W\|·M/BM, so wide-N wants BN = 256 |
| v8 | per-shape autotune over v6 / v7 / v9 (median of 15) | picks v9 on all four prefill shapes |
| v10 | split-K decode: M padded to 16 inside the mma tile by `cp.async` zero-fill, fp32 atomics | 90-102 GB/s |
| v11 | BK 32 -> 64: each W row read as one 128-byte segment | **+15-35%** on every decode shape; burst length mattered more than split-K |
| v12 | any-layout GEMM via `ldmatrix.trans` | dX = dY·W and dW = dY^T·X with no transpose in memory |
| v13 | FP4 E2M1 weights, group-32 fp16 scales, dequantized in registers | DRAM bytes / 3.3-3.5, kernel time / 3.15-3.24 |

Four things the profiler taught that intuition got wrong:

1. **Fixing the bottleneck you can see is not a speed-up.** The swizzle removed 99.99% of bank
   conflicts and the kernel got no faster, because it was DRAM-bound. The win came from
   rasterization, which cut DRAM bytes by 25%.
2. **Tile shape follows from a traffic model.** On up_proj (N = 14336), a 128 x 128 tile re-reads
   A 112 times; a 128 x 256 tile halves that, and the kernel went from 60% to 86% of cuBLAS.
3. **For decode, burst length beats parallelism.** Split-K was the obvious fix; the measured fix
   was reading 128 contiguous bytes per W row instead of 64.
4. **Fusion can add memory traffic.** Moving bias + SiLU + residual into the v9 epilogue made
   up_proj 11% slower: no spills, but 0.45-0.6 GB more DRAM reads per call, about 8x the residual
   itself, most likely because the residual loads evict the A/W tiles rasterization reuses from L2.

## FP4 on a GPU without FP4 tensor cores

Ampere has no FP4 MMA, so v13 is the weight-only path: W is stored as packed E2M1 (two codes per
byte) with one fp16 scale per 32 weights, 0.5625 bytes per weight instead of 2. Each code is turned
into fp16 with bit operations and no lookup table: placing its 3 magnitude bits at fp16 bits 11-9
and its sign at bit 15 gives exactly `value · 2^-14` (code 1 lands on the fp16 subnormal 2^-15),
so one `__hmul2` by `scale · 2^14` finishes the dequantization, straight into the `mma.sync` B
fragment.

Correctness is checked against cuBLAS run on the dequantized weights, which isolates the kernel
(max abs error <= 0.008) from quantization error (about 10% relative on Gaussian weights, reported
separately). On up_proj v13 drives DRAM at 89-91% of peak, as cuBLAS does, so nearly all of the
byte saving becomes time; down_proj trails at 79% vs 84%. Doubling the block to 8 warps doubles
occupancy and gains only ~1%, so that gap is not latency hiding.

## Methodology

- **Two baselines.** cuBLAS's default heuristic is not its fastest algorithm: on 4096^3 it runs at
  12.9 TFLOPS, while cuBLASLt's best is 16.6. Against the default, v9 would "beat cuBLAS" on two
  shapes. Every percentage here is against the faster of the two.
- **Correctness before timing:** element-wise against cuBLAS (abs 0.05 + rel 1%), NaN-safe, on every shape.
- **Timing:** CUDA events, L2 flushed before each rep (a memset over 4x the L2 size), median of 15-30 reps, warm-up first.
- **Clocks:** `nvidia-smi` sampled every 500 ms alongside every sweep (`results/*.clocks.csv`).
  Runs on battery are discarded. The power cap there slows cuBLAS more than these kernels and
  overstated percentages by up to 25 points.
- **Profiles:** Nsight Compute with its default clock lock, so kernels are compared at the same clock,
  and also on AC only: the lock fixes the SM clock, but on battery DRAM bandwidth was still limited.

## Build and run

Needs nvcc >= 12 and cuBLAS; `build.sh` points at a pip-installed CUDA 13 toolchain, so set
`CUDA_HOME`, `CUBLAS_HOME` and `CXX` for a system install.

```
bash build.sh
./build/bench                          # all kernels, all shapes: correctness + timing, JSON lines
SHAPE=llama8b_up_proj_m2048 ./build/bench
EPI=1 ./build/bench                    # fused epilogue vs cuBLAS + elementwise kernels
BWD=1 ./build/bench                    # backward dX / dW vs cuBLAS
FP4=1 ./build/bench                    # FP4-weight decode vs cuBLAS fp16
bash run_sweep.sh <tag> [env...]       # same, with clock telemetry, into results/
```

Source: [`src/kernels.cuh`](src/kernels.cuh) (all kernels), [`src/bench.cu`](src/bench.cu) (harness, baselines, checks).

## Limits

One laptop GPU, so absolute numbers are small and power-sensitive; the ratios are the result.
Shapes are multiples of the tile sizes (no ragged-edge handling outside decode M). No FP8 or FP4
tensor-core path: sm_86 has neither.
