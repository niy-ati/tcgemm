# Results: fp16 tensor-core GEMM vs cuBLAS (RTX 3050 6 GB Laptop, sm_86)

All timings on AC power, clocks sampled alongside every sweep (mean busy SM clock 1,600-1,650 MHz).
Every kernel is checked element-wise against cuBLAS before it is timed; all pass on all shapes.
Raw data: `sweep_ac_*.jsonl` (+ `.clocks.csv`). Profiles: `ncu_square_4096.md`.

## Two baselines

- **cuBLAS default**: `cublasGemmEx`, the library's own heuristic choice.
- **cuBLASLt best**: up to 16 algorithms from `cublasLtMatmulAlgoGetHeuristic`, each timed, fastest kept.

The default heuristic is not always the fastest algorithm cuBLAS has: on 4096^3 it picks a
256x128 split-K kernel at 12.9 TFLOPS, while cuBLASLt's best reaches 16.6.

## v9 (128x256 block tile, 64x64 warp tiles, cp.async 3-stage, swizzle, rasterization)

| Shape (M x N x K) | cuBLAS default | cuBLASLt best | v9 | v9 / cuBLASLt best |
|---|---|---|---|---|
| 4096 x 4096 x 4096 | 12.9 | 16.6 | 14.4 | **86%** |
| Llama-3-8B o_proj, 2048 x 4096 x 4096 | 14.5 | 15.9 | 14.6 | **92%** |
| Llama-3-8B up_proj, 2048 x 14336 x 4096 | 16.6 | 16.6 | 14.3 | **86%** |
| Llama-3-8B down_proj, 2048 x 4096 x 14336 | 16.1 | 16.0 | 14.6 | **91%** |

TFLOPS. v9 across 4 independent sweeps: 14.16-14.57 TFLOPS, spread < 1%.

## How it got there

| Step | Change | What the measurement showed |
|---|---|---|
| v3 | `mma.sync` m16n8k16 + `ldmatrix` | ~10 TFLOPS; 4-way bank conflicts (151M) |
| v4 | `cp.async` 3-stage pipeline | stalls moved off global loads; no speed-up |
| v5 | XOR swizzle | bank conflicts 151M -> 9.5K; still no speed-up: DRAM-bound at 80% |
| v6 | threadblock rasterization | DRAM reads -25%, L2 hit 35% -> 51% |
| v7 | 256x128 tile | better on some shapes, worse on others |
| v9 | 128x256 tile | up_proj 60% -> 86%: DRAM traffic ~ \|A\| N/BN + \|W\| M/BM, so wide-N GEMMs want BN = 256 |
| v8 | per-shape autotuning (v6 / v7 / v9) | picks v9 on all four shapes |

Battery runs are excluded: on battery the GPU is power-capped and cuBLAS's figures drop more than ours,
which overstated our percentages by up to ~25 points.

## Decode (M = 1-16): split-K kernel v11 vs cuBLAS

Decode GEMMs are bound by streaming W from DRAM, so the metric is effective bandwidth
(bytes of A, W, C each moved once / time). v10 pads M to 16 inside the mma tile
(rows >= M zero-filled by `cp.async` src-size 0, so they cost no DRAM traffic), splits K
across blocks (partials combined with fp32 atomics; memset + convert are inside the timed
region) and autotunes the split factor. v11 changes one thing: BK 32 -> 64, so each W row
is read as a contiguous 128-byte segment per step instead of 64 bytes.

| Shape | v10 (BK 32) | **v11 (BK 64)**, mean of 3 | best cuBLAS (default or cuBLASLt best), mean of 3 | v11 / best |
|---|---|---|---|---|
| o_proj, M=1 (4096x4096) | 101.5 GB/s | 107.9 | 119.5 | 90% (74-99) |
| o_proj, M=4 | 97.7 | 108.0 | 114.6 | 96% (71-119) |
| o_proj, M=16 | 92.2 | 115.6 | 121.4 | 95% (92-97) |
| up_proj, M=1 (14336x4096) | 94.0 | 121.8 | 124.7 | 98% (92-101) |
| up_proj, M=16 | 89.8 | 118.2 | 116.2 | 102% (99-103) |
| down_proj, M=1 (4096x14336) | 95.6 | 125.5 | 125.0 | 100% |
| down_proj, M=16 | 95.7 | 120.9 | 123.8 | 98% (91-104) |

Longer contiguous reads (128 B per row per step) lifted every shape by 15-35%, and once
bursts were long the autotuner mostly chose S = 1-4: the extra parallelism from split-K
mattered less than DRAM burst length. The smallest layer (o_proj, ~330 us/call) is noisy
for both kernels across runs; ranges are shown rather than a single number.

## Fused epilogue: y = SiLU(x W^T + bias) + residual (prefill, M = 2048)

Median of 3 runs on AC (spread <= 1.8%). All four paths are checked against an fp32 host
reference built from cuBLAS's GEMM output; all pass.

| Shape | cuBLAS + 3 kernels | cuBLAS + 1 fused elementwise | v9 + 1 fused elementwise | v9, epilogue inside the GEMM |
|---|---|---|---|---|
| up_proj (2048 x 14336 x 4096) | 17.83 ms (1.00x) | 16.04 (1.11x) | 18.39 (0.97x) | 20.34 (0.88x) |
| down_proj (2048 x 4096 x 14336) | 16.06 (1.00x) | 15.55 (1.03x) | 17.00 (0.94x) | 17.37 (0.92x) |
| o_proj (2048 x 4096 x 4096) | 5.79 (1.00x) | 5.22 (1.11x) | 5.18 (1.12x) | 5.16 (1.12x) |

Merging the three elementwise kernels into one saves 3-11%. Moving the epilogue into the
GEMM helps only on o_proj and is **slower** on up_proj (20.34 vs 18.39 ms). Nsight Compute
(`ncu_epilogue.md`): same 226 registers, no spills, but the fused kernel reads 0.45-0.6 GB
more from DRAM per call, about 8x the 58.7 MB residual tensor it loads. The likely cause is
the residual loads evicting the reused A/W tiles from the 1.5 MB L2; the obvious next
step is an L2 evict-first hint (or a coalesced shared-memory stage) for the residual.

## Backward pass: dX = dY W and dW = dY^T X, via `ldmatrix.trans` (v12)

Median of 3 runs on AC. Correctness: element-wise vs cuBLAS (0 mismatches), and in a
separate run both kernels vs an fp64 host reference on 512 sampled outputs per GEMM: ours
and cuBLAS have identical error on every GEMM (0.014-0.034, fp16 output rounding).

| Layer (tokens = 2048) | GEMM | cuBLAS TFLOPS | v12 TFLOPS | v12 / cuBLAS |
|---|---|---|---|---|
| o_proj | dX | 11.6 | 12.9 | 111% (96-131, noisy) |
| o_proj | dW | 16.2 | 14.0 | 87% |
| up_proj | dX | 16.1 | 13.3 | 83% |
| up_proj | dW | 15.7 | 12.2 | 78% |
| down_proj | dX | 16.4 | 10.7 | 65% |
| down_proj | dW | 15.7 | 12.0 | 77% |

v12 is a single 128 x 128 tile kernel with no per-shape tuning; it trails cuBLAS most
where the forward pass needed the 128 x 256 tile (wide N or long K). The o_proj dX
figure is high because cuBLAS's default algorithm is slow on it, and it varies widely
between runs. A fourth run (`sweep_ac_bwd_fp64check`) was power- and thermal-throttled
for its whole duration; it is kept for the fp64 check only, not for timing.

## FP4 (E2M1) weight-only decode (v13)

Timed with CUDA events at normal clocks, 8 warps, median of 3 runs, vs cuBLAS default on
fp16 weights. The fp32 workspace memset and fp16 convert are inside the timed region.
Kernel max abs error vs cuBLAS on the dequantized weights <= 0.008; quantization error
itself ~10% relative on Gaussian weights.

| Shape | v13 FP4 | cuBLAS fp16 | Speed-up |
|---|---|---|---|
| o_proj, M=1 | 0.107 ms | 0.281 ms | 2.63x |
| o_proj, M=16 | 0.119 | 0.282 | 2.37x |
| up_proj, M=1 | 0.289 | 0.940 | 3.25x |
| up_proj, M=16 | 0.360 | 1.044 | 2.86x (1.99-3.17) |
| down_proj, M=1 | 0.309 | 0.945 | 3.06x |
| down_proj, M=16 | 0.340 | 1.085 | 3.20x |

Small layers gain less because fixed costs (launch, memset, convert) are a larger share.
At locked clocks in Nsight Compute the kernels alone run 3.15-3.24x faster than cuBLAS
for 3.3-3.5x fewer bytes (`ncu_fp4_decode.md`). 4 vs 8 warps: within 1.5% everywhere.
