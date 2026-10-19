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
