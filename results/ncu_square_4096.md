# Nsight Compute: 4096 x 4096 x 4096 fp16 GEMM (fp32 accumulate)

RTX 3050 6 GB Laptop GPU (Ampere sm_86, 20 SMs, 1.5 MB L2). Nsight Compute
2026.3 with its default clock control (clocks locked to base, ~1.18 GHz), so
kernels are compared at the same clock. Single launch each, `SHAPE=square_4096`.

## Inside the block tile: v3 -> v5

| | v3 `mma.sync` | v4 + `cp.async` 3-stage | v5 + XOR swizzle |
|---|---|---|---|
| Duration | 14.25 ms | 14.46 ms | 14.85 ms |
| Shared-memory load bank conflicts | 150,994,944 | 151,020,250 | **9,533** |
| Shared-memory load wavefronts | 201,326,687 | 201,355,669 | **50,346,607** |
| DRAM throughput (% of peak) | 70.1 | 77.5 | 80.3 |
| Tensor pipe active | 40.4% | 40.8% | 38.7% |
| Registers / thread | 104 | 120 | 124 |
| Dominant stall | short + long scoreboard | barrier | barrier |

Both fixes did what they target: the swizzle removed the 4-way `ldmatrix` bank
conflict (wavefronts down 4x), and `cp.async` moved stalls off global-memory
loads. Neither made the kernel faster, because it was **DRAM-bound** at ~80%
of peak bandwidth: each 128x128 tile was re-read from DRAM by many blocks.

## Across tiles: v6 rasterization vs cuBLAS

| | v5 | **v6 + threadblock rasterization** | cuBLAS |
|---|---|---|---|
| Kernel | 128x128x32, 3 stages | 128x128x32, 3 stages, GROUP_M = 8 | CUTLASS 256x128x32, 3 stages, split-K 2 |
| Duration | 15.24 ms | **13.35 ms** | 12.57 ms |
| % of cuBLAS | 82% | **94%** | 100% |
| DRAM bytes read | 1.44 GB | **1.08 GB** | 1.14 GB |
| L2 hit rate | 35.0% | **51.1%** | 65.7% |
| Tensor pipe active | 37.7% | **43.1%** | 45.9% |

Launching blocks in groups of 8 block-rows keeps the A and W tiles they share
resident in L2: DRAM reads -25%, L2 hit rate 35% -> 51%. v6 now reads less DRAM
than cuBLAS; the remaining gap is cuBLAS's 2x larger block tile (256x128,
218 registers/thread), i.e. more math per shared-memory load.
