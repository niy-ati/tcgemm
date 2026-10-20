# Nsight Compute: FP4-weight decode GEMM (v13) vs cuBLAS fp16

RTX 3050 6 GB Laptop GPU (sm_86, 20 SMs), on AC power. Nsight Compute's default clock
control locks the SM clock to base (1.177 GHz); DRAM ran at 5.48 GHz for every kernel.
`FP4=1 SHAPE=<shape> FP4_SPLIT=2`, median of 35-51 profiled launches per kernel.
Compact data: `ncu_fp4_ac.csv`.

v13 reads W as packed E2M1 (0.5 B/weight) plus one fp16 scale per 32 weights:
0.5625 B/weight against 2 B for fp16, 3.56x fewer weight bytes. cuBLAS runs on the
original fp16 weights. Times include each library's reduction/convert kernel
(cuBLASLt `splitKreduce`, our `k10_to_half`).

| Shape (M x N x K) | cuBLAS fp16 | DRAM read | v13, 4 warps | v13, 8 warps | DRAM read | Bytes ratio | Time ratio (8 warps) |
|---|---|---|---|---|---|---|---|
| up_proj, 1 x 14336 x 4096 | 922 us (97% of peak) | 117.5 MB | 289 us (89%) | 285 us (90%) | 33.3 MB | 3.53x | **3.24x** |
| up_proj, 16 x 14336 x 4096 | 1029 us (91%) | 120.5 MB | 321 us (90%) | 319 us (91%) | 36.1 MB | 3.34x | **3.23x** |
| down_proj, 16 x 4096 x 14336 | 1077 us (84%) | 119.3 MB | 345 us (79%) | 342 us (79%) | 35.0 MB | 3.41x | **3.15x** |

On up_proj v13 runs at about the same fraction of DRAM peak as cuBLAS, so nearly all of
the byte saving turns into time. down_proj is the one shape where it trails (79% vs 84%).

## 8 warps per block: occupancy is not the limit

The 80 KB of shared-memory stages allow one block per SM. Doubling the block from 4 to
8 warps (same tile, 16 columns per warp) doubles achieved occupancy (8.3% -> 16.7% of
peak warps) and changes kernel time by only 0.7-1.5%. The remaining gap on down_proj is
therefore not latency hiding; it is not yet explained.

## Correction

An earlier version of this file reported 2.72-2.85x, measured on battery. Locking the SM
clock does not stop the battery power policy from limiting DRAM: the same v13 launch took
370 us at 75% of peak on battery and 286 us at 89% on AC, moving the same 33.2 MB. cuBLAS
slowed less (1013 -> 922 us), so the battery run understated the ratio.
