# Nsight Compute: FP4-weight decode GEMM (v13) vs cuBLAS fp16

RTX 3050 6 GB Laptop GPU (sm_86, 20 SMs). Nsight Compute with its default clock
control (SM clock locked to base, 1.17 GHz; DRAM at 5.0 GHz), so both kernels run
at the same clocks. Both are DRAM-bound, which makes kernel duration comparable.
`FP4=1 SHAPE=<shape> FP4_SPLIT=2`, v13 with 4 warps per block.

v13 reads W as packed E2M1 (0.5 B/weight) plus one fp16 scale per 32 weights
(0.0625 B/weight): 0.5625 B/weight against 2 B/weight for fp16, i.e. 3.56x fewer
weight bytes. cuBLAS runs on the original fp16 weights.

| Shape (M x N x K) | cuBLAS fp16 | DRAM read | v13 FP4 | DRAM read | Bytes ratio | Time ratio |
|---|---|---|---|---|---|---|
| up_proj, 1 x 14336 x 4096 | 1013 us (97% of peak DRAM) | 117.5 MB | 373 us (75%) | 33.3 MB | 3.53x | **2.72x** |
| up_proj, 16 x 14336 x 4096 | 1123 us (91%) | 120.8 MB | 394 us (80%) | 36.2 MB | 3.34x | **2.85x** |
| down_proj, 16 x 4096 x 14336 | 1162 us (86%) | 119.3 MB | 423 us (70%) | 35.0 MB | 3.41x | **2.75x** |

Times include each library's reduction/convert kernel (cuBLASLt `splitKreduce`,
our `k10_to_half`); the workspace memset is a copy-engine op, not in these rows.

## Why 2.75x and not 3.5x

The byte savings are all there; the gap is that v13 drives DRAM at 70-80% of peak
where cuBLAS reaches 86-97%. The 80 KB of shared-memory stages (3 x 26 KB at
BK = 256, chosen so each W row is one 128-byte read) allow one block per SM: 4 warps,
1 per scheduler, 8% occupancy. Each warp's dequantize -> mma chain then has nothing
to hide behind: 0.34 eligible warps per cycle, top stall "fixed-latency execution
dependency" (43% of cycles), no spills, 96 registers. The next step is the 8-warp
variant (`FP4_WARPS=8`, same tile, 16 columns per warp).
