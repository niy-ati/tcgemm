# Nsight Compute: v9 with and without the fused epilogue (up_proj, 2048 x 14336 x 4096)

On AC, Nsight Compute default clock lock (SM 1.177 GHz). Two launches of each kernel.

| | v9 (plain store) | v9 + fused bias/SiLU/residual |
|---|---|---|
| Duration | 23.88-23.93 ms | 24.79-24.82 ms |
| Registers / thread | 226 | 226 |
| Local-memory loads (spills) | 0 | 0 |
| DRAM read | 0.77-0.98 GB | 1.38-1.44 GB |
| DRAM write | 59 MB | 59 MB |
| Global load sectors (L1) | 88.1 M | 91.8 M |
| Tensor pipe active | 41.8% | 40.1% |

The residual tensor is 58.7 MB, yet DRAM reads grow by 0.45-0.6 GB. L1-side load
traffic grows only 4%, so the extra bytes are L2 misses: the residual stream most likely
evicts the A/W tiles that rasterization (GROUP_M = 8) relies on reusing from the 1.5 MB
L2. At locked clocks this costs 0.9 ms; at normal clocks, where the kernel is closer to
DRAM-bound, the timed loss is ~2 ms.
