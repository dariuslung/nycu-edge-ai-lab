ID: 314561005
Name: 龍偉亮
# Lab 1 Report

Machine: Ubuntu 26.04, driver 595.91.07, CUDA 13.1 (nvcc 13.1.115), built with `-arch=native` (sm_89).

## Part 0: Know your GPU

### Q1. GPU details and peaks

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti (8 GB), AD106 | `nvidia-smi` |
| Compute capability | 8.9 | `nvidia-smi --query-gpu=compute_cap`, harness reports `sm_89` |
| SM count | 34 | harness header (`cudaGetDeviceProperties`) |
| CUDA cores | 4352 (34 SMs x 128) | TechPowerUp GPU Database |
| Boost clock | 2.535 GHz | TechPowerUp GPU Database |
| Memory | 18 Gbps GDDR6, 128-bit bus | TechPowerUp GPU Database |

> The spec-sheet values (4352 shading units, 2535 MHz boost, 2250 MHz memory clock = 18 Gbps GDDR6, 128-bit bus, AD106) are from the TechPowerUp entry for the GeForce RTX 4060 Ti 8 GB. The 34 SM count and CC 8.9 were read from the device.

Peak FP32 = 2 x CUDA cores x boost clock = 2 x 4352 x 2.535 GHz = **22,065 GFLOP/s** (about 22.06 TFLOP/s).

Peak DRAM bandwidth = data rate x bus width / 8 = 18 Gbps x 128 / 8 = **288 GB/s**.

The harness prints its own estimate of 22,326 GFLOP/s. It is derived from the
2.565 GHz clock the driver reports, not the spec-sheet boost clock, so the
spec-sheet 22,065 GFLOP/s is used for all percentages below.

### Q2. Ridge point and arithmetic intensity

- Ridge point = peak FLOP/s / peak bytes/s = 22,065 / 288 = **76.6 FLOP/byte**.
- SGEMM at M = N = K = 4096: FLOPs = 2N^3. Minimum DRAM traffic = read A + read B + write C = 3 x N^2 x 4 bytes = 12N^2 bytes.
  Arithmetic intensity = 2N^3 / 12N^2 = N/6 = 4096/6 = **about 683 FLOP/byte**.
- 683 is about 9x the ridge point (76.6), so SGEMM at this size should be **compute-bound** on this GPU. A good kernel is limited by FP32 throughput, not by DRAM bandwidth.
  The naive kernel below is far below this limit, because it is limited by something else (L1 access), not by DRAM.

## Part 1: The naive kernel and the profiler

### Q1. Benchmark results (naive vs cuBLAS)

Commands: `./sgemm naive S S S` and `./sgemm cublas S S S` for S = 1024, 2048, 4096. All runs PASS.

| Size | naive ms | naive GFLOP/s | cuBLAS ms | cuBLAS GFLOP/s | naive / cuBLAS |
|---|---|---|---|---|---|
| 1024^3 | 13.24 | 162.2 | 0.183 | 11,716 | 1.4% |
| 2048^3 | 93.94 | 182.9 | 1.317 | 13,042 | 1.4% |
| 4096^3 | 748.82 | 183.5 | 10.923 | 12,582 | 1.5% |

![naive 1024](<screenshots/Screenshot_20261021_194611.png>)
![naive 2048](<screenshots/Screenshot_20261021_194644.png>)
![naive 4096](<screenshots/Screenshot_20261021_194655.png>)
![cublas 1024](<screenshots/Screenshot_20261021_194713.png>)
![cublas 2048](<screenshots/Screenshot_20261021_194729.png>)
![cublas 4096](<screenshots/Screenshot_20261021_194741.png>)

### Q2. Is the naive kernel limited by memory or by FP32 math?

Profile: `scripts/profile.sh naive 2048 2048 2048` (ncu `--set full`, `--clock-control none`).
The kernel runs in 93.96 ms.

GPU Speed Of Light Throughput:

| Metric | Value |
|---|---|
| Memory Throughput | **98.55%** |
| L1/TEX Cache Throughput | 99.02% |
| L2 Cache Throughput | 2.32% |
| DRAM Throughput | 0.66% |
| Compute (SM) Throughput | **11.95%** |

Memory Workload Analysis:

- L1/TEX hit rate 99.2%, L2 hit rate 94.2%.
- Global loads: 8,862,564,352 sectors / 537,001,984 requests = **16.5 sectors per request**.

Warp State Statistics:

- 192.1 cycles between issued instructions per warp.
- The dominant stall is **LG throttle** (waiting for the L1 queue for global memory operations to have room): 180.4 of 192.1 cycles (93.9%).

Compute Workload Analysis: issue slots busy 4.08%, executed IPC 0.16.

**Conclusion: the kernel is limited by memory access, specifically the L1/TEX path, not by FP32 math.**
Memory throughput is at 98.55% (L1/TEX 99.02%) while compute throughput is only 11.95%,
with the issue slots busy only 4% of the time. DRAM (0.66%) and L2 (2.32%) are
almost idle, so this is not a DRAM bandwidth problem: nearly all loads hit in L1/L2,
but the L1 is saturated by the number of sectors it has to process.

Why 16.5 sectors per request: in `naive.cuh`, `threadIdx.x` maps to the **row**, so the 32 lanes of a warp
have 32 different rows and the same column.

- `A[row*K + k]`: lanes are K floats apart, so the 32 lanes touch 32 different sectors (32 sectors per request).
- `B[k*N + col]`: all lanes read the same address (a broadcast), so 1 sector per request.
- The average over the two loads is (32 + 1) / 2 = 16.5, which matches the measured value exactly.

Percentage of peak FP32: 2 x 2048^3 / 93.96 ms = 182.8 GFLOP/s, which is 182.8 / 22,065 = **0.83% of peak FP32**
(also 0.83% at 4096^3, using 183.5 GFLOP/s). cuBLAS reaches about 53 to 59% of the same peak.

![naive: GPU Speed Of Light Throughput](<screenshots/ncu_naive_2048_2048_2048 - speed of light.png>)
![naive: Memory Workload Analysis](<screenshots/ncu_naive_2048_2048_2048 - memory workload analysis.png>)
![naive: L1/TEX Cache table (16.50 sectors/request)](<screenshots/ncu_naive_2048_2048_2048 - l1tex cache table.png>)
![naive: Warp State Statistics (LG throttle)](<screenshots/ncu_naive_2048_2048_2048 - warp state statistics.png>)

## Part 2: Coalescing

Change in `src/kernels/coalesced.cuh`: `threadIdx.x` now maps to the **column** and `threadIdx.y` to the row
(grid `ceil(N/32) x ceil(M/32)`, block 32x32). The algorithm and the order of loads are unchanged.
`check_all.sh` passes for `coalesced` on every shape.

### Q1. Benchmark results

| Size | naive ms | coalesced ms | speedup | coalesced GFLOP/s | % of cuBLAS |
|---|---|---|---|---|---|
| 4096^3 | 748.82 | 106.92 | 7.0x | 1285 | 10.4% |

At 4096^3 the coalesced kernel reaches 1285 / 22,065 = 5.8% of peak FP32 (naive: 0.83%).

![coalesced 4096](<screenshots/Screenshot_20261021_194842.png>)

### Q2. Why did the speed change?

ncu at 2048^3 (global loads, same 537,001,984 requests in both kernels):

| Metric | naive | coalesced |
|---|---|---|
| Global-load sectors | 8,862,564,352 | 1,342,701,568 |
| Sectors per request | 16.5 | **2.5** |
| Duration (ncu) | 93.96 ms | 11.42 ms |
| L1/TEX throughput | 99.02% | 98.89% |
| Dominant stall | LG throttle (180 of 192 cycles) | LG throttle (17.7 of 23.5 cycles) |
| Cycles between issued instructions per warp | 192.1 | 23.5 |

The number of requests (one per warp-level load instruction) and the FLOPs are identical. What changed is how many 32-byte sectors the L1 must process per request.

- Before: lanes had different rows, so `A[row*K+k]` hit 32 sectors per request and `B[k*N+col]` (same column) hit 1. Average 16.5.
- After: lanes have consecutive columns. `B[k*N+col]` reads 32 consecutive floats = 128 bytes = 4 sectors, and `A[row*K+k]` is a broadcast of one address = 1 sector. Average (4 + 1) / 2 = **2.5**, which matches the measurement.

The L1/TEX unit processes sectors at a fixed rate, and it was saturated in the naive kernel. With 6.6x fewer sectors for the same work, each warp waits far less for the L1 queue (192 down to 23.5 cycles per issued instruction), so the kernel runs about 8x faster. The L1 is still at about 99% of its throughput, so the kernel is still limited by L1 traffic, just much less so.

### Q3. Which memory level is busiest, and where do the repeated loads go?

- Busiest: **L1/TEX at 98.9%**. In the Memory Chart the red arrow between the Global unit and the L1/TEX cache (537.00 M requests) is the one near 100% of peak, while the links to L2 and device memory are dark purple (low utilization). L2 is at 10.4% and DRAM at 2.9%.
- The repeated loads (each element of A is loaded N times and each element of B M times) are served from the **L1 cache and then the L2 cache, not DRAM**:
  - L1/TEX hit rate is 94.96%.
  - Of the requests that miss L1, the L2 hit rate is 96.8%. L2 receives 67.6M sector reads (about 2.2 GB) against 1.34 billion sectors (about 43 GB) requested from L1.
  - DRAM reads only 68.6 MB, close to the 3 x 16.8 MB minimum for the three 2048^3 matrices, which means each element of A and B really is fetched from DRAM roughly once.
- So the data reuse is real, but it is happening through hardware caches. Every one of the 2N^3 loads still goes through the L1, which is why L1 stays saturated. Part 3 fixes this by keeping tiles in shared memory and registers explicitly.

![coalesced: Memory Chart](<screenshots/ncu_coalesced_2048_2048_2048 - memory chart.png>)
![coalesced: L1/TEX Cache table (2.50 sectors/request)](<screenshots/ncu_coalesced_2048_2048_2048 - l1tex cache table.png>)
![coalesced: GPU Speed Of Light Throughput](<screenshots/ncu_coalesced_2048_2048_2048 - speed of light.png>)

## Part 3: Tiling with shared memory and registers

Implemented `sgemm_smem<TILE>` (`smem16`, `smem32`) and `sgemm_reg2d` in `src/kernels/`. Six reg2d configurations are registered in `src/kernels.cuh`.
All pass the full `check_all.sh` including the sanitizer stage (203 PASS, 0 FAIL; memcheck and racecheck clean on every shape).

### Q1. Results

Benchmark at 4096^3 (cuBLAS: 11.14 ms, 12,337 GFLOP/s):

| Kernel | ms | GFLOP/s | % cuBLAS |
|---|---|---|---|
| smem16 | 81.29 | 1,691 | 13.7% |
| smem32 | 83.29 | 1,650 | 13.4% |
| reg2d_128x128x8_8x8 | 15.55 | 8,839 | 71.6% |
| reg2d_128x128x16_8x8 | **15.34** | **8,959** | **72.6%** |
| reg2d_128x64x8_8x4 | 18.09 | 7,599 | 61.6% |
| reg2d_64x64x32_4x4 | 24.04 | 5,716 | 46.3% |
| reg2d_128x128x8_4x4 | 24.42 | 5,628 | 45.6% |
| reg2d_64x64x8_4x4 | 25.54 | 5,381 | 43.6% |

![sgemm all 4096](<screenshots/Screenshot_20261021_194904.png>)

Experiment table (ncu `--set full` at 4096^3, `--clock-control none`; "time" is the ncu kernel duration, so it is slightly lower than the harness median).
Theoretical occupancy is `sm__maximum_warps_per_active_cycle_pct`, achieved is `sm__warps_active.avg.pct_of_peak_sustained_active`.
Bank conflicts are `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}`, in millions.
The dominant stall is the largest `warps_issue_stalled_*_per_issue_active` ratio, excluding `selected` and `not_selected`.

| Config | Time (ms) | Regs/thr | Smem/block | Occ. theor. | Occ. achieved | Dominant stall (ratio) | Conflicts ld | Conflicts st |
|---|---|---|---|---|---|---|---|---|
| 128x128x8, 8x8 | 15.10 | 120 | 8.2 KB | 33.3% | 33.3% | Long scoreboard (1.28) | 268M | 117M |
| 128x128x16, 8x8 | 14.87 | 125 | 16.4 KB | 33.3% | 33.3% | Long scoreboard (1.54) | 268M | 252M |
| 128x64x8, 8x4 | 17.63 | 80 | 6.1 KB | 50.0% | 49.6% | Long scoreboard (3.70) | 0 | 235M |
| 64x64x32, 4x4 | 23.13 | 64 | 16.4 KB | 66.7% | 66.1% | MIO throttle (5.43) | 0.2M | **1,040M** |
| 128x128x8, 4x4 | 24.29 | 64 | 8.2 KB | 66.7% | 66.7% | Long scoreboard (7.04) | 0 | 123M |
| 64x64x8, 4x4 | 25.49 | 58 | 4.1 KB | 66.7% | 66.2% | Long scoreboard (7.81) | 0 | 235M |

Registers and shared memory come from the `make` ptxas output. The 128x128 4x4 config uses 1024 threads per block, so only one block fits per SM.

### Q2. Shared-memory loads per FMA

- **3a (`smem`, one output per thread):** each FMA needs `As[ty][k]` and `Bs[k][tx]`, which is **2 shared loads per FMA**.
- **3b (reg2d, TM = TN = 8):** for each k a thread loads TM + TN = 16 values (8 of A, 8 of B) and does TM x TN = 64 FMAs, which is **16 / 64 = 0.25 loads per FMA**, 8x fewer than 3a.

The FMA units are idle while a thread waits for shared-memory data, and the shared-memory bandwidth (the LSU pipe) is limited.
In 3a the kernel spends most of its time issuing loads for every FMA.
In 3b each value loaded into a register is reused 8 times, so the same shared-memory bandwidth feeds 8x more FMAs, and the kernel becomes much less bound by shared-memory access.
That is the jump from about 1.6 TFLOP/s (smem) to about 8.5 TFLOP/s (reg2d 8x8), a 5.2x speedup.
A 4x4 thread tile gives 8 / 16 = 0.5 loads per FMA, which is only a 4x reduction, and those configs are indeed slower.

### Q3. Fastest configuration and a slower one

**Fastest:** `reg2d_128x128x16_8x8` (15.34 ms, 72.6% of cuBLAS), with `128x128x8_8x8` essentially tied (15.55 ms).
These have the lowest shared loads per FMA (0.25) and the most FMAs per thread between barriers.
For this config ncu raises no stall warning: it needs only 6.92 cycles between issued instructions (vs 17.3 for `64x64x8_4x4` and 17.7 for `64x64x32_4x4`), with long scoreboard the largest stall component at 1.54 cycles per issue. They use 120 to 125 registers per thread, which limits them to 2 blocks of 256 threads per SM, 33% theoretical occupancy, but the high instruction-level parallelism of the 64 independent accumulators makes up for it.

**Clearly slower: `reg2d_64x64x8_4x4` (25.49 ms, 1.7x slower than 128x128x8_8x8).**
It has *higher* occupancy (66.2% vs 33.3%) and fewer registers (58 vs 120), so occupancy is not the problem. The ncu metrics point to less work per thread:

- Long-scoreboard stall (waiting for global loads) is 7.81 per issued instruction, compared with 1.28 for the fast config. Each thread does only 16 FMAs per k and 128 per tile, so there is far less math between the global loads and the barriers to hide their latency, and extra warps do not make up for it.
- Barrier stalls are also higher (1.50 vs 0.46), because each block does a barrier-synchronized tile load for only 64x64 outputs.
- Shared loads per FMA are 0.5 instead of 0.25, and the shared-load wavefronts are 814M vs 672M.

**Another slow one: `reg2d_64x64x32_4x4` (23.13 ms in ncu): bank conflicts.**
Its shared-store conflicts are 1,040M (the Shared Memory table), about 9x those of the 128x128x8 8x8 config (117M). The transposed store `As[k][m]` has consecutive lanes writing consecutive `k` (since `k = idx % BK`) with `m` fixed, so the addresses are `BM` words apart. With BK = 32 and BM = 64 all 32 lanes of a warp hit the same bank (32-way conflict).
The matching ncu signs are MIO throttle (5.43, the top stall) and short scoreboard (3.89), which are both shared-memory pipeline stalls.
The store conflicts grow with BK (117M at BK = 8, 252M at BK = 16 for the 128x128 8x8 configs), so the larger BK that would reduce barriers is partly cancelled by conflicts here.

**`reg2d_128x128x8_4x4`** is limited by its 1024-thread blocks: one block per SM, with the highest barrier stall ratio (2.82) and long-scoreboard stall (7.04), so there is no other block to run while this one waits at a barrier.

Part 4 can address the conflicts: pad `As` or change the store mapping, and use vectorized loads.

![reg2d_128x128x16_8x8: Warp State Statistics](<screenshots/ncu_reg2d_128x128x16_8x8_4096 - warp state statistics.png>)
![reg2d_128x128x16_8x8: Occupancy (33.33% theoretical, 33.24% achieved)](<screenshots/ncu_reg2d_128x128x16_8x8_4096 - occupancy.png>)
![reg2d_128x128x16_8x8: Shared Memory table](<screenshots/ncu_reg2d_128x128x16_8x8_4096 - shared memory table.png>)
![reg2d_64x64x8_4x4: Warp State Statistics (long scoreboard)](<screenshots/ncu_reg2d_64x64x8_4x4_4096 - warp state statistics.png>)
![reg2d_64x64x8_4x4: Occupancy (66.67% theoretical, 66.16% achieved)](<screenshots/ncu_reg2d_64x64x8_4x4_4096 - occupancy.png>)
![reg2d_64x64x32_4x4: Warp State Statistics (MIO throttle)](<screenshots/ncu_reg2d_64x64x32_4x4_4096 - warp state statistics.png>)
![reg2d_64x64x32_4x4: Shared Memory table (store bank conflicts)](<screenshots/ncu_reg2d_64x64x32_4x4_4096 - shared memory table.png>)

## Part 4: Your own optimizations

Chain (`kPart4Chain`): `reg2d_128x128x16_8x8` -> `custom_pad` -> `custom_cf` -> `custom_vec4`.
Timings are from `scripts/grade.sh` at 4096^3 (best of 3 rounds, median of 20 launches).

### Step 1: padded `As` (`custom_pad`, row stride BM + 2)

- **What I tried:** in the transposed store `As[k][m]`, consecutive lanes write consecutive `k` with the same `m`, i.e. addresses `BM` words apart, which land in the same bank (4-way or worse conflicts, 252M measured by ncu). I padded the row stride from 128 to 130 words (2 odd/even interleaving: lanes with `k` go to even banks and the second `m` to odd banks, so the 32 lanes hit 32 distinct banks).
- **Time (`grade.sh`, final run):** 15.355 ms -> 14.780 ms = **1.039x** (an earlier run gave 15.348 -> 15.061 ms = 1.019x). Both are below the 1.05x threshold, so this step is a failed optimization. Five extra rounds with `--reps 40` gave 1.5% to 2.0%. A pad of 4 gave 0.7% to 1.8% and a pad of 1 was slower (16.35 ms), since with pad 1 half of the lanes still collide.
- **ncu (4096^3, baseline -> custom_pad):**

| Metric | reg2d_128x128x16_8x8 | custom_pad |
|---|---|---|
| Shared-store bank conflicts | 251.7M | **0** |
| Shared-store wavefronts | 285.2M | 33.6M |
| Short-scoreboard stall (per issue) | 1.26 | 0.49 |
| Shared-load bank conflicts | 268.4M | 268.4M (unchanged) |
| Shared-load wavefronts | 671.8M | 672.0M |
| Duration | 14.88 ms | 14.70 ms |
| FMA pipe active | 41.3% | 42.9% |
| Instructions executed | 3.117G | 3.218G (+3%) |

- **Why it helped so little:** the padding does what it was meant to do: the store conflicts are gone and the short-scoreboard stall dropped by 60%. But the stores were only about 30% of the shared-memory wavefronts (285M vs 672M for loads, and the stores happen once per tile while the loads happen in every one of the 16 k-steps), so they were not the bottleneck. The main loop is still bound by the 672M shared loads of the outer product (which keep their own 268M conflicts) and by the issue rate (FMA pipe active only about 43%, issue slots about 60% busy). The padded index also costs 3% more instructions. The remaining shared-load conflicts are the next thing to fix.

![baseline reg2d_128x128x16_8x8: Shared Memory table (252M store conflicts)](<screenshots/ncu_reg2d_128x128x16_8x8_4096 - shared memory table.png>)
![custom_pad: Shared Memory table (0 store conflicts)](<screenshots/ncu_custom_pad_4096 - shared memory table.png>)

### Step 2: conflict-free shared loads (`custom_cf`)

- **What I tried:** in the outer-product loop the thread reads its 8 values of B as `Bs[k][tc*8 .. tc*8+7]`. The compiler issues these as two 128-bit loads, and a 128-bit load is served 8 lanes at a time. Those 8 lanes (`tc` = 0..7) read 16-byte chunks 32 bytes apart, so lanes `tc` and `tc+4` hit the same banks (2-way conflict, 268M conflicts measured). I changed which columns a thread owns: instead of 8 contiguous columns it owns two groups of 4 contiguous columns, `tc*4 .. tc*4+3` and `BN/2 + tc*4 .. +3`. Now the 8 lanes read 8 consecutive 16-byte chunks, which is conflict-free. The write-back to C uses the same column mapping, so the results are identical.
- **Time (`grade.sh`, 4096^3):** 14.780 ms -> 14.311 ms = **1.033x** (an earlier run gave 15.045 -> 14.608 ms = 1.030x), below 1.05x, so this step is also a failed optimization. Repeated `--reps 40` runs gave 14.66 to 14.74 ms vs 15.04 to 15.42 ms (2.5% to 4.5%).
- **ncu (4096^3, custom_pad -> custom_cf):**

| Metric | custom_pad | custom_cf |
|---|---|---|
| Shared-load bank conflicts | 268.4M | **0** |
| Shared-load wavefronts | 672.0M | 403.3M (-40%) |
| Duration | 14.99 ms | 14.30 ms (-4.6%) |
| FMA pipe active | 41.9% | 44.3% |
| Issue slots active | 59.8% | 62.5% |
| Short-scoreboard stall (per issue) | 0.49 | 0.40 |
| MIO-throttle stall (per issue) | 0.32 | 0.20 |
| Instructions executed | 3.218G | 3.218G (unchanged) |

- **Why it helped so little:** the conflicts are completely gone and shared-load wavefronts dropped 40%, which lowered the MIO and short-scoreboard stalls. But the kernel is not mainly waiting on shared memory: it executes 3.22G warp instructions for 2.15G FMA warp instructions (only 67% of instructions are FMAs), and the issue slots are already 60 to 62% busy with the FMA pipe at 44%. Removing the conflicts did not change the instruction count, so the gain is limited to the cycles the warps used to stall on the LDS. The remaining stall is long scoreboard (1.37, global-load latency, which each block exposes at every tile boundary), and the remaining non-FMA instructions are index arithmetic and scalar global loads/stores. The next steps should therefore cut instruction count (vectorized `float4` loads) and hide global-load latency (double buffering).

![custom_pad: Shared Memory table (268M load conflicts)](<screenshots/ncu_custom_pad_4096 - shared memory table.png>)
![custom_cf: Shared Memory table (0 load conflicts)](<screenshots/ncu_custom_cf_4096 - shared memory table.png>)

### Step 3: `float4` global loads (`custom_vec4`)

- **What I tried:** the tile loads used one 32-bit global load per element (`BM*BK + BK*BN` = 4096 loads per block per tile, 16 per thread). I replaced them with 128-bit loads (`float4`). For B the `float4` is stored straight into `Bs` as a `float4`. For A, a `float4` holds 4 consecutive `k` of one row `m`, and I store it transposed into `As[k..k+3][m]` with four scalar stores. With the row padding from step 1 these stores are conflict-free: a warp covers 4 `k`-groups x 8 `m` values, so the banks are `k*8 + m` mod 32 = 32 distinct banks. The launcher returns false if A or B are not 16-byte aligned (all other requirements, M % 128 = N % 128 = K % 16 = 0, are the same as before).
- **Time (`grade.sh`, 4096^3):** 14.311 ms -> 12.855 ms = **1.113x**, a successful optimization (5 points). `--reps 40` runs gave 12.89 to 13.02 ms vs 14.44 to 14.56 ms.
- **ncu (4096^3, custom_cf -> custom_vec4):**

| Metric | custom_cf | custom_vec4 |
|---|---|---|
| Global-load requests | 34.08M | **8.91M** (-74%) |
| Global-load sectors | 142.6M | 142.6M (same bytes) |
| Instructions executed | 3.218G | **2.797G** (-13%) |
| FMA pipe active | 44.3% | 50.0% |
| Issue slots active | 62.5% | 63.6% |
| Registers per thread | 125 | 126 |
| Shared bank conflicts (ld + st) | 0 | 0 |
| Duration | 14.30 ms | 12.31 ms (-14%) |

- **Why it worked:** the bytes moved are identical (142.6M sectors), but each warp-level load request now moves 128 bits per lane, so there are 4x fewer load instructions (34.1M -> 8.9M requests) and the 4x fewer address computations that go with them. That removed 13% of all executed instructions. The kernel is bound by the issue rate (about 63% issue slots busy), so fewer non-FMA instructions turn directly into a higher FMA-pipe utilization (44% -> 50%) and a shorter kernel. Registers did not change, so occupancy stayed the same. The remaining stall is still long scoreboard (1.44 per issue), since the loads for the next tile are issued and waited for at each tile boundary, which is what double buffering would hide.

![custom_cf: L1/TEX Cache table (34.1M global-load requests)](<screenshots/ncu_custom_cf_4096 - l1tex cache table.png>)
![custom_vec4: L1/TEX Cache table (8.9M global-load requests)](<screenshots/ncu_custom_vec4_4096 - l1tex cache table.png>)
