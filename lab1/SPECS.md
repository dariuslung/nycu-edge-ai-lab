# Homework: FP32 Matrix Multiplication Kernel in CUDA

Deep learning inference ultimately comes down to executing a sequence of GPU kernels, and matrix multiplication is one of the most common ones: linear layers, attention, and convolutions all reduce to it. How fast a model runs is therefore largely a question of how well those kernels use the hardware.

In this lab you will take a single-precision matrix multiplication (SGEMM) kernel from a naive implementation to one that approaches cuBLAS. Although the lab is written in CUDA, the underlying ideas are not GPU-specific. Reasoning about arithmetic intensity, memory hierarchy, and parallelism carries over to most other AI inference hardware.

## Rules

1. **FP32 on CUDA cores only.** No tensor cores, TF32, WMMA, or `mma.sync`.
2. **No libraries inside your kernels.** cuBLAS is only the reference.
3. **Every kernel must pass `scripts/check_all.sh`.** Only `naive`, `coalesced`, `smem16` and `smem32` must support every shape; other kernels may skip shapes they don't support.
4. **Please include screenshots showing the profiler metrics and values referenced in your analysis.**

## Setup

You will need `gcc`, `make`, `nvcc`, the CUDA and cuBLAS headers, `ncu`, and `nsys` installed in your GPU environment, and [Nsight Compute(GUI)](https://developer.nvidia.com/nsight-compute) on your host. How you install them depends on your OS, package manager, and CUDA version. The following is only an example, for Ubuntu with CUDA 12.8 from NVIDIA's apt repository; adapt the package names and versions to your own setup:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 \
    cuda-nsight-compute-12-8 cuda-nsight-systems-12-8
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

```bash
make                         # builds ./sgemm, prints registers/smem per kernel
./sgemm --list               # kernel ids
./sgemm all 4096 4096 4096   # benchmark + check every kernel
./sgemm naive 1024 1024 1024 # one kernel
scripts/check_all.sh         # correctness on many shapes + compute-sanitizer (what graders run)
NO_SANITIZER=1 scripts/check_all.sh  # faster while iterating: skip the sanitizer stage
scripts/sweep.sh             # optional: performance sweep over many shapes -> results/*.csv
scripts/profile.sh naive     # nsys + ncu reports into reports/
scripts/grade.sh             # what graders run: Part 2/3 correctness + Part 4 score
```
Some scripts may require chmod to execute at the first time.

If `make` fails on `-arch=native`, use e.g. `make ARCH=-arch=sm_86`.

If `ncu` reports `ERR_NVGPUCTRPERM`: on Linux run with `sudo` or have the counters enabled; on Windows enable "Allow access to GPU performance counters to all users" in the NVIDIA Control Panel.

### Harness output

```
id  kernel          ms   GFLOP/s   GB/s  %cuBLAS  %peakF  %peakBW  check
```

GFLOP/s is `2·M·N·K / time`. GB/s is the minimum DRAM traffic (read A and B, write C) divided by time. Peak numbers are estimated from device attributes; use the spec-sheet values in your report. Timings are the median of 20 launches with L2 flushed before each launch.

---

## Part 0 — Know your GPU (15%)

Find your GPU's name, compute capability, SM count, peak FP32 GFLOP/s, and peak DRAM GB/s. Take the peaks from your GPU's spec sheet, with peak FP32 at the boost clock.

Where to find the spec sheet: look up your GPU in the [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/), which lists SM count, CUDA cores, boost clock, FP32 TFLOP/s, memory bus width, and bandwidth for every NVIDIA GPU. For data-center GPUs (e.g. T4, A100), NVIDIA's official datasheet PDF (search "NVIDIA <GPU name> datasheet") also lists FP32 TFLOP/s and memory bandwidth. Compute capability is listed at [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).

Your peaks should match these formulas:

- Peak FP32 (GFLOP/s) = 2 × CUDA cores × boost clock (GHz). The 2 is because one fused multiply-add counts as two floating-point operations. For example, T4: 2 × 2560 × 1.59 ≈ 8,141 GFLOP/s.
- Peak DRAM bandwidth (GB/s) = memory data rate (Gbps per pin) × bus width (bits) ÷ 8. For example, T4: 10 × 256 ÷ 8 = 320 GB/s.

**Questions**

1. Report your GPU's name, compute capability, SM count, peak FP32 GFLOP/s, and peak DRAM GB/s, and show that the peaks match the formulas above.
2. What is the ridge point of your GPU's roofline (FLOP/byte), and what is the arithmetic intensity of SGEMM with M = N = K = 4096, assuming A and B are each read from DRAM exactly once and C is written exactly once? Should SGEMM at this size be compute-bound or memory-bound on your GPU?

## Part 1 — The naive kernel and the profiler (15%)

`sgemm_naive` in `src/kernels/naive.cuh` is given. Do not modify it.

1. Benchmark `naive` against `cublas` at 1024, 2048, and 4096.
2. Run `scripts/profile.sh naive 2048 2048 2048`.

Open the `.ncu-rep` in Nsight Compute and look at:

- **GPU Speed Of Light Throughput.** Both values are already a percentage of the hardware peak. Which one is closer to 100%?
- **Memory Workload Analysis.** L1/L2/DRAM throughput, and the ratio of sectors to requests for global loads.
- **Warp State Statistics.** The dominant stall reason.

**Questions**

1. Screenshot the benchmark results.
2. Using ncu numbers, show whether the naive kernel is limited by memory access or by FP32 math (quote the metrics you used). What percentage of your GPU's peak FP32 GFLOP/s does it reach?

## Part 2 — Coalescing (30%)

CUDA threads execute in groups of 32 called *warps*. All threads in a warp execute the same instruction at the same time.

This makes the memory access pattern matter: if the threads in a warp access consecutive memory locations, the GPU can combine those accesses into a small number of memory transactions. This is called *coalescing*. If the accesses are scattered, each one may need its own transaction, which is much slower.

Let's try it out by implementing `sgemm_coalesced` in `src/kernels/coalesced.cuh`. Use the same algorithm as the naive kernel, but change how threads map to `(row, col)` so that threads in the same warp access contiguous memory.

**Questions**

1. Screenshot the benchmark results.
2. The code performs the same FLOPs and the same loads in the same order. Explain why the speed changed, using the global-load sectors-per-request (or the equivalent ncu metric) before and after.
3. In the Memory Workload Analysis chart, which level of the memory hierarchy (L1, L2, DRAM) is now busiest? Each element of A is loaded N times in total (once per column of C), and each element of B M times (once per row of C). Where do those repeated loads get served from?


## Part 3 — Tiling with shared memory and registers (30%)



### 3a. Shared-memory tiling

Implement `sgemm_smem<TILE>` (one output per thread) in `src/kernels/smem.cuh`. It must handle any M, N, and K. Registered configurations: `smem16`, `smem32`.

### 3b. 2D register blocking

Implement `sgemm_reg2d<BM, BN, BK, TM, TN>` in `src/kernels/reg2d.cuh`. Each thread computes a TM×TN block of outputs held in registers; each block of threads computes a BM×BN tile. The launcher is provided. Set `kReg2DImplemented = true` once it works.

The inner loop over k is **given**: for each k, each thread loads TM values of A and TN values of B from shared memory into registers and does all TM × TN FMAs with them (an outer product). You write the two remaining parts:

- **Loading the tiles into shared memory.** A block has (BM/TM)·(BN/TN) threads, but the A tile has BM·BK elements and the B tile BK·BN, so each thread must load several elements in a loop. Store A **transposed** (`As[k][m]`), because the given loop reads it that way.
- **Writing the results back:** `C = alpha * acc + beta * C` for the thread's TM×TN sub-tile.

### Experiment

Three reg2d configurations are already registered. Add more to `kKernels[]` in `src/kernels.cuh` for a total of six reg2d configurations that vary BM/BN, BK, and TM/TN. For example: BM, BN ∈ {64, 128}; BK ∈ {8, 16, 32}; TM, TN ∈ {4, 8}. For each configuration, fill in a table with: time; registers/thread (from `make` output); shared memory per block; theoretical and achieved occupancy (ncu Occupancy section); dominant warp stall; shared-memory bank conflicts (ncu Memory Workload Analysis → Shared Memory table).

**Questions**

1. Show the results: a screenshot of the benchmark results for `smem16`, `smem32` and all six reg2d configurations, and the table from the experiment above.
2. In 3a, how many shared-memory loads does each thread do per FMA? In 3b with TM = TN = 8? Relate this to why 3b is faster.
3. Which configuration is fastest on your GPU? Pick one that is clearly slower and explain the difference using ncu metrics (e.g. register pressure → lower occupancy, bank conflicts, too few blocks to fill the SMs, barrier stalls).


## Part 4 — Your own optimizations (10%)

Try as many optimizations as you like. Put your kernels in `src/kernels/custom.cuh` and register them in `kKernels[]`. Then list the ones you want graded, in order, in `kPart4Chain` at the bottom of `src/kernels.cuh`:

```cpp
static const std::vector<const char*> kPart4Chain = {
    "reg2d_128x128x8_8x8",  // starting point: one of your Part 3 kernels
    "custom_vec4",          // + float4 loads
    "custom_vec4_dbuf",     // + double buffering (on top of custom_vec4)
};
```

Each entry must be the previous entry plus **one** optimization, and every entry must be correct (it must never FAIL in `check_all.sh`). `scripts/grade.sh` times each step against the step before it at 4096³ and scores it as below; run it yourself to see your score.

**Questions**

1. For each optimization: what did you try, what were the times before and after (and the speedup), and why do you think it worked or did not work?

**Grading (10 points, capped at 10).** Each optimization counts separately:

- **Successful optimization: 5 points.** The step is at least **1.05× faster** than the previous entry (previous time ÷ new time ≥ 1.05).
- **Failed optimization: 2.5 points.** The step is less than 1.05× faster (or slower), and your report explains why with measurements.
- **Incorrect entry: 0 points.** The next step is then compared with the last correct entry.

For example, two successful optimizations, or one successful and two failed ones, give full marks.

Below are some optimization ideas you may explore, but you are not limited to these:

| Optimization | Notes |
|---|---|
| Vectorized loads (`float4`) | For global and/or shared memory. Watch alignment. |
| Transposed / padded shared-memory layout | Find the bank conflicts in ncu first, then fix them. |
| Shared-memory swizzling | XOR-based index remapping as an alternative to padding. |
| Thread-block swizzling / rasterization | Changes the launch order of blocks to improve L2 hit rate. |
| Warp tiling | Adds a warp-level tile between block tiles and thread tiles. |
| Double buffering | Two shared-memory stages, or register prefetch of the next tile while computing the current one. |
| `__launch_bounds__` | Trade registers against occupancy. |
| Autotuning | Automatically search the configuration space. |
| Split-K, stream-K | For shapes with few output tiles. |
| Shape-specific kernels | E.g. a GEMV kernel for N = 1 (one warp per row, warp-shuffle reduction) behind a shape-dispatching launcher. |
| Anything else | Loop unrolling, `__ldg`, … |


---

## What to submit
Zip your files into student_id.zip, eg: 314551011.zip
    
- `src/kernels.cuh` and the whole `src/kernels/` folder: all your kernels, registered in `kKernels[]`.
- `<student_id>.pdf`: report. Answer every question in the **Questions** list of each part. Nothing else needs to be in the report.

## Notes

- Never profile or benchmark a `-G` (debug) build.
- A missing `__syncthreads()` often passes at small sizes and fails at large ones. Run `check_all.sh`.
- `row * K` overflows `int` for large matrices; use `size_t` for indices.

