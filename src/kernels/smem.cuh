#pragma once
// =============================================================================
//  smem.cuh  --  Kernel 3: shared-memory tiling, one output per thread (Part 3a)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// Each block computes a TILE x TILE tile of C. Loop over K in steps of TILE:
// cooperatively load a tile of A and a tile of B into __shared__ memory,
// __syncthreads(), accumulate, __syncthreads().
// Must handle ANY M, N, K (zero-pad out-of-range tile elements).
template <int TILE>
__global__ void sgemm_smem(int M, int N, int K, float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B, float beta,
                           float* __restrict__ C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;  // tx -> column (coalesced)
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    const int ka = k0 + tx;  // column of A loaded by this thread
    const int kb = k0 + ty;  // row of B loaded by this thread
    As[ty][tx] = (row < M && ka < K) ? A[(size_t)row * K + ka] : 0.f;
    Bs[ty][tx] = (kb < K && col < N) ? B[(size_t)kb * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) {
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}

template <int TILE>
bool launch_smem(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C, cudaStream_t s) {
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  sgemm_smem<TILE><<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
