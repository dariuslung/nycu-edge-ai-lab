#pragma once
// =============================================================================
//  custom.cuh  --  Your own optimized kernels (Part 4)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// e.g. a vectorized / double-buffered / swizzled variant of the reg2d kernel,
// a shape-dispatching launcher, ... Register each one in src/kernels.cuh.

// -----------------------------------------------------------------------------
// custom_pad / custom_cf: reg2d_128x128x16_8x8 + padded As (one change: As row stride is
// BM + PAD instead of BM). In the transposed store As[k][m], consecutive lanes
// write consecutive k with the same m, i.e. addresses BM words apart, which all
// land in the same bank. Padding the row stride spreads them over the banks.
// -----------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN, int PAD, bool CF, bool VEC>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_custom(int M, int N, int K, float alpha,
                     const float* __restrict__ A, const float* __restrict__ B,
                     float beta, float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ float As[BK][BM + PAD];
  __shared__ __align__(16) float Bs[BK][BN];
  const int tid = threadIdx.x;
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  // Column (inside the BN tile) of this thread's j-th output. CF = false: TN
  // contiguous columns (tc*TN + j). CF = true: two halves of TN/2 contiguous
  // columns, BN/2 apart, so that the 8 lanes of a 128-bit shared load read 8
  // consecutive 16-byte chunks (conflict-free) instead of chunks 32 bytes apart.
  auto bcol = [&](int j) {
    if (!CF) return tc * TN + j;
    return j < TN / 2 ? tc * (TN / 2) + j : BN / 2 + tc * (TN / 2) + (j - TN / 2);
  };
  for (int k0 = 0; k0 < K; k0 += BK) {
    if constexpr (VEC) {
      // 128-bit global loads. A: each float4 is 4 consecutive k of one row m,
      // stored transposed into As[k][m] (4 scalar shared stores; with PAD = 2
      // the lanes of a warp hit 32 distinct banks). B: float4 -> float4.
      static_assert(BK % 4 == 0 && BN % 4 == 0, "float4 tiles");
      static_assert((BM * BK / 4) % NT == 0 && (BK * BN / 4) % NT == 0, "even split");
#pragma unroll
      for (int idx = tid; idx < BM * BK / 4; idx += NT) {
        const int m = idx / (BK / 4), k = (idx % (BK / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(
            &A[(size_t)(blockIdx.y * BM + m) * K + k0 + k]);
        As[k + 0][m] = v.x;
        As[k + 1][m] = v.y;
        As[k + 2][m] = v.z;
        As[k + 3][m] = v.w;
      }
#pragma unroll
      for (int idx = tid; idx < BK * BN / 4; idx += NT) {
        const int k = idx / (BN / 4), n = (idx % (BN / 4)) * 4;
        *reinterpret_cast<float4*>(&Bs[k][n]) = *reinterpret_cast<const float4*>(
            &B[(size_t)(k0 + k) * N + blockIdx.x * BN + n]);
      }
    } else {
      for (int idx = tid; idx < BM * BK; idx += NT) {
        const int m = idx / BK, k = idx % BK;
        As[k][m] = A[(size_t)(blockIdx.y * BM + m) * K + k0 + k];
      }
      for (int idx = tid; idx < BK * BN; idx += NT) {
        const int k = idx / BN, n = idx % BN;
        Bs[k][n] = B[(size_t)(k0 + k) * N + blockIdx.x * BN + n];
      }
    }
    __syncthreads();
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][bcol(j)];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; ++i) {
    const size_t r = (size_t)blockIdx.y * BM + tr * TM + i;
    for (int j = 0; j < TN; ++j) {
      const size_t idx = r * N + blockIdx.x * BN + bcol(j);
      C[idx] = alpha * acc[i][j] + beta * C[idx];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN, int PAD, bool CF, bool VEC>
bool launch_custom(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C, cudaStream_t s) {
  constexpr int kThreads = (BM / TM) * (BN / TN);
  if (M % BM || N % BN || K % BK) return false;
  if (VEC && (((uintptr_t)A | (uintptr_t)B) & 15)) return false;  // float4 needs 16B alignment
  dim3 grid(N / BN, M / BM);
  sgemm_custom<BM, BN, BK, TM, TN, PAD, CF, VEC>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
