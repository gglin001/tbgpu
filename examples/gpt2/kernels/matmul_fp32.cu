#include <math.h>

#define TILE 16

extern "C" __global__ void matmul_tiled_fp32(
  const float *__restrict__ a,
  const float *__restrict__ b,
  const float *__restrict__ bias,
  float *__restrict__ c,
  unsigned int m,
  unsigned int n,
  unsigned int k,
  unsigned int epilogue
) {
  __shared__ float a_tile[TILE][TILE];
  __shared__ float b_tile[TILE][TILE];

  const unsigned int row = blockIdx.y * TILE + threadIdx.y;
  const unsigned int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  for (unsigned int tile = 0; tile < k; tile += TILE) {
    const unsigned int a_col = tile + threadIdx.x;
    const unsigned int b_row = tile + threadIdx.y;

    a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (unsigned int kk = 0; kk < TILE; ++kk) {
      acc += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    const unsigned int idx = row * n + col;
    float value = acc;
    if (bias != nullptr) value += bias[col];
    if (epilogue == 1) {
      const float cube = 0.044715f * value * value * value;
      value = 0.5f * value * (1.0f + tanhf(0.7978845608028654f * (value + cube)));
    }
    c[idx] = value;
  }
}
