#define TILE 16

extern "C" __global__ void matmul_tiled_fp32(
  float *__restrict__ c,
  const float *__restrict__ a,
  const float *__restrict__ b,
  unsigned int m,
  unsigned int n,
  unsigned int k,
  float alpha,
  float beta
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
    c[idx] = alpha * acc + beta * c[idx];
  }
}
