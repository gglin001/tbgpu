#include <mma.h>
#include <math.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

extern "C" __global__ void matmul_wmma_tf32(
  const float *__restrict__ a,
  const float *__restrict__ b,
  const float *__restrict__ bias,
  float *__restrict__ d,
  unsigned int m,
  unsigned int n,
  unsigned int k,
  unsigned int epilogue
) {
#if __CUDA_ARCH__ >= 800
  const unsigned int warp_m = blockIdx.x;
  const unsigned int warp_n = blockIdx.y * blockDim.y + threadIdx.y;

  if (warp_m * WMMA_M >= m || warp_n * WMMA_N >= n) return;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  for (unsigned int kk = 0; kk < k; kk += WMMA_K) {
    const unsigned int a_row = warp_m * WMMA_M;
    const unsigned int b_col = warp_n * WMMA_N;

    wmma::load_matrix_sync(a_frag, a + a_row * k + kk, k);
    wmma::load_matrix_sync(b_frag, b + kk + b_col * k, k);

    #pragma unroll
    for (int i = 0; i < a_frag.num_elements; ++i) a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
    #pragma unroll
    for (int i = 0; i < b_frag.num_elements; ++i) b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  const unsigned int row_offset = warp_m * WMMA_M;
  const unsigned int col_offset = warp_n * WMMA_N;
  float *tile_ptr = d + row_offset * n + col_offset;
  wmma::store_matrix_sync(tile_ptr, acc_frag, n, wmma::mem_row_major);
  __syncwarp();

  for (unsigned int index = threadIdx.x; index < WMMA_M * WMMA_N; index += 32) {
    const unsigned int tile_row = index / WMMA_N;
    const unsigned int tile_col = index % WMMA_N;
    const unsigned int global_row = row_offset + tile_row;
    const unsigned int global_col = col_offset + tile_col;
    if (global_row >= m || global_col >= n) continue;
    float value = tile_ptr[tile_row * n + tile_col];
    if (bias != nullptr) value += bias[global_col];
    if (epilogue == 1) {
      const float cube = 0.044715f * value * value * value;
      value = 0.5f * value * (1.0f + tanhf(0.7978845608028654f * (value + cube)));
    }
    tile_ptr[tile_row * n + tile_col] = value;
  }
#endif
}
