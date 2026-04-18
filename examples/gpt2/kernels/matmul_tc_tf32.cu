#include <mma.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

extern "C" __global__ void matmul_wmma_tf32(
  const float *__restrict__ a,
  const float *__restrict__ b,
  float *__restrict__ d,
  unsigned int m,
  unsigned int n,
  unsigned int k
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

  wmma::store_matrix_sync(d + warp_m * WMMA_M * n + warp_n * WMMA_N, acc_frag, n, wmma::mem_row_major);
#endif
}
