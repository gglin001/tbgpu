extern "C" __global__ void add_bias(
  float *__restrict__ out,
  const float *__restrict__ bias,
  unsigned int rows,
  unsigned int cols
) {
  const unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = vec_idx * 4;
  const unsigned int total = rows * cols;
  if (base >= total) return;

  #pragma unroll
  for (unsigned int i = 0; i < 4; ++i) {
    const unsigned int idx = base + i;
    if (idx >= total) return;
    out[idx] += bias[idx % cols];
  }
}
