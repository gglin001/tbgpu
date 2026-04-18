extern "C" __global__ void pad_rows(
  const float *__restrict__ inp,
  float *__restrict__ out,
  unsigned int rows,
  unsigned int padded_rows,
  unsigned int cols
) {
  const unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = vec_idx * 4;
  const unsigned int total = padded_rows * cols;
  if (base >= total) return;

  #pragma unroll
  for (unsigned int i = 0; i < 4; ++i) {
    const unsigned int idx = base + i;
    if (idx >= total) return;
    const unsigned int row = idx / cols;
    out[idx] = row < rows ? inp[idx] : 0.0f;
  }
}
