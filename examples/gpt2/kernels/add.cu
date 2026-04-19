extern "C" __global__ void residual_add(
  const float *__restrict__ a,
  const float *__restrict__ b,
  float *__restrict__ out,
  unsigned int count
) {
  const unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = vec_idx * 4;
  if (base >= count) return;

  #pragma unroll
  for (unsigned int i = 0; i < 4; ++i) {
    const unsigned int idx = base + i;
    if (idx >= count) return;
    out[idx] = a[idx] + b[idx];
  }
}
