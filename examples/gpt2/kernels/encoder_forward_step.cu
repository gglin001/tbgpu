extern "C" __global__ void encoder_forward_step(
  unsigned int token,
  unsigned int position,
  const float *__restrict__ wte,
  const float *__restrict__ wpe,
  float *__restrict__ out,
  unsigned int channels
) {
  const unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = vec_idx * 4;
  if (base >= channels) return;

  #pragma unroll
  for (unsigned int i = 0; i < 4; ++i) {
    const unsigned int idx = base + i;
    if (idx >= channels) return;
    out[idx] = wte[token * channels + idx] + wpe[position * channels + idx];
  }
}
