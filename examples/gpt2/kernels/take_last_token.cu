extern "C" __global__ void take_last_token(
  const float *__restrict__ inp,
  float *__restrict__ out,
  unsigned int batch_size,
  unsigned int seq_len,
  unsigned int channels
) {
  const unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = vec_idx * 4;
  const unsigned int total = batch_size * channels;
  if (base >= total) return;

  #pragma unroll
  for (unsigned int i = 0; i < 4; ++i) {
    const unsigned int idx = base + i;
    if (idx >= total) return;
    const unsigned int b = idx / channels;
    const unsigned int c = idx % channels;
    out[idx] = inp[((b * seq_len) + (seq_len - 1)) * channels + c];
  }
}
