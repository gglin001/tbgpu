extern "C" __global__ void encoder_forward(
  const int *__restrict__ tokens,
  const float *__restrict__ wte,
  const float *__restrict__ wpe,
  float *__restrict__ out,
  unsigned int batch_size,
  unsigned int seq_len,
  unsigned int channels
) {
  const unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = vec_idx * 4;
  const unsigned int total = batch_size * seq_len * channels;
  if (base >= total) return;

  const unsigned int bt = base / channels;
  const unsigned int t = bt % seq_len;
  const unsigned int token = (unsigned int)tokens[bt];
  const unsigned int c = base % channels;

  #pragma unroll
  for (unsigned int i = 0; i < 4; ++i) {
    const unsigned int idx = base + i;
    if (idx >= total || c + i >= channels) return;
    out[idx] = wte[token * channels + c + i] + wpe[t * channels + c + i];
  }
}
