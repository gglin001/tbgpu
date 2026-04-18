extern "C" __global__ void write_kv_cache(
  const float *__restrict__ qkv,
  float *__restrict__ key_cache,
  float *__restrict__ value_cache,
  unsigned int rows,
  unsigned int channels,
  unsigned int start_pos
) {
  const unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = vec_idx * 4;
  const unsigned int total = rows * channels;
  if (base >= total) return;

  #pragma unroll
  for (unsigned int i = 0; i < 4; ++i) {
    const unsigned int idx = base + i;
    if (idx >= total) return;
    const unsigned int row = idx / channels;
    const unsigned int col = idx % channels;
    const unsigned int cache_idx = (start_pos + row) * channels + col;
    const unsigned int qkv_idx = row * channels * 3 + channels + col;
    key_cache[cache_idx] = qkv[qkv_idx];
    value_cache[cache_idx] = qkv[qkv_idx + channels];
  }
}
