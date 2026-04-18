extern "C" __global__ void split_qkv_heads(
  const float *__restrict__ qkv,
  float *__restrict__ q,
  float *__restrict__ k,
  float *__restrict__ v,
  unsigned int batch_size,
  unsigned int seq_len,
  unsigned int channels,
  unsigned int head_dim,
  unsigned int num_heads
) {
  const unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int base = vec_idx * 4;
  const unsigned int total = batch_size * seq_len * channels;
  if (base >= total) return;

  #pragma unroll
  for (unsigned int i = 0; i < 4; ++i) {
    const unsigned int idx = base + i;
    if (idx >= total) return;
    const unsigned int bt = idx / channels;
    const unsigned int c = idx % channels;
    const unsigned int b = bt / seq_len;
    const unsigned int t = bt % seq_len;
    const unsigned int h = c / head_dim;
    const unsigned int d = c % head_dim;
    const unsigned int out_idx = (((b * num_heads + h) * seq_len + t) * head_dim) + d;
    const unsigned int qkv_row = bt * (channels * 3) + c;
    q[out_idx] = qkv[qkv_row];
    k[out_idx] = qkv[qkv_row + channels];
    v[out_idx] = qkv[qkv_row + channels * 2];
  }
}
