#include <math.h>

__device__ __forceinline__ float fused_warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

extern "C" __global__ void fused_residual_layernorm(
  const float *__restrict__ residual_in,
  const float *__restrict__ update,
  const float *__restrict__ weight,
  const float *__restrict__ bias,
  float *__restrict__ residual_out,
  float *__restrict__ norm_out,
  unsigned int rows,
  unsigned int channels
) {
  const unsigned int lane = threadIdx.x;
  const unsigned int warp = threadIdx.y;
  const unsigned int row = blockIdx.x * blockDim.y + warp;
  if (row >= rows) return;

  const float *a = residual_in + row * channels;
  const float *b = update + row * channels;
  float *res = residual_out + row * channels;
  float *norm = norm_out + row * channels;

  float sum = 0.0f;
  float sum_sq = 0.0f;
  for (unsigned int c = lane; c < channels; c += 32) {
    const float v = a[c] + b[c];
    res[c] = v;
    sum += v;
    sum_sq += v * v;
  }
  sum = fused_warp_sum(sum);
  sum_sq = fused_warp_sum(sum_sq);
  sum = __shfl_sync(0xffffffffu, sum, 0);
  sum_sq = __shfl_sync(0xffffffffu, sum_sq, 0);

  const float mean = sum / channels;
  const float var = fmaxf(sum_sq / channels - mean * mean, 0.0f);
  const float inv_std = rsqrtf(var + 1e-5f);

  for (unsigned int c = lane; c < channels; c += 32) {
    norm[c] = (res[c] - mean) * inv_std * weight[c] + bias[c];
  }
}
