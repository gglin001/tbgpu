#include <math.h>

extern "C" __global__ void gelu_forward(
  const float *__restrict__ inp,
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
    const float x = inp[idx];
    const float cube = 0.044715f * x * x * x;
    out[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + cube)));
  }
}
