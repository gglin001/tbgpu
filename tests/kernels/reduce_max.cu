#include <float.h>
#include <math.h>

#define ITEMS_PER_THREAD 4

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

extern "C" __global__ void reduce_max(const float *__restrict__ inp, float *__restrict__ out, unsigned int n) {
  extern __shared__ float shared[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * ITEMS_PER_THREAD + tid;
  unsigned int grid_stride = gridDim.x * blockDim.x * ITEMS_PER_THREAD;
  float local_max = -FLT_MAX;

  while (i < n) {
    local_max = fmaxf(local_max, inp[i]);
    if (i + blockDim.x < n) local_max = fmaxf(local_max, inp[i + blockDim.x]);
    if (i + 2 * blockDim.x < n) local_max = fmaxf(local_max, inp[i + 2 * blockDim.x]);
    if (i + 3 * blockDim.x < n) local_max = fmaxf(local_max, inp[i + 3 * blockDim.x]);
    i += grid_stride;
  }

  shared[tid] = local_max;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 32; s >>= 1) {
    if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    __syncthreads();
  }

  float block_max = shared[tid];
  if (tid < 32) {
    if (blockDim.x >= 64) block_max = fmaxf(block_max, shared[tid + 32]);
    block_max = warp_reduce_max(block_max);
    if (tid == 0) out[blockIdx.x] = block_max;
  }
}
