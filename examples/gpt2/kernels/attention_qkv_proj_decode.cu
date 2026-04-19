#include <float.h>
#include <math.h>
#include <stdint.h>

#define BLOCK_N 32
#define WARP_SIZE 32
#define MAX_HEAD_DIM 128
#define MAX_D_PER_LANE (MAX_HEAD_DIM / WARP_SIZE)

__device__ __forceinline__ float warp_reduce_sum_decode(float value) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

extern "C" __global__ void flash_attn_qkv_proj_decode_fwd(
  const float *__restrict__ qkv,
  const float *__restrict__ proj_weight_t,
  const float *__restrict__ proj_bias,
  const float *__restrict__ key_cache,
  const float *__restrict__ value_cache,
  float *__restrict__ out,
  unsigned int num_heads,
  unsigned int cache_len,
  unsigned int head_dim,
  float softmax_scale
) {
  const unsigned int lane = threadIdx.x;
  const unsigned int warp = threadIdx.y;
  const unsigned int out_col = blockIdx.x * WARP_SIZE + lane;
  const unsigned int channels = num_heads * head_dim;

  extern __shared__ float smem[];
  float *score_smem = smem;
  float *alpha_smem = score_smem + num_heads * BLOCK_N;
  float *beta_smem = alpha_smem + num_heads;
  float *row_l_smem = beta_smem + num_heads;
  float *partial_smem = row_l_smem + num_heads;
  float *attn_smem = partial_smem + num_heads * WARP_SIZE;

  float q_frag[MAX_D_PER_LANE];
  float acc_frag[MAX_D_PER_LANE];
  const unsigned int frag_count = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
  const unsigned int head_offset = warp * head_dim;
  float row_m = -INFINITY;
  float row_l = 0.0f;

  for (unsigned int frag = 0; frag < frag_count; ++frag) {
    const unsigned int dim = lane + frag * WARP_SIZE;
    q_frag[frag] = dim < head_dim ? qkv[head_offset + dim] : 0.0f;
    acc_frag[frag] = 0.0f;
  }

  for (unsigned int kv_start = 0; kv_start < cache_len; kv_start += BLOCK_N) {
    const unsigned int valid_k = min(BLOCK_N, cache_len - kv_start);
    float tile_m = -INFINITY;
    float tile_l = 0.0f;
    float alpha = 0.0f;
    float beta = 0.0f;
    float tile_acc_frag[MAX_D_PER_LANE];

    for (unsigned int frag = 0; frag < frag_count; ++frag) tile_acc_frag[frag] = 0.0f;

    for (unsigned int key = 0; key < valid_k; ++key) {
      float partial = 0.0f;
      const unsigned int key_base = (kv_start + key) * channels + head_offset;
      for (unsigned int frag = 0; frag < frag_count; ++frag) {
        const unsigned int dim = lane + frag * WARP_SIZE;
        if (dim < head_dim) partial += q_frag[frag] * key_cache[key_base + dim];
      }
      const float score = warp_reduce_sum_decode(partial) * softmax_scale;
      if (lane == 0) score_smem[warp * BLOCK_N + key] = score;
    }
    __syncthreads();

    if (lane == 0) {
      for (unsigned int key = 0; key < valid_k; ++key) tile_m = fmaxf(tile_m, score_smem[warp * BLOCK_N + key]);
      for (unsigned int key = 0; key < valid_k; ++key) {
        const float weight = expf(score_smem[warp * BLOCK_N + key] - tile_m);
        score_smem[warp * BLOCK_N + key] = weight;
        tile_l += weight;
      }

      const float row_m_new = fmaxf(row_m, tile_m);
      alpha = row_m == -INFINITY ? 0.0f : expf(row_m - row_m_new);
      beta = expf(tile_m - row_m_new);
      row_l = alpha * row_l + beta * tile_l;
      row_m = row_m_new;
      alpha_smem[warp] = alpha;
      beta_smem[warp] = beta;
      row_l_smem[warp] = row_l;
    }
    __syncthreads();

    alpha = alpha_smem[warp];
    beta = beta_smem[warp];
    for (unsigned int key = 0; key < valid_k; ++key) {
      const float weight = score_smem[warp * BLOCK_N + key];
      const unsigned int value_base = (kv_start + key) * channels + head_offset;
      for (unsigned int frag = 0; frag < frag_count; ++frag) {
        const unsigned int dim = lane + frag * WARP_SIZE;
        if (dim < head_dim) tile_acc_frag[frag] += weight * value_cache[value_base + dim];
      }
    }
    for (unsigned int frag = 0; frag < frag_count; ++frag) acc_frag[frag] = alpha * acc_frag[frag] + beta * tile_acc_frag[frag];
    __syncthreads();
  }

  const float inv_row_l = 1.0f / row_l_smem[warp];
  for (unsigned int frag = 0; frag < frag_count; ++frag) {
    const unsigned int dim = lane + frag * WARP_SIZE;
    if (dim < head_dim) attn_smem[warp * head_dim + dim] = acc_frag[frag] * inv_row_l;
  }
  __syncthreads();

  float partial = 0.0f;
  if (out_col < channels) {
    for (unsigned int dim = 0; dim < head_dim; ++dim) {
      partial += attn_smem[warp * head_dim + dim] * proj_weight_t[(head_offset + dim) * channels + out_col];
    }
  }
  partial_smem[warp * WARP_SIZE + lane] = partial;
  __syncthreads();

  if (warp == 0 && out_col < channels) {
    float total = 0.0f;
    for (unsigned int head = 0; head < num_heads; ++head) total += partial_smem[head * WARP_SIZE + lane];
    out[out_col] = total + proj_bias[out_col];
  }
}
