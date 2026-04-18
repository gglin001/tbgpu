#include <float.h>
#include <math.h>
#include <stdint.h>

#define BLOCK_M 4
#define BLOCK_N 32
#define WARP_SIZE 32
#define MAX_HEAD_DIM 128
#define MAX_D_PER_LANE (MAX_HEAD_DIM / WARP_SIZE)

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

extern "C" __global__ void flash_attn_v2_fwd(
  const float *__restrict__ q,
  const float *__restrict__ k,
  const float *__restrict__ v,
  float *__restrict__ o,
  float *__restrict__ lse,
  unsigned int batch_size,
  unsigned int num_heads,
  unsigned int seq_len,
  unsigned int head_dim,
  float softmax_scale,
  unsigned int causal
) {
  (void)batch_size;
  (void)num_heads;

  const unsigned int lane = threadIdx.x;
  const unsigned int warp = threadIdx.y;
  const unsigned int batch_idx = blockIdx.z;
  const unsigned int head_idx = blockIdx.y;
  const unsigned int query_idx = blockIdx.x * BLOCK_M + warp;
  const bool active_row = query_idx < seq_len && head_dim <= MAX_HEAD_DIM;

  const unsigned int threads_per_block = blockDim.x * blockDim.y;
  const unsigned int linear_tid = warp * blockDim.x + lane;

  extern __shared__ float smem[];
  float *k_smem = smem;
  float *v_smem = k_smem + BLOCK_N * head_dim;
  float *score_smem = v_smem + BLOCK_N * head_dim;
  float *alpha_smem = score_smem + BLOCK_M * BLOCK_N;
  float *beta_smem = alpha_smem + BLOCK_M;
  float *row_l_smem = beta_smem + BLOCK_M;

  const uint64_t bh_offset = ((uint64_t)batch_idx * num_heads + head_idx) * seq_len * head_dim;
  const uint64_t bh_lse_offset = ((uint64_t)batch_idx * num_heads + head_idx) * seq_len;
  const float *q_ptr = q + bh_offset;
  const float *k_ptr = k + bh_offset;
  const float *v_ptr = v + bh_offset;
  float *o_ptr = o + bh_offset;
  float *lse_ptr = lse + bh_lse_offset;

  float q_frag[MAX_D_PER_LANE];
  float acc_frag[MAX_D_PER_LANE];
  const unsigned int frag_count = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
  float row_m = -INFINITY;
  float row_l = 0.0f;

  if (active_row) {
    const uint64_t q_row_offset = (uint64_t)query_idx * head_dim;
    for (unsigned int frag = 0; frag < frag_count; ++frag) {
      const unsigned int dim = lane + frag * WARP_SIZE;
      q_frag[frag] = dim < head_dim ? q_ptr[q_row_offset + dim] : 0.0f;
      acc_frag[frag] = 0.0f;
    }
  }

  for (unsigned int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_N) {
    const unsigned int valid_k = min(BLOCK_N, seq_len - kv_start);
    const unsigned int tile_elems = valid_k * head_dim;
    float tile_m = -INFINITY;
    float tile_l = 0.0f;
    float alpha = 0.0f;
    float beta = 0.0f;
    float tile_acc_frag[MAX_D_PER_LANE];

    for (unsigned int idx = linear_tid; idx < tile_elems; idx += threads_per_block) {
      k_smem[idx] = k_ptr[(uint64_t)kv_start * head_dim + idx];
      v_smem[idx] = v_ptr[(uint64_t)kv_start * head_dim + idx];
    }
    __syncthreads();

    for (unsigned int frag = 0; frag < frag_count; ++frag) tile_acc_frag[frag] = 0.0f;

    if (active_row) {
      for (unsigned int key = 0; key < valid_k; ++key) {
        float partial = 0.0f;
        for (unsigned int frag = 0; frag < frag_count; ++frag) {
          const unsigned int dim = lane + frag * WARP_SIZE;
          if (dim < head_dim) partial += q_frag[frag] * k_smem[key * head_dim + dim];
        }
        float score = warp_reduce_sum(partial) * softmax_scale;
        if (lane == 0) {
          if (causal != 0 && kv_start + key > query_idx) score = -INFINITY;
          score_smem[warp * BLOCK_N + key] = score;
        }
      }
    }

    __syncthreads();

    if (active_row && lane == 0) {
      for (unsigned int key = 0; key < valid_k; ++key) tile_m = fmaxf(tile_m, score_smem[warp * BLOCK_N + key]);

      if (tile_m != -INFINITY) {
        for (unsigned int key = 0; key < valid_k; ++key) {
          const float weight = score_smem[warp * BLOCK_N + key] == -INFINITY ? 0.0f : expf(score_smem[warp * BLOCK_N + key] - tile_m);
          score_smem[warp * BLOCK_N + key] = weight;
          tile_l += weight;
        }
      } else {
        for (unsigned int key = 0; key < valid_k; ++key) score_smem[warp * BLOCK_N + key] = 0.0f;
      }

      const float row_m_new = fmaxf(row_m, tile_m);
      alpha = row_m == -INFINITY ? 0.0f : expf(row_m - row_m_new);
      beta = tile_m == -INFINITY ? 0.0f : expf(tile_m - row_m_new);
      row_l = alpha * row_l + beta * tile_l;
      row_m = row_m_new;
      alpha_smem[warp] = alpha;
      beta_smem[warp] = beta;
      row_l_smem[warp] = row_l;
    }

    __syncthreads();

    if (active_row) {
      alpha = alpha_smem[warp];
      beta = beta_smem[warp];

      for (unsigned int key = 0; key < valid_k; ++key) {
        const float weight = score_smem[warp * BLOCK_N + key];
        for (unsigned int frag = 0; frag < frag_count; ++frag) {
          const unsigned int dim = lane + frag * WARP_SIZE;
          if (dim < head_dim) tile_acc_frag[frag] += weight * v_smem[key * head_dim + dim];
        }
      }

      for (unsigned int frag = 0; frag < frag_count; ++frag) acc_frag[frag] = alpha * acc_frag[frag] + beta * tile_acc_frag[frag];
    }

    __syncthreads();
  }

  if (!active_row) return;

  const float final_row_l = row_l_smem[warp];
  const float inv_row_l = final_row_l == 0.0f ? 0.0f : 1.0f / final_row_l;
  const uint64_t out_row_offset = (uint64_t)query_idx * head_dim;
  for (unsigned int frag = 0; frag < frag_count; ++frag) {
    const unsigned int dim = lane + frag * WARP_SIZE;
    if (dim < head_dim) o_ptr[out_row_offset + dim] = acc_frag[frag] * inv_row_l;
  }
  if (lane == 0) lse_ptr[query_idx] = row_l == 0.0f ? -INFINITY : row_m + logf(row_l);
}
