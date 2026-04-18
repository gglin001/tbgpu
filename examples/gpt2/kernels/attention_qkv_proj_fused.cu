#include <float.h>
#include <math.h>
#include <stdint.h>

#define BLOCK_M 4
#define BLOCK_N 32
#define WARP_SIZE 32
#define OUT_TILE 32
#define MAX_HEAD_DIM 128
#define MAX_D_PER_LANE (MAX_HEAD_DIM / WARP_SIZE)

__device__ __forceinline__ float warp_reduce_sum_qkv_proj(float value) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

extern "C" __global__ void flash_attn_qkv_proj_fused_fwd(
  const float *__restrict__ qkv,
  const float *__restrict__ proj_weight_t,
  const float *__restrict__ proj_bias,
  float *__restrict__ o,
  unsigned int batch_size,
  unsigned int num_heads,
  unsigned int seq_len,
  unsigned int head_dim,
  float softmax_scale,
  unsigned int causal
) {
  (void)batch_size;

  const unsigned int lane = threadIdx.x;
  const unsigned int warp = threadIdx.y;
  const unsigned int batch_idx = blockIdx.z;
  const unsigned int query_idx = blockIdx.x * BLOCK_M + warp;
  const unsigned int out_base = blockIdx.y * OUT_TILE;
  const bool active_row = query_idx < seq_len && head_dim <= MAX_HEAD_DIM;

  const unsigned int channels = num_heads * head_dim;
  const unsigned int qkv_channels = channels * 3;
  const unsigned int out_col = out_base + lane;
  const unsigned int threads_per_block = blockDim.x * blockDim.y;
  const unsigned int linear_tid = warp * blockDim.x + lane;

  extern __shared__ float smem[];
  float *k_smem = smem;
  float *v_smem = k_smem + BLOCK_N * head_dim;
  float *score_smem = v_smem + BLOCK_N * head_dim;
  float *alpha_smem = score_smem + BLOCK_M * BLOCK_N;
  float *beta_smem = alpha_smem + BLOCK_M;
  float *row_l_smem = beta_smem + BLOCK_M;
  float *attn_smem = row_l_smem + BLOCK_M;

  float proj_acc = 0.0f;

  for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
    float q_frag[MAX_D_PER_LANE];
    float acc_frag[MAX_D_PER_LANE];
    const unsigned int frag_count = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float row_m = -INFINITY;
    float row_l = 0.0f;

    if (active_row) {
      const uint64_t q_row_base = ((uint64_t)batch_idx * seq_len + query_idx) * qkv_channels + head_idx * head_dim;
      for (unsigned int frag = 0; frag < frag_count; ++frag) {
        const unsigned int dim = lane + frag * WARP_SIZE;
        q_frag[frag] = dim < head_dim ? qkv[q_row_base + dim] : 0.0f;
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
        const unsigned int key = idx / head_dim;
        const unsigned int dim = idx % head_dim;
        const uint64_t k_base = ((uint64_t)batch_idx * seq_len + (kv_start + key)) * qkv_channels + channels + head_idx * head_dim;
        k_smem[idx] = qkv[k_base + dim];
        v_smem[idx] = qkv[k_base + channels + dim];
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
          float score = warp_reduce_sum_qkv_proj(partial) * softmax_scale;
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

    const float final_row_l = active_row ? row_l_smem[warp] : 0.0f;
    const float inv_row_l = final_row_l == 0.0f ? 0.0f : 1.0f / final_row_l;
    for (unsigned int frag = 0; frag < frag_count; ++frag) {
      const unsigned int dim = lane + frag * WARP_SIZE;
      if (dim < head_dim) attn_smem[warp * head_dim + dim] = active_row ? (acc_frag[frag] * inv_row_l) : 0.0f;
    }
    __syncthreads();

    if (active_row && out_col < channels) {
      const unsigned int head_offset = head_idx * head_dim;
      for (unsigned int dim = 0; dim < head_dim; ++dim) {
        proj_acc += attn_smem[warp * head_dim + dim] * proj_weight_t[(head_offset + dim) * channels + out_col];
      }
    }
    __syncthreads();
  }

  if (!active_row || out_col >= channels) return;
  const uint64_t out_idx = ((uint64_t)batch_idx * seq_len + query_idx) * channels + out_col;
  o[out_idx] = proj_acc + (proj_bias == nullptr ? 0.0f : proj_bias[out_col]);
}
