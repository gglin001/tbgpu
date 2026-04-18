from __future__ import annotations

import argparse
import array
import ctypes
import math
from pathlib import Path

import tbgpu.cuda_compat as cuda
from tbgpu.compiler import compile_cuda_to_ptx

KERNEL_NAME = "flash_attn_v2_fwd"
BLOCK_M = 4
BLOCK_N = 32
WARP_SIZE = 32
MAX_HEAD_DIM = 128


def init_c_var(ty, create_cb):
  value = ty()
  create_cb(value)
  return value


def _check(status: int):
  if status != 0:
    err = ctypes.POINTER(ctypes.c_char)()
    cuda.cuGetErrorString(status, ctypes.byref(err))
    raise RuntimeError(f"CUDA shim error {status}: {ctypes.string_at(err).decode()}")


def _buffer_ptr(buf: array.array) -> int:
  return ctypes.addressof((ctypes.c_float * len(buf)).from_buffer(buf))


def _tensor_offset(num_heads: int, seq_len: int, head_dim: int, batch: int, head: int, row: int, col: int) -> int:
  return (((batch * num_heads + head) * seq_len + row) * head_dim) + col


def _lse_offset(num_heads: int, seq_len: int, batch: int, head: int, row: int) -> int:
  return ((batch * num_heads + head) * seq_len) + row


def _build_tensor(batch_size: int, num_heads: int, seq_len: int, head_dim: int, seed: int) -> array.array:
  total = batch_size * num_heads * seq_len * head_dim
  return array.array("f", (float((((seed + i * 17) % 113) - 56) / 16.0) for i in range(total)))


def _cpu_flash_attn(
  q: array.array,
  k: array.array,
  v: array.array,
  batch_size: int,
  num_heads: int,
  seq_len: int,
  head_dim: int,
  causal: bool,
):
  scale = 1.0 / math.sqrt(head_dim)
  out = array.array("f", [0.0] * (batch_size * num_heads * seq_len * head_dim))
  lse = array.array("f", [0.0] * (batch_size * num_heads * seq_len))

  for batch in range(batch_size):
    for head in range(num_heads):
      for row in range(seq_len):
        scores = []
        max_score = -math.inf
        for key in range(seq_len):
          if causal and key > row:
            scores.append(-math.inf)
            continue
          score = 0.0
          for dim in range(head_dim):
            q_idx = _tensor_offset(num_heads, seq_len, head_dim, batch, head, row, dim)
            k_idx = _tensor_offset(num_heads, seq_len, head_dim, batch, head, key, dim)
            score += q[q_idx] * k[k_idx]
          score *= scale
          scores.append(score)
          if score > max_score:
            max_score = score

        exps = [0.0] * seq_len
        denom = 0.0
        for key, score in enumerate(scores):
          if score == -math.inf:
            continue
          weight = math.exp(score - max_score)
          exps[key] = weight
          denom += weight

        inv_denom = 0.0 if denom == 0.0 else 1.0 / denom
        lse[_lse_offset(num_heads, seq_len, batch, head, row)] = -math.inf if denom == 0.0 else max_score + math.log(denom)

        for dim in range(head_dim):
          value = 0.0
          for key, weight in enumerate(exps):
            if weight == 0.0:
              continue
            v_idx = _tensor_offset(num_heads, seq_len, head_dim, batch, head, key, dim)
            value += weight * v[v_idx]
          out[_tensor_offset(num_heads, seq_len, head_dim, batch, head, row, dim)] = value * inv_denom

  return out, lse


def _max_abs_diff(got: array.array, expected: array.array) -> float:
  return max(abs(g - e) for g, e in zip(got, expected)) if got else 0.0


def _make_kernel_params(
  q_ptr: int,
  k_ptr: int,
  v_ptr: int,
  o_ptr: int,
  lse_ptr: int,
  batch_size: int,
  num_heads: int,
  seq_len: int,
  head_dim: int,
  softmax_scale: float,
  causal: bool,
):
  scalars = [
    ctypes.c_uint64(q_ptr),
    ctypes.c_uint64(k_ptr),
    ctypes.c_uint64(v_ptr),
    ctypes.c_uint64(o_ptr),
    ctypes.c_uint64(lse_ptr),
    ctypes.c_uint32(batch_size),
    ctypes.c_uint32(num_heads),
    ctypes.c_uint32(seq_len),
    ctypes.c_uint32(head_dim),
    ctypes.c_float(softmax_scale),
    ctypes.c_uint32(int(causal)),
  ]
  params = (ctypes.c_void_p * len(scalars))(*[ctypes.addressof(value) for value in scalars])
  return params, scalars


def _kernel_source() -> str:
  kernel_path = Path(__file__).with_name("kernels") / "flash_attn_v2.cu"
  return kernel_path.read_text()


def run_flash_attn_v2(
  batch_size: int,
  num_heads: int,
  seq_len: int,
  head_dim: int,
  causal: bool = True,
):
  if batch_size <= 0 or num_heads <= 0 or seq_len <= 0:
    raise ValueError("batch_size, num_heads, and seq_len must be positive")
  if head_dim <= 0 or head_dim > MAX_HEAD_DIM:
    raise ValueError(f"head_dim must be between 1 and {MAX_HEAD_DIM}")

  scale = 1.0 / math.sqrt(head_dim)
  q = _build_tensor(batch_size, num_heads, seq_len, head_dim, seed=11)
  k = _build_tensor(batch_size, num_heads, seq_len, head_dim, seed=23)
  v = _build_tensor(batch_size, num_heads, seq_len, head_dim, seed=37)
  expected_out, expected_lse = _cpu_flash_attn(q, k, v, batch_size, num_heads, seq_len, head_dim, causal)
  host_out = array.array("f", [0.0] * len(expected_out))
  host_lse = array.array("f", [0.0] * len(expected_lse))

  _check(cuda.cuInit(0))
  dev = init_c_var(cuda.CUdevice, lambda x: _check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
  ctx = init_c_var(cuda.CUcontext, lambda x: _check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, dev.value)))
  _check(cuda.cuCtxSetCurrent(ctx))

  module = None
  q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr = [cuda.CUdeviceptr() for _ in range(5)]
  try:
    _check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), dev.value))
    arch = f"sm_{major.value}{minor.value}"
    kernel_image = compile_cuda_to_ptx(_kernel_source().strip() + "\n", arch, kernel_name=KERNEL_NAME)
    module = init_c_var(cuda.CUmodule, lambda x: _check(cuda.cuModuleLoadData(ctypes.byref(x), kernel_image)))
    func = init_c_var(cuda.CUfunction, lambda x: _check(cuda.cuModuleGetFunction(ctypes.byref(x), module, KERNEL_NAME.encode())))

    q_nbytes = len(q) * ctypes.sizeof(ctypes.c_float)
    k_nbytes = len(k) * ctypes.sizeof(ctypes.c_float)
    v_nbytes = len(v) * ctypes.sizeof(ctypes.c_float)
    out_nbytes = len(host_out) * ctypes.sizeof(ctypes.c_float)
    lse_nbytes = len(host_lse) * ctypes.sizeof(ctypes.c_float)

    _check(cuda.cuMemAlloc_v2(ctypes.byref(q_ptr), q_nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(k_ptr), k_nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(v_ptr), v_nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(o_ptr), out_nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(lse_ptr), lse_nbytes))

    _check(cuda.cuMemcpyHtoDAsync_v2(q_ptr, _buffer_ptr(q), q_nbytes, None))
    _check(cuda.cuMemcpyHtoDAsync_v2(k_ptr, _buffer_ptr(k), k_nbytes, None))
    _check(cuda.cuMemcpyHtoDAsync_v2(v_ptr, _buffer_ptr(v), v_nbytes, None))

    params, _ = _make_kernel_params(
      q_ptr.value,
      k_ptr.value,
      v_ptr.value,
      o_ptr.value,
      lse_ptr.value,
      batch_size,
      num_heads,
      seq_len,
      head_dim,
      scale,
      causal,
    )
    grid = ((seq_len + BLOCK_M - 1) // BLOCK_M, num_heads, batch_size)
    block = (WARP_SIZE, BLOCK_M, 1)
    shared_nbytes = (2 * BLOCK_N * head_dim + BLOCK_M * BLOCK_N + 3 * BLOCK_M) * ctypes.sizeof(ctypes.c_float)
    _check(cuda.cuLaunchKernel(func, *grid, *block, shared_nbytes, None, params, None))
    _check(cuda.cuCtxSynchronize())

    _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(host_out), o_ptr, out_nbytes))
    _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(host_lse), lse_ptr, lse_nbytes))
  finally:
    for ptr in [lse_ptr, o_ptr, v_ptr, k_ptr, q_ptr]:
      if ptr.value not in (None, 0):
        cuda.cuMemFree_v2(ptr)
    if module is not None:
      cuda.cuModuleUnload(module)
    cuda.cuCtxDestroy_v2(ctx)

  out_diff = _max_abs_diff(host_out, expected_out)
  lse_diff = _max_abs_diff(host_lse, expected_lse)
  if out_diff > 2e-4:
    raise AssertionError(f"flash_attn_v2 output mismatch: max_abs_diff={out_diff}")
  if lse_diff > 2e-4:
    raise AssertionError(f"flash_attn_v2 lse mismatch: max_abs_diff={lse_diff}")
  return out_diff, lse_diff


def main():
  parser = argparse.ArgumentParser(description="Run a flash attention v2 forward kernel through the standalone TinyGPU CUDA compatibility layer")
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--num-heads", type=int, default=2)
  parser.add_argument("--seq-len", type=int, default=33)
  parser.add_argument("--head-dim", type=int, default=64)
  parser.add_argument("--non-causal", action="store_true")
  parser.add_argument("--verify-suite", action="store_true", help="run a small set of non-trivial shapes instead of the single CLI shape")
  args = parser.parse_args()

  cases = (
    [
      (1, 2, 33, 64, True),
      (2, 1, 47, 96, True),
      (1, 1, 29, 128, False),
    ]
    if args.verify_suite
    else [(args.batch_size, args.num_heads, args.seq_len, args.head_dim, not args.non_causal)]
  )

  for batch_size, num_heads, seq_len, head_dim, causal in cases:
    out_diff, lse_diff = run_flash_attn_v2(
      batch_size=batch_size,
      num_heads=num_heads,
      seq_len=seq_len,
      head_dim=head_dim,
      causal=causal,
    )
    mode = "causal" if causal else "non-causal"
    print(
      f"flash attn v2 ok, batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, "
      f"head_dim={head_dim}, mode={mode}, max_out_diff={out_diff:.3e}, max_lse_diff={lse_diff:.3e}"
    )


if __name__ == "__main__":
  main()
