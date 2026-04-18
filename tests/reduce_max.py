from __future__ import annotations

import argparse
import array
import ctypes
import struct

import tbgpu.cuda_compat as cuda
from tbgpu.compiler import compile_cuda_to_ptx

KERNEL_NAME = "reduce_max"
ITEMS_PER_THREAD = 4
DEFAULT_MAX_BLOCKS = 1024

REDUCE_MAX_CUDA = f"""
#include <float.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce_max(float val) {{
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {{
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }}
  return val;
}}

extern "C" __global__ void reduce_max(const float *__restrict__ inp, float *__restrict__ out, unsigned int n) {{
  extern __shared__ float shared[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * {ITEMS_PER_THREAD} + tid;
  unsigned int grid_stride = gridDim.x * blockDim.x * {ITEMS_PER_THREAD};
  float local_max = -FLT_MAX;

  while (i < n) {{
    local_max = fmaxf(local_max, inp[i]);
    if (i + blockDim.x < n) local_max = fmaxf(local_max, inp[i + blockDim.x]);
    if (i + 2 * blockDim.x < n) local_max = fmaxf(local_max, inp[i + 2 * blockDim.x]);
    if (i + 3 * blockDim.x < n) local_max = fmaxf(local_max, inp[i + 3 * blockDim.x]);
    i += grid_stride;
  }}

  shared[tid] = local_max;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 32; s >>= 1) {{
    if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    __syncthreads();
  }}

  float block_max = shared[tid];
  if (tid < 32) {{
    if (blockDim.x >= 64) block_max = fmaxf(block_max, shared[tid + 32]);
    block_max = warp_reduce_max(block_max);
    if (tid == 0) out[blockIdx.x] = block_max;
  }}
}}
"""


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


def _launch_reduce_max(func, inp, out, size: int, blocks: int, block_size: int, shared_nbytes: int):
  arg_blob = struct.pack("<QQI", inp.value, out.value, size)
  arg_buf = ctypes.create_string_buffer(arg_blob)
  arg_size = ctypes.c_size_t(len(arg_blob))
  extra = (ctypes.c_void_p * 5)(
    ctypes.c_void_p(1),
    ctypes.cast(arg_buf, ctypes.c_void_p),
    ctypes.c_void_p(2),
    ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
    ctypes.c_void_p(0),
  )
  _check(cuda.cuLaunchKernel(func, blocks, 1, 1, block_size, 1, 1, shared_nbytes, None, None, extra))


def _ceil_div(x: int, y: int) -> int:
  return (x + y - 1) // y


def _validate_block_size(block_size: int):
  if block_size < 32 or block_size > 1024 or block_size & (block_size - 1):
    raise ValueError("block_size must be a power of two between 32 and 1024")


def _grid_blocks(size: int, block_size: int, max_blocks: int) -> int:
  return max(1, min(max_blocks, _ceil_div(size, block_size * ITEMS_PER_THREAD)))


def _build_input(size: int) -> tuple[array.array, float]:
  data = array.array("f", (float(((i * 37 + 17) % 2048) - 1024) for i in range(size)))
  peak_index = (size * 11) // 17
  peak_value = 12345.25
  data[peak_index] = peak_value
  if size > 3:
    data[(size * 5) // 13] = peak_value - 1.0
    data[(size * 7) // 19] = peak_value - 2.0
  return data, peak_value


def run_reduce_max(
  size: int = 1 << 20,
  block_size: int = 256,
  max_blocks: int = DEFAULT_MAX_BLOCKS,
):
  if size <= 0:
    raise ValueError("size must be positive")
  if max_blocks <= 0:
    raise ValueError("max_blocks must be positive")
  _validate_block_size(block_size)

  _check(cuda.cuInit(0))
  dev = init_c_var(cuda.CUdevice, lambda x: _check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
  ctx = init_c_var(cuda.CUcontext, lambda x: _check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, dev.value)))
  _check(cuda.cuCtxSetCurrent(ctx))

  module = None
  d_in, d_tmp0, d_tmp1 = cuda.CUdeviceptr(), cuda.CUdeviceptr(), cuda.CUdeviceptr()
  try:
    _check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), dev.value))
    arch = f"sm_{major.value}{minor.value}"
    kernel_image = compile_cuda_to_ptx(REDUCE_MAX_CUDA.strip() + "\n", arch, kernel_name=KERNEL_NAME)
    module = init_c_var(cuda.CUmodule, lambda x: _check(cuda.cuModuleLoadData(ctypes.byref(x), kernel_image)))
    func = init_c_var(cuda.CUfunction, lambda x: _check(cuda.cuModuleGetFunction(ctypes.byref(x), module, KERNEL_NAME.encode())))

    host_in, expected = _build_input(size)
    nbytes = size * ctypes.sizeof(ctypes.c_float)
    max_partials = _grid_blocks(size, block_size, max_blocks)
    temp_nbytes = max_partials * ctypes.sizeof(ctypes.c_float)
    shared_nbytes = block_size * ctypes.sizeof(ctypes.c_float)

    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_in), nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_tmp0), temp_nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_tmp1), temp_nbytes))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_in, _buffer_ptr(host_in), nbytes, None))

    current_ptr = d_in
    current_size = size
    temp_ptrs = [d_tmp0, d_tmp1]
    stages = 0

    while current_size > 1:
      out_ptr = temp_ptrs[stages & 1]
      blocks = _grid_blocks(current_size, block_size, max_blocks)
      _launch_reduce_max(func, current_ptr, out_ptr, current_size, blocks, block_size, shared_nbytes)
      current_ptr = out_ptr
      current_size = blocks
      stages += 1

    _check(cuda.cuCtxSynchronize())
    host_out = array.array("f", [0.0])
    _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(host_out), current_ptr, ctypes.sizeof(ctypes.c_float)))
  finally:
    for ptr in [d_tmp1, d_tmp0, d_in]:
      if ptr.value not in (None, 0):
        cuda.cuMemFree_v2(ptr)
    if module is not None:
      cuda.cuModuleUnload(module)
    cuda.cuCtxDestroy_v2(ctx)

  result = host_out[0]
  if abs(result - expected) > 1e-5:
    raise AssertionError(f"reduce max mismatch: {result} != {expected}")
  return result, stages


def main():
  parser = argparse.ArgumentParser(description="Run a CUDA reduce_max example through the standalone TinyGPU CUDA compatibility layer")
  parser.add_argument("--size", type=int, default=1 << 20)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
  args = parser.parse_args()

  result, stages = run_reduce_max(
    size=args.size,
    block_size=args.block_size,
    max_blocks=args.max_blocks,
  )
  print(f"reduce max ok, size={args.size}, block_size={args.block_size}, max_blocks={args.max_blocks}, stages={stages}, result={result}")


if __name__ == "__main__":
  main()
