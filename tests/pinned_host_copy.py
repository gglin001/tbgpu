from __future__ import annotations

import argparse
import array
import ctypes

import tbgpu.cuda_compat as cuda
from tests.vector_add import KERNEL_NAME, VecAddArgs, _buffer_ptr, _check, _make_extra, init_c_var, load_kernel_image


def _host_view(ptr: int, size: int):
  return (ctypes.c_float * size).from_address(ptr)


def _copy_into_host(ptr: int, values: array.array) -> None:
  ctypes.memmove(ptr, _buffer_ptr(values), len(values) * ctypes.sizeof(ctypes.c_float))


def _copy_from_host(ptr: int, size: int) -> array.array:
  return array.array("f", _host_view(ptr, size))


def _launch_vector_add(func, size: int, block_size: int, a_ptr: int, b_ptr: int, out_ptr: int, stream) -> None:
  if size == 0:
    return
  args = VecAddArgs(a_ptr, b_ptr, out_ptr, size)
  extra, _ = _make_extra(args)
  grid = ((size + block_size - 1) // block_size, 1, 1)
  block = (block_size, 1, 1)
  _check(cuda.cuLaunchKernel(func, *grid, *block, 0, stream, None, extra))


def _assert_matches(label: str, got: array.array, expected: array.array) -> None:
  for got_item, exp_item in zip(got, expected):
    if abs(got_item - exp_item) > 1e-5:
      raise AssertionError(f"{label} mismatch: {got_item} != {exp_item}")


def run_pinned_host_copy(size: int = 4096, block_size: int = 256, kernel_input: str = "ptx") -> None:
  _check(cuda.cuInit(0))
  dev = init_c_var(cuda.CUdevice, lambda x: _check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
  ctx = init_c_var(cuda.CUcontext, lambda x: _check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, dev.value)))
  _check(cuda.cuCtxSetCurrent(ctx))
  _check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), dev.value))
  arch = f"sm_{major.value}{minor.value}"
  kernel_image = load_kernel_image(arch, kernel_input)
  module = init_c_var(cuda.CUmodule, lambda x: _check(cuda.cuModuleLoadData(ctypes.byref(x), kernel_image)))
  func = init_c_var(cuda.CUfunction, lambda x: _check(cuda.cuModuleGetFunction(ctypes.byref(x), module, KERNEL_NAME.encode())))
  stream = init_c_var(cuda.CUstream, lambda x: _check(cuda.cuStreamCreate(ctypes.byref(x), cuda.CU_STREAM_NON_BLOCKING)))

  a = array.array("f", (float(i) for i in range(size)))
  b = array.array("f", (float(2 * i + 1) for i in range(size)))
  expected = array.array("f", (x + y for x, y in zip(a, b)))
  expected_zero_copy = array.array("f", (x + 2.0 * y for x, y in zip(a, b)))
  nbytes = size * ctypes.sizeof(ctypes.c_float)
  host_flags = cuda.CU_MEMHOSTALLOC_PORTABLE | cuda.CU_MEMHOSTALLOC_DEVICEMAP

  h_a = ctypes.c_void_p()
  h_b = ctypes.c_void_p()
  h_out = ctypes.c_void_p()
  h_zero_copy = ctypes.c_void_p()
  d_a = cuda.CUdeviceptr()
  d_b = cuda.CUdeviceptr()
  d_out = cuda.CUdeviceptr()
  zc_a = cuda.CUdeviceptr()
  zc_b = cuda.CUdeviceptr()
  zc_out = cuda.CUdeviceptr()

  try:
    for host_ptr in [h_a, h_b, h_out, h_zero_copy]:
      _check(cuda.cuMemHostAlloc(ctypes.byref(host_ptr), nbytes, host_flags))

    _copy_into_host(h_a.value, a)
    _copy_into_host(h_b.value, b)

    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_a), nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_b), nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), nbytes))

    _check(cuda.cuMemcpyHtoDAsync_v2(d_a, h_a.value, nbytes, stream))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_b, h_b.value, nbytes, stream))
    _launch_vector_add(func, size, block_size, d_a.value, d_b.value, d_out.value, stream)
    _check(cuda.cuMemcpyDtoHAsync_v2(h_out.value, d_out, nbytes, stream))
    _check(cuda.cuStreamSynchronize(stream))
    _assert_matches("pinned memcpy roundtrip", _copy_from_host(h_out.value, size), expected)

    for dev_ptr, host_ptr in [(zc_a, h_a), (zc_b, h_b), (zc_out, h_zero_copy)]:
      _check(cuda.cuMemHostGetDevicePointer(ctypes.byref(dev_ptr), host_ptr, 0))

    _launch_vector_add(func, size, block_size, zc_a.value, zc_b.value, zc_out.value, stream)
    _launch_vector_add(func, size, block_size, zc_out.value, zc_b.value, zc_out.value, stream)
    _check(cuda.cuStreamSynchronize(stream))
    _assert_matches("mapped host zero-copy launch", _copy_from_host(h_zero_copy.value, size), expected_zero_copy)
  finally:
    for ptr in [d_out, d_b, d_a]:
      if ptr.value not in (None, 0):
        cuda.cuMemFree_v2(ptr)
    for host_ptr in [h_zero_copy, h_out, h_b, h_a]:
      if host_ptr.value not in (None, 0):
        cuda.cuMemFreeHost(host_ptr)
    if stream.value not in (None, 0):
      cuda.cuStreamDestroy_v2(stream)
    cuda.cuModuleUnload(module)
    cuda.cuCtxDestroy_v2(ctx)


def main():
  parser = argparse.ArgumentParser(description="Validate pinned host copies and mapped host device pointers on the TinyGPU CUDA shim")
  parser.add_argument("--size", type=int, default=4096)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--kernel-input", choices=["cuda", "ptx"], default="ptx")
  args = parser.parse_args()
  run_pinned_host_copy(size=args.size, block_size=args.block_size, kernel_input=args.kernel_input)
  print(f"pinned host copy ok, size={args.size}, block_size={args.block_size}, kernel_input={args.kernel_input}")


if __name__ == "__main__":
  main()
