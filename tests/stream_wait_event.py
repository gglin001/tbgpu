from __future__ import annotations

import argparse
import array
import ctypes

import tbgpu.cuda_compat as cuda
from tests.vector_add import KERNEL_NAME, VecAddArgs, _buffer_ptr, _check, _make_extra, init_c_var, load_kernel_image


def _launch_vector_add(func, size: int, block_size: int, a_ptr: int, b_ptr: int, out_ptr: int, stream) -> None:
  if size == 0:
    return
  args = VecAddArgs(a_ptr, b_ptr, out_ptr, size)
  extra, _ = _make_extra(args)
  grid = ((size + block_size - 1) // block_size, 1, 1)
  block = (block_size, 1, 1)
  _check(cuda.cuLaunchKernel(func, *grid, *block, 0, stream, None, extra))


def _assert_matches(label: str, got: array.array, expected: array.array):
  for got_item, exp_item in zip(got, expected):
    if abs(got_item - exp_item) > 1e-5:
      raise AssertionError(f"{label} mismatch: {got_item} != {exp_item}")


def _run_independent_streams(func, size: int, block_size: int, d_a, d_b, d_out0, d_out1, stream0, stream1, expected: array.array):
  out0 = array.array("f", [0.0] * size)
  out1 = array.array("f", [0.0] * size)

  _launch_vector_add(func, size, block_size, d_a.value, d_b.value, d_out0.value, stream0)
  _launch_vector_add(func, size, block_size, d_a.value, d_b.value, d_out1.value, stream1)
  _check(cuda.cuStreamSynchronize(stream0))
  _check(cuda.cuStreamSynchronize(stream1))
  _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(out0), d_out0, len(out0) * ctypes.sizeof(ctypes.c_float)))
  _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(out1), d_out1, len(out1) * ctypes.sizeof(ctypes.c_float)))

  _assert_matches("independent stream result", out0, expected)
  _assert_matches("independent stream result", out1, expected)


def _run_wait_event_dependency(func, size: int, block_size: int, d_a, d_b, d_tmp, d_out1, stream0, stream1, event, expected: array.array):
  out1 = array.array("f", [0.0] * size)

  _launch_vector_add(func, size, block_size, d_a.value, d_b.value, d_tmp.value, stream0)
  _check(cuda.cuEventRecord(event, stream0))
  _check(cuda.cuStreamWaitEvent(stream1, event, 0))
  _launch_vector_add(func, size, block_size, d_tmp.value, d_b.value, d_out1.value, stream1)
  _check(cuda.cuStreamSynchronize(stream1))
  _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(out1), d_out1, len(out1) * ctypes.sizeof(ctypes.c_float)))

  _assert_matches("stream wait event result", out1, expected)


def _run_optional_multi_lane_smoke(func, size: int, block_size: int, d_a, d_b, d_out0, d_out1):
  stream0 = init_c_var(cuda.CUstream, lambda x: _check(cuda.cuStreamCreate(ctypes.byref(x), 0)))
  stream1 = init_c_var(cuda.CUstream, lambda x: _check(cuda.cuStreamCreate(ctypes.byref(x), 0)))
  try:
    lane_ids = {cuda._STREAMS[stream0.value].lane_index, cuda._STREAMS[stream1.value].lane_index}
    if len(lane_ids) <= 1:
      raise AssertionError(f"expected streams to span multiple compute lanes, got {sorted(lane_ids)}")
    _launch_vector_add(func, size, block_size, d_a.value, d_b.value, d_out0.value, stream0)
    _launch_vector_add(func, size, block_size, d_a.value, d_b.value, d_out1.value, stream1)
    _check(cuda.cuStreamSynchronize(stream0))
    _check(cuda.cuStreamSynchronize(stream1))
  finally:
    cuda.cuStreamDestroy_v2(stream1)
    cuda.cuStreamDestroy_v2(stream0)


def run_stream_wait_event(size: int = 4096, block_size: int = 256, kernel_input: str = "ptx"):
  _check(cuda.cuInit(0))
  dev = init_c_var(cuda.CUdevice, lambda x: _check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
  ctx = init_c_var(cuda.CUcontext, lambda x: _check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, dev.value)))
  _check(cuda.cuCtxSetCurrent(ctx))
  _check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), dev.value))
  arch = f"sm_{major.value}{minor.value}"
  kernel_image = load_kernel_image(arch, kernel_input)
  module = init_c_var(cuda.CUmodule, lambda x: _check(cuda.cuModuleLoadData(ctypes.byref(x), kernel_image)))
  func = init_c_var(cuda.CUfunction, lambda x: _check(cuda.cuModuleGetFunction(ctypes.byref(x), module, KERNEL_NAME.encode())))
  upload_stream = init_c_var(cuda.CUstream, lambda x: _check(cuda.cuStreamCreate(ctypes.byref(x), 0)))
  stream0 = init_c_var(cuda.CUstream, lambda x: _check(cuda.cuStreamCreate(ctypes.byref(x), 0)))
  stream1 = init_c_var(cuda.CUstream, lambda x: _check(cuda.cuStreamCreate(ctypes.byref(x), 0)))
  event = init_c_var(cuda.CUevent, lambda x: _check(cuda.cuEventCreate(ctypes.byref(x), 0)))

  a = array.array("f", (float(i) for i in range(size)))
  b = array.array("f", (float(2 * i + 1) for i in range(size)))
  nbytes = size * ctypes.sizeof(ctypes.c_float)

  d_a, d_b, d_tmp, d_out0, d_out1 = (cuda.CUdeviceptr() for _ in range(5))
  try:
    for ptr in [d_a, d_b, d_tmp, d_out0, d_out1]:
      _check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes))

    _check(cuda.cuMemcpyHtoDAsync_v2(d_a, _buffer_ptr(a), nbytes, upload_stream))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_b, _buffer_ptr(b), nbytes, upload_stream))
    _check(cuda.cuStreamSynchronize(upload_stream))

    expected = array.array("f", (x + y for x, y in zip(a, b)))
    _run_independent_streams(func, size, block_size, d_a, d_b, d_out0, d_out1, stream0, stream1, expected)

    expected_wait = array.array("f", (x + 2.0 * y for x, y in zip(a, b)))
    _run_wait_event_dependency(func, size, block_size, d_a, d_b, d_tmp, d_out1, stream0, stream1, event, expected_wait)
    _run_optional_multi_lane_smoke(func, size, block_size, d_a, d_b, d_out0, d_out1)
  finally:
    cuda.cuEventDestroy_v2(event)
    cuda.cuStreamDestroy_v2(upload_stream)
    cuda.cuStreamDestroy_v2(stream1)
    cuda.cuStreamDestroy_v2(stream0)
    cuda.cuModuleUnload(module)
    for ptr in [d_out1, d_out0, d_tmp, d_b, d_a]:
      if ptr.value not in (None, 0):
        cuda.cuMemFree_v2(ptr)
    cuda.cuCtxDestroy_v2(ctx)


def main():
  parser = argparse.ArgumentParser(description="Exercise multi-stream execution and stream wait events on the TinyGPU CUDA shim")
  parser.add_argument("--size", type=int, default=4096)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--kernel-input", choices=["cuda", "ptx"], default="ptx")
  args = parser.parse_args()
  run_stream_wait_event(size=args.size, block_size=args.block_size, kernel_input=args.kernel_input)
  print(f"stream wait event ok, size={args.size}, block_size={args.block_size}, kernel_input={args.kernel_input}")


if __name__ == "__main__":
  main()
