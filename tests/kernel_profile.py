from __future__ import annotations

import argparse
import array
import ctypes

import tbgpu.cuda_compat as cuda
from tests.vector_add import KERNEL_NAME, VecAddArgs, _buffer_ptr, _check, _make_extra, init_c_var, load_kernel_image


def run_profile(
  size: int = 4096,
  block_size: int = 256,
  iters: int = 100,
  launch_mode: str = "extra",
  kernel_input: str = "ptx",
):
  if hasattr(cuda, "pti_enable"):
    cuda.pti_enable(True)
  if hasattr(cuda, "pti_reset"):
    cuda.pti_reset()

  _check(cuda.cuInit(0))
  dev = init_c_var(cuda.CUdevice, lambda x: _check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
  ctx = init_c_var(cuda.CUcontext, lambda x: _check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, dev.value)))
  _check(cuda.cuCtxSetCurrent(ctx))
  _check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), dev.value))
  arch = f"sm_{major.value}{minor.value}"
  kernel_image = load_kernel_image(arch, kernel_input)
  module = init_c_var(cuda.CUmodule, lambda x: _check(cuda.cuModuleLoadData(ctypes.byref(x), kernel_image)))
  func = init_c_var(cuda.CUfunction, lambda x: _check(cuda.cuModuleGetFunction(ctypes.byref(x), module, KERNEL_NAME.encode())))

  nbytes = size * ctypes.sizeof(ctypes.c_float)
  a = array.array("f", (float(i) for i in range(size)))
  b = array.array("f", (float(2 * i + 1) for i in range(size)))
  out = array.array("f", [0.0] * size)
  d_a, d_b, d_out = cuda.CUdeviceptr(), cuda.CUdeviceptr(), cuda.CUdeviceptr()

  try:
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_a), nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_b), nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), nbytes))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_a, _buffer_ptr(a), nbytes, None))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_b, _buffer_ptr(b), nbytes, None))

    grid = ((size + block_size - 1) // block_size, 1, 1)
    block = (block_size, 1, 1)
    args = VecAddArgs(d_a.value, d_b.value, d_out.value, size)

    start = init_c_var(cuda.CUevent, lambda x: _check(cuda.cuEventCreate(ctypes.byref(x), 0)))
    end = init_c_var(cuda.CUevent, lambda x: _check(cuda.cuEventCreate(ctypes.byref(x), 0)))
    _check(cuda.cuEventRecord(start, None))
    for _ in range(iters):
      if launch_mode == "kernel_params":
        scalars = [ctypes.c_uint64(args.a), ctypes.c_uint64(args.b), ctypes.c_uint64(args.c), ctypes.c_uint32(args.n)]
        params = (ctypes.c_void_p * len(scalars))(*[ctypes.addressof(v) for v in scalars])
        _check(cuda.cuLaunchKernel(func, *grid, *block, 0, None, params, None))
      else:
        extra, _ = _make_extra(args)
        _check(cuda.cuLaunchKernel(func, *grid, *block, 0, None, None, extra))
    _check(cuda.cuEventRecord(end, None))
    _check(cuda.cuEventSynchronize(end))
    elapsed_ms = ctypes.c_float(0.0)
    _check(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), start, end))
    _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(out), d_out, nbytes))

    expected = array.array("f", (x + y for x, y in zip(a, b)))
    for got, exp in zip(out, expected):
      if abs(got - exp) > 1e-5:
        raise AssertionError(f"vector add mismatch: {got} != {exp}")

    pti_records = cuda.pti_collect(clear=True) if hasattr(cuda, "pti_collect") else []
    total_us = sum(float(rec["duration_us"]) for rec in pti_records)
    avg_us = (total_us / len(pti_records)) if pti_records else 0.0
    avg_ms = elapsed_ms.value / iters

    print(f"event elapsed: total={elapsed_ms.value:.3f} ms, avg={avg_ms:.6f} ms/launch, launches={iters}")
    print(f"pti records: count={len(pti_records)}, total={total_us:.3f} us, avg={avg_us:.3f} us/launch")
    if pti_records:
      first = pti_records[0]
      print(f"pti first: kernel={first['kernel']}, grid={first['grid']}, block={first['block']}, duration_us={first['duration_us']:.3f}")
  finally:
    for ptr in [d_out, d_b, d_a]:
      if ptr.value not in (None, 0):
        cuda.cuMemFree_v2(ptr)
    cuda.cuModuleUnload(module)
    cuda.cuCtxDestroy_v2(ctx)


def main():
  parser = argparse.ArgumentParser(description="Profile CUDA kernels on tbgpu using CUDA events + PTI-like records")
  parser.add_argument("--size", type=int, default=4096)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--iters", type=int, default=100)
  parser.add_argument("--launch-mode", choices=["extra", "kernel_params"], default="extra")
  parser.add_argument("--kernel-input", choices=["cuda", "ptx"], default="ptx")
  args = parser.parse_args()
  run_profile(size=args.size, block_size=args.block_size, iters=args.iters, launch_mode=args.launch_mode, kernel_input=args.kernel_input)


if __name__ == "__main__":
  main()
