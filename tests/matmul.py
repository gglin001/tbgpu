from __future__ import annotations

import argparse
import array
import ctypes
from pathlib import Path

import tbgpu.cuda_compat as cuda
from tbgpu.compiler import compile_cuda_to_ptx

KERNEL_NAME = "matmul_tiled"
TILE = 16
MAX_ABS_DIFF = 2e-4


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


def _kernel_source() -> str:
  kernel_path = Path(__file__).with_name("kernels") / "matmul.cu"
  return kernel_path.read_text()


def _build_matrix(size: int, seed: int, stride: int, scale: float) -> array.array:
  return array.array("f", (float((((seed + idx * stride) % 97) - 48) * scale) for idx in range(size)))


def _cpu_matmul(
  a: array.array,
  b: array.array,
  c: array.array,
  m: int,
  n: int,
  k: int,
  alpha: float,
  beta: float,
) -> array.array:
  out = array.array("f", c)
  for row in range(m):
    row_off = row * k
    out_row_off = row * n
    for col in range(n):
      acc = 0.0
      for kk in range(k):
        acc += a[row_off + kk] * b[kk * n + col]
      out[out_row_off + col] = alpha * acc + beta * out[out_row_off + col]
  return out


def _max_abs_diff(got: array.array, expected: array.array) -> float:
  return max(abs(g - e) for g, e in zip(got, expected)) if got else 0.0


def _make_kernel_params(
  c_ptr: int,
  a_ptr: int,
  b_ptr: int,
  m: int,
  n: int,
  k: int,
  alpha: float,
  beta: float,
):
  scalars = [
    ctypes.c_uint64(c_ptr),
    ctypes.c_uint64(a_ptr),
    ctypes.c_uint64(b_ptr),
    ctypes.c_uint32(m),
    ctypes.c_uint32(n),
    ctypes.c_uint32(k),
    ctypes.c_float(alpha),
    ctypes.c_float(beta),
  ]
  params = (ctypes.c_void_p * len(scalars))(*[ctypes.addressof(value) for value in scalars])
  return params, scalars


def run_matmul(
  m: int,
  n: int,
  k: int,
  alpha: float = 1.0,
  beta: float = 0.0,
):
  if m <= 0 or n <= 0 or k <= 0:
    raise ValueError("m, n, and k must be positive")

  a = _build_matrix(m * k, seed=11, stride=7, scale=0.125)
  b = _build_matrix(k * n, seed=23, stride=5, scale=0.2)
  c = _build_matrix(m * n, seed=37, stride=3, scale=0.1)
  expected = _cpu_matmul(a, b, c, m, n, k, alpha, beta)
  host_out = array.array("f", c)

  _check(cuda.cuInit(0))
  dev = init_c_var(cuda.CUdevice, lambda x: _check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
  ctx = init_c_var(cuda.CUcontext, lambda x: _check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, dev.value)))
  _check(cuda.cuCtxSetCurrent(ctx))

  module = None
  a_ptr, b_ptr, c_ptr = [cuda.CUdeviceptr() for _ in range(3)]
  try:
    _check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), dev.value))
    arch = f"sm_{major.value}{minor.value}"
    kernel_image = compile_cuda_to_ptx(_kernel_source().strip() + "\n", arch, kernel_name=KERNEL_NAME)
    module = init_c_var(cuda.CUmodule, lambda x: _check(cuda.cuModuleLoadData(ctypes.byref(x), kernel_image)))
    func = init_c_var(cuda.CUfunction, lambda x: _check(cuda.cuModuleGetFunction(ctypes.byref(x), module, KERNEL_NAME.encode())))

    a_nbytes = len(a) * ctypes.sizeof(ctypes.c_float)
    b_nbytes = len(b) * ctypes.sizeof(ctypes.c_float)
    c_nbytes = len(host_out) * ctypes.sizeof(ctypes.c_float)

    _check(cuda.cuMemAlloc_v2(ctypes.byref(a_ptr), a_nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(b_ptr), b_nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(c_ptr), c_nbytes))

    _check(cuda.cuMemcpyHtoDAsync_v2(a_ptr, _buffer_ptr(a), a_nbytes, None))
    _check(cuda.cuMemcpyHtoDAsync_v2(b_ptr, _buffer_ptr(b), b_nbytes, None))
    _check(cuda.cuMemcpyHtoDAsync_v2(c_ptr, _buffer_ptr(host_out), c_nbytes, None))

    params, _ = _make_kernel_params(c_ptr.value, a_ptr.value, b_ptr.value, m, n, k, alpha, beta)
    grid = ((n + TILE - 1) // TILE, (m + TILE - 1) // TILE, 1)
    block = (TILE, TILE, 1)
    _check(cuda.cuLaunchKernel(func, *grid, *block, 0, None, params, None))
    _check(cuda.cuCtxSynchronize())
    _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(host_out), c_ptr, c_nbytes))
  finally:
    for ptr in [c_ptr, b_ptr, a_ptr]:
      if ptr.value not in (None, 0):
        cuda.cuMemFree_v2(ptr)
    if module is not None:
      cuda.cuModuleUnload(module)
    cuda.cuCtxDestroy_v2(ctx)

  max_diff = _max_abs_diff(host_out, expected)
  if max_diff > MAX_ABS_DIFF:
    raise AssertionError(f"matmul mismatch: max_abs_diff={max_diff}, tolerance={MAX_ABS_DIFF}")
  return max_diff


def main():
  parser = argparse.ArgumentParser(description="Run a tiled CUDA matmul kernel through the standalone TinyGPU CUDA compatibility layer")
  parser.add_argument("--m", type=int, default=192)
  parser.add_argument("--n", type=int, default=256)
  parser.add_argument("--k", type=int, default=128)
  parser.add_argument("--alpha", type=float, default=1.25)
  parser.add_argument("--beta", type=float, default=-0.5)
  parser.add_argument("--verify-suite", action="store_true", help="run a few non-square and non-tile-aligned shapes")
  args = parser.parse_args()

  cases = (
    [
      (192, 256, 128, 1.25, -0.5),
      (255, 257, 129, 0.75, 0.25),
      (128, 384, 96, -0.5, 1.0),
    ]
    if args.verify_suite
    else [(args.m, args.n, args.k, args.alpha, args.beta)]
  )

  for m, n, k, alpha, beta in cases:
    max_diff = run_matmul(m=m, n=n, k=k, alpha=alpha, beta=beta)
    print(f"matmul ok, m={m}, n={n}, k={k}, alpha={alpha}, beta={beta}, max_abs_diff={max_diff}")


if __name__ == "__main__":
  main()
