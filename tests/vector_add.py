from __future__ import annotations

import argparse
import array
import ctypes
import struct

import tbgpu.cuda_compat as cuda
from tbgpu.compiler import compile_cuda_to_ptx, compile_ptx_to_cubin

KERNEL_NAME = "vector_add"

VECTOR_ADD_CUDA = r"""
extern "C" __global__ void vector_add(const float *a, const float *b, float *c, unsigned int n) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}
"""

VECTOR_ADD_PTX = r""".version VERSION
.target TARGET
.address_size 64

.visible .entry vector_add(
  .param .u64 a,
  .param .u64 b,
  .param .u64 c,
  .param .u32 n
)
{
  .reg .pred %p<2>;
  .reg .f32 %f<4>;
  .reg .b32 %r<6>;
  .reg .b64 %rd<8>;

  ld.param.u64 %rd1, [a];
  ld.param.u64 %rd2, [b];
  ld.param.u64 %rd3, [c];
  ld.param.u32 %r1, [n];

  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r5, %r2, %r3, %r4;
  setp.ge.u32 %p1, %r5, %r1;
  @%p1 bra DONE;

  mul.wide.u32 %rd4, %r5, 4;
  add.s64 %rd5, %rd1, %rd4;
  add.s64 %rd6, %rd2, %rd4;
  add.s64 %rd7, %rd3, %rd4;
  ld.global.f32 %f1, [%rd5];
  ld.global.f32 %f2, [%rd6];
  add.f32 %f3, %f1, %f2;
  st.global.f32 [%rd7], %f3;

DONE:
  ret;
}
"""


class VecAddArgs(ctypes.Structure):
  _fields_ = [("a", ctypes.c_uint64), ("b", ctypes.c_uint64), ("c", ctypes.c_uint64), ("n", ctypes.c_uint32)]


def init_c_var(ty, create_cb):
  value = ty()
  create_cb(value)
  return value


def _ptx_version_for_arch(arch: str) -> str:
  ver = int(arch.removeprefix("sm_"))
  if ver >= 120:
    return "8.7"
  if ver >= 89:
    return "7.8"
  return "7.5"


def render_vector_add_ptx(arch: str) -> bytes:
  return VECTOR_ADD_PTX.replace("TARGET", arch).replace("VERSION", _ptx_version_for_arch(arch)).encode()


def _write_if_requested(path: str | None, data: bytes):
  if path is not None:
    with open(path, "wb") as fh:
      fh.write(data)


def load_kernel_image(arch: str, kernel_input: str, emit_ptx: str | None = None, emit_cubin: str | None = None) -> bytes:
  if kernel_input == "cuda":
    ptx = compile_cuda_to_ptx(VECTOR_ADD_CUDA.strip() + "\n", arch, kernel_name=KERNEL_NAME)
    _write_if_requested(emit_ptx, ptx)
    cubin = compile_ptx_to_cubin(ptx, arch)
    _write_if_requested(emit_cubin, cubin)
    return ptx
  if kernel_input == "ptx":
    ptx = render_vector_add_ptx(arch)
    _write_if_requested(emit_ptx, ptx)
    cubin = compile_ptx_to_cubin(ptx, arch)
    _write_if_requested(emit_cubin, cubin)
    return ptx
  raise ValueError(f"unsupported kernel_input {kernel_input}")


def _check(status: int):
  if status != 0:
    err = ctypes.POINTER(ctypes.c_char)()
    cuda.cuGetErrorString(status, ctypes.byref(err))
    raise RuntimeError(f"CUDA shim error {status}: {ctypes.string_at(err).decode()}")


def _buffer_ptr(buf: array.array) -> int:
  return ctypes.addressof((ctypes.c_float * len(buf)).from_buffer(buf))


def _encode_args_blob(args: VecAddArgs) -> bytes:
  return struct.pack("<QQQI", args.a, args.b, args.c, args.n)


def _make_extra(args: VecAddArgs):
  arg_blob = _encode_args_blob(args)
  arg_buf = ctypes.create_string_buffer(arg_blob)
  arg_size = ctypes.c_size_t(len(arg_blob))
  extra = (ctypes.c_void_p * 5)(
    ctypes.c_void_p(1),
    ctypes.cast(arg_buf, ctypes.c_void_p),
    ctypes.c_void_p(2),
    ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
    ctypes.c_void_p(0),
  )
  return extra, (arg_buf, arg_size)


def _make_kernel_params(args: VecAddArgs):
  scalars = [ctypes.c_uint64(args.a), ctypes.c_uint64(args.b), ctypes.c_uint64(args.c), ctypes.c_uint32(args.n)]
  params = (ctypes.c_void_p * len(scalars))(*[ctypes.addressof(v) for v in scalars])
  return params, scalars


def run_vector_add(
  size: int = 256,
  block_size: int = 64,
  launch_mode: str = "extra",
  kernel_input: str = "ptx",
  emit_ptx: str | None = None,
  emit_cubin: str | None = None,
):
  _check(cuda.cuInit(0))
  dev = init_c_var(cuda.CUdevice, lambda x: _check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
  ctx = init_c_var(cuda.CUcontext, lambda x: _check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, dev.value)))
  _check(cuda.cuCtxSetCurrent(ctx))
  _check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), dev.value))
  arch = f"sm_{major.value}{minor.value}"
  kernel_image = load_kernel_image(arch, kernel_input, emit_ptx=emit_ptx, emit_cubin=emit_cubin)
  module = init_c_var(cuda.CUmodule, lambda x: _check(cuda.cuModuleLoadData(ctypes.byref(x), kernel_image)))
  func = init_c_var(cuda.CUfunction, lambda x: _check(cuda.cuModuleGetFunction(ctypes.byref(x), module, KERNEL_NAME.encode())))
  a = array.array("f", (float(i) for i in range(size)))
  b = array.array("f", (float(2 * i + 1) for i in range(size)))
  out = array.array("f", [0.0] * size)
  nbytes = size * ctypes.sizeof(ctypes.c_float)
  d_a, d_b, d_out = cuda.CUdeviceptr(), cuda.CUdeviceptr(), cuda.CUdeviceptr()
  try:
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_a), nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_b), nbytes))
    _check(cuda.cuMemAlloc_v2(ctypes.byref(d_out), nbytes))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_a, _buffer_ptr(a), nbytes, None))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_b, _buffer_ptr(b), nbytes, None))
    if size > 0:
      grid = ((size + block_size - 1) // block_size, 1, 1)
      block = (block_size, 1, 1)
      args = VecAddArgs(d_a.value, d_b.value, d_out.value, size)
      if launch_mode == "kernel_params":
        params, _ = _make_kernel_params(args)
        _check(cuda.cuLaunchKernel(func, *grid, *block, 0, None, params, None))
      else:
        extra, _ = _make_extra(args)
        _check(cuda.cuLaunchKernel(func, *grid, *block, 0, None, None, extra))
    _check(cuda.cuCtxSynchronize())
    _check(cuda.cuMemcpyDtoH_v2(_buffer_ptr(out), d_out, nbytes))
  finally:
    for ptr in [d_out, d_b, d_a]:
      if ptr.value not in (None, 0):
        cuda.cuMemFree_v2(ptr)
    cuda.cuModuleUnload(module)
    cuda.cuCtxDestroy_v2(ctx)
  expected = array.array("f", (x + y for x, y in zip(a, b)))
  for got, exp in zip(out, expected):
    if abs(got - exp) > 1e-5:
      raise AssertionError(f"vector add mismatch: {got} != {exp}")
  return out


def main():
  parser = argparse.ArgumentParser(description="Run vector_add through the standalone TinyGPU CUDA compatibility layer")
  parser.add_argument("--size", type=int, default=256)
  parser.add_argument("--block-size", type=int, default=64)
  parser.add_argument("--launch-mode", choices=["extra", "kernel_params"], default="extra")
  parser.add_argument("--kernel-input", choices=["cuda", "ptx"], default="ptx")
  parser.add_argument("--emit-ptx")
  parser.add_argument("--emit-cubin")
  args = parser.parse_args()
  run_vector_add(
    size=args.size,
    block_size=args.block_size,
    launch_mode=args.launch_mode,
    kernel_input=args.kernel_input,
    emit_ptx=args.emit_ptx,
    emit_cubin=args.emit_cubin,
  )
  print(f"vector add ok, size={args.size}, launch_mode={args.launch_mode}, kernel_input={args.kernel_input}")


if __name__ == "__main__":
  main()
