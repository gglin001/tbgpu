from __future__ import annotations

import array
import ctypes
import functools
import itertools
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from tbgpu.compiler import compile_ptx_to_cubin
from tbgpu.runtime.device import TBGPUDevice
from tbgpu.runtime.program import TBGPUProgram

DEBUG = int(os.getenv("DEBUG", "0"))

CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUevent = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64

CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_INVALID_DEVICE = 101
CUDA_ERROR_INVALID_IMAGE = 200
CUDA_ERROR_INVALID_CONTEXT = 201
CUDA_ERROR_INVALID_PTX = 218
CUDA_ERROR_INVALID_HANDLE = 400
CUDA_ERROR_NOT_SUPPORTED = 801
CUDA_ERROR_LAUNCH_FAILED = 719

_ERROR_NAMES = {
  CUDA_SUCCESS: "CUDA_SUCCESS",
  CUDA_ERROR_INVALID_VALUE: "CUDA_ERROR_INVALID_VALUE",
  CUDA_ERROR_INVALID_DEVICE: "CUDA_ERROR_INVALID_DEVICE",
  CUDA_ERROR_INVALID_IMAGE: "CUDA_ERROR_INVALID_IMAGE",
  CUDA_ERROR_INVALID_CONTEXT: "CUDA_ERROR_INVALID_CONTEXT",
  CUDA_ERROR_INVALID_PTX: "CUDA_ERROR_INVALID_PTX",
  CUDA_ERROR_INVALID_HANDLE: "CUDA_ERROR_INVALID_HANDLE",
  CUDA_ERROR_NOT_SUPPORTED: "CUDA_ERROR_NOT_SUPPORTED",
  CUDA_ERROR_LAUNCH_FAILED: "CUDA_ERROR_LAUNCH_FAILED",
}


def round_up(value: int, alignment: int) -> int:
  return ((value + alignment - 1) // alignment) * alignment


@dataclass
class _ContextState:
  device_ordinal: int
  nv_device: Any


@dataclass
class _KernelParam:
  size: int
  align: int


@dataclass
class _ModuleState:
  ctx_id: int
  raw_image: bytes
  program_image: bytes
  signatures: dict[str, list[_KernelParam]] = field(default_factory=dict)


@dataclass
class _FunctionState:
  module_id: int
  name: str
  attrs: dict[int, int] = field(default_factory=dict)
  program: Any | None = None


@dataclass
class _DeviceAllocation:
  base: int
  size: int
  buf: Any
  owner: Any


@dataclass
class _HostAllocation:
  base: int
  size: int
  buf: Any


@dataclass
class _EventState:
  timestamp_ns: int = 0


class _RawArgsState:
  def __init__(self, buf):
    self.buf = buf


_HANDLE_COUNTER = itertools.count(1)
_TLS = threading.local()
_NV_DEVICE_CACHE: dict[int, Any] = {}
_CONTEXTS: dict[int, _ContextState] = {}
_MODULES: dict[int, _ModuleState] = {}
_FUNCTIONS: dict[int, _FunctionState] = {}
_DEVICE_ALLOCS: dict[int, _DeviceAllocation] = {}
_HOST_ALLOCS: dict[int, _HostAllocation] = {}
_EVENTS: dict[int, _EventState] = {}
_ERROR_BUFS: dict[int, ctypes.Array] = {}

_ENTRY_RE = re.compile(r"\.visible\s+\.entry\s+(?P<name>[\w$@.]+)\s*\((?P<params>.*?)\)\s*\{", re.S)
_PARAM_RE = re.compile(r"\.param\s+(?:(?:\.align\s+(?P<align>\d+)\s+)?\.(?P<type>[a-z]\d+|pred)\s+(?P<name>[\w$@.]+)(?:\[(?P<count>\d+)\])?)")
_TYPE_SIZES = {
  "pred": 1,
  "b8": 1,
  "u8": 1,
  "s8": 1,
  "b16": 2,
  "u16": 2,
  "s16": 2,
  "f16": 2,
  "b32": 4,
  "u32": 4,
  "s32": 4,
  "f32": 4,
  "b64": 8,
  "u64": 8,
  "s64": 8,
  "f64": 8,
}


def _new_handle() -> int:
  return next(_HANDLE_COUNTER)


def _set_current_context(ctx_id: int | None):
  _TLS.current_context = ctx_id


def _get_current_context_id() -> int | None:
  return getattr(_TLS, "current_context", None)


def _as_int(value) -> int:
  if value is None:
    return 0
  if isinstance(value, int):
    return value
  if hasattr(value, "value") and value.value is not None:
    return int(value.value)
  return int(value)


def _get_ctx(ctx=None) -> _ContextState:
  ctx_id = _get_current_context_id() if ctx is None else _as_int(ctx)
  if ctx_id not in _CONTEXTS:
    raise RuntimeError("invalid context")
  return _CONTEXTS[ctx_id]


def _get_nv_device(ordinal: int):
  if ordinal not in _NV_DEVICE_CACHE:
    _NV_DEVICE_CACHE[ordinal] = TBGPUDevice(ordinal)
  return _NV_DEVICE_CACHE[ordinal]


def _cc_from_arch(arch: str) -> tuple[int, int]:
  num = arch.removeprefix("sm_")
  return int(num[:-1]), int(num[-1])


def _module_is_ptx(image: bytes) -> bool:
  prefix = image.lstrip()
  return prefix.startswith(b".version") or b".visible .entry" in prefix


def _parse_ptx_signatures(image: bytes) -> dict[str, list[_KernelParam]]:
  try:
    src = image.decode("utf-8")
  except UnicodeDecodeError:
    return {}
  signatures: dict[str, list[_KernelParam]] = {}
  for match in _ENTRY_RE.finditer(src):
    params: list[_KernelParam] = []
    for param in _PARAM_RE.finditer(match.group("params")):
      if (typ := param.group("type")) not in _TYPE_SIZES:
        continue
      size = int(param.group("count")) if param.group("count") is not None else _TYPE_SIZES[typ]
      align = int(param.group("align")) if param.group("align") is not None else _TYPE_SIZES[typ]
      params.append(_KernelParam(size=size, align=align))
    signatures[match.group("name")] = params
  return signatures


def _align_up(value: int, alignment: int) -> int:
  if alignment <= 1:
    return value
  return round_up(value, alignment)


def _read_void_p_array(ptr, count: int) -> list[int]:
  arr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))
  return [int(arr[i]) if arr[i] is not None else 0 for i in range(count)]


def _extract_extra_blob(extra) -> bytes | None:
  if not extra:
    return None
  vals = _read_void_p_array(extra, 5)
  if vals[0] != 1 or vals[2] != 2 or vals[4] != 0:
    return None
  size = ctypes.cast(vals[3], ctypes.POINTER(ctypes.c_size_t)).contents.value
  return ctypes.string_at(vals[1], size)


def _marshal_kernel_params(kernel_params, signature: list[_KernelParam]) -> bytes | None:
  if not kernel_params:
    return None
  vals = _read_void_p_array(kernel_params, len(signature))
  blob = bytearray()
  for val_ptr, param in zip(vals, signature):
    blob.extend(b"\x00" * (_align_up(len(blob), param.align) - len(blob)))
    blob.extend(ctypes.string_at(val_ptr, param.size))
  return bytes(blob)


def _build_arg_blob(function: _FunctionState, kernel_params, extra) -> bytes | None:
  if (blob := _extract_extra_blob(extra)) is not None:
    return blob
  signature = _MODULES[function.module_id].signatures.get(function.name)
  if signature is None:
    return None
  return _marshal_kernel_params(kernel_params, signature)


def _find_device_alloc(ptr: int):
  for alloc in _DEVICE_ALLOCS.values():
    if alloc.base <= ptr < alloc.base + alloc.size:
      return alloc, ptr - alloc.base
  return None


def _copy_to_device(dst_ptr: int, src_ptr: int, size: int):
  if (dst_info := _find_device_alloc(dst_ptr)) is None:
    raise RuntimeError(f"unknown device pointer 0x{dst_ptr:x}")
  alloc, offset = dst_info
  alloc.owner.allocator._copyin(alloc.buf.offset(offset=offset, size=size), memoryview(bytearray(ctypes.string_at(src_ptr, size))))


def _copy_from_device(dst_ptr: int, src_ptr: int, size: int):
  if (src_info := _find_device_alloc(src_ptr)) is None:
    raise RuntimeError(f"unknown device pointer 0x{src_ptr:x}")
  alloc, offset = src_info
  tmp = bytearray(size)
  alloc.owner.allocator._copyout(memoryview(tmp), alloc.buf.offset(offset=offset, size=size))
  ctypes.memmove(dst_ptr, bytes(tmp), size)


def _copy_device_to_device(dst_ptr: int, src_ptr: int, size: int):
  if (dst_info := _find_device_alloc(dst_ptr)) is None or (src_info := _find_device_alloc(src_ptr)) is None:
    raise RuntimeError("unknown device pointer")
  dst_alloc, dst_off = dst_info
  src_alloc, src_off = src_info
  src_alloc.owner.allocator._transfer(
    dst_alloc.buf.offset(offset=dst_off, size=size), src_alloc.buf.offset(offset=src_off, size=size), size, src_alloc.owner, dst_alloc.owner
  )


def _patch_shared_mem(prg, shared_mem_bytes: int) -> bytes | None:
  if shared_mem_bytes == 0:
    return None
  original = bytes(prg.qmd.mv)
  total = round_up(prg.shmem_usage + shared_mem_bytes, 128)
  smem_cfg = min(conf * 1024 for conf in [32, 64, 100] if conf * 1024 >= total) // 4096 + 1
  if prg.qmd.ver >= 5:
    prg.qmd.write(
      shared_memory_size_shifted7=total >> 7,
      min_sm_config_shared_mem_size=smem_cfg,
      target_sm_config_shared_mem_size=smem_cfg,
      max_sm_config_shared_mem_size=0x1A,
    )
  else:
    prg.qmd.write(
      shared_memory_size=total, min_sm_config_shared_mem_size=smem_cfg, target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1A
    )
  return original


def _restore_shared_mem(prg, original: bytes | None):
  if original is not None:
    prg.qmd.mv[:] = original


def _build_launch_cbuf0(prg, grid: tuple[int, int, int], block: tuple[int, int, int]) -> bytes:
  if not prg.cbuf_0:
    return b""
  cbuf_words = list(prg.cbuf_0)
  if len(cbuf_words) >= 6:
    cbuf_words[0:6] = [int(block[0]), int(block[1]), int(block[2]), int(grid[0]), int(grid[1]), int(grid[2])]
  return array.array("I", cbuf_words).tobytes()


@functools.lru_cache(maxsize=None)
def _compile_ptx_to_cubin(ptx: bytes, arch: str) -> bytes:
  return compile_ptx_to_cubin(ptx, arch)


def _ensure_program(function: _FunctionState):
  if function.program is None:
    module = _MODULES[function.module_id]
    function.program = TBGPUProgram(_CONTEXTS[module.ctx_id].nv_device, function.name, module.program_image)
  return function.program


def _launch(function: _FunctionState, arg_blob: bytes, grid: tuple[int, int, int], block: tuple[int, int, int], shared_mem_bytes: int):
  from tbgpu.runtime.device import NVComputeQueue

  prg = _ensure_program(function)
  dev = prg.dev
  argsbuf = dev.kernargs_buf.offset(offset=dev.kernargs_offset_allocator.alloc(prg.kernargs_alloc_size, 8), size=prg.kernargs_alloc_size)
  prefix = _build_launch_cbuf0(prg, grid, block)
  view = argsbuf.cpu_view().view(fmt="B")
  view[: len(prefix)] = prefix
  view[len(prefix) : len(prefix) + len(arg_blob)] = arg_blob
  q = NVComputeQueue().wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
  original_qmd = _patch_shared_mem(prg, shared_mem_bytes)
  try:
    q.exec(prg, argsbuf, grid, block)
  finally:
    _restore_shared_mem(prg, original_qmd)
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)


def cuInit(flags: int) -> int:
  return CUDA_SUCCESS


def cuDeviceGet(device, ordinal: int) -> int:
  try:
    _get_nv_device(ordinal)
  except Exception:
    return CUDA_ERROR_INVALID_DEVICE
  device._obj.value = ordinal
  return CUDA_SUCCESS


def cuCtxCreate_v2(pctx, flags: int, dev: int) -> int:
  try:
    nv_device = _get_nv_device(dev)
  except Exception:
    return CUDA_ERROR_INVALID_DEVICE
  ctx_id = _new_handle()
  _CONTEXTS[ctx_id] = _ContextState(device_ordinal=dev, nv_device=nv_device)
  _set_current_context(ctx_id)
  pctx._obj.value = ctx_id
  return CUDA_SUCCESS


def cuCtxDestroy_v2(ctx) -> int:
  ctx_id = _as_int(ctx)
  if ctx_id not in _CONTEXTS:
    return CUDA_ERROR_INVALID_CONTEXT
  if _get_current_context_id() == ctx_id:
    _set_current_context(None)
  del _CONTEXTS[ctx_id]
  return CUDA_SUCCESS


def cuCtxSetCurrent(context) -> int:
  ctx_id = _as_int(context)
  if ctx_id == 0:
    _set_current_context(None)
    return CUDA_SUCCESS
  if ctx_id not in _CONTEXTS:
    return CUDA_ERROR_INVALID_CONTEXT
  _set_current_context(ctx_id)
  return CUDA_SUCCESS


def cuCtxSynchronize() -> int:
  try:
    _get_ctx().nv_device.synchronize()
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  except Exception:
    return CUDA_ERROR_INVALID_CONTEXT
  return CUDA_SUCCESS


def cuDeviceComputeCapability(major, minor, dev: int) -> int:
  try:
    maj, minr = _cc_from_arch(_get_nv_device(dev).arch)
  except Exception:
    return CUDA_ERROR_INVALID_DEVICE
  major._obj.value, minor._obj.value = maj, minr
  return CUDA_SUCCESS


def cuDeviceCanAccessPeer(canAccessPeer, dev: int, peerDev: int) -> int:
  try:
    _get_nv_device(dev)
    _get_nv_device(peerDev)
  except Exception:
    return CUDA_ERROR_INVALID_DEVICE
  canAccessPeer._obj.value = 1
  return CUDA_SUCCESS


def cuCtxEnablePeerAccess(peerContext, flags: int) -> int:
  return CUDA_SUCCESS


def cuMemAlloc_v2(dptr, bytesize: int) -> int:
  try:
    ctx = _get_ctx()
    buf = ctx.nv_device.allocator.alloc(bytesize)
  except Exception:
    return CUDA_ERROR_INVALID_CONTEXT
  _DEVICE_ALLOCS[buf.va_addr] = _DeviceAllocation(base=buf.va_addr, size=bytesize, buf=buf, owner=ctx.nv_device)
  dptr._obj.value = buf.va_addr
  return CUDA_SUCCESS


def cuMemFree_v2(dptr) -> int:
  ptr = _as_int(dptr)
  if ptr not in _DEVICE_ALLOCS:
    return CUDA_ERROR_INVALID_VALUE
  alloc = _DEVICE_ALLOCS.pop(ptr)
  alloc.owner.allocator.free(alloc.buf, alloc.size)
  return CUDA_SUCCESS


def cuMemHostAlloc(pp, bytesize: int, flags: int) -> int:
  host_buf = ctypes.create_string_buffer(bytesize)
  ptr = ctypes.addressof(host_buf)
  _HOST_ALLOCS[ptr] = _HostAllocation(base=ptr, size=bytesize, buf=host_buf)
  pp._obj.value = ptr
  return CUDA_SUCCESS


def cuMemFreeHost(p) -> int:
  ptr = _as_int(p)
  if ptr not in _HOST_ALLOCS:
    return CUDA_ERROR_INVALID_VALUE
  del _HOST_ALLOCS[ptr]
  return CUDA_SUCCESS


def cuMemcpyHtoDAsync_v2(dst, src, bytesize: int, stream) -> int:
  try:
    _copy_to_device(_as_int(dst), _as_int(src), bytesize)
  except Exception:
    return CUDA_ERROR_INVALID_VALUE
  return CUDA_SUCCESS


def cuMemcpyDtoH_v2(dst, src, bytesize: int) -> int:
  try:
    _copy_from_device(_as_int(dst), _as_int(src), bytesize)
  except Exception:
    return CUDA_ERROR_INVALID_VALUE
  return CUDA_SUCCESS


def cuMemcpyDtoDAsync_v2(dst, src, bytesize: int, stream) -> int:
  try:
    _copy_device_to_device(_as_int(dst), _as_int(src), bytesize)
  except Exception:
    return CUDA_ERROR_INVALID_VALUE
  return CUDA_SUCCESS


def cuModuleLoadData(module, image) -> int:
  ctx_id = _get_current_context_id()
  if ctx_id not in _CONTEXTS:
    return CUDA_ERROR_INVALID_CONTEXT
  if not isinstance(image, (bytes, bytearray, memoryview)):
    return CUDA_ERROR_INVALID_IMAGE
  raw_image = bytes(image)
  try:
    program_image = raw_image if not _module_is_ptx(raw_image) else _compile_ptx_to_cubin(raw_image, _CONTEXTS[ctx_id].nv_device.arch)
  except RuntimeError as exc:
    if DEBUG >= 1:
      print(f"cuda_compat: {exc}")
    return CUDA_ERROR_INVALID_PTX if _module_is_ptx(raw_image) else CUDA_ERROR_INVALID_IMAGE
  module_id = _new_handle()
  _MODULES[module_id] = _ModuleState(ctx_id=ctx_id, raw_image=raw_image, program_image=program_image, signatures=_parse_ptx_signatures(raw_image))
  module._obj.value = module_id
  return CUDA_SUCCESS


def cuModuleUnload(hmod) -> int:
  module_id = _as_int(hmod)
  if module_id not in _MODULES:
    return CUDA_ERROR_INVALID_HANDLE
  for fn_id, fn in [entry for entry in _FUNCTIONS.items() if entry[1].module_id == module_id]:
    if fn.program is not None:
      del fn.program
    del _FUNCTIONS[fn_id]
  del _MODULES[module_id]
  return CUDA_SUCCESS


def cuModuleGetFunction(hfunc, hmod, name: bytes) -> int:
  module_id = _as_int(hmod)
  if module_id not in _MODULES:
    return CUDA_ERROR_INVALID_HANDLE
  fn_id = _new_handle()
  _FUNCTIONS[fn_id] = _FunctionState(module_id=module_id, name=name.decode("utf-8"))
  hfunc._obj.value = fn_id
  return CUDA_SUCCESS


def cuFuncSetAttribute(hfunc, attrib: int, value: int) -> int:
  fn_id = _as_int(hfunc)
  if fn_id not in _FUNCTIONS:
    return CUDA_ERROR_INVALID_HANDLE
  _FUNCTIONS[fn_id].attrs[attrib] = value
  return CUDA_SUCCESS


def cuLaunchKernel(f, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, sharedMemBytes: int, hStream, kernelParams, extra) -> int:
  fn_id = _as_int(f)
  if fn_id not in _FUNCTIONS:
    return CUDA_ERROR_INVALID_HANDLE
  if (arg_blob := _build_arg_blob(_FUNCTIONS[fn_id], kernelParams, extra)) is None:
    return CUDA_ERROR_NOT_SUPPORTED
  try:
    _launch(_FUNCTIONS[fn_id], arg_blob, (gx, gy, gz), (lx, ly, lz), sharedMemBytes)
  except Exception as exc:
    if DEBUG >= 1:
      print(f"cuda_compat launch failed: {exc}")
    return CUDA_ERROR_LAUNCH_FAILED
  return CUDA_SUCCESS


def cuEventCreate(phEvent, flags: int) -> int:
  event_id = _new_handle()
  _EVENTS[event_id] = _EventState()
  phEvent._obj.value = event_id
  return CUDA_SUCCESS


def cuEventRecord(hEvent, hStream) -> int:
  event_id = _as_int(hEvent)
  if event_id not in _EVENTS:
    return CUDA_ERROR_INVALID_HANDLE
  try:
    _get_ctx().nv_device.synchronize()
  except Exception:
    return CUDA_ERROR_INVALID_CONTEXT
  _EVENTS[event_id].timestamp_ns = time.perf_counter_ns()
  return CUDA_SUCCESS


def cuEventSynchronize(hEvent) -> int:
  event_id = _as_int(hEvent)
  if event_id not in _EVENTS:
    return CUDA_ERROR_INVALID_HANDLE
  return CUDA_SUCCESS


def cuEventElapsedTime(pMilliseconds, hStart, hEnd) -> int:
  st_id, en_id = _as_int(hStart), _as_int(hEnd)
  if st_id not in _EVENTS or en_id not in _EVENTS:
    return CUDA_ERROR_INVALID_HANDLE
  pMilliseconds._obj.value = (_EVENTS[en_id].timestamp_ns - _EVENTS[st_id].timestamp_ns) / 1e6
  return CUDA_SUCCESS


def cuEventDestroy_v2(hEvent) -> int:
  _EVENTS.pop(_as_int(hEvent), None)
  return CUDA_SUCCESS


def cuStreamWaitEvent(stream, event, flags: int) -> int:
  return CUDA_SUCCESS


def cuGetErrorString(error: int, pStr) -> int:
  if error not in _ERROR_BUFS:
    _ERROR_BUFS[error] = ctypes.create_string_buffer(_ERROR_NAMES.get(error, "Unknown CUDA error").encode())
  ctypes.cast(pStr, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))[0] = ctypes.cast(_ERROR_BUFS[error], ctypes.POINTER(ctypes.c_char))
  return CUDA_SUCCESS
