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
from tbgpu.runtime.device import Fence, TBGPUDevice
from tbgpu.runtime.program import TBGPUProgram

DEBUG = int(os.getenv("DEBUG", "0"))
CUDA_PTI = bool(os.getenv("CUDA_PTI", "0"))

CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUevent = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64

CU_STREAM_DEFAULT = 0x0
CU_STREAM_NON_BLOCKING = 0x1
CU_STREAM_LEGACY = 0x1
CU_STREAM_PER_THREAD = 0x2

CU_EVENT_DEFAULT = 0x0
CU_EVENT_BLOCKING_SYNC = 0x1
CU_EVENT_DISABLE_TIMING = 0x2
CU_EVENT_INTERPROCESS = 0x4

CU_MEMHOSTALLOC_PORTABLE = 0x1
CU_MEMHOSTALLOC_DEVICEMAP = 0x2
CU_MEMHOSTALLOC_WRITECOMBINED = 0x4

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
  default_stream: _StreamState
  stream_ids: set[int] = field(default_factory=set)
  per_thread_streams: dict[int, _StreamState] = field(default_factory=dict)
  next_lane: int = 0


@dataclass
class _StreamState:
  ctx_id: int
  lane_index: int
  flags: int = CU_STREAM_DEFAULT
  default_kind: str | None = None
  tail: Fence | None = None
  pending_waits: list[Fence] = field(default_factory=list)
  tail_kind: str | None = None


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
  owner: Any
  flags: int = 0


@dataclass
class _EventState:
  ctx_id: int | None = None
  fence: Fence | None = None
  flags: int = CU_EVENT_DEFAULT
  timestamp_ns: int = 0


@dataclass
class _PTIPendingKernel:
  ctx_id: int
  device_ordinal: int
  kernel: str
  grid: tuple[int, int, int]
  block: tuple[int, int, int]
  shared_mem_bytes: int
  launch_index: int
  start_signal: Any
  end_signal: Any
  signal_value: int = 1
  queued_ns: int = field(default_factory=time.perf_counter_ns)


class _RawArgsState:
  def __init__(self, buf):
    self.buf = buf


_HANDLE_COUNTER = itertools.count(3)
_TLS = threading.local()
_NV_DEVICE_CACHE: dict[int, Any] = {}
_CONTEXTS: dict[int, _ContextState] = {}
_MODULES: dict[int, _ModuleState] = {}
_FUNCTIONS: dict[int, _FunctionState] = {}
_DEVICE_ALLOCS: dict[int, _DeviceAllocation] = {}
_HOST_ALLOCS: dict[int, _HostAllocation] = {}
_EVENTS: dict[int, _EventState] = {}
_STREAMS: dict[int, _StreamState] = {}
_ERROR_BUFS: dict[int, ctypes.Array] = {}
_PTI_PENDING: list[_PTIPendingKernel] = []
_PTI_COMPLETED: list[dict[str, Any]] = []
_PTI_LOCK = threading.Lock()
_PTI_LAUNCH_COUNTER = itertools.count(1)

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


def _signal_timestamp_us(signal) -> float:
  return float(signal.timestamp)


def _flush_pti(ctx_id: int | None = None):
  ready: list[dict[str, Any]] = []
  pending: list[_PTIPendingKernel] = []
  for item in _PTI_PENDING:
    if ctx_id is not None and item.ctx_id != ctx_id:
      pending.append(item)
      continue
    if item.end_signal.value < item.signal_value:
      pending.append(item)
      continue
    st_us = _signal_timestamp_us(item.start_signal)
    en_us = _signal_timestamp_us(item.end_signal)
    ready.append({
      "ctx_id": item.ctx_id,
      "device_ordinal": item.device_ordinal,
      "kernel": item.kernel,
      "grid": item.grid,
      "block": item.block,
      "shared_mem_bytes": item.shared_mem_bytes,
      "launch_index": item.launch_index,
      "queued_ns": item.queued_ns,
      "start_us": st_us,
      "end_us": en_us,
      "duration_us": en_us - st_us,
    })
  _PTI_PENDING[:] = pending
  _PTI_COMPLETED.extend(ready)


def pti_enable(enabled: bool = True):
  global CUDA_PTI
  CUDA_PTI = bool(enabled)


def pti_reset():
  with _PTI_LOCK:
    _PTI_PENDING.clear()
    _PTI_COMPLETED.clear()


def pti_collect(clear: bool = True) -> list[dict[str, Any]]:
  with _PTI_LOCK:
    _flush_pti()
    records = [dict(item) for item in _PTI_COMPLETED]
    if clear:
      _PTI_COMPLETED.clear()
    return records


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


def _create_default_stream(ctx_id: int, nv_device: TBGPUDevice) -> _StreamState:
  return _StreamState(
    ctx_id=ctx_id,
    lane_index=0 if nv_device.compute_lane_count > 0 else -1,
    flags=CU_STREAM_DEFAULT,
    default_kind="legacy",
  )


def _get_per_thread_stream(ctx_id: int, ctx: _ContextState) -> _StreamState:
  thread_id = threading.get_ident()
  if thread_id not in ctx.per_thread_streams:
    ctx.nv_device.ensure_compute_lanes(len(ctx.stream_ids) + len(ctx.per_thread_streams) + 1)
    ctx.per_thread_streams[thread_id] = _StreamState(
      ctx_id=ctx_id,
      lane_index=_next_stream_lane(ctx),
      flags=CU_STREAM_DEFAULT,
      default_kind="per_thread",
    )
  return ctx.per_thread_streams[thread_id]


def _stream_is_non_blocking(stream: _StreamState) -> bool:
  return stream.default_kind is None and bool(stream.flags & CU_STREAM_NON_BLOCKING)


def _stream_syncs_with_legacy(stream: _StreamState) -> bool:
  if stream.default_kind == "legacy":
    return False
  return not _stream_is_non_blocking(stream)


def _get_stream(ctx_id: int, stream) -> _StreamState:
  handle = _as_int(stream)
  if ctx_id not in _CONTEXTS:
    raise RuntimeError("invalid context")
  ctx = _CONTEXTS[ctx_id]
  if handle in (0, CU_STREAM_LEGACY):
    return ctx.default_stream
  if handle == CU_STREAM_PER_THREAD:
    return _get_per_thread_stream(ctx_id, ctx)
  if handle not in _STREAMS or _STREAMS[handle].ctx_id != ctx_id:
    raise RuntimeError("invalid stream")
  return _STREAMS[handle]


def _iter_ctx_streams(ctx: _ContextState):
  yield ctx.default_stream
  for stream in list(ctx.per_thread_streams.values()):
    yield stream
  for stream_id in list(ctx.stream_ids):
    if stream_id in _STREAMS:
      yield _STREAMS[stream_id]


def _next_stream_lane(ctx: _ContextState) -> int:
  lane_count = max(1, ctx.nv_device.compute_lane_count)
  lane = ctx.next_lane % lane_count
  ctx.next_lane += 1
  return lane


def _synchronize_stream(stream: _StreamState, timeout: int | None = None):
  for fence in _stream_submission_waits(stream):
    fence.wait(timeout=timeout)
  stream.pending_waits.clear()


def _synchronize_ctx_streams(ctx: _ContextState, timeout: int | None = None):
  for stream in _iter_ctx_streams(ctx):
    _synchronize_stream(stream, timeout=timeout)


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


def _find_host_alloc(ptr: int):
  for alloc in _HOST_ALLOCS.values():
    if alloc.base <= ptr < alloc.base + alloc.size:
      return alloc, ptr - alloc.base
  return None


def _dedupe_fences(*fences: Fence | None) -> list[Fence]:
  unique: list[Fence] = []
  seen: set[tuple[int, int]] = set()
  for fence in fences:
    if fence is None:
      continue
    key = (fence.signal.value_addr, fence.value)
    if key in seen:
      continue
    seen.add(key)
    unique.append(fence)
  return unique


def _legacy_stream_waits(stream: _StreamState) -> list[Fence]:
  ctx = _CONTEXTS[stream.ctx_id]
  waits: list[Fence | None] = [stream.tail, *stream.pending_waits]
  for other in _iter_ctx_streams(ctx):
    if other is stream or not _stream_syncs_with_legacy(other):
      continue
    waits.extend([other.tail, *other.pending_waits])
  return _dedupe_fences(*waits)


def _stream_submission_waits(stream: _StreamState, *extra_waits: Fence | None) -> list[Fence]:
  waits: list[Fence | None] = [stream.tail, *stream.pending_waits, *extra_waits]
  if stream.default_kind == "legacy":
    waits.extend(_legacy_stream_waits(stream))
  elif _stream_syncs_with_legacy(stream):
    legacy = _CONTEXTS[stream.ctx_id].default_stream
    waits.extend([legacy.tail, *legacy.pending_waits])
  return _dedupe_fences(*waits)


def _advance_stream(stream: _StreamState, completion: Fence | None, kind: str | None = None):
  stream.pending_waits.clear()
  if completion is not None:
    stream.tail = completion
    stream.tail_kind = kind


def _submit_stream_marker(stream: _StreamState, wait_fences: list[Fence] | None = None, *, timestamp: bool = False) -> Fence:
  from tbgpu.runtime.device import NVComputeQueue

  ctx = _CONTEXTS[stream.ctx_id]
  lane = ctx.nv_device.compute_lanes[stream.lane_index]
  completion = ctx.nv_device.new_fence()
  q = NVComputeQueue()
  for fence in _stream_submission_waits(stream, *(wait_fences or [])):
    q.wait(fence.signal, fence.value)
  if timestamp:
    q.timestamp(completion.signal, completion.value)
  else:
    q.signal(completion.signal, completion.value)
  q.submit(lane)
  lane.record_submission(completion)
  _advance_stream(stream, completion, kind="marker")
  return completion


def _submit_buffer_copy(dst, src, size: int, stream: _StreamState):
  waits = _stream_submission_waits(stream)
  completion = src.owner.allocator._transfer(dst, src, size, src.owner, dst.owner, waits=waits)
  _advance_stream(stream, completion, kind="copy")


def _copy_to_device(dst_ptr: int, src_ptr: int, size: int, stream: _StreamState):
  if (dst_info := _find_device_alloc(dst_ptr)) is None:
    raise RuntimeError(f"unknown device pointer 0x{dst_ptr:x}")
  dst_alloc, dst_off = dst_info
  if (src_info := _find_host_alloc(src_ptr)) is not None:
    src_alloc, src_off = src_info
    _submit_buffer_copy(dst_alloc.buf.offset(offset=dst_off, size=size), src_alloc.buf.offset(offset=src_off, size=size), size, stream)
    return
  alloc, offset = dst_info
  waits = _stream_submission_waits(stream)
  completion = alloc.owner.allocator._copyin(
    alloc.buf.offset(offset=offset, size=size), memoryview(bytearray(ctypes.string_at(src_ptr, size))), waits=waits
  )
  if completion is None:
    for fence in waits:
      fence.wait()
  _advance_stream(stream, completion, kind="copy")


def _copy_from_device(dst_ptr: int, src_ptr: int, size: int):
  if (src_info := _find_device_alloc(src_ptr)) is None:
    raise RuntimeError(f"unknown device pointer 0x{src_ptr:x}")
  if (dst_info := _find_host_alloc(dst_ptr)) is not None:
    dst_alloc, dst_off = dst_info
    src_alloc, src_off = src_info
    stream = _get_stream(_as_int(_get_current_context_id()), CU_STREAM_LEGACY)
    _submit_buffer_copy(dst_alloc.buf.offset(offset=dst_off, size=size), src_alloc.buf.offset(offset=src_off, size=size), size, stream)
    _synchronize_stream(stream)
    return
  alloc, offset = src_info
  tmp = bytearray(size)
  alloc.owner.allocator._copyout(memoryview(tmp), alloc.buf.offset(offset=offset, size=size))
  ctypes.memmove(dst_ptr, bytes(tmp), size)


def _copy_from_device_async(dst_ptr: int, src_ptr: int, size: int, stream: _StreamState):
  if (src_info := _find_device_alloc(src_ptr)) is None:
    raise RuntimeError(f"unknown device pointer 0x{src_ptr:x}")
  if (dst_info := _find_host_alloc(dst_ptr)) is not None:
    dst_alloc, dst_off = dst_info
    src_alloc, src_off = src_info
    _submit_buffer_copy(dst_alloc.buf.offset(offset=dst_off, size=size), src_alloc.buf.offset(offset=src_off, size=size), size, stream)
    return
  _copy_from_device(dst_ptr, src_ptr, size)


def _copy_device_to_device(dst_ptr: int, src_ptr: int, size: int, stream: _StreamState):
  if (dst_info := _find_device_alloc(dst_ptr)) is None or (src_info := _find_device_alloc(src_ptr)) is None:
    raise RuntimeError("unknown device pointer")
  dst_alloc, dst_off = dst_info
  src_alloc, src_off = src_info
  _submit_buffer_copy(dst_alloc.buf.offset(offset=dst_off, size=size), src_alloc.buf.offset(offset=src_off, size=size), size, stream)


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


def _launch(
  function: _FunctionState, stream: _StreamState, arg_blob: bytes, grid: tuple[int, int, int], block: tuple[int, int, int], shared_mem_bytes: int
):
  from tbgpu.runtime.device import NVComputeQueue

  prg = _ensure_program(function)
  dev = prg.dev
  module = _MODULES[function.module_id]
  lane = dev.compute_lanes[stream.lane_index]
  argsbuf = dev.reserve_kernargs(stream.lane_index, prg.kernargs_alloc_size, 8)
  prefix = _build_launch_cbuf0(prg, grid, block)
  view = argsbuf.cpu_view().view(fmt="B")
  view[: len(prefix)] = prefix
  view[len(prefix) : len(prefix) + len(arg_blob)] = arg_blob
  q = NVComputeQueue()
  waits = _stream_submission_waits(stream)
  for fence in waits:
    q.wait(fence.signal, fence.value)
  if stream.tail_kind != "launch" or any(fence is not stream.tail for fence in waits):
    q.memory_barrier()
  prof_st = dev.new_signal() if CUDA_PTI else None
  prof_en = dev.new_signal() if CUDA_PTI else None
  if prof_st is not None:
    q.timestamp(prof_st, 1)
  q.exec(prg, argsbuf, grid, block, shared_mem_bytes=shared_mem_bytes)
  if prof_en is not None:
    q.timestamp(prof_en, 1)
  completion = dev.new_fence()
  q.signal(completion.signal, completion.value).submit(lane)
  lane.record_submission(completion)
  _advance_stream(stream, completion, kind="launch")
  if prof_st is not None and prof_en is not None:
    with _PTI_LOCK:
      _PTI_PENDING.append(
        _PTIPendingKernel(
          ctx_id=module.ctx_id,
          device_ordinal=_CONTEXTS[module.ctx_id].device_ordinal,
          kernel=function.name,
          grid=grid,
          block=block,
          shared_mem_bytes=shared_mem_bytes,
          launch_index=next(_PTI_LAUNCH_COUNTER),
          start_signal=prof_st,
          end_signal=prof_en,
        )
      )


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
  _CONTEXTS[ctx_id] = _ContextState(device_ordinal=dev, nv_device=nv_device, default_stream=_create_default_stream(ctx_id, nv_device))
  _set_current_context(ctx_id)
  pctx._obj.value = ctx_id
  return CUDA_SUCCESS


def cuCtxDestroy_v2(ctx) -> int:
  ctx_id = _as_int(ctx)
  if ctx_id not in _CONTEXTS:
    return CUDA_ERROR_INVALID_CONTEXT
  ctx = _CONTEXTS[ctx_id]
  try:
    _synchronize_ctx_streams(ctx)
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  with _PTI_LOCK:
    _flush_pti(ctx_id)
    _PTI_PENDING[:] = [item for item in _PTI_PENDING if item.ctx_id != ctx_id]
  for event in _EVENTS.values():
    if event.ctx_id == ctx_id:
      event.ctx_id, event.fence = None, None
  ctx.per_thread_streams.clear()
  for stream_id in list(ctx.stream_ids):
    _STREAMS.pop(stream_id, None)
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
    ctx_id = _get_current_context_id()
    ctx = _get_ctx()
    _synchronize_ctx_streams(ctx)
    if ctx_id is not None:
      with _PTI_LOCK:
        _flush_pti(ctx_id)
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
  try:
    alloc.owner.synchronize()
  except RuntimeError:
    _DEVICE_ALLOCS[ptr] = alloc
    return CUDA_ERROR_LAUNCH_FAILED
  alloc.owner.allocator.free(alloc.buf, alloc.size)
  return CUDA_SUCCESS


def cuMemHostAlloc(pp, bytesize: int, flags: int) -> int:
  if flags & ~(CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED):
    return CUDA_ERROR_INVALID_VALUE
  try:
    ctx = _get_ctx()
    host_buf = ctx.nv_device.allocator.alloc(
      bytesize,
      host=True,
      cpu_access=True,
      contiguous=bool(flags & CU_MEMHOSTALLOC_WRITECOMBINED),
    )
  except Exception:
    return CUDA_ERROR_INVALID_CONTEXT
  ptr = host_buf.cpu_view().addr
  _HOST_ALLOCS[ptr] = _HostAllocation(base=ptr, size=bytesize, buf=host_buf, owner=ctx.nv_device, flags=flags)
  pp._obj.value = ptr
  return CUDA_SUCCESS


def cuMemHostGetDevicePointer(pdptr, p, flags: int) -> int:
  if flags != 0:
    return CUDA_ERROR_INVALID_VALUE
  ptr = _as_int(p)
  if (alloc_info := _find_host_alloc(ptr)) is None:
    return CUDA_ERROR_INVALID_VALUE
  alloc, offset = alloc_info
  if not (alloc.flags & CU_MEMHOSTALLOC_DEVICEMAP):
    return CUDA_ERROR_INVALID_VALUE
  pdptr._obj.value = alloc.buf.va_addr + offset
  return CUDA_SUCCESS


def cuMemFreeHost(p) -> int:
  ptr = _as_int(p)
  if ptr not in _HOST_ALLOCS:
    return CUDA_ERROR_INVALID_VALUE
  alloc = _HOST_ALLOCS.pop(ptr)
  try:
    alloc.owner.synchronize()
  except RuntimeError:
    _HOST_ALLOCS[ptr] = alloc
    return CUDA_ERROR_LAUNCH_FAILED
  alloc.owner.allocator.free(alloc.buf, alloc.size)
  return CUDA_SUCCESS


def _legacy_stream(ctx_id: int) -> _StreamState:
  return _get_stream(ctx_id, CU_STREAM_LEGACY)


def cuMemcpyHtoDAsync_v2(dst, src, bytesize: int, stream) -> int:
  try:
    ctx_id = _get_current_context_id()
    if ctx_id not in _CONTEXTS:
      return CUDA_ERROR_INVALID_CONTEXT
    _copy_to_device(_as_int(dst), _as_int(src), bytesize, _get_stream(ctx_id, stream))
  except Exception:
    return CUDA_ERROR_INVALID_VALUE
  return CUDA_SUCCESS


def cuMemcpyHtoD_v2(dst, src, bytesize: int) -> int:
  try:
    ctx_id = _get_current_context_id()
    if ctx_id not in _CONTEXTS:
      return CUDA_ERROR_INVALID_CONTEXT
    stream = _legacy_stream(ctx_id)
    _copy_to_device(_as_int(dst), _as_int(src), bytesize, stream)
    _synchronize_stream(stream)
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  except Exception:
    return CUDA_ERROR_INVALID_VALUE
  return CUDA_SUCCESS


def cuMemcpyDtoH_v2(dst, src, bytesize: int) -> int:
  try:
    ctx_id = _get_current_context_id()
    if ctx_id not in _CONTEXTS:
      return CUDA_ERROR_INVALID_CONTEXT
    stream = _legacy_stream(ctx_id)
    _copy_from_device_async(_as_int(dst), _as_int(src), bytesize, stream)
    _synchronize_stream(stream)
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  except Exception:
    return CUDA_ERROR_INVALID_VALUE
  return CUDA_SUCCESS


def cuMemcpyDtoHAsync_v2(dst, src, bytesize: int, stream) -> int:
  try:
    ctx_id = _get_current_context_id()
    if ctx_id not in _CONTEXTS:
      return CUDA_ERROR_INVALID_CONTEXT
    _copy_from_device_async(_as_int(dst), _as_int(src), bytesize, _get_stream(ctx_id, stream))
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  except Exception:
    return CUDA_ERROR_INVALID_VALUE
  return CUDA_SUCCESS


def cuMemcpyDtoD_v2(dst, src, bytesize: int) -> int:
  try:
    ctx_id = _get_current_context_id()
    if ctx_id not in _CONTEXTS:
      return CUDA_ERROR_INVALID_CONTEXT
    stream = _legacy_stream(ctx_id)
    _copy_device_to_device(_as_int(dst), _as_int(src), bytesize, stream)
    _synchronize_stream(stream)
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  except Exception:
    return CUDA_ERROR_INVALID_VALUE
  return CUDA_SUCCESS


def cuMemcpyDtoDAsync_v2(dst, src, bytesize: int, stream) -> int:
  try:
    ctx_id = _get_current_context_id()
    if ctx_id not in _CONTEXTS:
      return CUDA_ERROR_INVALID_CONTEXT
    _copy_device_to_device(_as_int(dst), _as_int(src), bytesize, _get_stream(ctx_id, stream))
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
  try:
    _synchronize_ctx_streams(_CONTEXTS[_MODULES[module_id].ctx_id])
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
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
  if kernelParams and extra:
    return CUDA_ERROR_INVALID_VALUE
  if extra and _extract_extra_blob(extra) is None:
    return CUDA_ERROR_INVALID_VALUE
  if kernelParams and _MODULES[_FUNCTIONS[fn_id].module_id].signatures.get(_FUNCTIONS[fn_id].name) is None:
    return CUDA_ERROR_INVALID_IMAGE
  if (arg_blob := _build_arg_blob(_FUNCTIONS[fn_id], kernelParams, extra)) is None:
    return CUDA_ERROR_INVALID_VALUE
  try:
    ctx_id = _MODULES[_FUNCTIONS[fn_id].module_id].ctx_id
    _launch(_FUNCTIONS[fn_id], _get_stream(ctx_id, hStream), arg_blob, (gx, gy, gz), (lx, ly, lz), sharedMemBytes)
  except Exception as exc:
    if DEBUG >= 1:
      print(f"cuda_compat launch failed: {exc}")
    return CUDA_ERROR_LAUNCH_FAILED
  return CUDA_SUCCESS


def cuEventCreate(phEvent, flags: int) -> int:
  if flags & ~(CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS):
    return CUDA_ERROR_INVALID_VALUE
  if flags & CU_EVENT_INTERPROCESS and not flags & CU_EVENT_DISABLE_TIMING:
    return CUDA_ERROR_INVALID_VALUE
  event_id = _new_handle()
  _EVENTS[event_id] = _EventState(flags=flags)
  phEvent._obj.value = event_id
  return CUDA_SUCCESS


def cuEventRecord(hEvent, hStream) -> int:
  event_id = _as_int(hEvent)
  if event_id not in _EVENTS:
    return CUDA_ERROR_INVALID_HANDLE
  event = _EVENTS[event_id]
  try:
    ctx_id = _get_current_context_id()
    if ctx_id not in _CONTEXTS:
      return CUDA_ERROR_INVALID_CONTEXT
    fence = _submit_stream_marker(_get_stream(ctx_id, hStream), timestamp=not (event.flags & CU_EVENT_DISABLE_TIMING))
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  except Exception:
    return CUDA_ERROR_INVALID_CONTEXT
  _EVENTS[event_id] = _EventState(ctx_id=ctx_id, fence=fence, flags=event.flags, timestamp_ns=time.perf_counter_ns())
  return CUDA_SUCCESS


def cuEventSynchronize(hEvent) -> int:
  event_id = _as_int(hEvent)
  if event_id not in _EVENTS:
    return CUDA_ERROR_INVALID_HANDLE
  event = _EVENTS[event_id]
  if event.fence is None:
    return CUDA_SUCCESS
  try:
    event.fence.wait()
    if event.ctx_id is not None:
      with _PTI_LOCK:
        _flush_pti(event.ctx_id)
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  except Exception:
    return CUDA_ERROR_INVALID_CONTEXT
  return CUDA_SUCCESS


def cuEventElapsedTime(pMilliseconds, hStart, hEnd) -> int:
  st_id, en_id = _as_int(hStart), _as_int(hEnd)
  if st_id not in _EVENTS or en_id not in _EVENTS:
    return CUDA_ERROR_INVALID_HANDLE
  st_ev, en_ev = _EVENTS[st_id], _EVENTS[en_id]
  if st_ev.flags & CU_EVENT_DISABLE_TIMING or en_ev.flags & CU_EVENT_DISABLE_TIMING:
    return CUDA_ERROR_INVALID_HANDLE
  if st_ev.fence is not None and en_ev.fence is not None:
    try:
      st_ev.fence.wait()
      en_ev.fence.wait()
    except RuntimeError:
      return CUDA_ERROR_LAUNCH_FAILED
    except Exception:
      return CUDA_ERROR_INVALID_CONTEXT
    pMilliseconds._obj.value = (_signal_timestamp_us(en_ev.fence.signal) - _signal_timestamp_us(st_ev.fence.signal)) / 1000.0
    return CUDA_SUCCESS
  pMilliseconds._obj.value = (en_ev.timestamp_ns - st_ev.timestamp_ns) / 1e6
  return CUDA_SUCCESS


def cuEventDestroy_v2(hEvent) -> int:
  _EVENTS.pop(_as_int(hEvent), None)
  return CUDA_SUCCESS


def cuEventCreateWithFlags(phEvent, flags: int) -> int:
  return cuEventCreate(phEvent, flags)


def cuStreamCreate(phStream, flags: int = 0) -> int:
  ctx_id = _get_current_context_id()
  if ctx_id not in _CONTEXTS:
    return CUDA_ERROR_INVALID_CONTEXT
  if flags not in (CU_STREAM_DEFAULT, CU_STREAM_NON_BLOCKING):
    return CUDA_ERROR_INVALID_VALUE
  ctx = _CONTEXTS[ctx_id]
  ctx.nv_device.ensure_compute_lanes(len(ctx.stream_ids) + len(ctx.per_thread_streams) + 1)
  stream_id = _new_handle()
  _STREAMS[stream_id] = _StreamState(ctx_id=ctx_id, lane_index=_next_stream_lane(ctx), flags=flags)
  ctx.stream_ids.add(stream_id)
  phStream._obj.value = stream_id
  return CUDA_SUCCESS


def cuStreamCreateWithFlags(phStream, flags: int) -> int:
  return cuStreamCreate(phStream, flags)


def cuStreamDestroy_v2(hStream) -> int:
  stream_id = _as_int(hStream)
  if stream_id in (0, CU_STREAM_LEGACY, CU_STREAM_PER_THREAD):
    return CUDA_ERROR_INVALID_HANDLE
  if stream_id not in _STREAMS:
    return CUDA_ERROR_INVALID_HANDLE
  stream = _STREAMS.pop(stream_id)
  try:
    _synchronize_stream(stream)
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  if stream.ctx_id in _CONTEXTS:
    _CONTEXTS[stream.ctx_id].stream_ids.discard(stream_id)
  return CUDA_SUCCESS


def cuStreamSynchronize(hStream) -> int:
  try:
    ctx_id = _get_current_context_id()
    if ctx_id not in _CONTEXTS:
      return CUDA_ERROR_INVALID_CONTEXT
    stream = _get_stream(ctx_id, hStream)
    _synchronize_stream(stream)
    with _PTI_LOCK:
      _flush_pti(ctx_id)
  except RuntimeError:
    return CUDA_ERROR_LAUNCH_FAILED
  except Exception:
    return CUDA_ERROR_INVALID_CONTEXT
  return CUDA_SUCCESS


def cuStreamWaitEvent(stream, event, flags: int) -> int:
  ctx_id = _get_current_context_id()
  if ctx_id not in _CONTEXTS:
    return CUDA_ERROR_INVALID_CONTEXT
  if flags != 0:
    return CUDA_ERROR_INVALID_VALUE
  event_id = _as_int(event)
  if event_id not in _EVENTS:
    return CUDA_ERROR_INVALID_HANDLE
  event_state = _EVENTS[event_id]
  if event_state.ctx_id is not None:
    if event_state.ctx_id not in _CONTEXTS:
      return CUDA_ERROR_INVALID_CONTEXT
    if _CONTEXTS[event_state.ctx_id].nv_device is not _CONTEXTS[ctx_id].nv_device:
      return CUDA_ERROR_INVALID_CONTEXT
  if event_state.fence is None:
    return CUDA_SUCCESS
  try:
    target_stream = _get_stream(ctx_id, stream)
    target_stream.pending_waits = _dedupe_fences(*target_stream.pending_waits, event_state.fence)
  except Exception:
    return CUDA_ERROR_INVALID_CONTEXT
  return CUDA_SUCCESS


cuCtxCreate = cuCtxCreate_v2
cuCtxDestroy = cuCtxDestroy_v2
cuEventDestroy = cuEventDestroy_v2
cuMemAlloc = cuMemAlloc_v2
cuMemAllocHost = cuMemHostAlloc
cuMemFree = cuMemFree_v2
cuMemcpyDtoD = cuMemcpyDtoD_v2
cuMemcpyDtoDAsync = cuMemcpyDtoDAsync_v2
cuMemcpyDtoH = cuMemcpyDtoH_v2
cuMemcpyDtoHAsync = cuMemcpyDtoHAsync_v2
cuMemcpyHtoD = cuMemcpyHtoD_v2
cuMemcpyHtoDAsync = cuMemcpyHtoDAsync_v2
cuStreamDestroy = cuStreamDestroy_v2


def cuGetErrorString(error: int, pStr) -> int:
  if error not in _ERROR_BUFS:
    _ERROR_BUFS[error] = ctypes.create_string_buffer(_ERROR_NAMES.get(error, "Unknown CUDA error").encode())
  ctypes.cast(pStr, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))[0] = ctypes.cast(_ERROR_BUFS[error], ctypes.POINTER(ctypes.c_char))
  return CUDA_SUCCESS
