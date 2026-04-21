from __future__ import annotations

import argparse
import array
import contextlib
import ctypes
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tbgpu.cuda_compat as cuda
from tbgpu.runtime.transport import RemotePCIDevice
from tests.vector_add import KERNEL_NAME, VecAddArgs, _check, _make_extra, init_c_var, load_kernel_image


FLOAT_SZ = ctypes.sizeof(ctypes.c_float)
HOST_FLAGS = cuda.CU_MEMHOSTALLOC_PORTABLE | cuda.CU_MEMHOSTALLOC_DEVICEMAP
HOST_FLAGS_WC = HOST_FLAGS | cuda.CU_MEMHOSTALLOC_WRITECOMBINED


@dataclass
class BenchResult:
  name: str
  size_bytes: int
  iterations: int
  elapsed_s: float
  payload_bytes: int
  transfer_stats: dict[str, int]
  transport_stats: dict[str, float | int]
  extra: dict[str, float | int | str]

  def as_dict(self) -> dict[str, object]:
    gbps = (self.payload_bytes / self.elapsed_s) / (1 << 30) if self.elapsed_s > 0 else 0.0
    return {
      "name": self.name,
      "size_bytes": self.size_bytes,
      "iterations": self.iterations,
      "elapsed_s": self.elapsed_s,
      "avg_ms": (self.elapsed_s * 1000.0 / self.iterations) if self.iterations else 0.0,
      "payload_bytes": self.payload_bytes,
      "payload_gbps": gbps,
      "transfer_stats": self.transfer_stats,
      "transport_stats": self.transport_stats,
      "extra": self.extra,
    }


class BenchSession:
  def __init__(self, kernel_input: str, block_size: int, needs_kernel: bool):
    _check(cuda.cuInit(0))
    self.dev = init_c_var(cuda.CUdevice, lambda x: _check(cuda.cuDeviceGet(ctypes.byref(x), 0)))
    self.ctx = init_c_var(cuda.CUcontext, lambda x: _check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, self.dev.value)))
    _check(cuda.cuCtxSetCurrent(self.ctx))
    self.block_size = block_size
    self.module = None
    self.func = None
    if needs_kernel:
      _check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), self.dev.value))
      arch = f"sm_{major.value}{minor.value}"
      kernel_image = load_kernel_image(arch, kernel_input)
      self.module = init_c_var(cuda.CUmodule, lambda x: _check(cuda.cuModuleLoadData(ctypes.byref(x), kernel_image)))
      self.func = init_c_var(cuda.CUfunction, lambda x: _check(cuda.cuModuleGetFunction(ctypes.byref(x), self.module, KERNEL_NAME.encode())))

  def create_stream(self, flags: int = 0):
    return init_c_var(cuda.CUstream, lambda x: _check(cuda.cuStreamCreate(ctypes.byref(x), flags)))

  def create_event(self):
    return init_c_var(cuda.CUevent, lambda x: _check(cuda.cuEventCreate(ctypes.byref(x), 0)))

  def launch_vector_add(self, size: int, a_ptr: int, b_ptr: int, out_ptr: int, stream) -> None:
    if size == 0:
      return
    if self.func is None:
      raise RuntimeError("vector_add kernel was not initialized for this benchmark session")
    args = VecAddArgs(a_ptr, b_ptr, out_ptr, size)
    extra, _ = _make_extra(args)
    grid = ((size + self.block_size - 1) // self.block_size, 1, 1)
    block = (self.block_size, 1, 1)
    _check(cuda.cuLaunchKernel(self.func, *grid, *block, 0, stream, None, extra))

  def close(self):
    if self.module is not None:
      cuda.cuModuleUnload(self.module)
    cuda.cuCtxDestroy_v2(self.ctx)


class HostBuffer:
  def __init__(self, nbytes: int, flags: int):
    self.nbytes = nbytes
    self.ptr = ctypes.c_void_p()
    _check(cuda.cuMemHostAlloc(ctypes.byref(self.ptr), nbytes, flags))

  @property
  def value(self) -> int:
    assert self.ptr.value is not None
    return self.ptr.value

  def free(self):
    if self.ptr.value not in (None, 0):
      cuda.cuMemFreeHost(self.ptr)
      self.ptr = ctypes.c_void_p()


class DeviceBuffer:
  def __init__(self, nbytes: int):
    self.ptr = cuda.CUdeviceptr()
    _check(cuda.cuMemAlloc_v2(ctypes.byref(self.ptr), nbytes))

  @property
  def value(self) -> int:
    return int(self.ptr.value or 0)

  def free(self):
    if self.ptr.value not in (None, 0):
      cuda.cuMemFree_v2(self.ptr)
      self.ptr = cuda.CUdeviceptr()


def _buffer_ptr(buf: array.array) -> int:
  return ctypes.addressof((ctypes.c_float * len(buf)).from_buffer(buf))


def _fill_host(ptr: int, values: array.array) -> None:
  ctypes.memmove(ptr, _buffer_ptr(values), len(values) * FLOAT_SZ)


def _host_array(ptr: int, size: int) -> array.array:
  return array.array("f", (ctypes.c_float * size).from_address(ptr))


def _assert_close(label: str, got: array.array, expected: array.array) -> None:
  for idx, (got_item, exp_item) in enumerate(zip(got, expected)):
    if abs(got_item - exp_item) > 1e-5:
      raise AssertionError(f"{label} mismatch at {idx}: {got_item} != {exp_item}")


def _reset_stats() -> None:
  cuda.transfer_stats_reset()
  RemotePCIDevice.reset_stats()


def _snapshot_stats() -> tuple[dict[str, int], dict[str, float | int]]:
  transfer_stats = cuda.transfer_stats_snapshot(0)
  transport_stats = RemotePCIDevice.snapshot_stats()
  return transfer_stats, transport_stats


def bench_h2d(session: BenchSession, elems: int, iters: int, *, write_combined: bool = False) -> BenchResult:
  nbytes = elems * FLOAT_SZ
  stream = session.create_stream(cuda.CU_STREAM_NON_BLOCKING)
  host = HostBuffer(nbytes, HOST_FLAGS_WC if write_combined else HOST_FLAGS)
  device = DeviceBuffer(nbytes)
  payload = array.array("f", (float(i % 251) for i in range(elems)))
  _fill_host(host.value, payload)
  try:
    _check(cuda.cuMemcpyHtoDAsync_v2(device.ptr, host.value, nbytes, stream))
    _check(cuda.cuStreamSynchronize(stream))
    _reset_stats()
    start = time.perf_counter()
    for _ in range(iters):
      _check(cuda.cuMemcpyHtoDAsync_v2(device.ptr, host.value, nbytes, stream))
    _check(cuda.cuStreamSynchronize(stream))
    elapsed = time.perf_counter() - start
    transfer_stats, transport_stats = _snapshot_stats()
    return BenchResult(
      name="h2d_wc" if write_combined else "h2d_pinned",
      size_bytes=nbytes,
      iterations=iters,
      elapsed_s=elapsed,
      payload_bytes=nbytes * iters,
      transfer_stats=transfer_stats,
      transport_stats=transport_stats,
      extra={"write_combined": int(write_combined)},
    )
  finally:
    device.free()
    host.free()
    cuda.cuStreamDestroy_v2(stream)


def bench_d2h(session: BenchSession, elems: int, iters: int) -> BenchResult:
  nbytes = elems * FLOAT_SZ
  stream = session.create_stream(cuda.CU_STREAM_NON_BLOCKING)
  host = HostBuffer(nbytes, HOST_FLAGS)
  device = DeviceBuffer(nbytes)
  seed = array.array("f", (float((i * 3) % 257) for i in range(elems)))
  try:
    _check(cuda.cuMemcpyHtoDAsync_v2(device.ptr, _buffer_ptr(seed), nbytes, stream))
    _check(cuda.cuStreamSynchronize(stream))
    _reset_stats()
    start = time.perf_counter()
    for _ in range(iters):
      _check(cuda.cuMemcpyDtoHAsync_v2(host.value, device.ptr, nbytes, stream))
    _check(cuda.cuStreamSynchronize(stream))
    elapsed = time.perf_counter() - start
    got = _host_array(host.value, elems)
    _assert_close("dtoh", got, seed)
    transfer_stats, transport_stats = _snapshot_stats()
    return BenchResult(
      name="d2h_pinned",
      size_bytes=nbytes,
      iterations=iters,
      elapsed_s=elapsed,
      payload_bytes=nbytes * iters,
      transfer_stats=transfer_stats,
      transport_stats=transport_stats,
      extra={},
    )
  finally:
    device.free()
    host.free()
    cuda.cuStreamDestroy_v2(stream)


def bench_zero_copy(session: BenchSession, elems: int, iters: int) -> BenchResult:
  nbytes = elems * FLOAT_SZ
  stream = session.create_stream(cuda.CU_STREAM_NON_BLOCKING)
  h_a = HostBuffer(nbytes, HOST_FLAGS_WC)
  h_b = HostBuffer(nbytes, HOST_FLAGS_WC)
  h_out = HostBuffer(nbytes, HOST_FLAGS)
  zc_a, zc_b, zc_out = cuda.CUdeviceptr(), cuda.CUdeviceptr(), cuda.CUdeviceptr()
  a = array.array("f", (float(i % 257) for i in range(elems)))
  b = array.array("f", (float((2 * i + 1) % 263) for i in range(elems)))
  expected = array.array("f", (x + y for x, y in zip(a, b)))
  try:
    _fill_host(h_a.value, a)
    _fill_host(h_b.value, b)
    ctypes.memset(h_out.value, 0, nbytes)
    _check(cuda.cuMemHostGetDevicePointer(ctypes.byref(zc_a), h_a.ptr, 0))
    _check(cuda.cuMemHostGetDevicePointer(ctypes.byref(zc_b), h_b.ptr, 0))
    _check(cuda.cuMemHostGetDevicePointer(ctypes.byref(zc_out), h_out.ptr, 0))
    session.launch_vector_add(elems, zc_a.value, zc_b.value, zc_out.value, stream)
    _check(cuda.cuStreamSynchronize(stream))
    _reset_stats()
    start = time.perf_counter()
    for _ in range(iters):
      session.launch_vector_add(elems, zc_a.value, zc_b.value, zc_out.value, stream)
    _check(cuda.cuStreamSynchronize(stream))
    elapsed = time.perf_counter() - start
    got = _host_array(h_out.value, elems)
    _assert_close("zero-copy", got, expected)
    transfer_stats, transport_stats = _snapshot_stats()
    return BenchResult(
      name="zero_copy_wc_inputs",
      size_bytes=nbytes,
      iterations=iters,
      elapsed_s=elapsed,
      payload_bytes=nbytes * iters * 3,
      transfer_stats=transfer_stats,
      transport_stats=transport_stats,
      extra={"kernel_bytes_touched": nbytes * iters * 3},
    )
  finally:
    h_out.free()
    h_b.free()
    h_a.free()
    cuda.cuStreamDestroy_v2(stream)


def _allocate_slot_buffers(slot_count: int, chunk_elems: int) -> tuple[list[HostBuffer], list[HostBuffer], list[HostBuffer], list[DeviceBuffer], list[DeviceBuffer], list[DeviceBuffer]]:
  nbytes = chunk_elems * FLOAT_SZ
  h_a = [HostBuffer(nbytes, HOST_FLAGS_WC) for _ in range(slot_count)]
  h_b = [HostBuffer(nbytes, HOST_FLAGS_WC) for _ in range(slot_count)]
  h_out = [HostBuffer(nbytes, HOST_FLAGS) for _ in range(slot_count)]
  d_a = [DeviceBuffer(nbytes) for _ in range(slot_count)]
  d_b = [DeviceBuffer(nbytes) for _ in range(slot_count)]
  d_out = [DeviceBuffer(nbytes) for _ in range(slot_count)]
  return h_a, h_b, h_out, d_a, d_b, d_out


def _free_slots(*groups) -> None:
  for group in groups:
    for item in group:
      item.free()


def _seed_slot_buffers(h_a: list[HostBuffer], h_b: list[HostBuffer], h_out: list[HostBuffer], chunk_elems: int) -> array.array:
  expected = array.array("f", (float((i % 97) + ((i * 5 + 3) % 89)) for i in range(chunk_elems)))
  a = array.array("f", (float(i % 97) for i in range(chunk_elems)))
  b = array.array("f", (float((i * 5 + 3) % 89) for i in range(chunk_elems)))
  for host_a, host_b, host_out in zip(h_a, h_b, h_out):
    _fill_host(host_a.value, a)
    _fill_host(host_b.value, b)
    ctypes.memset(host_out.value, 0, chunk_elems * FLOAT_SZ)
  return expected


def bench_overlap(session: BenchSession, chunk_elems: int, iters: int, depth: int) -> list[BenchResult]:
  nbytes = chunk_elems * FLOAT_SZ
  total_transfer_bytes = nbytes * iters * 3
  expected = None
  upload_stream = session.create_stream(cuda.CU_STREAM_NON_BLOCKING)
  compute_stream = session.create_stream(cuda.CU_STREAM_NON_BLOCKING)
  download_stream = session.create_stream(cuda.CU_STREAM_NON_BLOCKING)
  seq_stream = session.create_stream(cuda.CU_STREAM_NON_BLOCKING)
  upload_done = [session.create_event() for _ in range(depth)]
  compute_done = [session.create_event() for _ in range(depth)]
  download_done = [session.create_event() for _ in range(depth)]
  h_a, h_b, h_out, d_a, d_b, d_out = _allocate_slot_buffers(depth, chunk_elems)
  try:
    expected = _seed_slot_buffers(h_a, h_b, h_out, chunk_elems)

    _check(cuda.cuMemcpyHtoDAsync_v2(d_a[0].ptr, h_a[0].value, nbytes, seq_stream))
    _check(cuda.cuMemcpyHtoDAsync_v2(d_b[0].ptr, h_b[0].value, nbytes, seq_stream))
    session.launch_vector_add(chunk_elems, d_a[0].value, d_b[0].value, d_out[0].value, seq_stream)
    _check(cuda.cuMemcpyDtoHAsync_v2(h_out[0].value, d_out[0].ptr, nbytes, seq_stream))
    _check(cuda.cuStreamSynchronize(seq_stream))

    _reset_stats()
    start = time.perf_counter()
    for idx in range(iters):
      slot = idx % depth
      _check(cuda.cuMemcpyHtoDAsync_v2(d_a[slot].ptr, h_a[slot].value, nbytes, seq_stream))
      _check(cuda.cuMemcpyHtoDAsync_v2(d_b[slot].ptr, h_b[slot].value, nbytes, seq_stream))
      session.launch_vector_add(chunk_elems, d_a[slot].value, d_b[slot].value, d_out[slot].value, seq_stream)
      _check(cuda.cuMemcpyDtoHAsync_v2(h_out[slot].value, d_out[slot].ptr, nbytes, seq_stream))
    _check(cuda.cuStreamSynchronize(seq_stream))
    sequential_elapsed = time.perf_counter() - start
    seq_transfer_stats, seq_transport_stats = _snapshot_stats()
    _assert_close("sequential overlap", _host_array(h_out[(iters - 1) % depth].value, chunk_elems), expected)

    _reset_stats()
    start = time.perf_counter()
    for idx in range(iters):
      slot = idx % depth
      if idx >= depth:
        _check(cuda.cuEventSynchronize(download_done[slot]))
      _check(cuda.cuMemcpyHtoDAsync_v2(d_a[slot].ptr, h_a[slot].value, nbytes, upload_stream))
      _check(cuda.cuMemcpyHtoDAsync_v2(d_b[slot].ptr, h_b[slot].value, nbytes, upload_stream))
      _check(cuda.cuEventRecord(upload_done[slot], upload_stream))
      _check(cuda.cuStreamWaitEvent(compute_stream, upload_done[slot], 0))
      session.launch_vector_add(chunk_elems, d_a[slot].value, d_b[slot].value, d_out[slot].value, compute_stream)
      _check(cuda.cuEventRecord(compute_done[slot], compute_stream))
      _check(cuda.cuStreamWaitEvent(download_stream, compute_done[slot], 0))
      _check(cuda.cuMemcpyDtoHAsync_v2(h_out[slot].value, d_out[slot].ptr, nbytes, download_stream))
      _check(cuda.cuEventRecord(download_done[slot], download_stream))
    _check(cuda.cuStreamSynchronize(download_stream))
    pipelined_elapsed = time.perf_counter() - start
    pipe_transfer_stats, pipe_transport_stats = _snapshot_stats()
    _assert_close("pipelined overlap", _host_array(h_out[(iters - 1) % depth].value, chunk_elems), expected)

    return [
      BenchResult(
        name="sequential_copy_compute_copy",
        size_bytes=nbytes,
        iterations=iters,
        elapsed_s=sequential_elapsed,
        payload_bytes=total_transfer_bytes,
        transfer_stats=seq_transfer_stats,
        transport_stats=seq_transport_stats,
        extra={"chunk_bytes": nbytes, "pipeline_depth": depth},
      ),
      BenchResult(
        name="pipelined_copy_compute_copy",
        size_bytes=nbytes,
        iterations=iters,
        elapsed_s=pipelined_elapsed,
        payload_bytes=total_transfer_bytes,
        transfer_stats=pipe_transfer_stats,
        transport_stats=pipe_transport_stats,
        extra={
          "chunk_bytes": nbytes,
          "pipeline_depth": depth,
          "speedup_vs_sequential": (sequential_elapsed / pipelined_elapsed) if pipelined_elapsed > 0 else 0.0,
        },
      ),
    ]
  finally:
    for event in [*upload_done, *compute_done, *download_done]:
      cuda.cuEventDestroy_v2(event)
    for stream in [download_stream, compute_stream, upload_stream, seq_stream]:
      cuda.cuStreamDestroy_v2(stream)
    _free_slots(d_out, d_b, d_a, h_out, h_b, h_a)


def _parse_benches(raw: str) -> list[str]:
  benches = [item.strip() for item in raw.split(",") if item.strip()]
  valid = {"h2d", "h2d_wc", "d2h", "zero_copy", "overlap"}
  invalid = [item for item in benches if item not in valid]
  if invalid:
    raise ValueError(f"unknown benches: {', '.join(invalid)}")
  return benches


def run_transfer_bench(
  size_mb: int,
  chunk_mb: int,
  iters: int,
  pipeline_depth: int,
  block_size: int,
  kernel_input: str,
  remote_sock: str | None,
  benches: list[str],
) -> list[BenchResult]:
  if remote_sock:
    os.environ["APL_REMOTE_SOCK"] = remote_sock

  elems = (size_mb << 20) // FLOAT_SZ
  chunk_elems = (chunk_mb << 20) // FLOAT_SZ
  if elems <= 0 or chunk_elems <= 0:
    raise ValueError("bench sizes must be positive")

  session = BenchSession(kernel_input=kernel_input, block_size=block_size, needs_kernel=any(item in {"zero_copy", "overlap"} for item in benches))
  try:
    results: list[BenchResult] = []
    if "h2d" in benches:
      results.append(bench_h2d(session, elems, iters, write_combined=False))
    if "h2d_wc" in benches:
      results.append(bench_h2d(session, elems, iters, write_combined=True))
    if "d2h" in benches:
      results.append(bench_d2h(session, elems, iters))
    if "zero_copy" in benches:
      results.append(bench_zero_copy(session, elems, iters))
    if "overlap" in benches:
      results.extend(bench_overlap(session, chunk_elems, iters, pipeline_depth))
    return results
  finally:
    session.close()


def main() -> None:
  parser = argparse.ArgumentParser(description="Benchmark USB4-sensitive transfer paths on the TinyGPU CUDA shim")
  parser.add_argument("--size-mb", type=int, default=64, help="buffer size for the HtoD, DtoH, and zero-copy benches")
  parser.add_argument("--chunk-mb", type=int, default=8, help="chunk size for the overlap bench")
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--pipeline-depth", type=int, default=3)
  parser.add_argument("--block-size", type=int, default=256)
  parser.add_argument("--kernel-input", choices=["cuda", "ptx"], default="ptx")
  parser.add_argument("--remote-sock", help="optional Unix socket path for the TinyGPU server")
  parser.add_argument("--benches", default="h2d,h2d_wc,d2h,zero_copy,overlap", help="comma-separated subset of: h2d,h2d_wc,d2h,zero_copy,overlap")
  args = parser.parse_args()

  with contextlib.redirect_stdout(io.StringIO()):
    results = run_transfer_bench(
      size_mb=args.size_mb,
      chunk_mb=args.chunk_mb,
      iters=args.iters,
      pipeline_depth=args.pipeline_depth,
      block_size=args.block_size,
      kernel_input=args.kernel_input,
      remote_sock=args.remote_sock,
      benches=_parse_benches(args.benches),
    )
  print(json.dumps([result.as_dict() for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
