from __future__ import annotations

import atexit
import contextlib
import ctypes
import enum
import fcntl
import functools
import itertools
import mmap
import os
import socket
import struct
import subprocess
import time

from tbgpu.helpers import DEBUG, OSX, ceildiv, getenv, temp, unwrap
from tbgpu.runtime.common import FileIOInterface, MAP_FIXED, MMIOInterface


class _System:
  @functools.cached_property
  def libsys(self):
    import ctypes.util
    return ctypes.CDLL(ctypes.util.find_library("System") or "/usr/lib/libSystem.B.dylib")

  def memory_barrier(self):
    self.libsys.atomic_thread_fence(5)

  def flock_acquire(self, name: str) -> int:
    os.umask(0)
    path = temp(name)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
      raise RuntimeError(f"failed to acquire lock {path}") from exc
    return fd


System = _System()


class RemoteCmd(enum.IntEnum):
  PROBE, MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, RESET, MMIO_READ, MMIO_WRITE, MAP_SYSMEM, SYSMEM_READ, SYSMEM_WRITE, RESIZE_BAR, PING = range(13)


class RemoteMMIOInterface(MMIOInterface):
  def __init__(self, dev: "RemotePCIDevice", residx: int, nbytes: int, fmt="B", off=0, rd_cmd=RemoteCmd.MMIO_READ, wr_cmd=RemoteCmd.MMIO_WRITE):
    self.dev, self.residx, self.nbytes, self.fmt, self.off = dev, residx, nbytes, fmt, off
    self.el_sz = struct.calcsize(fmt)
    self.rd_cmd, self.wr_cmd = rd_cmd, wr_cmd

  def __len__(self):
    return self.nbytes // self.el_sz

  def __getitem__(self, index):
    sl = index if isinstance(index, slice) else slice(index, index + 1)
    start, stop = (sl.start or 0) * self.el_sz, (sl.stop or len(self)) * self.el_sz
    data = self.dev._bulk_read(self.rd_cmd, self.residx, self.off + start, stop - start)
    result = data if self.fmt == "B" else list(struct.unpack(f"<{(stop - start) // self.el_sz}{self.fmt}", data))
    return result if isinstance(index, slice) else result[0]

  def __setitem__(self, index, value):
    start = (index.start or 0) * self.el_sz if isinstance(index, slice) else index * self.el_sz
    payload = (value if self.fmt == "B" else struct.pack(f"<{len(value)}{self.fmt}", *value)) if isinstance(index, slice) else struct.pack(f"<{self.fmt}", value)
    self.dev._bulk_write(self.wr_cmd, self.residx, self.off + start, payload)

  def view(self, offset=0, size=None, fmt=None):
    return RemoteMMIOInterface(self.dev, self.residx, size or (self.nbytes - offset), fmt or self.fmt, self.off + offset, self.rd_cmd, self.wr_cmd)


class RemotePCIDevice:
  _bulk_sent = 0
  _bulk_recv = 0
  _rpc_count = 0
  _start_time = 0.0

  @staticmethod
  def _recvall(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n and (chunk := sock.recv(n - len(data))):
      data += chunk
    if len(data) < n:
      raise RuntimeError("connection closed")
    return data

  @staticmethod
  def _rpc(sock: socket.socket, dev_id: int, cmd: int, *args: int, bar: int = 0, readout_size: int = 0, payload: bytes = b"", has_fd=False):
    sock.sendall(struct.pack("<BIIQQQ", cmd, dev_id, bar, *(*args, 0, 0, 0)[:3]) + payload)
    if has_fd:
      msg, anc, _, _ = sock.recvmsg(17, socket.CMSG_LEN(4))
      fd = struct.unpack("<i", anc[0][2][:4])[0]
    else:
      msg, fd = RemotePCIDevice._recvall(sock, 17), None
    resp = struct.unpack("<BQQ", msg)
    if resp[0] != 0:
      raise RuntimeError(RemotePCIDevice._recvall(sock, resp[1]).decode("utf-8") if resp[1] else "rpc failed")
    RemotePCIDevice._rpc_count += 1
    return (resp[1], resp[2]) + ((RemotePCIDevice._recvall(sock, readout_size) if readout_size else None),) + (fd,)

  def __init__(self, devpref: str, dev_id: int, sock: socket.socket):
    self.sock, self.dev_id = sock, dev_id
    self.pcibus = f"remote:{dev_id}"
    self.peer_group = "tinygpu"
    self.lock_fd = System.flock_acquire(f"{devpref.lower()}_{dev_id}.lock")
    for opt in [socket.SO_SNDBUF, socket.SO_RCVBUF]:
      self.sock.setsockopt(socket.SOL_SOCKET, opt, 64 << 20)

  def _bulk_read(self, cmd: int, idx: int, offset: int, size: int) -> bytes:
    RemotePCIDevice._bulk_recv += size
    return unwrap(self._rpc(self.sock, self.dev_id, cmd, offset, size, bar=idx, readout_size=size)[2])

  def _bulk_write(self, cmd: int, idx: int, offset: int, data: bytes):
    RemotePCIDevice._bulk_sent += len(data)
    self.sock.sendall(struct.pack("<BIIQQQ", cmd, self.dev_id, idx, offset, len(data), 0) + data)

  def alloc_sysmem(self, size: int, vaddr: int = 0, contiguous: bool = False):
    mapped_size, _, _, fd = self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_SYSMEM_FD, size, int(contiguous), has_fd=True)
    memview = MMIOInterface(FileIOInterface(fd=fd).mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, 0), mapped_size, fmt="B")
    paddrs_raw = list(itertools.takewhile(lambda pair: pair[1] != 0, zip(memview.view(fmt="Q")[0::2], memview.view(fmt="Q")[1::2])))
    return memview, [p + i for p, sz in paddrs_raw for i in range(0, sz, 0x1000)][:ceildiv(size, 0x1000)]

  def reset(self):
    self._rpc(self.sock, self.dev_id, RemoteCmd.RESET)

  def read_config(self, offset: int, size: int):
    return self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_READ, offset, size)[0]

  def write_config(self, offset: int, value: int, size: int):
    self._rpc(self.sock, self.dev_id, RemoteCmd.CFG_WRITE, offset, size, value)

  @functools.cache
  def bar_info(self, bar_idx: int):
    return self._rpc(self.sock, self.dev_id, RemoteCmd.MAP_BAR, bar=bar_idx)[:2]

  def map_bar(self, bar: int, off: int = 0, addr: int = 0, size: int | None = None, fmt="B"):
    return RemoteMMIOInterface(self, bar, size or self.bar_info(bar)[1], fmt).view(off, size, fmt)

  def resize_bar(self, bar_idx: int):
    self._rpc(self.sock, self.dev_id, RemoteCmd.RESIZE_BAR, bar=bar_idx)


class APLRemotePCIDevice(RemotePCIDevice):
  APP_PATH = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"

  @classmethod
  def connect(cls) -> socket.socket:
    sock_path = getenv("APL_REMOTE_SOCK", temp("tinygpu.sock"))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for attempt in range(100):
      with contextlib.suppress(ConnectionRefusedError, FileNotFoundError):
        sock.connect(sock_path)
        return sock
      if attempt == 0:
        subprocess.Popen([cls.APP_PATH, "server", sock_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      time.sleep(0.05)
    raise RuntimeError(f"failed to connect to TinyGPU server at {sock_path}")

  def __init__(self, devpref: str, dev_id: int):
    super().__init__(devpref, dev_id, self.connect())

