from __future__ import annotations

import ctypes
import os
import struct
import time
from dataclasses import dataclass

from tbgpu.autogen import libc
from tbgpu.helpers import getenv, to_mv, unwrap

MAP_FIXED = 0x10


class MMIOInterface:
  def __init__(self, addr: int, nbytes: int, fmt: str = "B"):
    self.addr, self.nbytes, self.fmt = addr, nbytes, fmt
    self.mv = to_mv(addr, nbytes).cast(fmt)

  def __len__(self) -> int:
    return self.nbytes // struct.calcsize(self.fmt)

  def __getitem__(self, key):
    return (self.mv[key] if self.fmt == "B" else self.mv[key].tolist()) if isinstance(key, slice) else self.mv[key]

  def __setitem__(self, key, value):
    self.mv[key] = value

  def view(self, offset: int = 0, size: int | None = None, fmt: str | None = None) -> "MMIOInterface":
    return MMIOInterface(self.addr + offset, self.nbytes - offset if size is None else size, fmt=fmt or self.fmt)


class FileIOInterface:
  def __init__(self, path: str = "", flags: int = os.O_RDONLY, fd: int | None = None):
    self.path = path
    self.fd = os.open(path, flags) if fd is None else fd

  def __del__(self):
    if hasattr(self, "fd"):
      try:
        os.close(self.fd)
      except OSError:
        pass

  def mmap(self, start: int, size: int, prot: int, flags: int, offset: int) -> int:
    return FileIOInterface._mmap(start, size, prot, flags, self.fd, offset)

  def ioctl(self, request, arg):
    import fcntl

    return fcntl.ioctl(self.fd, request, arg)

  def read(self, size=None, binary=False, offset=None):
    if offset is not None:
      self.seek(offset)
    with open(self.fd, "rb" if binary else "r", closefd=False) as fh:
      return fh.read(size)

  def write(self, content, binary=False, offset=None):
    if offset is not None:
      self.seek(offset)
    with open(self.fd, "wb" if binary else "w", closefd=False) as fh:
      fh.write(content)

  def seek(self, offset: int):
    os.lseek(self.fd, offset, os.SEEK_SET)

  @staticmethod
  def _mmap(start: int, size: int, prot: int, flags: int, fd: int, offset: int) -> int:
    addr = libc.mmap(start, size, prot, flags, fd, offset)
    if int(ctypes.cast(addr, ctypes.c_void_p).value or 0) == 0xFFFFFFFFFFFFFFFF:
      raise OSError(f"failed to mmap {size:#x} bytes")
    return unwrap(ctypes.cast(addr, ctypes.c_void_p).value)

  @staticmethod
  def anon_mmap(start: int, size: int, prot: int, flags: int, offset: int) -> int:
    return FileIOInterface._mmap(start, size, prot, flags, -1, offset)

  @staticmethod
  def munmap(addr: int, size: int):
    return libc.munmap(addr, size)


@dataclass
class Buffer:
  va_addr: int
  size: int
  meta: object | None = None
  _base: "Buffer | None" = None
  view: MMIOInterface | None = None
  owner: object | None = None

  def offset(self, offset: int = 0, size: int | None = None) -> "Buffer":
    return Buffer(
      self.va_addr + offset,
      self.size - offset if size is None else size,
      meta=self.meta,
      _base=self._base or self,
      view=self.view.view(offset=offset, size=size) if self.view is not None else None,
      owner=self.owner,
    )

  def cpu_view(self) -> MMIOInterface:
    assert self.view is not None, "buffer has no cpu mapping"
    return self.view

  @property
  def base(self) -> "Buffer":
    return self._base or self


class Signal:
  def __init__(self, base_buf: Buffer, owner=None, timestamp_divider=1000):
    self.base_buf = base_buf
    self.owner = owner
    self.timestamp_divider = timestamp_divider
    self.value = 0

  @property
  def value_addr(self) -> int:
    return self.base_buf.va_addr

  @property
  def timestamp_addr(self) -> int:
    return self.base_buf.va_addr + 8

  @property
  def value(self) -> int:
    return self.base_buf.cpu_view().view(size=8, fmt="Q")[0]

  @value.setter
  def value(self, new_value: int):
    self.base_buf.cpu_view().view(size=8, fmt="Q")[0] = new_value

  @property
  def timestamp(self):
    return self.base_buf.cpu_view().view(offset=8, size=8, fmt="Q")[0] / self.timestamp_divider

  def wait(self, value: int, timeout: int | None = None):
    timeout = timeout or getenv("TBGPU_WAIT_TIMEOUT_MS", 30000)
    start = int(time.perf_counter() * 1000)
    while (current := self.value) < value and int(time.perf_counter() * 1000) - start < timeout:
      if self.owner is not None and hasattr(self.owner, "iface"):
        self.owner.iface.sleep(int(time.perf_counter() * 1000) - start)
      else:
        time.sleep(0.001)
      if current != self.value:
        start = int(time.perf_counter() * 1000)
    if self.value < value:
      raise RuntimeError(f"wait timeout: signal={self.value}, expected>={value}")
