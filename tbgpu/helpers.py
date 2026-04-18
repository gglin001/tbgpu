from __future__ import annotations

import contextlib
import ctypes
import hashlib
import math
import os
import pathlib
import tempfile
import time
import urllib.parse
import urllib.request
from typing import Any, TypeVar

T = TypeVar("T")

OSX = os.name == "posix" and os.uname().sysname == "Darwin"
WIN = os.name == "nt"
DEBUG = int(os.getenv("DEBUG", "0"))


def getenv(name: str, default: T) -> T:
  raw = os.getenv(name)
  if raw is None:
    return default
  if isinstance(default, bool):
    return raw not in ("", "0", "false", "False")  # type: ignore[return-value]
  if isinstance(default, int):
    return int(raw)  # type: ignore[return-value]
  if isinstance(default, float):
    return float(raw)  # type: ignore[return-value]
  return raw  # type: ignore[return-value]


def round_up(value: int, alignment: int) -> int:
  if alignment <= 1:
    return value
  return ((value + alignment - 1) // alignment) * alignment


def round_down(value: int, alignment: int) -> int:
  if alignment <= 1:
    return value
  return (value // alignment) * alignment


def ceildiv(value: int, divisor: int) -> int:
  return -(-value // divisor)


def lo32(value: Any) -> Any:
  return value & 0xFFFFFFFF


def hi32(value: Any) -> Any:
  return value >> 32


def data64(value: Any) -> tuple[Any, Any]:
  return (value >> 32, value & 0xFFFFFFFF)


def data64_le(value: Any) -> tuple[Any, Any]:
  return (value & 0xFFFFFFFF, value >> 32)


def getbits(value: int, start: int, end: int) -> int:
  return (value >> start) & ((1 << (end - start + 1)) - 1)


def i2u(bits: int, value: int) -> int:
  return value & ((1 << bits) - 1)


def unwrap(value: T | None) -> T:
  assert value is not None
  return value


def prod(values) -> int:
  return math.prod(values)


def wait_cond(cb, *args, value=True, timeout_ms=10000, msg=""):
  start = int(time.perf_counter() * 1000)
  while int(time.perf_counter() * 1000) - start < timeout_ms:
    if (current := cb(*args)) == value:
      return current
  raise TimeoutError(f"{msg}. Timed out after {timeout_ms} ms, condition not met: {current} != {value}")


def to_mv(ptr: int, size: int) -> memoryview:
  return memoryview((ctypes.c_uint8 * size).from_address(ptr)).cast("B")


def from_mv(mv: memoryview, to_type: type[ctypes._SimpleCData] = ctypes.c_char):
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents


def temp(name: str) -> str:
  return str(pathlib.Path(tempfile.gettempdir()) / name)


def _downloads_dir() -> pathlib.Path:
  path = pathlib.Path(os.getenv("TBGPU_CACHE_DIR", pathlib.Path.home() / ".cache" / "tbgpu"))
  path.mkdir(parents=True, exist_ok=True)
  return path


def fetch(url: str, *, name: str | None = None, subdir: str = "downloads") -> pathlib.Path:
  root = _downloads_dir() / subdir
  root.mkdir(parents=True, exist_ok=True)
  if name is None:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    name = f"{digest}-{pathlib.Path(urllib.parse.urlparse(url).path).name or 'download'}"
  path = root / name
  if path.exists():
    return path
  with contextlib.closing(urllib.request.urlopen(url)) as resp:
    path.write_bytes(resp.read())
  return path
