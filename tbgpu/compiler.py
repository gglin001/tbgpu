from __future__ import annotations

import functools
import hashlib
import pathlib
import shutil
import subprocess
import tempfile


def require_nvcc() -> str:
  if (nvcc := shutil.which("nvcc")) is None:
    raise RuntimeError("nvcc not found, run extra/setup_nvcc_osx.sh or provide nvcc in PATH")
  return nvcc


def _run_nvcc(args: list[str]):
  proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
  if proc.returncode != 0:
    raise RuntimeError((proc.stderr or proc.stdout).decode("utf-8", "ignore").strip())


def compile_cuda_to_ptx(cuda_src: str, arch: str, kernel_name="kernel") -> bytes:
  require_nvcc()
  cache_root = pathlib.Path.home() / ".cache" / "tbgpu" / "nvcc-tmp"
  cache_root.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="tbgpu_cuda_", dir=cache_root) as tmpdir:
    src = pathlib.Path(tmpdir) / f"{kernel_name}.cu"
    out = pathlib.Path(tmpdir) / f"{kernel_name}.ptx"
    src.write_text(cuda_src)
    _run_nvcc(["nvcc", f"-arch={arch}", "-ptx", "-o", out.as_posix(), src.as_posix()])
    return out.read_bytes()


@functools.lru_cache(maxsize=None)
def compile_ptx_to_cubin(ptx: bytes, arch: str) -> bytes:
  require_nvcc()
  digest = hashlib.sha256(ptx + arch.encode()).hexdigest()[:16]
  root = pathlib.Path.home() / ".cache" / "tbgpu" / "nvcc-cache"
  root.mkdir(parents=True, exist_ok=True)
  workdir = root / digest
  workdir.mkdir(parents=True, exist_ok=True)
  ptx_path = workdir / "module.ptx"
  cubin_path = workdir / "module.cubin"
  ptx_path.write_bytes(ptx)
  _run_nvcc(["nvcc", f"-arch={arch}", "-cubin", "-o", cubin_path.as_posix(), ptx_path.as_posix()])
  return cubin_path.read_bytes()
