from __future__ import annotations

import contextlib
import ctypes
import math
from dataclasses import dataclass
from pathlib import Path

import torch

import tbgpu.cuda_compat as cuda
from tbgpu.compiler import compile_cuda_to_ptx

MATMUL_TC_KERNEL = "matmul_wmma_tf32"
ENCODER_KERNEL = "encoder_forward"
ENCODER_STEP_KERNEL = "encoder_forward_step"
LAYERNORM_KERNEL = "layernorm_forward"
FUSED_RESIDUAL_LN_KERNEL = "fused_residual_layernorm"
ADD_KERNEL = "residual_add"
PAD_ROWS_KERNEL = "pad_rows"
TAKE_LAST_TOKEN_KERNEL = "take_last_token"
WRITE_KV_CACHE_KERNEL = "write_kv_cache"
FLASH_ATTN_QKV_PROJ_FUSED_KERNEL = "flash_attn_qkv_proj_fused_fwd"
FLASH_ATTN_QKV_PROJ_DECODE_KERNEL = "flash_attn_qkv_proj_decode_fwd"

TC_BLOCK_WARPS = 4
FLASH_BLOCK_M = 4
FLASH_BLOCK_N = 32
FLASH_WARP_SIZE = 32
WARP_SIZE = 32
WARPS_PER_BLOCK = 4


class CUDACompatError(RuntimeError):
  pass


def _check(status: int):
  if status == 0:
    return
  err = ctypes.POINTER(ctypes.c_char)()
  cuda.cuGetErrorString(status, ctypes.byref(err))
  raise CUDACompatError(f"CUDA shim error {status}: {ctypes.string_at(err).decode()}")


def _kernel_file(name: str) -> Path:
  return Path(__file__).with_name("kernels") / name


def _shape_numel(shape: tuple[int, ...]) -> int:
  numel = 1
  for dim in shape:
    numel *= dim
  return numel


def _as_cpu_contiguous(tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
  if tensor.device.type != "cpu":
    raise ValueError("expected a CPU tensor")
  if dtype is not None:
    tensor = tensor.to(dtype=dtype, device="cpu")
  return tensor.detach().contiguous()


def _round_up(value: int, alignment: int) -> int:
  return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class DeviceTensor:
  ptr: cuda.CUdeviceptr
  shape: tuple[int, ...]
  dtype: torch.dtype
  alloc_nbytes: int
  static: bool = False

  @property
  def numel(self) -> int:
    return _shape_numel(self.shape)

  @property
  def nbytes(self) -> int:
    return self.numel * self.dtype.itemsize

  @property
  def matrix_rows(self) -> int:
    return _shape_numel(self.shape[:-1]) if len(self.shape) > 1 else 1

  @property
  def matrix_cols(self) -> int:
    return self.shape[-1]


class TBGPUContext:
  def __init__(self):
    self._closed = False
    self._modules: list[cuda.CUmodule] = []
    self._owned_ptrs: list[cuda.CUdeviceptr] = []
    self._deferred_free: list[DeviceTensor] = []

    _check(cuda.cuInit(0))
    self.device = cuda.CUdevice()
    _check(cuda.cuDeviceGet(ctypes.byref(self.device), 0))
    self.context = cuda.CUcontext()
    _check(cuda.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device.value))
    _check(cuda.cuCtxSetCurrent(self.context))
    self.stream = cuda.CUstream()
    _check(cuda.cuStreamCreate(ctypes.byref(self.stream), 0))

    major = ctypes.c_int()
    minor = ctypes.c_int()
    _check(cuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), self.device.value))
    if (major.value, minor.value) < (8, 0):
      raise RuntimeError(f"examples/gpt2 requires an Ampere-or-newer NVIDIA GPU, got cc={major.value}.{minor.value}")
    self.arch = f"sm_{major.value}{minor.value}"

    self._functions = {
      MATMUL_TC_KERNEL: self._load_kernel("matmul_tc_tf32.cu", MATMUL_TC_KERNEL),
      ENCODER_KERNEL: self._load_kernel("encoder_forward.cu", ENCODER_KERNEL),
      ENCODER_STEP_KERNEL: self._load_kernel("encoder_forward_step.cu", ENCODER_STEP_KERNEL),
      LAYERNORM_KERNEL: self._load_kernel("layernorm.cu", LAYERNORM_KERNEL),
      FUSED_RESIDUAL_LN_KERNEL: self._load_kernel("fused_residual_layernorm.cu", FUSED_RESIDUAL_LN_KERNEL),
      ADD_KERNEL: self._load_kernel("add.cu", ADD_KERNEL),
      PAD_ROWS_KERNEL: self._load_kernel("pad_rows.cu", PAD_ROWS_KERNEL),
      TAKE_LAST_TOKEN_KERNEL: self._load_kernel("take_last_token.cu", TAKE_LAST_TOKEN_KERNEL),
      WRITE_KV_CACHE_KERNEL: self._load_kernel("write_kv_cache.cu", WRITE_KV_CACHE_KERNEL),
      FLASH_ATTN_QKV_PROJ_FUSED_KERNEL: self._load_kernel("attention_qkv_proj_fused.cu", FLASH_ATTN_QKV_PROJ_FUSED_KERNEL),
      FLASH_ATTN_QKV_PROJ_DECODE_KERNEL: self._load_kernel("attention_qkv_proj_decode.cu", FLASH_ATTN_QKV_PROJ_DECODE_KERNEL),
    }

  def close(self):
    if self._closed:
      return
    self._closed = True
    with contextlib.suppress(Exception):
      self.synchronize()
    for ptr in reversed(self._owned_ptrs):
      with contextlib.suppress(Exception):
        cuda.cuMemFree_v2(ptr)
    self._owned_ptrs.clear()
    for module in self._modules:
      with contextlib.suppress(Exception):
        cuda.cuModuleUnload(module)
    self._modules.clear()
    with contextlib.suppress(Exception):
      if self.stream.value not in (None, 0):
        cuda.cuStreamDestroy_v2(self.stream)
    with contextlib.suppress(Exception):
      cuda.cuCtxDestroy_v2(self.context)

  def __del__(self):
    with contextlib.suppress(Exception):
      self.close()

  def synchronize(self):
    _check(cuda.cuCtxSetCurrent(self.context))
    _check(cuda.cuStreamSynchronize(self.stream))

  def _load_kernel(self, filename: str, kernel_name: str) -> cuda.CUfunction:
    _check(cuda.cuCtxSetCurrent(self.context))
    ptx = compile_cuda_to_ptx(_kernel_file(filename).read_text().strip() + "\n", self.arch, kernel_name=kernel_name)
    module = cuda.CUmodule()
    _check(cuda.cuModuleLoadData(ctypes.byref(module), ptx))
    function = cuda.CUfunction()
    _check(cuda.cuModuleGetFunction(ctypes.byref(function), module, kernel_name.encode()))
    self._modules.append(module)
    return function

  def _new_ptr(self, nbytes: int) -> cuda.CUdeviceptr:
    _check(cuda.cuCtxSetCurrent(self.context))
    ptr = cuda.CUdeviceptr()
    _check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), nbytes))
    self._owned_ptrs.append(ptr)
    return ptr

  def empty(self, shape: tuple[int, ...], dtype: torch.dtype, *, alloc_shape: tuple[int, ...] | None = None, static: bool = False) -> DeviceTensor:
    alloc = alloc_shape or shape
    return DeviceTensor(
      ptr=self._new_ptr(_shape_numel(alloc) * dtype.itemsize),
      shape=shape,
      dtype=dtype,
      alloc_nbytes=_shape_numel(alloc) * dtype.itemsize,
      static=static,
    )

  def upload(self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None, static: bool = False) -> DeviceTensor:
    cpu = _as_cpu_contiguous(tensor, dtype=dtype)
    out = self.empty(tuple(cpu.shape), cpu.dtype, static=static)
    _check(cuda.cuMemcpyHtoDAsync_v2(out.ptr, cpu.data_ptr(), cpu.numel() * cpu.element_size(), self.stream))
    self.synchronize()
    return out

  def download(self, tensor: DeviceTensor) -> torch.Tensor:
    out = torch.empty(tensor.shape, dtype=tensor.dtype)
    _check(cuda.cuMemcpyDtoH_v2(out.data_ptr(), tensor.ptr, tensor.nbytes))
    return out

  def free(self, tensor: DeviceTensor | None):
    if tensor is None or tensor.static:
      return
    if tensor.ptr in self._owned_ptrs:
      self._owned_ptrs.remove(tensor.ptr)
      _check(cuda.cuMemFree_v2(tensor.ptr))

  def defer_free(self, tensor: DeviceTensor | None):
    if tensor is None or tensor.static:
      return
    self._deferred_free.append(tensor)

  def flush_deferred_frees(self):
    pending = self._deferred_free
    self._deferred_free = []
    for tensor in pending:
      with contextlib.suppress(Exception):
        self.free(tensor)

  def _launch(self, kernel_name: str, params: list[ctypes._SimpleCData], grid: tuple[int, int, int], block: tuple[int, int, int], shared: int = 0):
    kernel_params = (ctypes.c_void_p * len(params))(*[ctypes.addressof(value) for value in params])
    _check(cuda.cuLaunchKernel(self._functions[kernel_name], *grid, *block, shared, self.stream, kernel_params, None))

  def encoder_forward(self, tokens: torch.Tensor, wte: DeviceTensor, wpe: DeviceTensor) -> DeviceTensor:
    tokens_i32 = _as_cpu_contiguous(tokens.to(dtype=torch.int32, device="cpu"))
    batch_size, seq_len = tokens_i32.shape
    channels = wte.shape[1]
    tokens_gpu = self.upload(tokens_i32, dtype=torch.int32)
    out = self.empty((batch_size, seq_len, channels), torch.float32)
    total_vec = _round_up(batch_size * seq_len * channels, 4) // 4
    params = [
      ctypes.c_uint64(tokens_gpu.ptr.value),
      ctypes.c_uint64(wte.ptr.value),
      ctypes.c_uint64(wpe.ptr.value),
      ctypes.c_uint64(out.ptr.value),
      ctypes.c_uint32(batch_size),
      ctypes.c_uint32(seq_len),
      ctypes.c_uint32(channels),
    ]
    self._launch(ENCODER_KERNEL, params, ((total_vec + 255) // 256, 1, 1), (256, 1, 1))
    self.synchronize()
    self.free(tokens_gpu)
    return out

  def encoder_forward_step(self, token: int, position: int, wte: DeviceTensor, wpe: DeviceTensor) -> DeviceTensor:
    channels = wte.shape[1]
    out = self.empty((1, channels), torch.float32)
    total_vec = _round_up(channels, 4) // 4
    params = [
      ctypes.c_uint32(int(token)),
      ctypes.c_uint32(int(position)),
      ctypes.c_uint64(wte.ptr.value),
      ctypes.c_uint64(wpe.ptr.value),
      ctypes.c_uint64(out.ptr.value),
      ctypes.c_uint32(channels),
    ]
    self._launch(ENCODER_STEP_KERNEL, params, ((total_vec + 255) // 256, 1, 1), (256, 1, 1))
    return out

  def layernorm(self, x: DeviceTensor, weight: DeviceTensor, bias: DeviceTensor) -> DeviceTensor:
    out = self.empty(x.shape, torch.float32)
    params = [
      ctypes.c_uint64(x.ptr.value),
      ctypes.c_uint64(weight.ptr.value),
      ctypes.c_uint64(bias.ptr.value),
      ctypes.c_uint64(out.ptr.value),
      ctypes.c_uint32(x.matrix_rows),
      ctypes.c_uint32(x.matrix_cols),
    ]
    self._launch(LAYERNORM_KERNEL, params, (((x.matrix_rows + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK), 1, 1), (WARP_SIZE, WARPS_PER_BLOCK, 1))
    return out

  def fused_residual_layernorm(
    self, residual_in: DeviceTensor, update: DeviceTensor, weight: DeviceTensor, bias: DeviceTensor
  ) -> tuple[DeviceTensor, DeviceTensor]:
    residual_out = self.empty(residual_in.shape, torch.float32)
    norm_out = self.empty(residual_in.shape, torch.float32)
    params = [
      ctypes.c_uint64(residual_in.ptr.value),
      ctypes.c_uint64(update.ptr.value),
      ctypes.c_uint64(weight.ptr.value),
      ctypes.c_uint64(bias.ptr.value),
      ctypes.c_uint64(residual_out.ptr.value),
      ctypes.c_uint64(norm_out.ptr.value),
      ctypes.c_uint32(residual_in.matrix_rows),
      ctypes.c_uint32(residual_in.matrix_cols),
    ]
    self._launch(
      FUSED_RESIDUAL_LN_KERNEL,
      params,
      (((residual_in.matrix_rows + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK), 1, 1),
      (WARP_SIZE, WARPS_PER_BLOCK, 1),
    )
    return residual_out, norm_out

  def residual_add(self, a: DeviceTensor, b: DeviceTensor) -> DeviceTensor:
    out = self.empty(a.shape, torch.float32)
    total_vec = _round_up(a.numel, 4) // 4
    params = [
      ctypes.c_uint64(a.ptr.value),
      ctypes.c_uint64(b.ptr.value),
      ctypes.c_uint64(out.ptr.value),
      ctypes.c_uint32(a.numel),
    ]
    self._launch(ADD_KERNEL, params, ((total_vec + 255) // 256, 1, 1), (256, 1, 1))
    return out

  def take_last_token(self, x: DeviceTensor, *, batch_size: int, seq_len: int) -> DeviceTensor:
    channels = x.shape[-1]
    out = self.empty((batch_size, channels), torch.float32)
    total_vec = _round_up(batch_size * channels, 4) // 4
    params = [
      ctypes.c_uint64(x.ptr.value),
      ctypes.c_uint64(out.ptr.value),
      ctypes.c_uint32(batch_size),
      ctypes.c_uint32(seq_len),
      ctypes.c_uint32(channels),
    ]
    self._launch(TAKE_LAST_TOKEN_KERNEL, params, ((total_vec + 255) // 256, 1, 1), (256, 1, 1))
    return out

  def write_kv_cache(self, qkv: DeviceTensor, key_cache: DeviceTensor, value_cache: DeviceTensor, *, start_pos: int, rows: int):
    channels = qkv.shape[-1] // 3
    total_vec = _round_up(rows * channels, 4) // 4
    params = [
      ctypes.c_uint64(qkv.ptr.value),
      ctypes.c_uint64(key_cache.ptr.value),
      ctypes.c_uint64(value_cache.ptr.value),
      ctypes.c_uint32(rows),
      ctypes.c_uint32(channels),
      ctypes.c_uint32(start_pos),
    ]
    self._launch(WRITE_KV_CACHE_KERNEL, params, ((total_vec + 255) // 256, 1, 1), (256, 1, 1))

  def flash_attention_qkv_proj_fused(
    self,
    qkv: DeviceTensor,
    proj_weight_t: DeviceTensor,
    proj_bias: DeviceTensor,
    *,
    batch_size: int,
    seq_len: int,
  ) -> DeviceTensor:
    channels = qkv.shape[-1] // 3
    head_dim = channels // 12
    out = self.empty((batch_size, seq_len, channels), torch.float32)
    shared_floats = 2 * FLASH_BLOCK_N * head_dim + FLASH_BLOCK_M * FLASH_BLOCK_N + 4 * FLASH_BLOCK_M + FLASH_BLOCK_M * head_dim
    params = [
      ctypes.c_uint64(qkv.ptr.value),
      ctypes.c_uint64(proj_weight_t.ptr.value),
      ctypes.c_uint64(proj_bias.ptr.value),
      ctypes.c_uint64(out.ptr.value),
      ctypes.c_uint32(batch_size),
      ctypes.c_uint32(12),
      ctypes.c_uint32(seq_len),
      ctypes.c_uint32(head_dim),
      ctypes.c_float(1.0 / math.sqrt(head_dim)),
      ctypes.c_uint32(1),
    ]
    self._launch(
      FLASH_ATTN_QKV_PROJ_FUSED_KERNEL,
      params,
      ((seq_len + FLASH_BLOCK_M - 1) // FLASH_BLOCK_M, (channels + WARP_SIZE - 1) // WARP_SIZE, batch_size),
      (FLASH_WARP_SIZE, FLASH_BLOCK_M, 1),
      shared=shared_floats * ctypes.sizeof(ctypes.c_float),
    )
    return out

  def flash_attention_qkv_proj_decode(
    self,
    qkv: DeviceTensor,
    proj_weight_t: DeviceTensor,
    proj_bias: DeviceTensor,
    key_cache: DeviceTensor,
    value_cache: DeviceTensor,
    *,
    cache_len: int,
  ) -> DeviceTensor:
    channels = qkv.shape[-1] // 3
    head_dim = channels // 12
    out = self.empty((1, channels), torch.float32)
    shared_floats = 12 * FLASH_BLOCK_N + 3 * 12 + 12 * WARP_SIZE + 12 * head_dim
    params = [
      ctypes.c_uint64(qkv.ptr.value),
      ctypes.c_uint64(proj_weight_t.ptr.value),
      ctypes.c_uint64(proj_bias.ptr.value),
      ctypes.c_uint64(key_cache.ptr.value),
      ctypes.c_uint64(value_cache.ptr.value),
      ctypes.c_uint64(out.ptr.value),
      ctypes.c_uint32(12),
      ctypes.c_uint32(cache_len),
      ctypes.c_uint32(head_dim),
      ctypes.c_float(1.0 / math.sqrt(head_dim)),
    ]
    self._launch(
      FLASH_ATTN_QKV_PROJ_DECODE_KERNEL,
      params,
      ((channels + WARP_SIZE - 1) // WARP_SIZE, 1, 1),
      (FLASH_WARP_SIZE, 12, 1),
      shared=shared_floats * ctypes.sizeof(ctypes.c_float),
    )
    return out

  def _pad_rows(self, x: DeviceTensor, padded_rows: int) -> DeviceTensor:
    out = self.empty((x.matrix_rows, x.matrix_cols), torch.float32, alloc_shape=(padded_rows, x.matrix_cols))
    total_vec = _round_up(padded_rows * x.matrix_cols, 4) // 4
    params = [
      ctypes.c_uint64(x.ptr.value),
      ctypes.c_uint64(out.ptr.value),
      ctypes.c_uint32(x.matrix_rows),
      ctypes.c_uint32(padded_rows),
      ctypes.c_uint32(x.matrix_cols),
    ]
    self._launch(PAD_ROWS_KERNEL, params, ((total_vec + 255) // 256, 1, 1), (256, 1, 1))
    return out

  def matmul_tc(self, x: DeviceTensor, weight: DeviceTensor, bias: DeviceTensor | None, *, gelu: bool = False) -> DeviceTensor:
    padded_rows = _round_up(x.matrix_rows, 16)
    a_padded = self._pad_rows(x, padded_rows) if padded_rows != x.matrix_rows else x
    out = self.empty((*x.shape[:-1], weight.shape[0]), torch.float32, alloc_shape=(padded_rows, weight.shape[0]))
    params = [
      ctypes.c_uint64(a_padded.ptr.value),
      ctypes.c_uint64(weight.ptr.value),
      ctypes.c_uint64(0 if bias is None else bias.ptr.value),
      ctypes.c_uint64(out.ptr.value),
      ctypes.c_uint32(padded_rows),
      ctypes.c_uint32(weight.shape[0]),
      ctypes.c_uint32(weight.shape[1]),
      ctypes.c_uint32(int(gelu)),
    ]
    self._launch(
      MATMUL_TC_KERNEL,
      params,
      (padded_rows // 16, (weight.shape[0] + 16 * TC_BLOCK_WARPS - 1) // (16 * TC_BLOCK_WARPS), 1),
      (32, TC_BLOCK_WARPS, 1),
    )
    if a_padded is not x:
      self.defer_free(a_padded)
    return out
