from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from .runtime import DeviceTensor, TBGPUContext

GPT2_BLOCK_SIZE = 1024
GPT2_VOCAB_SIZE = 50257
GPT2_N_LAYER = 12
GPT2_N_HEAD = 12
GPT2_N_EMBD = 768

GPT2_DEBUG_AGENT_DIR = Path(__file__).resolve().parents[2] / "debug_agent"
GPT2_WEIGHTS_PATH = GPT2_DEBUG_AGENT_DIR / "pytorch_model.bin"
TIKTOKEN_CACHE_DIR = GPT2_DEBUG_AGENT_DIR / "data-gym-cache"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))


def _round_up(value: int, alignment: int) -> int:
  return ((value + alignment - 1) // alignment) * alignment


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
  if temperature <= 0.0:
    return torch.argmax(logits, dim=-1, keepdim=True)
  logits = logits / temperature
  if top_k is not None:
    top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits = torch.where(logits < top_values[:, [-1]], torch.full_like(logits, float("-inf")), logits)
  probs = F.softmax(logits, dim=-1)
  return torch.multinomial(probs, num_samples=1)


class LayerNorm(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(GPT2_N_EMBD))
    self.bias = nn.Parameter(torch.zeros(GPT2_N_EMBD))
    self.eps = 1e-5

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.c_attn = nn.Linear(GPT2_N_EMBD, 3 * GPT2_N_EMBD, bias=True)
    self.c_proj = nn.Linear(GPT2_N_EMBD, GPT2_N_EMBD, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, channels = x.shape
    q, k, v = self.c_attn(x).split(GPT2_N_EMBD, dim=2)
    head_dim = channels // GPT2_N_HEAD
    q = q.view(batch_size, seq_len, GPT2_N_HEAD, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, GPT2_N_HEAD, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, GPT2_N_HEAD, head_dim).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
    return self.c_proj(y)


class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.c_fc = nn.Linear(GPT2_N_EMBD, 4 * GPT2_N_EMBD, bias=True)
    self.c_proj = nn.Linear(4 * GPT2_N_EMBD, GPT2_N_EMBD, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.c_proj(F.gelu(self.c_fc(x), approximate="none"))


class Block(nn.Module):
  def __init__(self):
    super().__init__()
    self.ln_1 = LayerNorm()
    self.attn = CausalSelfAttention()
    self.ln_2 = LayerNorm()
    self.mlp = MLP()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.transformer = nn.ModuleDict({
      "wte": nn.Embedding(GPT2_VOCAB_SIZE, GPT2_N_EMBD),
      "wpe": nn.Embedding(GPT2_BLOCK_SIZE, GPT2_N_EMBD),
      "h": nn.ModuleList([Block() for _ in range(GPT2_N_LAYER)]),
      "ln_f": LayerNorm(),
    })
    self.lm_head = nn.Linear(GPT2_N_EMBD, GPT2_VOCAB_SIZE, bias=False)
    self.transformer.wte.weight = self.lm_head.weight

  def forward(self, idx: torch.Tensor) -> torch.Tensor:
    _, seq_len = idx.shape
    if seq_len > GPT2_BLOCK_SIZE:
      raise ValueError(f"sequence length {seq_len} exceeds block size {GPT2_BLOCK_SIZE}")

    pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
    x = self.transformer.wte(idx) + self.transformer.wpe(pos)
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)
    return self.lm_head(x[:, [-1], :])

  @classmethod
  def load(cls) -> "GPT":
    if not GPT2_WEIGHTS_PATH.exists():
      raise FileNotFoundError(f"missing GPT-2 weights: {GPT2_WEIGHTS_PATH}")
    model = cls()
    _load_model_state(model, torch.load(GPT2_WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model

  @torch.no_grad()
  def generate(self, idx: torch.Tensor, max_new_tokens: int, *, temperature: float = 0.0, top_k: int | None = None) -> torch.Tensor:
    idx = idx.to(dtype=torch.long, device=idx.device).contiguous()
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <= GPT2_BLOCK_SIZE else idx[:, -GPT2_BLOCK_SIZE:]
      logits = self(idx_cond)[:, -1, :]
      next_token = _sample_next_token(logits, temperature, top_k)
      idx = torch.cat((idx, next_token), dim=1)
    return idx


def _normalize_external_key(key: str) -> str:
  if key.startswith("transformer."):
    return key
  if key.startswith("h.") or key.startswith("wte.") or key.startswith("wpe.") or key.startswith("ln_f."):
    return f"transformer.{key}"
  return key


def _load_model_state(model: GPT, external_state: dict[str, torch.Tensor]):
  normalized = {_normalize_external_key(key): value for key, value in external_state.items()}
  model_state = model.state_dict()
  transposed_suffixes = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
  )

  with torch.no_grad():
    for model_key, model_value in model_state.items():
      source_value = normalized["transformer.wte.weight"] if model_key == "lm_head.weight" else normalized[model_key]

      if source_value.shape == model_value.shape:
        model_value.copy_(source_value.to(dtype=model_value.dtype))
        continue

      if any(model_key.endswith(suffix) for suffix in transposed_suffixes) and source_value.shape[::-1] == model_value.shape:
        model_value.copy_(source_value.t().to(dtype=model_value.dtype))
        continue

      raise ValueError(f"shape mismatch for {model_key}: expected {tuple(model_value.shape)}, got {tuple(source_value.shape)}")

  model.load_state_dict(model_state)


@dataclass
class _GPUBlock:
  ln_1_weight: DeviceTensor
  ln_1_bias: DeviceTensor
  attn_qkv_weight: DeviceTensor
  attn_qkv_bias: DeviceTensor
  attn_proj_weight_t: DeviceTensor
  attn_proj_bias: DeviceTensor
  ln_2_weight: DeviceTensor
  ln_2_bias: DeviceTensor
  mlp_fc_weight: DeviceTensor
  mlp_fc_bias: DeviceTensor
  mlp_proj_weight: DeviceTensor
  mlp_proj_bias: DeviceTensor


class TBGPUGPT:
  def __init__(self, model: GPT):
    self.model = model.eval()
    self.runtime = TBGPUContext()

    self.wte = self.runtime.upload(model.transformer.wte.weight.detach().to(dtype=torch.float32, device="cpu"), static=True)
    self.wpe = self.runtime.upload(model.transformer.wpe.weight.detach().to(dtype=torch.float32, device="cpu"), static=True)
    self.ln_f_weight = self.runtime.upload(model.transformer.ln_f.weight.detach().to(dtype=torch.float32, device="cpu"), static=True)
    self.ln_f_bias = self.runtime.upload(model.transformer.ln_f.bias.detach().to(dtype=torch.float32, device="cpu"), static=True)

    lm_head_weight = model.lm_head.weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
    self.lm_head_vocab_size = lm_head_weight.shape[0]
    self.lm_head_padded_vocab_size = _round_up(self.lm_head_vocab_size, 16)
    if self.lm_head_padded_vocab_size != self.lm_head_vocab_size:
      padded = torch.zeros((self.lm_head_padded_vocab_size, lm_head_weight.shape[1]), dtype=lm_head_weight.dtype)
      padded[: self.lm_head_vocab_size] = lm_head_weight
      lm_head_weight = padded
    self.lm_head_weight = self.runtime.upload(lm_head_weight, dtype=torch.float32, static=True)

    self.blocks = []
    for block in model.transformer.h:
      self.blocks.append(
        _GPUBlock(
          ln_1_weight=self.runtime.upload(block.ln_1.weight.detach().to(dtype=torch.float32, device="cpu"), static=True),
          ln_1_bias=self.runtime.upload(block.ln_1.bias.detach().to(dtype=torch.float32, device="cpu"), static=True),
          attn_qkv_weight=self.runtime.upload(block.attn.c_attn.weight.detach().to(dtype=torch.float32, device="cpu"), static=True),
          attn_qkv_bias=self.runtime.upload(block.attn.c_attn.bias.detach().to(dtype=torch.float32, device="cpu"), static=True),
          attn_proj_weight_t=self.runtime.upload(
            block.attn.c_proj.weight.detach().t().contiguous().to(dtype=torch.float32, device="cpu"), static=True
          ),
          attn_proj_bias=self.runtime.upload(block.attn.c_proj.bias.detach().to(dtype=torch.float32, device="cpu"), static=True),
          ln_2_weight=self.runtime.upload(block.ln_2.weight.detach().to(dtype=torch.float32, device="cpu"), static=True),
          ln_2_bias=self.runtime.upload(block.ln_2.bias.detach().to(dtype=torch.float32, device="cpu"), static=True),
          mlp_fc_weight=self.runtime.upload(block.mlp.c_fc.weight.detach().to(dtype=torch.float32, device="cpu"), static=True),
          mlp_fc_bias=self.runtime.upload(block.mlp.c_fc.bias.detach().to(dtype=torch.float32, device="cpu"), static=True),
          mlp_proj_weight=self.runtime.upload(block.mlp.c_proj.weight.detach().to(dtype=torch.float32, device="cpu"), static=True),
          mlp_proj_bias=self.runtime.upload(block.mlp.c_proj.bias.detach().to(dtype=torch.float32, device="cpu"), static=True),
        )
      )
    self.key_caches = [self.runtime.empty((GPT2_BLOCK_SIZE, GPT2_N_EMBD), torch.float32) for _ in range(GPT2_N_LAYER)]
    self.value_caches = [self.runtime.empty((GPT2_BLOCK_SIZE, GPT2_N_EMBD), torch.float32) for _ in range(GPT2_N_LAYER)]

  def close(self):
    self.runtime.close()

  def __del__(self):
    with contextlib.suppress(Exception):
      self.close()

  def forward(self, idx: torch.Tensor) -> torch.Tensor:
    return self._prefill(idx)

  def _prefill(self, idx: torch.Tensor) -> torch.Tensor:
    idx = idx.to(dtype=torch.long, device="cpu").contiguous()
    batch_size, seq_len = idx.shape
    if seq_len > GPT2_BLOCK_SIZE:
      raise ValueError(f"sequence length {seq_len} exceeds block size {GPT2_BLOCK_SIZE}")

    logits_dev = None
    try:
      x = self.runtime.encoder_forward(idx, self.wte, self.wpe)
      ln_1 = self.runtime.layernorm(x, self.blocks[0].ln_1_weight, self.blocks[0].ln_1_bias)

      for block_index, block in enumerate(self.blocks):
        qkv = self.runtime.matmul_tc(ln_1, block.attn_qkv_weight, block.attn_qkv_bias)
        self.runtime.defer_free(ln_1)
        self.runtime.write_kv_cache(qkv, self.key_caches[block_index], self.value_caches[block_index], start_pos=0, rows=seq_len)
        attn_proj = self.runtime.flash_attention_qkv_proj_fused(
          qkv,
          block.attn_proj_weight_t,
          block.attn_proj_bias,
          batch_size=batch_size,
          seq_len=seq_len,
        )
        self.runtime.defer_free(qkv)

        residual_1, ln_2 = self.runtime.fused_residual_layernorm(x, attn_proj, block.ln_2_weight, block.ln_2_bias)
        self.runtime.defer_free(x)
        self.runtime.defer_free(attn_proj)

        mlp_hidden = self.runtime.matmul_tc(ln_2, block.mlp_fc_weight, block.mlp_fc_bias, gelu=True)
        self.runtime.defer_free(ln_2)
        mlp_proj = self.runtime.matmul_tc(mlp_hidden, block.mlp_proj_weight, block.mlp_proj_bias)
        self.runtime.defer_free(mlp_hidden)

        if block_index + 1 < len(self.blocks):
          next_block = self.blocks[block_index + 1]
          x, ln_1 = self.runtime.fused_residual_layernorm(residual_1, mlp_proj, next_block.ln_1_weight, next_block.ln_1_bias)
          self.runtime.defer_free(residual_1)
          self.runtime.defer_free(mlp_proj)
        else:
          x = self.runtime.residual_add(residual_1, mlp_proj)
          self.runtime.defer_free(residual_1)
          self.runtime.defer_free(mlp_proj)

      x = self.runtime.layernorm(x, self.ln_f_weight, self.ln_f_bias)
      last = self.runtime.take_last_token(x, batch_size=batch_size, seq_len=seq_len)
      self.runtime.defer_free(x)
      logits_dev = self.runtime.matmul_tc(last, self.lm_head_weight, None)
      self.runtime.defer_free(last)
      self.runtime.synchronize()
      logits = self.runtime.download(logits_dev).view(batch_size, 1, self.lm_head_padded_vocab_size)[..., : self.lm_head_vocab_size]
      self.runtime.free(logits_dev)
      logits_dev = None
      return logits
    finally:
      with contextlib.suppress(Exception):
        self.runtime.synchronize()
      if logits_dev is not None:
        with contextlib.suppress(Exception):
          self.runtime.free(logits_dev)
      self.runtime.flush_deferred_frees()

  def _decode_step(self, token: int, position: int) -> torch.Tensor:
    logits_dev = None
    try:
      x = self.runtime.encoder_forward_step(token, position, self.wte, self.wpe)
      ln_1 = self.runtime.layernorm(x, self.blocks[0].ln_1_weight, self.blocks[0].ln_1_bias)

      for block_index, block in enumerate(self.blocks):
        qkv = self.runtime.matmul_tc(ln_1, block.attn_qkv_weight, block.attn_qkv_bias)
        self.runtime.defer_free(ln_1)
        self.runtime.write_kv_cache(qkv, self.key_caches[block_index], self.value_caches[block_index], start_pos=position, rows=1)
        attn_proj = self.runtime.flash_attention_qkv_proj_decode(
          qkv,
          block.attn_proj_weight_t,
          block.attn_proj_bias,
          self.key_caches[block_index],
          self.value_caches[block_index],
          cache_len=position + 1,
        )
        self.runtime.defer_free(qkv)

        residual_1, ln_2 = self.runtime.fused_residual_layernorm(x, attn_proj, block.ln_2_weight, block.ln_2_bias)
        self.runtime.defer_free(x)
        self.runtime.defer_free(attn_proj)

        mlp_hidden = self.runtime.matmul_tc(ln_2, block.mlp_fc_weight, block.mlp_fc_bias, gelu=True)
        self.runtime.defer_free(ln_2)
        mlp_proj = self.runtime.matmul_tc(mlp_hidden, block.mlp_proj_weight, block.mlp_proj_bias)
        self.runtime.defer_free(mlp_hidden)

        if block_index + 1 < len(self.blocks):
          next_block = self.blocks[block_index + 1]
          x, ln_1 = self.runtime.fused_residual_layernorm(residual_1, mlp_proj, next_block.ln_1_weight, next_block.ln_1_bias)
          self.runtime.defer_free(residual_1)
          self.runtime.defer_free(mlp_proj)
        else:
          x = self.runtime.residual_add(residual_1, mlp_proj)
          self.runtime.defer_free(residual_1)
          self.runtime.defer_free(mlp_proj)

      x = self.runtime.layernorm(x, self.ln_f_weight, self.ln_f_bias)
      logits_dev = self.runtime.matmul_tc(x, self.lm_head_weight, None)
      self.runtime.defer_free(x)
      self.runtime.synchronize()
      logits = self.runtime.download(logits_dev).view(1, 1, self.lm_head_padded_vocab_size)[..., : self.lm_head_vocab_size]
      self.runtime.free(logits_dev)
      logits_dev = None
      return logits
    finally:
      with contextlib.suppress(Exception):
        self.runtime.synchronize()
      if logits_dev is not None:
        with contextlib.suppress(Exception):
          self.runtime.free(logits_dev)
      self.runtime.flush_deferred_frees()

  __call__ = forward

  @torch.no_grad()
  def generate(self, idx: torch.Tensor, max_new_tokens: int, *, temperature: float = 0.0, top_k: int | None = None) -> torch.Tensor:
    idx = idx.to(dtype=torch.long, device="cpu").contiguous()
    if idx.shape[0] != 1:
      raise ValueError("TBGPUGPT.generate with KV-cache only supports batch_size=1")
    if idx.shape[1] + max_new_tokens > GPT2_BLOCK_SIZE:
      raise ValueError(f"prompt length + generated tokens exceed block size {GPT2_BLOCK_SIZE}")

    logits = self._prefill(idx)[:, -1, :]
    for step in range(max_new_tokens):
      next_token = _sample_next_token(logits, temperature, top_k)
      idx = torch.cat((idx, next_token.to(dtype=torch.long, device="cpu")), dim=1)
      if step + 1 < max_new_tokens:
        logits = self._decode_step(int(next_token.item()), idx.shape[1] - 1)[:, -1, :]
    return idx
