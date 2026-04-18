from __future__ import annotations

import contextlib
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from .runtime import LinearWeights, TBGPUContext


class LayerNorm(nn.Module):
  def __init__(self, ndim: int, bias: bool, eps: float = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
  def __init__(self, config: "GPTConfig"):
    super().__init__()
    if config.n_embd % config.n_head != 0:
      raise ValueError("n_embd must be divisible by n_head")
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    self.n_head = config.n_head
    self.n_embd = config.n_embd

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, channels = x.shape
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    head_dim = channels // self.n_head
    q = q.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
    return self.c_proj(y)


class MLP(nn.Module):
  def __init__(self, config: "GPTConfig"):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.c_proj(F.gelu(self.c_fc(x), approximate="none"))


class Block(nn.Module):
  def __init__(self, config: "GPTConfig"):
    super().__init__()
    self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
    self.mlp = MLP(config)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


@dataclass
class GPTConfig:
  block_size: int = 1024
  vocab_size: int = 50304
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768
  dropout: float = 0.0
  bias: bool = True


MODEL_CONFIGS = {
  "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
  "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
  "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
  "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
}


def _round_up(value: int, alignment: int) -> int:
  return ((value + alignment - 1) // alignment) * alignment


class GPT(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict({
      "wte": nn.Embedding(config.vocab_size, config.n_embd),
      "wpe": nn.Embedding(config.block_size, config.n_embd),
      "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
      "ln_f": LayerNorm(config.n_embd, bias=config.bias),
    })
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight

    self.apply(self._init_weights)
    for name, param in self.named_parameters():
      if name.endswith("c_proj.weight"):
        torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

  def _init_weights(self, module: nn.Module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    _, seq_len = idx.shape
    if seq_len > self.config.block_size:
      raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")

    pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
    x = self.transformer.wte(idx) + self.transformer.wpe(pos)
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)

    if targets is None:
      logits = self.lm_head(x[:, [-1], :])
      return logits, None

    logits = self.lm_head(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return logits, loss

  @classmethod
  def from_pretrained(cls, model_type: str, *, cache_dir: str | Path | None = None, state_dict_path: str | Path | None = None) -> "GPT":
    if model_type not in MODEL_CONFIGS:
      raise ValueError(f"unsupported model_type: {model_type}")
    config = GPTConfig(vocab_size=50257, block_size=1024, bias=True, dropout=0.0, **MODEL_CONFIGS[model_type])
    model = cls(config)
    state_dict = _load_external_state_dict(model_type=model_type, cache_dir=cache_dir, state_dict_path=state_dict_path)
    _load_model_state(model, state_dict)
    model.eval()
    return model

  @torch.no_grad()
  def generate(self, idx: torch.Tensor, max_new_tokens: int, *, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    idx = idx.to(dtype=torch.long, device=idx.device).contiguous()
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]
      if temperature <= 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
      else:
        logits = logits / temperature
        if top_k is not None:
          top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
          logits = torch.where(logits < top_values[:, [-1]], torch.full_like(logits, float("-inf")), logits)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
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
      if model_key == "lm_head.weight" and "lm_head.weight" not in normalized and "transformer.wte.weight" in normalized:
        source_value = normalized["transformer.wte.weight"]
      else:
        if model_key not in normalized:
          raise KeyError(f"missing key in external state dict: {model_key}")
        source_value = normalized[model_key]

      if source_value.shape == model_value.shape:
        model_value.copy_(source_value.to(dtype=model_value.dtype))
        continue

      if any(model_key.endswith(suffix) for suffix in transposed_suffixes) and source_value.shape[::-1] == model_value.shape:
        model_value.copy_(source_value.t().to(dtype=model_value.dtype))
        continue

      raise ValueError(f"shape mismatch for {model_key}: expected {tuple(model_value.shape)}, got {tuple(source_value.shape)}")

  model.load_state_dict(model_state)


def _load_external_state_dict(
  *,
  model_type: str,
  cache_dir: str | Path | None = None,
  state_dict_path: str | Path | None = None,
) -> dict[str, Any]:
  if state_dict_path is not None:
    return torch.load(Path(state_dict_path), map_location="cpu")

  cache_root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "tbgpu" / "gpt2"
  cache_root.mkdir(parents=True, exist_ok=True)
  target = cache_root / f"{model_type}.pytorch_model.bin"
  if not target.exists():
    url = f"https://huggingface.co/{model_type}/resolve/main/pytorch_model.bin"
    urllib.request.urlretrieve(url, target)
  return torch.load(target, map_location="cpu")


@dataclass
class _GPUBlock:
  ln_1_weight: Any
  ln_1_bias: Any
  attn_qkv: LinearWeights
  attn_proj: LinearWeights
  ln_2_weight: Any
  ln_2_bias: Any
  mlp_fc: LinearWeights
  mlp_proj: LinearWeights
  num_heads: int
  head_dim: int


class TBGPUGPT:
  def __init__(self, model: GPT, *, device_ordinal: int = 0):
    self.model = model.eval()
    self.runtime = TBGPUContext(device_ordinal=device_ordinal)
    self.config = model.config

    self.wte = self.runtime.upload(model.transformer.wte.weight.detach().to(dtype=torch.float32, device="cpu"), static=True)
    self.wpe = self.runtime.upload(model.transformer.wpe.weight.detach().to(dtype=torch.float32, device="cpu"), static=True)
    self.ln_f_weight = self.runtime.upload(model.transformer.ln_f.weight.detach().to(dtype=torch.float32, device="cpu"), static=True)
    self.ln_f_bias = self.runtime.upload(model.transformer.ln_f.bias.detach().to(dtype=torch.float32, device="cpu"), static=True)
    self.lm_head_vocab_size = model.lm_head.weight.shape[0]
    self.lm_head_padded_vocab_size = _round_up(self.lm_head_vocab_size, 16)
    lm_head_weight = model.lm_head.weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if self.lm_head_padded_vocab_size != self.lm_head_vocab_size:
      padded = torch.zeros((self.lm_head_padded_vocab_size, lm_head_weight.shape[1]), dtype=lm_head_weight.dtype)
      padded[: self.lm_head_vocab_size] = lm_head_weight
      lm_head_weight = padded
    self.lm_head = self.runtime.upload_linear_weights(lm_head_weight, None, prefer_tensor_cores=True)

    self.blocks = []
    for block in model.transformer.h:
      self.blocks.append(
        _GPUBlock(
          ln_1_weight=self.runtime.upload(block.ln_1.weight.detach().to(dtype=torch.float32, device="cpu"), static=True),
          ln_1_bias=self.runtime.upload(block.ln_1.bias.detach().to(dtype=torch.float32, device="cpu"), static=True),
          attn_qkv=self.runtime.upload_linear_weights(block.attn.c_attn.weight, block.attn.c_attn.bias, prefer_tensor_cores=True),
          attn_proj=self.runtime.upload_linear_weights(block.attn.c_proj.weight, block.attn.c_proj.bias, prefer_tensor_cores=True),
          ln_2_weight=self.runtime.upload(block.ln_2.weight.detach().to(dtype=torch.float32, device="cpu"), static=True),
          ln_2_bias=self.runtime.upload(block.ln_2.bias.detach().to(dtype=torch.float32, device="cpu"), static=True),
          mlp_fc=self.runtime.upload_linear_weights(block.mlp.c_fc.weight, block.mlp.c_fc.bias, prefer_tensor_cores=True),
          mlp_proj=self.runtime.upload_linear_weights(block.mlp.c_proj.weight, block.mlp.c_proj.bias, prefer_tensor_cores=True),
          num_heads=block.attn.n_head,
          head_dim=block.attn.n_embd // block.attn.n_head,
        )
      )

  def close(self):
    self.runtime.close()

  def __del__(self):
    with contextlib.suppress(Exception):
      self.close()

  def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if targets is not None:
      raise NotImplementedError("TBGPUGPT currently implements inference-only forward")

    idx = idx.to(dtype=torch.long, device="cpu").contiguous()
    batch_size, seq_len = idx.shape
    if seq_len > self.config.block_size:
      raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")

    logits_dev = None
    try:
      x = self.runtime.encoder_forward(idx, self.wte, self.wpe)
      ln_1 = self.runtime.layernorm(x, self.blocks[0].ln_1_weight, self.blocks[0].ln_1_bias)

      for block_index, block in enumerate(self.blocks):
        qkv = self.runtime.matmul_linear(ln_1, block.attn_qkv, prefer_tensor_cores=True)
        self.runtime.defer_free(ln_1)
        attn_proj = self.runtime.flash_attention_qkv_proj_fused(
          qkv,
          block.attn_proj,
          batch_size=batch_size,
          seq_len=seq_len,
          num_heads=block.num_heads,
          causal=True,
        )
        self.runtime.defer_free(qkv)

        residual_1, ln_2 = self.runtime.fused_residual_layernorm(x, attn_proj, block.ln_2_weight, block.ln_2_bias)
        self.runtime.defer_free(x)
        self.runtime.defer_free(attn_proj)

        mlp_hidden = self.runtime.matmul_linear(ln_2, block.mlp_fc, prefer_tensor_cores=True, gelu=True)
        self.runtime.defer_free(ln_2)
        mlp_proj = self.runtime.matmul_linear(mlp_hidden, block.mlp_proj, prefer_tensor_cores=True)
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
          ln_1 = None

      x = self.runtime.layernorm(x, self.ln_f_weight, self.ln_f_bias)
      last = self.runtime.take_last_token(x, batch_size=batch_size, seq_len=seq_len)
      self.runtime.defer_free(x)
      logits_dev = self.runtime.matmul_linear(last, self.lm_head, prefer_tensor_cores=True)
      self.runtime.defer_free(last)
      self.runtime.synchronize()
      logits = self.runtime.download(logits_dev).view(batch_size, 1, self.lm_head_padded_vocab_size)[..., : self.lm_head_vocab_size]
      self.runtime.free(logits_dev)
      logits_dev = None
      return logits, None
    finally:
      with contextlib.suppress(Exception):
        self.runtime.synchronize()
      if logits_dev is not None:
        with contextlib.suppress(Exception):
          self.runtime.free(logits_dev)
      self.runtime.flush_deferred_frees()

  __call__ = forward

  @torch.no_grad()
  def generate(self, idx: torch.Tensor, max_new_tokens: int, *, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    idx = idx.to(dtype=torch.long, device="cpu").contiguous()
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]
      if temperature <= 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
      else:
        logits = logits / temperature
        if top_k is not None:
          top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
          logits = torch.where(logits < top_values[:, [-1]], torch.full_like(logits, float("-inf")), logits)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, next_token.to(dtype=torch.long, device="cpu")), dim=1)
    return idx
