from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import torch

DEBUG_AGENT_DIR = Path(__file__).resolve().parents[2] / "debug_agent"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(DEBUG_AGENT_DIR / "data-gym-cache"))

import tiktoken

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  from examples.gpt2.model import GPT, GPT2_WEIGHTS_PATH, TBGPUGPT
else:
  from .model import GPT, GPT2_WEIGHTS_PATH, TBGPUGPT


def _assert_close(name: str, got: torch.Tensor, expected: torch.Tensor, tolerance: float):
  max_diff = (got - expected).abs().max().item() if got.numel() else 0.0
  if max_diff > tolerance:
    raise AssertionError(f"{name} mismatch: max_abs_diff={max_diff}, tolerance={tolerance}")
  return max_diff


def main():
  parser = argparse.ArgumentParser(description="Compare the TBGPU GPT-2 path against the torch-cpu baseline")
  parser.add_argument("--prompt", type=str, default="The answer to life is")
  parser.add_argument("--count", type=int, default=4, help="number of greedy tokens to generate")
  parser.add_argument(
    "--warmup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run a GPU warmup pass before measuring latency",
  )
  args = parser.parse_args()

  if not GPT2_WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"missing GPT-2 weights: {GPT2_WEIGHTS_PATH}")

  tokenizer = tiktoken.get_encoding("gpt2")
  model = GPT.load().eval()
  tokens = torch.tensor(tokenizer.encode(args.prompt, allowed_special={"<|endoftext|>"}), dtype=torch.long).unsqueeze(0)

  with torch.no_grad():
    t0 = time.perf_counter()
    ref_logits = model(tokens)
    cpu_logits_ms = (time.perf_counter() - t0) * 1000.0

  runner = TBGPUGPT(model)
  try:
    if args.warmup:
      runner.warmup(prompt_len=max(tokens.shape[1], 16), decode_steps=max(args.count - 1, 0))
    with torch.no_grad():
      t0 = time.perf_counter()
      got_logits = runner(tokens)
      gpu_logits_ms = (time.perf_counter() - t0) * 1000.0
    logits_diff = _assert_close("gpt2 logits", got_logits, ref_logits, tolerance=1e-2)

    with torch.no_grad():
      t0 = time.perf_counter()
      ref_generated = model.generate(tokens.clone(), args.count)
      cpu_gen_ms = (time.perf_counter() - t0) * 1000.0

      t0 = time.perf_counter()
      got_generated = runner.generate(tokens.clone(), args.count)
      gpu_gen_ms = (time.perf_counter() - t0) * 1000.0

    token_diff = _assert_close("generated tokens", got_generated.float(), ref_generated.float(), tolerance=0.0)
  finally:
    runner.close()

  print(
    "gpt2 baseline compare ok, "
    f"prompt_tokens={tokens.numel()}, gen_tokens={args.count}, "
    f"max_logits_diff={logits_diff:.3e}, token_diff={token_diff:.3e}, "
    f"cpu_logits_ms={cpu_logits_ms:.2f}, gpu_logits_ms={gpu_logits_ms:.2f}, "
    f"cpu_gen_ms={cpu_gen_ms:.2f}, gpu_gen_ms={gpu_gen_ms:.2f}"
  )
  print("generated text:")
  print(tokenizer.decode(got_generated[0].tolist()))


if __name__ == "__main__":
  main()
