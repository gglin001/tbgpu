from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import time

import torch

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  from examples.gpt2.model import GPT, GPTConfig, TBGPUGPT
else:
  from .model import GPT, GPTConfig, TBGPUGPT


def _assert_close(name: str, got: torch.Tensor, expected: torch.Tensor, tolerance: float):
  max_diff = (got - expected).abs().max().item() if got.numel() else 0.0
  if max_diff > tolerance:
    raise AssertionError(f"{name} mismatch: max_abs_diff={max_diff}, tolerance={tolerance}")
  return max_diff


def _run_synthetic_case(config: GPTConfig, batch_size: int, seq_len: int, seed: int, gen_tokens: int):
  torch.manual_seed(seed)
  model = GPT(config).eval()
  tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)

  with torch.no_grad():
    ref_logits, _ = model(tokens)

  runner = TBGPUGPT(model)
  try:
    with torch.no_grad():
      got_logits, _ = runner(tokens)
    logits_diff = _assert_close("synthetic logits", got_logits, ref_logits, tolerance=3e-3)

    input_tokens = tokens[:, : max(1, seq_len - gen_tokens)]
    torch.manual_seed(seed + 1)
    ref_generated = model.generate(input_tokens.clone(), gen_tokens, temperature=0.0)
    got_generated = runner.generate(input_tokens.clone(), gen_tokens, temperature=0.0)
    token_diff = _assert_close("synthetic generated tokens", got_generated.float(), ref_generated.float(), tolerance=0.0)
  finally:
    runner.close()

  print(
    "synthetic gpt2 ok, "
    f"block_size={config.block_size}, vocab_size={config.vocab_size}, "
    f"layers={config.n_layer}, heads={config.n_head}, embd={config.n_embd}, "
    f"batch_size={batch_size}, seq_len={seq_len}, "
    f"max_logits_diff={logits_diff:.3e}, token_diff={token_diff:.3e}"
  )


def _load_tiktoken():
  try:
    import tiktoken
  except ImportError as exc:
    raise RuntimeError("tiktoken is required for the pretrained GPT-2 test") from exc
  os.environ.setdefault("TIKTOKEN_CACHE_DIR", "debug_agent/data-gym-cache")
  return tiktoken.get_encoding("gpt2")


def _run_real_weight_case(state_dict_path: Path, prompt: str, gen_tokens: int):
  if not state_dict_path.exists():
    raise FileNotFoundError(f"missing GPT-2 weight file: {state_dict_path}")

  tokenizer = _load_tiktoken()
  model = GPT.from_pretrained("gpt2", state_dict_path=state_dict_path).eval()
  tokens = torch.tensor(tokenizer.encode(prompt, allowed_special={"<|endoftext|>"}), dtype=torch.long).unsqueeze(0)

  with torch.no_grad():
    t0 = time.perf_counter()
    ref_logits, _ = model(tokens)
    cpu_logits_ms = (time.perf_counter() - t0) * 1000.0

  runner = TBGPUGPT(model)
  try:
    with torch.no_grad():
      t0 = time.perf_counter()
      got_logits, _ = runner(tokens)
      gpu_logits_ms = (time.perf_counter() - t0) * 1000.0
    logits_diff = _assert_close("pretrained logits", got_logits, ref_logits, tolerance=2e-2)

    with torch.no_grad():
      t0 = time.perf_counter()
      ref_generated = model.generate(tokens.clone(), gen_tokens, temperature=0.0)
      cpu_gen_ms = (time.perf_counter() - t0) * 1000.0

      t0 = time.perf_counter()
      got_generated = runner.generate(tokens.clone(), gen_tokens, temperature=0.0)
      gpu_gen_ms = (time.perf_counter() - t0) * 1000.0

    token_diff = _assert_close("pretrained generated tokens", got_generated.float(), ref_generated.float(), tolerance=0.0)
  finally:
    runner.close()

  print(
    "pretrained gpt2 ok, "
    f"prompt_tokens={tokens.numel()}, gen_tokens={gen_tokens}, "
    f"max_logits_diff={logits_diff:.3e}, token_diff={token_diff:.3e}, "
    f"cpu_logits_ms={cpu_logits_ms:.2f}, gpu_logits_ms={gpu_logits_ms:.2f}, "
    f"cpu_gen_ms={cpu_gen_ms:.2f}, gpu_gen_ms={gpu_gen_ms:.2f}"
  )
  print("generated text:")
  print(tokenizer.decode(got_generated[0].tolist()))


def main():
  parser = argparse.ArgumentParser(description="Verify GPT-2 inference with GPU-resident fused TBGPU kernels")
  parser.add_argument("--verify-suite", action="store_true", help="run the small synthetic correctness suite")
  parser.add_argument(
    "--state-dict",
    type=Path,
    default=Path(__file__).resolve().parents[2] / "debug_agent" / "pytorch_model.bin",
    help="path to a GPT-2 pytorch_model.bin checkpoint",
  )
  parser.add_argument("--real-weight", action="store_true", help="run the real GPT-2 checkpoint test")
  parser.add_argument("--prompt", type=str, default="The answer to life is")
  parser.add_argument("--count", type=int, default=4, help="number of greedy tokens to generate in the real-weight test")
  args = parser.parse_args()

  if args.verify_suite:
    _run_synthetic_case(GPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=True), 2, 11, 7, 3)
    _run_synthetic_case(GPTConfig(block_size=19, vocab_size=80, n_layer=3, n_head=4, n_embd=32, dropout=0.0, bias=True), 1, 13, 11, 4)

  if args.real_weight:
    _run_real_weight_case(args.state_dict, args.prompt, args.count)

  if not args.verify_suite and not args.real_weight:
    parser.error("choose at least one of --verify-suite or --real-weight")


if __name__ == "__main__":
  main()
