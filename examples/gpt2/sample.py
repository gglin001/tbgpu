from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  from examples.gpt2.model import GPT, TBGPUGPT
else:
  from .model import GPT, TBGPUGPT


def _parse_tokens(value: str) -> torch.Tensor:
  tokens = [int(part.strip()) for part in value.split(",") if part.strip()]
  if not tokens:
    raise ValueError("token list is empty")
  return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def _load_tiktoken():
  try:
    import tiktoken
  except ImportError:
    return None
  return tiktoken.get_encoding("gpt2")


def main():
  parser = argparse.ArgumentParser(description="Run GPT-2 inference with torch-cpu orchestration and TBGPU CUDA kernels")
  parser.add_argument("--model-type", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
  parser.add_argument("--state-dict", type=Path, help="optional local pytorch_model.bin path")
  parser.add_argument("--cache-dir", type=Path, help="optional cache directory for downloaded HuggingFace weights")
  parser.add_argument("--tokens", type=str, help="comma-separated GPT-2 token ids")
  parser.add_argument("--prompt", type=str, help="text prompt, requires tiktoken")
  parser.add_argument("--count", type=int, default=16, help="number of new tokens to generate")
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--top-k", type=int, help="optional top-k sampling")
  parser.add_argument("--device-ordinal", type=int, default=0)
  args = parser.parse_args()

  if (args.tokens is None) == (args.prompt is None):
    raise SystemExit("provide exactly one of --tokens or --prompt")

  tokenizer = None
  if args.prompt is not None:
    tokenizer = _load_tiktoken()
    if tokenizer is None:
      raise SystemExit("tiktoken is required for --prompt. Use --tokens or install tiktoken.")
    input_tokens = torch.tensor(tokenizer.encode(args.prompt, allowed_special={"<|endoftext|>"}), dtype=torch.long).unsqueeze(0)
  else:
    input_tokens = _parse_tokens(args.tokens)

  model = GPT.from_pretrained(args.model_type, cache_dir=args.cache_dir, state_dict_path=args.state_dict)
  runner = TBGPUGPT(model, device_ordinal=args.device_ordinal)
  try:
    output = runner.generate(input_tokens, args.count, temperature=args.temperature, top_k=args.top_k)
  finally:
    runner.close()

  token_ids = output[0].tolist()
  print("tokens:", ",".join(str(token) for token in token_ids))
  if tokenizer is not None:
    print("text:")
    print(tokenizer.decode(token_ids))


if __name__ == "__main__":
  main()
