from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

DEBUG_AGENT_DIR = Path(__file__).resolve().parents[2] / "debug_agent"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(DEBUG_AGENT_DIR / "data-gym-cache"))

import tiktoken

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  from examples.gpt2.model import GPT, TBGPUGPT
else:
  from .model import GPT, TBGPUGPT


def main():
  parser = argparse.ArgumentParser(description="Run GPT-2 inference with default debug_agent weights and tiktoken cache")
  parser.add_argument("--prompt", type=str, default="The answer to life is")
  parser.add_argument("--count", type=int, default=16, help="number of new tokens to generate")
  parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature, 0 uses greedy decode")
  parser.add_argument("--top-k", type=int, help="optional top-k sampling cutoff")
  args = parser.parse_args()

  tokenizer = tiktoken.get_encoding("gpt2")
  input_tokens = torch.tensor(tokenizer.encode(args.prompt, allowed_special={"<|endoftext|>"}), dtype=torch.long).unsqueeze(0)
  model = GPT.load()
  runner = TBGPUGPT(model)
  try:
    output = runner.generate(input_tokens, args.count, temperature=args.temperature, top_k=args.top_k)
  finally:
    runner.close()

  token_ids = output[0].tolist()
  print("tokens:", ",".join(str(token) for token in token_ids))
  print("text:")
  print(tokenizer.decode(token_ids))


if __name__ == "__main__":
  main()
