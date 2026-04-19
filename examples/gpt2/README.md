# gpt2

This directory only supports the standard GPT-2 124M configuration.

Required local files:

- `debug_agent/pytorch_model.bin`
- `debug_agent/data-gym-cache/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6`
- `debug_agent/data-gym-cache/6d1cbeee0f20b3d9449abfede4726ed8212e3aee`

Download cache manually:

```bash
curl -L https://huggingface.co/gpt2/resolve/main/pytorch_model.bin \
  -o debug_agent/pytorch_model.bin

CACHE_DIR=debug_agent/data-gym-cache
mkdir -p "$CACHE_DIR"
curl -L "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe" \
  -o "$CACHE_DIR/6d1cbeee0f20b3d9449abfede4726ed8212e3aee"
curl -L "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json" \
  -o "$CACHE_DIR/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6"
```

Run a sample:

```bash
python examples/gpt2/sample.py --prompt "The answer to life is" --count 4
```

Compare against the torch-cpu baseline:

```bash
python examples/gpt2/test_inference.py --prompt "The answer to life is" --count 4
```
