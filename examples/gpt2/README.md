# gpt2

Download the `gpt2` weights cache manually:

```bash
curl -L https://huggingface.co/gpt2/resolve/main/pytorch_model.bin \
  -o debug_agent/pytorch_model.bin
```

Download the `tiktoken` cache manually:

```bash
CACHE_DIR=debug_agent/data-gym-cache
mkdir -p "$CACHE_DIR"
curl -L "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe" \
  -o "$CACHE_DIR/6d1cbeee0f20b3d9449abfede4726ed8212e3aee"
curl -L "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json" \
  -o "$CACHE_DIR/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6"
```
