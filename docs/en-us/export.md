# Export & Inference Department (en-US)

## 1. Department Overview

This department covers two post-training operations:

- `export.py` exports the pre and post models to ExecuTorch `.pte` programs.
- `demo.py` runs greedy or beam-search inference from a PyTorch checkpoint.

## 2. `export.py` — ExecuTorch Export

The exporter loads `pre_model/` and `post_model/` from a checkpoint directory.
It produces a multi-method `pre_model.pte`, a `post_model.pte`, and selective
build manifests containing the operators and dtype variants required by both
programs.

The pre program contains separate methods for prefix decoding, conditional
decoding, and cross-attention KV projection. Export uses dynamic sequence
dimensions and optionally applies XNNPACK dynamic per-channel quantization.

Run the current script with `python export.py`. Its checkpoint, output path,
model metadata, beam width, dtype, and quantization mode are currently module
constants near the top of the file.

## 3. `demo.py` — Inference Demo

The demo accepts the checkpoint, vocabulary configuration, context text,
pinyin syllables, beam size, device, and dtype as command-line options:

```bash
python demo.py \
  --checkpoint checkpoints/<run>/final_model \
  --text "" \
  --pinyin ni hao \
  --beam-size 3
```

It first runs greedy decoding and then, when the beam size is greater than one,
prints N-best beam-search results and execution time.
