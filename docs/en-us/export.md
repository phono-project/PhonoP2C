# Export & Inference Department (en-US)

## 1. Department Overview

This department covers two post-training operations:

- `export/` implements the Hydra task that exports the pre and post models to
  ExecuTorch `.pte` programs.
- `demo.py` runs greedy or beam-search inference from a PyTorch checkpoint.

## 2. `export/` — ExecuTorch Export Task

The exporter loads `pre_model/` and `post_model/` from a checkpoint directory.
It produces a multi-method `pre_model.pte`, a `post_model.pte`, and selective
build manifests containing the operators and dtype variants required by both
programs.

The pre program contains separate methods for prefix decoding, conditional
decoding, and cross-attention KV projection. Export uses dynamic sequence
dimensions and optionally applies XNNPACK dynamic per-channel quantization.

Run the task through the common entry point:

```bash
python main.py task=export
```

`config/task/export.yaml` defines the input checkpoint, output directory and
filenames, target device and dtype, model metadata, representative input
dimensions, quantization mode (`none`, `w8a8`, or `w4a8`), graph strictness and
printing, memory-planning behavior, and selective-build manifest settings.
Every field can be overridden through Hydra, for example:

```bash
python main.py task=export \
  task.checkpoint_dir=checkpoints/my-run/final_model \
  task.output_dir=export_output/my-run \
  task.quantization.mode=w8a8
```

`export/task.py` validates the runtime configuration, resolves paths against
Hydra's original working directory, captures the four dynamic graphs, applies
quantization, lowers them with XNNPACK, writes both programs, and optionally
generates per-model and merged operator manifests.

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
