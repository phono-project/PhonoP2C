# Training and model semantics

This document describes current source and defaults, not the historical configuration of a published checkpoint. Reproduction requires the resolved training configuration, data manifest and source revision.

## Entry point and environment

Run from this project using its own pixi environment:

```sh
pixi run python main.py task=train dataset=pretrain_v2
pixi run python main.py task=train --cfg job --resolve
pixi run test
```

The current manifest supports Linux and depends on CUDA 13. A sibling project's Windows environment is not an equivalent training environment. See [Hydra configuration](../../config/config.yaml). Distributed training uses torchrun and the [distributed utilities](../../utils/distributed.py); consult Trainer's distributed configuration checks for batch-size semantics.

With `system.set_seed=true`, `main.py` seeds Python, NumPy and Torch. The default is false. The standalone preprocessing CLI bypasses that initialization. Record the actual entry point and resolved configuration; multiprocessing, devices and operators can still affect reproducibility.

## Current defaults

Configuration files remain authoritative. At this documentation revision:

| Configuration | Setting |
| --- | --- |
| [base.yaml](../../config/model/base.yaml) | Dimension 768; 8-layer causal pre decoder; 4-layer bidirectional pinyin post encoder; self-attention dimension 256 with 4 heads; FFN dimension 2048; Chinese/pinyin length limits 128/32 |
| [train.yaml](../../config/task/train.yaml) | 7 epochs; batchsize 4096; AdamW; max/min LR 4e-4/4e-5; betas [0.9, 0.95]; weight decay 1e-3; 7500 warmup and 150000 decay steps; gradient clipping 1.0 |
| Loss | CE; unconditional loss weight 1.0; focal loss and label smoothing are optional, not current defaults |
| Free-decoding validation | Beam width 3; chunk size 64; stride 32 in dataset order, not random source-stratified sampling |
| [system/default.yaml](../../config/system/default.yaml) | CUDA, bf16, float8 linear-layer training acceleration, 8-bit optimizer and compilation; gradient checkpointing enabled by default but currently disabled with a warning under DDP |

Float8 training acceleration and W8A8 deployment quantization are separate stages. See [export.md](export.md) for export and calibration.

## Two passes and the cache contract

The [pre model](../../model/model.py) is the Chinese causal decoder: self-attention, optional cross-attention to pinyin, then FFN; its `lm_proj` produces Chinese logits. The post model is a bidirectional pinyin encoder returning hidden states and per-position legal-character masks. Post neither emits Chinese logits nor cross-attends to pre.

Pass 1 has no pinyin condition. It builds Chinese history self-attention KV and optionally computes unconditional loss. Pass 2 starts with the last history token, followed by shifted target characters; the decoder attends to the full pinyin sequence and constrains each output position. The [training wrapper](../../model/wrapper.py) combines:

`loss = conditional_loss + unconditional_loss_lambda * unconditional_loss`

A zero lambda skips unconditional loss computation and logging, not the history forward pass required by conditional decoding.

[Python beam search](../../model/beam_search.py) prefills `prefix[:-1]`, then processes the last prefix token in pass 2. Its conditional self-attention KV must remain visible throughout that generation, but must not become unconditional history for later committed text. Pinyin cross-attention KV is a different tensor. Runtime commits must prefill selected text through the unconditional path.

## Training, validation and artifacts

[Trainer](../../tasks/train.py) loads the tokenizer, training stream and frozen validation data; constructs pre/post and their loss wrapper; and performs forward/backward passes, clipping and scheduling. Distributed execution handles token-count weighting. The main rank saves pre/post separately with `save_pretrained`.

Validation token ACC, Top-K ACC, S-ACC and ECE use teacher-forced conditional logits. They are not exact whole-candidate accuracy under free decoding. Trainer separately performs beam decoding on a strided subset. Comparisons must specify metric, denominator, sample selection, search width and displayed candidate count.

Attach resolved configuration and seeds, source revision, data/split identity, vocabulary and pronunciation-frequency hashes, and checkpoint hashes to training artifacts. Deployment additionally requires export/calibration settings, tool versions and binary build provenance. Today's YAML cannot substitute for historical records.

Cross-project research reports and experiments live under the collection root's `research/`; this document describes the PhonoP2C training implementation.
