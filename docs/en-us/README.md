# PhonoP2C — Project Documentation (en-US)

## 1. What is PhonoP2C

PhonoP2C (Fast Pinyin-to-Chinese) is a research sub-project of the PhonoP2C-collection. It implements a **Pinyin-to-Chinese (P2C) conversion model** built on a two-stage "PostfixLM" architecture:

- **PhonoP2CPreModel** — a causal decoder with two exported passes: an unconditional pass that fills self-attention history and a conditional pass that cross-attends to post hidden states and predicts Chinese characters.
- **PhonoP2CPostModel** — a bidirectional pinyin encoder that returns hidden states and a **pinyin->Chinese possibility mask**; it does not own a cross-KV cache or produce Chinese logits.

The input is therefore "Chinese context + pinyin syllables", and the output is "Chinese characters". Training is joint: both sub-models are trained end-to-end on `(prefix context, pinyin, Chinese target)` samples.

## 2. Repository Layout

| Path | Purpose |
|---|---|
| `main.py` | Hydra entry point; dispatches train / preprocess tasks |
| `preprocessor.py` | preprocessing pipeline: raw corpus to MDS / Arrow datasets |
| `subset.py` | extracts a random subset of a corpus into parquet |
| `tokenizer.py` | three-vocabulary tokenizer (chinese / context / pinyin) |
| `dataset.py` | training-time data transforms, NJT collate, streaming dataset |
| `loss.py` | loss functions (cross-entropy, focal, mask-aware label smoothing) |
| `export/` | Hydra task that exports pre/post models to ExecuTorch `.pte` files |
| `demo.py` | parameterized inference CLI (greedy / beam search) |
| `tasks/train.py` | Trainer: joint training loop, validation, checkpointing |
| `model/config.py` | PreModelConfig / PostModelConfig + YAML-to-config builder |
| `model/model.py` | PhonoP2CPreModel / PhonoP2CPostModel |
| `model/attn.py` | MHSALayer (self-attention) / MHCALayer (cross-attention) |
| `model/ffn.py` | SwiGLU feed-forward |
| `model/moe.py` | MoE_EC_FFN (expert-choice MoE FFN, optional) |
| `model/utils.py` | RoPE (RotaryEmbedding) helpers, local position ids |
| `model/custom_ops.py` | custom torch.library KV-cache update ops (export-friendly) |
| `metrics/accumulator.py` | ACC / Top-k ACC / Sentence-ACC / Adaptive ECE accumulator |
| `utils/float8.py` | module filter for torchao float8 training conversion |
| `config/` | hydra configuration (model, dataset, task, system, logging, output) |
| `vocabs/` | chinese / context / pinyin vocabularies + config.yaml |
| `datasets/` | raw corpora (in pretrain_base) and generated datasets (pretrain_v2) |
| `checkpoints/` | training run outputs (pre_model / post_model subdirs) |
| `pixi.toml` | pixi environment definition (Python 3.13, CUDA 13, torch cu130) |
| `docs/en-us/` | this documentation set (English) |
| `docs/zh-cn/` | Chinese translation of this documentation set |

## 3. The Three Departments

The documentation is organized into three departments (phases of the project lifecycle). Each department has its own document:

| Department | Document | Covers | Entry points |
|---|---|---|---|
| Preprocess | `preprocess.md` | Turning raw text corpora into model-ready samples: `subset.py`, `preprocessor.py`, `tokenizer.py`, `vocabs/`, and the data layer of `dataset.py` | `python subset.py`, `python preprocessor.py --preprocess` |
| Train | `train.md` | Model architecture and the joint training pipeline: `main.py`, `config/`, `tasks/train.py`, `model/`, `loss.py`, `metrics/`, `utils/float8.py` | `python main.py` |
| Export & Inference | `export.md` | ExecuTorch export (`export/`) and the inference demo (`demo.py`) | `python main.py task=export`, `python demo.py` |

### 3.1 Preprocess department

Raw corpora (JSONL, plain text, parquet) are normalized (traditional to simplified Chinese, NFKC, emoji stripping, context-vocab filtering), split into runs of Chinese / pause-punctuation / non-Chinese segments, sliced into samples of bounded length, and annotated with nested per-character pinyin readings. The train split is written to MDS (MosaicML StreamingDataset) format in parallel, the validation split is written to HF Arrow format with prefix/suffix/pinyin materialized. `subset.py` is a helper that extracts a random sub-corpus (used e.g. to build a smaller fineweb subset).

### 3.2 Train department

A hydra-driven joint trainer builds the two sub-models from config, computes the pinyin->Chinese possibility mask from the tokenizer, and trains both models end-to-end on nested-jagged-tensor (NJT) batches with BF16 mixed precision, optional torchao float8 linear acceleration, optional `torch.compile`, AdamW/AdEMAMix family optimizers, cosine LR schedule, gradient clipping, periodic validation (loss, ACC, Top-k ACC, Sentence-ACC, Adaptive ECE), wandb logging, and `save_pretrained` checkpoints.

### 3.3 Export & Inference department

After training, the `export` task freezes and exports a multi-method pre program plus a post encoder to ExecuTorch `.pte` files with configurable XNNPACK quantization. `demo.py` provides parameterized greedy and beam-search inference in PyTorch.

## 4. End-to-End Data Flow (prose)

1. **Corpus acquisition** — It is recommended to store raw data in `datasets/pretrain_base`. The data processor supports data in JSONL / Parquet formats from sources such as LCCC, MMC, CLUE, Wikipedia, Zhihu-KOL, and FineWeb. For very large corpora, using `subset.py` to extract a subset is recommended.
2. **Preprocessing** — `preprocessor.py` normalizes each text, segments it, slices samples of 16–64 characters, computes nested per-character pinyin, and writes `datasets/pretrain_v2/train` (MDS, zstd) plus `datasets/pretrain_v2/val` (HF Arrow with materialized prefix/suffix/pinyin pairs).
3. **Training** — `main.py` loads hydra config; `Trainer` builds the tokenizer, the possibility mask, both models, and streams MDS training batches. Each batch is transformed online (span selection, pinyin augmentation), collated into NJTs, and fed to pre -> post models. The post logits are masked by the possibility mask, the loss is computed, and both models are optimized jointly. Validation runs periodically on the Arrow val set. Checkpoints are saved per epoch as `pre_model` / `post_model` subdirectories.
4. **Export** — `main.py task=export` loads the configured checkpoint, exports the pre methods and post encoder, applies the configured XNNPACK quantization, and writes `pre_model.pte` / `post_model.pte`.
5. **Inference** — `demo.py` encodes pinyin once, then runs the pre decoder's conditional pass with self-KV cache state and decodes with greedy or beam search.

## 5. Environment

The project uses pixi (`pixi.toml`): Python 3.13, CUDA 13 runtime, PyTorch cu130 build, plus jieba, pypinyin, zhconv-rs, streaming (MosaicML), datasets, hydra-core, wandb, netcal, torchao, bitsandbytes, executorch, and quality tools (black, isort, flake8, mypy). Typical workflow commands:

- `pixi install` / `pixi run python ...` to run any script.
- `python subset.py` — build a corpus subset.
- `python preprocessor.py --preprocess` — full preprocessing.
- `python preprocessor.py --generate_val` — materialize the val dataset.
- `python main.py` — train (configuration can be overridden via Hydra).
- `python main.py task=export` — ExecuTorch export.
- `python demo.py --checkpoint <dir> --pinyin <syllables...>` — inference demo.

> **Temporary note on known issues:** In the current Python 3.13 environment, the stable version of PyTorch has a known upstream bug regarding support for NJT’s `torch.compile`, which manifests as a symbol generation error `torch._inductor.exc.InductorError: AssertionError`.
> If you require the `torch.compile` feature, please **use PyTorch Nightly as is currently the default in pixi.toml**; it is known that `torch.compile` works correctly when compiling this model with PyTorch Nightly.
> If you prefer to use the stable release, you can set `compile_model` to `false` in `config/task/train.yaml`.

## 6. Reading Guide

Each department document describes components at two levels of abstraction:

- **Department level** — the department's purpose, inputs, outputs, and the overall processing flow.
- **Function level** — every module, class, and function: its functionality (what it does), its usage (how it is invoked and with what parameters), and its behavior (what happens when it runs, edge cases included).
