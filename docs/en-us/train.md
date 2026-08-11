# Train Department (en-US)

## 1. Department Overview

The train department builds the two sub-models from hydra configuration and trains them jointly on the preprocessed datasets. It covers the entry point (`main.py`), all configuration (`config/`), the training loop (`tasks/train.py`), the model architecture (`model/`), loss functions (`loss.py`), validation metrics (`metrics/`), and the float8 conversion filter (`utils/float8.py`).

Training-time data representation (`dataset.py`) is documented in the preprocess department; the decoding-parameter search (`tasks/param_search.py`) is documented in the export & inference department.

Key properties of the training pipeline:

- **Joint training** — the causal pre model encodes the Chinese context prefix; its output is projected to shared K/V; the bidirectional post model decodes pinyin to Chinese characters with cross-attention over that K/V. One loss over the post model's masked logits back-propagates into both models.
- **NJT batching** — samples of different lengths are collated into nested jagged tensors; the trainer computes min/max sequence lengths as Python ints so the compiled graph stays free of data-dependent control flow.
- **Acceleration stack** — BF16 autocast, optional torchao float8 linear layers, optional `torch.compile`, expandable CUDA memory segments, TF32 enabled.
- **Optimization** — AdamW / AdEMAMix (bitsandbytes, 8-bit or 32-bit, paged or not) with cosine schedule and warmup, weight-decay grouping, gradient clipping.
- **Validation & logging** — periodic validation computing loss, ACC, Top-3/Top-5 ACC, Sentence-ACC and adaptive ECE; wandb logging; rich progress bars; per-epoch `save_pretrained` checkpoints plus a final `final_model` checkpoint.

## 2. `main.py` — Entry Point

**Functionality:** The hydra-decorated entry point (`@hydra.main(version_base=None, config_path="config", config_name="config")`). Loads the full configuration, applies global runtime settings, and dispatches to the requested task runner.

**Usage:** `python main.py` (train, per `config/task/train.yaml`), or `python main.py task=param_search` (plus task-specific overrides, see `export.md`).

**Behavior:**

- Sets `WANDB_CONSOLE=off`; enables PyTorch `expandable_segments` CUDA allocator when `system.expandable_segments_enabled`.
- Applies a global `logging.basicConfig` when `output.logging.log_level` is not `DEFAULT` (also re-levels the `hydra` and `torch` loggers).
- Enables TF32 for matmul and cudnn when `system.tf32_enabled`.
- Optionally seeds CPU/GPU when `system.set_seed` (default seed 42).
- Sets `torch.set_float32_matmul_precision("high")` and disables the inductor FX graph cache (NJT incompatibility).
- Dispatches: `task_type == "param_search"` -> `ParamSearchRunner`; anything else -> `Trainer.train()`.

## 3. `config/` — Hydra Configuration

**Functionality:** All run-time configuration. `config.yaml` composes the groups: `model: base`, `dataset: pretrain_v1`, `system: default`, `task: train`, `logging: train`, `output: default`.

### `config/model/base.yaml` — architecture
- `common`: `model_dim: 768`, `rope_theta: 1000.0`.
- `pre_model`: `max_seqlen: 128`, `mhsa_layers: 8`, `mhsa_heads: 4`, `attn_dim: 256`, `ffn_common_dim: 4096`.
- `post_model`: `max_seqlen: 32`, `mhsa_layers: 12`, `mhsa_heads: 4`, `attn_dim: 256`, `mhca_heads: 12`, `mhca_attn_dim: 768`, `ffn_common_dim: 4096`.

### `config/dataset/pretrain_v1.yaml` — data
- Paths: `dataset_dir`, `vocabs_config`, `train_dir_mds` (`${dataset.dataset_dir}/train`), `val_dir` (`${dataset.dataset_dir}/val`).
- `augmentation`: `drop_vowels: 0.2`, `drop_last_vowel: 0.3`, `vowels_droprate: [0.5, 1.0]`, `heteronym_confusion: 0.1`.
- `online_policy`: `strategy: "golden-sequence"`, `no_context_prob: 0.2`, `forward_ratio: 0.0` (note: the transform reads `forward_prob`, so the forward/backward mix falls back to the default 0.5).
- `shuffle`: MDS streaming shuffle (`algo: py1e`, `blocksize: 200000000000`, `cache_limit: 140gb`).

### `config/system/default.yaml` — runtime
`seed: 'no'`, `set_seed: false`, `num_workers: 4`, `keep_in_memory: true`, `tf32_enabled: true`, `mixed_precision: 'bf16'`, `ao_acceleration: 'float8'`, `optim_8bit: true`, `optim_paged: false`, `device: 'cuda'`, `expandable_segments_enabled: true`.

### `config/task/train.yaml` — training hyperparameters
`epochs: 15`, `batchsize: 1280`, `optimizer: "adamw"`, `learning_rate: 2e-4`, `weight_decay: 1e-2`, `warmup_steps: 10000`, `gradient_clip_val: 1.0`, `compile_model: true`, `compile_mode: 'default'`, `loss_type: "focal"`, `focal_loss_gamma: 1.0`, `focal_loss_alpha: 0.5`, `ece_bins: 15`, `ece_top_k: 5`, `task_type: "train"`.

### `config/logging/train.yaml` — logging & checkpoints
`log_with_wandb: true`, `project_name: "PhonoP2C"`, `run_name: ...`, `checkpoint_dir: ./checkpoints/<run>`, `histogram_interval: 10000`, `val_interval: 5000`, `save_interval: 1` (epochs).

### `config/output/default.yaml` — output behavior
- `progress_bar`: bar width, refresh rate, time elapsed/remaining, speed, ratio columns, `transient: true`.
- `logging.log_level: "ERROR"` (or `DEFAULT`).

### `config/task/param_search.yaml` — decoding calibration
See `export.md` (used by `main.py task=param_search`).

## 4. `tasks/train.py` — Trainer

**Functionality:** The joint training loop and validation procedure for both sub-models, using NJT batches.

### Progress bar support

#### `IterationSpeedColumn(ProgressColumn)`
- Functionality: renders iterations/second for a rich task.
- Behavior: shows `? it/s` while speed is unknown, else `X.XX it/s`.

#### `MofNCompleteColumn(ProgressColumn)`
- Functionality: renders `completed/total` for a task.
- Behavior: `?` for unknown totals.

#### `_build_progress(cfg)`
- Functionality: constructs a rich `Progress` object from `cfg.output.progress_bar`.
- Behavior: assembles columns (description, bar, percentage, optional elapsed/remaining/ratio/speed), with configurable refresh rate and transient mode; the task postfix field is always appended.

### Config helpers

#### `build_model_configs(cfg, tokenizer)`
- Functionality: builds `PreModelConfig` and `PostModelConfig` from the `model.*` YAML, filling vocab sizes from the tokenizer.
- Behavior: resolves `cfg.model` to a dict; feeds `context` / `pinyin` / `chinese` vocab sizes into `build_configs_from_dict`.

### `Trainer`

#### `__init__(cfg)`
- Functionality: builds every training component.
- Behavior, in order: sets device and logging; builds the tokenizer; builds both model configs; builds the pinyin->Chinese possibility mask on device and installs it on `post_model.logits_mask`; instantiates both models and prints parameter counts; optionally converts Linear layers to float8 training (torchao, `pad_inner_dim=True`, `module_filter_fn`); optionally `torch.compile`s both models (`dynamic=True`, with dynamo config `capture_scalar_outputs` / `capture_dynamic_output_shape_ops`); sets up the checkpoint dir and (optionally) wandb init + `wandb.watch` with histogram logging; builds the online policy / augmentation config dicts; builds the MDS streaming training set and `StreamingDataLoader` (shuffle algo/block size/cache limit, NJT collate); derives `epoch_steps` / `total_steps`; computes warmup steps from `warmup_steps` or `warmup_ratio`; builds the val set (`transform_pinyin_predict_val`) and a `DataLoader` (pin_memory, NJT collate); partitions parameters into decay / no-decay groups (decay: `nn.Linear` weights; no decay: biases, LayerNorm/RMSNorm/Embedding weights; asserts every parameter is separated exactly once); builds the optimizer (AdamW or AdEMAMix, 8/32-bit, paged or not) and a cosine scheduler; builds the loss via `get_loss_fn`; sets `use_amp` from `system.mixed_precision == "bf16"`.

#### `validate(pre_model, post_model, loader, epoch, global_step, progress)`
- Functionality: evaluates both models on the val loader.
- Behavior: switches to eval mode; for each batch moves NJTs to device, extracts flat ids/offsets, computes min/max seq lens as Python ints, runs pre -> post under bf16 autocast, computes the loss over the flat masked logits; accumulates loss and metrics (`MetricsAccumulator`); afterwards computes average val loss, perplexity (`exp(loss)` if loss < 20 else `inf`), and logs `val/loss`, `val/ACC`, `val/Top3-ACC`, `val/Top5-ACC`, `val/S-ACC`, `val/ECE` (plus `val/PPL` for CE loss) to wandb with the global step; restores train mode.

#### `train()`
- Functionality: the main training loop.
- Behavior: iterates epochs; for each batch: moves NJTs to device, extracts flat values/offsets, computes min/max seq lens, zeroes grads, runs the forward pass under autocast (pre model on flat prefix ids + offsets -> pre_embed/pre_K/pre_V; post model on flat postfix ids + offsets, with `pre_K/pre_V/pre_offsets` and both min/max bounds -> NJT logits), computes the loss on the flat masked logits, back-propagates, clips gradient norm (when `gradient_clip_val > 0`), steps optimizer and scheduler; logs train/loss, LR, grad-norm to wandb per step; validates every `val_interval` steps and on the final step; saves checkpoints every `save_interval` epochs (`epoch_<n>`/pre_model and post_model via `save_pretrained`, unwrapping `_orig_mod` for compiled models); after all epochs saves `final_model`; finishes wandb.

## 5. `model/` — Model Architecture

### `config.py`

#### `PreModelConfig(PretrainedConfig)`
- Functionality: configuration dataclass for the pre model (`model_type = "phono_p2c_pre"`).
- Fields: `model_dim` (768), `attn_dim`, `rope_theta`, `max_seqlen` (128), `mhsa_layers` (8), `mhsa_heads` (4), `ffn_common_dim` (4096), `vocab_size` (context vocab size), `mhca_attn_dim`, `cross_attn_heads` (post-side heads used for the shared K/V projection).

#### `PostModelConfig(PretrainedConfig)`
- Functionality: configuration dataclass for the post model (`model_type = "phono_p2c_post"`).
- Fields: `model_dim`, `attn_dim`, `rope_theta`, `pre_max_seqlen` (pre's max_seqlen, bounds cross-attention), `max_seqlen` (32), `mhsa_layers` (12), `mhsa_heads` (4), `use_moe_ffn`, `ffn_common_dim` (4096), `ffn_num_experts`, `ffn_choice`, `ffn_expert_dim`, `vocab_size` (pinyin vocab size), `proj_size` (chinese vocab size), `mhca_heads` (12), `mhca_attn_dim` (768).

#### `build_configs_from_dict(d, vocab_sizes)`
- Functionality: builds both configs from a nested YAML dict plus vocab sizes (`context`, `pinyin`, `chinese`).
- Behavior: reads `common` / `pre_model` / `post_model` sections with defaults; wires `pre.vocab_size = vocab_sizes["context"]`, `post.vocab_size = vocab_sizes["pinyin"]`, `post.proj_size = vocab_sizes["chinese"]`; derives the pre model's `mhca_attn_dim` and `cross_attn_heads` from the post-side settings.

### `model.py`

#### `PhonoP2CPreModel(PreTrainedModel)`
- Functionality: causal encoder for the Chinese context prefix.
- Structure: embedding (`vocab_size -> model_dim`), `mhsa_layers` blocks of `{mhsa, ffn(SwiGLU), norm1, norm2}`, final RMSNorm, and a shared `kv_proj` (`model_dim -> 2·mhca_attn_dim`) whose output is split into the cross-attention K and V.
- `forward(input_ids, offsets=None, kv_cache_memory=None, current_seqlen=None, min_seqlen=None, max_seqlen=None, pre_cross_kv_cache=None, pre_cross_cache_pos=None)` — three paths:
- NJT path (`offsets` given): flat embeddings, per-token RoPE positions via `make_local_position_ids`; each MHSA layer runs the NJT path (causal); returns `(hidden, pre_K, pre_V)` reshaped to `[total_tokens, cross_attn_heads, head_dim]`.
- Batched-with-cache path (`kv_cache_memory` + `current_seqlen`): each MHSA layer reads/writes the full self-attention KV cache (`update_mhsa_kv`); `pre_K/pre_V` are reshaped per head; when `pre_cross_kv_cache` and `pre_cross_cache_pos` are given, the K/V are written in-place into the shared cross cache via `update_cross_kv` and the updated cache is returned directly (used at inference/export).
- Plain batched path: dense SDPA, returns `(hidden, pre_K, pre_V)`.

#### `PhonoP2CPostModel(PreTrainedModel)`
- Functionality: bidirectional pinyin->Chinese decoder.
- Structure: embedding (`pinyin vocab -> model_dim`), `mhsa_layers` blocks of `{mhsa (bidirectional), mhca (cross-attn over pre K/V), norm1..3, ffn (SwiGLU or MoE_EC_FFN)}`, final RMSNorm, `lm_head` (`model_dim -> proj_size`), and the registered `logits_mask` buffer (pinyin->Chinese possibility map).
- `forward(input_ids, input_offsets=None, pre_K=None, pre_V=None, pre_offsets=None, pre_cross_kv_cache=None, current_seqlen=None, min_seqlen=None, max_seqlen=None, min_seqlen_pre=None, max_seqlen_pre=None, return_last_hidden=False)` — two paths:
- NJT path (`input_offsets` given): per-token self positions; cross position ids from `MHCALayer.compute_position_ids`; per layer: bidirectional MHSA, MHCA over `pre_K/pre_V` with separate query/KV bounds, FFN (MoE receives offsets); final norm; `lm_head` over the flat hidden; the per-position mask is applied by indexing `logits_mask` with the input token ids (impossible classes -> `-inf`); the flat logits are re-wrapped into an NJT (or returned together with hidden when `return_last_hidden`).
- Batched path (no offsets): same block layout; when using the cache, MHCA reads the shared cross cache with `current_seqlen` as the total pre-context length; returns dense `[B, S, proj_size]` logits (masked) or `(logits, hidden)`.

### `attn.py`

#### `MHSALayer(nn.Module)`
- Functionality: multi-head self-attention with RoPE; NJT, cached-batched, and plain batched paths.
- Structure: `qkv_proj` (`model_dim -> 3·attn_dim`, no bias), `out_proj`, rotary embedding.
- `forward(...)` behavior:
- NJT path: projects the flat hidden, reshapes per head, applies RoPE with local position ids, builds q/k/v NJTs (using precomputed min/max seq lens when given, else deriving from offsets), runs `scaled_dot_product_attention` (causal or not), projects back.
- Cached-batched path: computes q/k/v for the new chunk, writes k/v into the per-layer full cache (`update_mhsa_kv`), slices the valid KV window, applies RoPE to q (new positions) and the full valid K window, builds a causal mask (new positions attend to all cached positions up to themselves), runs SDPA with that mask.
- Plain batched path: RoPE over `arange(S)`, standard causal SDPA.

#### `MHCALayer(nn.Module)`
- Functionality: multi-head cross-attention: queries from the post model, K/V from the pre model's shared projection.
- Structure: `q_proj` (`model_dim -> mhca_attn_dim`), `out_proj`, rotary.
- `compute_position_ids(offsets, pre_offsets)` (static): computes query position ids (local post position + pre prefix length per batch row) and KV position ids (local pre positions) for the NJT path.
- `forward(...)` behavior:
- NJT path: projects queries, applies RoPE with the computed cross positions, applies RoPE to pre K/V, builds q/k/v NJTs with separate query and KV min/max bounds, runs non-causal SDPA.
- Batched path: reads the shared `pre_kv_cache` tuple up to `total_kv = cache_pos[0]` (K stored pre-RoPE), RoPEs the whole valid K window and the new query positions, runs non-causal SDPA.

### `ffn.py`

#### `SwiGLU(nn.Module)`
- Functionality: SwiGLU feed-forward block.
- Structure: `up_proj`, `gate_proj` (`in -> hidden`, no bias), `down_proj` (`hidden -> out`, no bias).
- Behavior: `down(silu(gate(x)) * up(x))`.

### `moe.py`

#### `MoE_EC_FFN(nn.Module)`
- Functionality: expert-choice MoE FFN with one always-on shared expert and N routed experts (batched matmul over all experts).
- Structure: `affinity` gate (`dim -> num_experts`), shared `expert_common` SwiGLU, stacked expert weights `experts_up_proj` / `experts_gate_proj` (`[num_experts, dim, expert_dim]`) and `experts_down_proj` (`[num_experts, expert_dim, dim]`), xavier-initialized.
- `forward(hidden, offsets=None)` behavior: flattens input to `[total_tokens, dim]`; expert capacity `max(1, total·choice / num_experts)`; gate scores -> per-expert `topk` over tokens (expert choice); `_batched_swiglu` computes every expert's output in one `bmm`; combine weights are `sigmoid(topk_vals)`; routed outputs are summed back into the per-token positions with `index_add` (out-of-place), added to the shared expert output; returns flat (NJT) or `[B, S, dim]` output.

### `utils.py`

#### `RotaryEmbedding(nn.Module)`
- Functionality: RoPE frequency table (`inv_freq = theta^(-2i/dim)`).
- Behavior: `forward(position_ids)` returns `(cos, sin)` of shape `[T, dim]` (freqs duplicated along the last dim).

#### `rotate_half(x)`
- Functionality: half-rotation of the last dim (RoPE helper).

#### `apply_rotary_pos_emb(v, cos, sin)`
- Functionality: applies RoPE to `[..., seq, heads, head_dim]` tensors.
- Behavior: `v·cos + rotate_half(v)·sin`, broadcasting cos/sin over batch and heads.

#### `make_local_position_ids(offsets)`
- Functionality: converts NJT offsets into per-token local positions.
- Behavior: global ids minus the start of each row's segment (`repeat_interleave` of offsets[:-1]).

### `custom_ops.py`
- Functionality: registers custom `torch.library` ops under the `phono` namespace that implement in-place KV-cache updates with functional semantics (clone + `index_copy_`), so torch.export / ExecuTorch can capture the cache updates without aliasing issues.
- Ops: `update_kv_cache(cache, value, start_pos)` (generic, dim-1 `index_copy_`), `update_cross_kv(cache, pre_K, pre_V, start_pos)` (K and V into a `[2, B, S, ...]` cache), `update_mhsa_kv(cache, k, v, start_pos, layer_idx)` (per-layer self-attention cache). Each has a functional and an `.out` variant, with CPU implementations and `fake` (meta) implementations.
- Behavior: `start_pos` tensor args are unwrapped via `.item()` in the wrapper helpers (`kv_cache_write`, `update_cross_kv`, `update_mhsa_kv`). `kv_cache_write` is exported in `model/__init__.py`; the current model forward paths use `update_cross_kv` / `update_mhsa_kv` directly.

### `__init__.py`
- Functionality: re-exports configs, models, layers, RoPE helpers and `kv_cache_write` as the package's public API.

## 6. `loss.py` — Loss Functions

### `FocalLoss(nn.Module)`
- Functionality: focal loss for class-imbalanced classification: `FL = -alpha · (1 - p_t)^gamma · log(p_t)`.
- Usage: selected with `loss_type="focal"`; configured via `focal_loss_alpha` / `focal_loss_gamma`.
- Behavior: computes per-token CE (ignoring `ignore_index` targets), derives `p_t = exp(-ce)`, applies the focal weight, masks ignored positions, and reduces by mean (over valid tokens) or sum.

### `LabelSmoothingCrossEntropy(nn.Module)`
- Functionality: label smoothing aware of the model's `logits_mask`.
- Usage: selected with `loss_type="ce"` plus a `label_smoothing_epsilon`.
- Behavior: smooths only over classes with finite logits (the masked classes are `-inf` and never receive probability mass); puts `1 - epsilon` on the target and spreads `epsilon` uniformly over the other valid classes; rows whose target is the only valid class fall back to plain CE (epsilon treated as 0, avoiding divide-by-zero); masks ignored rows; reduces by mean or sum.

### `get_loss_fn(loss_type="ce", ignore_index=-100, focal_loss_alpha=0.25, focal_loss_gamma=2.0, label_smoothing_epsilon=None)`
- Functionality: loss factory.
- Behavior: `"ce"` -> `nn.CrossEntropyLoss` (or `LabelSmoothingCrossEntropy` when `label_smoothing_epsilon` is set); `"focal"` -> `FocalLoss`; anything else raises `ValueError`.

## 7. `metrics/accumulator.py` — MetricsAccumulator

**Functionality:** Running metric accumulator for validation: per-token top-1 ACC, top-3 / top-5 ACC, sentence-level ACC, and Top-K adaptive ECE (quantile binning via `netcal.metrics.confidence.ACE`).

**Usage:** `acc = MetricsAccumulator(ece_bins=15, ece_top_k=5)`; `acc.update(logits, targets, target_offsets)` per batch; `acc.compute()` returns the metric dict; `acc.reset()`.

#### `__init__(ece_bins=15, ece_top_k=1)`
- Functionality: configures bin count and the k used to define ECE confidence/correct signals.

#### `update(logits, targets, target_offsets)`
- Functionality: ingests one batch of flat logits, flat targets, and sentence offsets.
- Behavior: computes softmax probs; accumulates token counts and correct counts for top-1/top-3/top-5; accumulates sentence-level correctness (all tokens of a sentence correct); accumulates per-token top-k confidence and correctness arrays (float64 CPU) for ECE. Ignores empty batches.

#### `compute()`
- Functionality: finalizes all metrics.
- Behavior: returns `{acc, top3_acc, top5_acc, s_acc, ece}`; ECE is computed with the ACE detector metric over the accumulated confidence / correct arrays (0.0 when no data).

#### `reset()`
- Functionality: clears all running statistics.

#### `__len__()`
- Functionality: number of accumulated tokens.

## 8. `utils/float8.py` — Float8 Conversion Filter

#### `module_filter_fn(mod, fqn)`
- Functionality: decides which modules torchao's `convert_to_float8_training` converts.
- Behavior: returns True only for `nn.Linear` modules whose fully-qualified name contains `expert`, `qkv_proj`, `out_proj`, `up_proj`, `gate_proj`, `down_proj`, `q_proj`, or `kv_proj`; everything else (norms, embeddings, gates) stays in its original dtype.