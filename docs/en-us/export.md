# Export & Inference Department (en-US)

## 1. Department Overview

The export & inference department covers everything that happens after training with frozen model weights:

- **Decoding calibration** — `tasks/param_search.py` runs the trained model over a validation subset, extracts per-position candidate probabilities, builds a dictionary trie, and uses Optuna to find the best Viterbi decoding priors (`beta_single`, `beta_word`). The algorithms live in `algo/` and the calibration dictionary in `dicts/`.
- **On-device export** — `export.py` freezes and exports both sub-models to ExecuTorch `.pte` files with XNNPACK dynamic per-channel quantization and shared cross-attention KV cache semantics.
- **Inference demo** — `demo.py` runs the full pipeline in PyTorch: KV cache allocation, greedy / top-k decoding, and dictionary-constrained Viterbi N-best decoding.

Components:

| Component | File | Role |
|---|---|---|
| Decoding calibration | `tasks/param_search.py` | Dictionary+trie build, probs extraction, Optuna search of Viterbi priors |
| Trie matching | `algo/trie.py` | Dictionary load/build/save, per-position word matching |
| Viterbi DP | `algo/viterbi_dp.py` | N-best beam-search decoding with priors |
| Dictionary asset | `dicts/dict_v1.txt` | Calibration dictionary (multi-char words) |
| ExecuTorch export | `export.py` | torch.export + XNNPACK quantization -> `.pte` files |
| Inference demo | `demo.py` | KV-cache inference: greedy, top-k, Viterbi |

## 2. `tasks/param_search.py` — Decoding Hyperparameter Search

**Functionality:** Calibrates the dictionary-constrained Viterbi decoding on a pretrained (frozen) model. Two subtasks:

1. **Preprocess**: filter the dictionary (chinese-vocab check, multi-char only, length < max, dedup), build and save the trie, run model inference on a validation subset, extract per-position non-zero candidate probabilities sorted descending, convert token ids to Chinese characters, and save them as `probs.json`.
2. **Optuna search**: load `probs.json` + `dict_trie.json`, precompute trie word matches per sample position (in parallel), evaluate Viterbi N-best decoding with score `sum(ln probs) + n_words·ln(beta)` across trials, optimize `beta_single` / `beta_word`, and save `results.json`.

**Usage:** `python main.py task=param_search` (config `config/task/param_search.yaml`; key fields: `pretrained_pre_model`, `pretrained_post_model`, `vocabs_config`, `dict_path`, `val_dir`, `output_dir`, `num_samples`, `batchsize`, `epsilon`, `subtasks.*`, `dict.max_len: 7`, `optuna.n_trials: 100`, `optuna.n_best: 5`, `optuna.objective: "sentence_acc"`, beta ranges).

### Module state and workers

#### `init_worker_trie(trie)`
- Functionality: process-pool initializer that installs the shared trie into each worker's `_global_trie`.
- Behavior: assigns the global; subsequent worker calls can use it.

#### `process_json_lines_chunk(lines)`
- Functionality: parses a chunk of `probs.json` lines in a worker and precomputes trie word matches.
- Behavior: raises `ValueError` if the trie is not initialized; for each line builds per-position `{char: prob}` candidate dicts, runs `find_matching_words` for every position, and returns `(candidates, words_at, target)` tuples.

#### `evaluate_chunk_viterbi(chunk, beta_single, beta_word, N)`
- Functionality: evaluates Viterbi decoding on a precomputed chunk in a worker process.
- Behavior: runs `viterbi_nbest` per sample; counts how many samples have the target as the best (rank-1) result and how many have it within the top-N; returns `(correct_1, correct_N, total)`.

### `ParamSearchRunner`

#### `__init__(cfg)`
- Functionality: prepares the runner.
- Behavior: reads the device and task config; creates the output dir; builds the tokenizer.

#### `run()`
- Functionality: runs the enabled subtasks.
- Behavior: runs `_subtask_preprocess` when `subtasks.preprocess` (default true), then `_subtask_optuna_search` when `subtasks.optuna_search` (default true).

#### `_subtask_preprocess()`
- Functionality: dictionary processing + inference probs extraction.
- Behavior: loads the dictionary (`load_dictionary`), filters words that are not entirely in the chinese vocab, are single-char, or are ≥ `dict.max_len`, deduplicates preserving order; builds and saves `dict_trie.json`; loads the pretrained models (and installs the possibility map on the post model); loads the val dataset, shuffles it, takes the first `num_samples`, applies the val transform; runs inference under bf16 autocast (pre -> post, NJT path); for each sample converts logits to probabilities, keeps candidates above `epsilon`, sorts descending, maps ids to characters via the tokenizer, and appends a JSON line `{"target", "positions"}` to `probs.json`; prints progress every 10 batches.

#### `_subtask_optuna_search()`
- Functionality: Optuna search of decoding priors.
- Behavior: loads the trie; reads all `probs.json` lines; chunks them across `os.cpu_count()` workers and precomputes word matches in a `ProcessPoolExecutor` (chunk size ≈ lines / (workers · 4)); splits the precomputed data into per-worker eval chunks; creates an Optuna study (maximize, `GPSampler(seed=42)`); runs `n_trials` of the selected objective; writes `results.json` with the best `beta_single`, `beta_word`, sentence-ACC, N-sentence-ACC, `n_best`, and `n_trials`; prints a summary.

#### `_objective_s_acc(trial)`
- Functionality: Optuna objective optimizing rank-1 sentence accuracy.
- Behavior: suggests `beta_single` in [`beta_single_low`, `beta_single_high`] (linear) and `beta_word` in [`beta_word_low`, `beta_word_high`] (linear); evaluates all chunks in parallel; computes sentence-ACC (target equals best result) and N-sentence-ACC (target within top-N); stores both as trial user attrs; returns sentence-ACC.

#### `_objective_n_s_acc(trial)`
- Functionality: Optuna objective optimizing top-N sentence accuracy.
- Behavior: same as above but suggests betas with `log=True` sampling and returns N-sentence-ACC.

## 3. `algo/trie.py` — Trie Dictionary Matching

**Functionality:** Builds and loads a nested-dict trie from a Chinese word list and finds all dictionary words that can start at a given position given per-position character probabilities.

### Functions

#### `load_dictionary(path)`
- Functionality: reads a newline-separated word list.
- Behavior: strips whitespace; skips empty lines.

#### `build_trie(words)`
- Functionality: builds a nested-dict trie.
- Behavior: each word is inserted character by character with `setdefault`; a `"#"` key marks a word end; returns the trie root dict.

#### `load_trie(path)` / `save_trie(trie, path)`
- Functionality: JSON (de)serialization of a trie.
- Behavior: `ensure_ascii=False` so Chinese characters stay readable.

#### `find_matching_words(trie, candidates, start)`
- Functionality: finds every dictionary word that can start at position `start`.
- Usage: precomputed once per sample by `param_search.py` and computed live by `demo.py`.
- Behavior: iterative DFS over trie nodes; a word is only extended through characters whose candidate probability at the current position is present and > 0; each completed word is recorded as `(length, word_str, log_prob_sum)`; results are sorted longest-first.

## 4. `algo/viterbi_dp.py` — Viterbi N-Best Decoding

**Functionality:** Viterbi DP beam search for dictionary-constrained P2C decoding. A path's score is `sum(ln probs)` plus a prior penalty per transition: `ln(beta_single)` for single-character steps and `ln(beta_word)` for multi-character dictionary words.

### `viterbi_nbest(candidates, words_at, beta_single, beta_word, N)`
- Functionality: returns the top-N segmentations of the candidate sequence.
- Usage: called by `evaluate_chunk_viterbi` (param search) and by `demo.py` (`predict_step_viterbi`).
- Behavior: converts candidate probabilities to log space (`-inf` for non-positive); maintains `paths[pos]` as a dict keyed by the decoded prefix string (merging identical prefixes) holding `(score, backpointer node)`; at each position prunes the beam to the top-N unique prefixes; advances single characters and multi-char trie words; at the end sorts the final states, keeps the top-N, and reconstructs word lists via the backpointers. Returns `[(score, words), ...]` sorted by score descending; empty candidate list yields `[(0.0, [])]`; an unreachable end state yields `[]`.

## 5. `dicts/` — Calibration Dictionary

`dicts/dict_v1.txt` — a large newline-separated Chinese word list (~349k entries) used as the decoding dictionary. Words must pass the chinese-vocab check, be multi-char, and be shorter than `dict.max_len` (7) before entering the trie. Consumed by `tasks/param_search.py`; the produced `dict_trie.json` is consumed by `demo.py`.

## 6. `export.py` — ExecuTorch Export

**Functionality:** Exports the trained `PhonoP2CPreModel` and `PhonoP2CPostModel` to ExecuTorch `.pte` programs with XNNPACK dynamic per-channel quantization, preserving the shared cross-attention KV cache semantics between the two models.

**Usage:** `python export.py`. Config constants: `CHECKPOINT_DIR` (`./checkpoints/v1_0-base/final_model`), `MODEL_TYPE` (`torch.float32`), `SAVE_DIR` (`./export_output`).

### `load_model_from_checkpoint(checkpoint_dir, device)`
- Functionality: loads both sub-models from a checkpoint directory.
- Behavior: expects `pre_model/` and `post_model/` subdirectories produced by `save_pretrained` (config.json + safetensors); loads via `from_pretrained`, moves to device, and switches both to eval mode; returns `(pre_model, post_model)`.

### Top-level script behavior
- Loads the models, casts them to `MODEL_TYPE`, and freezes all parameters (`requires_grad = False`).
- Derives cache geometry from the configs: pre self-attention cache `(mhsa_layers, 2, B, pre_max, pre_nheads, pre_head_dim)`, shared cross-attention cache `(2, B, pre_max, post_nheads, post_head_dim)`.
- Builds dummy inputs and example kwargs for both models with the batched (cache) paths; `pre_example_kwargs` includes the shared `pre_cross_kv_cache` and `pre_cross_cache_pos` so the pre model writes the cross cache in place and returns it; `post_example_kwargs` reads it with `current_seqlen` = total pre-encoded prefix length.
- Declares dynamic dimensions: `new_prefix_len` (1..pre_max) for the pre input ids, `post_len` (1..post_cfg.pre_max_seqlen) for the post input ids.
- Exports both models with `torch.export.export` (dynamic shapes) and prints the graphs.
- Quantizes with `XNNPACKQuantizer` + `get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)` via torchao's `prepare_pt2e` / `convert_pt2e`, running a dummy forward (`no_grad`) between prepare and convert to calibrate; re-exports the quantized models.
- Lowers with `to_edge_transform_and_lower` using the `XnnpackPartitioner`, builds ExecuTorch programs with `MemoryPlanningPass(alloc_graph_input=False)`, and writes `pre_model.pte` / `post_model.pte` under `SAVE_DIR`.

## 7. `demo.py` — Inference Demo

**Functionality:** Runs the trained pipeline in PyTorch with KV caches: greedy decoding, per-position top-k output, and dictionary-constrained Viterbi N-best decoding.

**Usage:** `python demo.py` runs a hardcoded example (prefix "这难道不会变得很" + pinyin "luan ma"). The module functions are importable for interactive use.

### Module constants
`BETA_SINGLE=0.4636`, `BETA_WORD=0.4839`, `TRIE_PATH=./param_search_output/ dict_trie.json`, `EPSILON=0.001`, `N_BEST=3`.

### `load_model_from_checkpoint(checkpoint_dir, device)`
- Functionality: same loader as `export.py`.
- Behavior: loads `pre_model/` and `post_model/` from the checkpoint directory, eval mode.

### `create_kv_caches(pre_model, post_model, device, batch_size=1, dtype=torch.float32)`
- Functionality: allocates the inference KV caches.
- Behavior: pre self-attention cache shaped `(pre_num_layers, 2, B, pre_max, pre_nheads, pre_head_dim)`; shared cross-attention cache shaped `(2, B, pre_max, post_nheads, post_head_dim)` (dimensions taken from the first layers of each model); returns `(pre_kv_cache, pre_cross_kv_cache)`.

### `predict_step(text, pinyin_list, pre_model, post_model, tokenizer, device, pre_kv_cache=None, pre_cross_kv_cache=None, current_seqlen=0, topk=1)`
- Functionality: one inference step with KV caches.
- Behavior: encodes the BOS token and runs the pre model to seed the caches (self-attention KV plus cross cache write); encodes the prefix text (`encode_context`) and runs the pre model again from `current_seqlen`, updating both caches; when the text is empty the BOS is re-encoded with cross-cache position 0 and `current_seqlen` is reset to 1; encodes the pinyin (`encode_pinyin`) and runs the post model with `pre_cross_kv_cache` and `current_seqlen` = total pre-context length; returns per `topk`:
- `topk == 0` — `full_logits` and the new `current_seqlen`;
- `topk == 1` — argmax ids, decoded string via `ids_to_text`, and `current_seqlen`;
- `topk > 1` — per-position top-k ids/chars, probabilities, logits, and per-position entropy, plus `current_seqlen`.

### `predict_step_viterbi(text, pinyin_list, pre_model, post_model, tokenizer, device, pre_kv_cache=None, pre_cross_kv_cache=None, current_seqlen=0, beta_single=BETA_SINGLE, beta_word=BETA_WORD, trie_path=TRIE_PATH, epsilon=EPSILON, n_best=N_BEST)`
- Functionality: inference step with dictionary-constrained Viterbi N-best decoding.
- Behavior: calls `predict_step(topk=0)`; converts full logits to probabilities; builds per-position candidate dicts from probabilities above `epsilon` (characters resolved via the tokenizer id->char map); loads the trie (cached in `_trie_cache`); precomputes `find_matching_words` per position; runs `viterbi_nbest`; returns `{nbest: [{score, words, text}], current_seqlen, candidates}`.

### `_load_trie_cached(path)`
- Functionality: lazily loads and caches the trie in a module global.
- Behavior: loads only once per process.

### `__main__` block
- Behavior: loads the checkpoint and tokenizer, creates caches, runs greedy (`topk=1`), top-5, and Viterbi (`n_best=5`, `epsilon=0.0`) decoding on the example, and prints timing and per-position probability/logit/entropy details.