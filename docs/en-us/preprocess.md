# Preprocess Department (en-US)

## 1. Department Overview

The preprocess department turns raw text corpora into model-ready samples consumed by the train department. Its inputs are raw corpus files (JSONL, plain text, parquet) and vocabulary definition files; its outputs are:

- `datasets/pretrain_v1/train` — MDS (MosaicML StreamingDataset) training split, sharded and compressed (zstd), each sample holding `text_list`, `pinyin_list` (nested per-character readings, JSON), and `labels`.
- `datasets/pretrain_v1/val` — HF Arrow validation split with `prefix` / `suffix` / `pinyin` materialized per sample.
- Intermediate artifacts `datasets/pretrain_v1/train_original` and `datasets/pretrain_v1/val_original`.

The department comprises:

| Component | File | Role |
|---|---|---|
| Corpus subsetter | `subset.py` | Extract a random subset of a large corpus into parquet |
| Main pipeline | `preprocessor.py` | Normalize, segment, sample, annotate pinyin, write train/val datasets |
| Tokenizer | `tokenizer.py` | Three-vocabulary tokenizer; encoding/decoding; pinyin->Chinese possibility map |
| Vocabulary assets | `vocabs/` | chinese_vocab.txt, context_vocab.txt, pinyin_vocab.txt, config.yaml |
| Data layer for training | `dataset.py` | Online transforms (span selection, pinyin augmentation), NJT collate, streaming dataset |

`dataset.py` is described here because it is the data-representation layer; the train department consumes it (see `train.md`).

## 2. `subset.py` — Corpus Subset Extractor

**Functionality:** A standalone helper that extracts a random subset of an existing dataset (HF `save_to_disk` directory or parquet file/directory) and saves it as parquet shards. Used to build smaller working corpora, e.g. the fineweb subset consumed by `preprocessor.py`.

**Usage:** Run directly: `python subset.py`. Configuration is via module constants: `INPUT_PATH`, `OUTPUT_PATH`, `INPUT_FORMAT` (`"hf"` or `"parquet"`), `SUBSET_SIZE` (fraction or absolute row count), `SEED` (default 114514), `NUM_SHARDS` (default 20).

### Functions

#### `_find_parquet_files(path)`
- Functionality: locates parquet files below a path.
- Usage: called by `load_source`; `path` may be a single `.parquet` file or a directory.
- Behavior: returns `[path]` if it is a file ending in `.parquet`; otherwise returns a sorted recursive glob of `*.parquet` under the directory; returns `[]` for missing paths.

#### `load_source(path, fmt)`
- Functionality: loads the input corpus.
- Usage: `fmt="hf"` loads via `load_from_disk`; `fmt="parquet"` loads all found parquet files via `Dataset.from_parquet`.
- Behavior: raises `FileNotFoundError` when no parquet is found; raises `ValueError` for unknown formats.

#### `extract_subset(ds, subset_size, seed)`
- Functionality: selects a random subset of rows.
- Usage: `subset_size` is a float fraction in (0, 1] or an integer row count; `seed` drives the RNG.
- Behavior: computes `n_keep` (clamped to `[0, n]`), permutes row indices with a numpy `default_rng`, keeps the first `n_keep` indices, sorts them (so sequential read-back is efficient), and returns `ds.select(...)`.

#### `save_parquet(ds, out_dir, num_shards)`
- Functionality: writes a dataset as parquet, optionally sharded.
- Usage: `num_shards <= 1` writes a single `data.parquet`.
- Behavior: for multiple shards writes `data-<shard>-of-<num_shards>.parquet` per contiguous shard; creates `out_dir` if needed.

#### `main()`
- Functionality: orchestrates load -> subset -> save and prints source/subset row counts.

## 3. `preprocessor.py` — Main Preprocessing Pipeline

**Functionality:** The core preprocessing pipeline. It normalizes raw text, segments it into Chinese / pause-punctuation / non-Chinese runs, slices segments into samples of bounded length, computes nested per-character pinyin for the Chinese segments, splits the result into train/val, and writes the train split as MDS (in parallel) and the val split as HF Arrow.

**Usage:** CLI with two flags:

- `python preprocessor.py --preprocess` — runs the full pipeline (all sources -> concatenation -> character count -> train/val split -> MDS train -> Arrow val-original).
- `python preprocessor.py --generate_val` — reads `val_original` and materializes a prepared val dataset (`prefix`/`suffix`/`pinyin` columns).

The tokenizer is built from `vocabs/config.yaml` (`P2CTokenizer.from_config`).

### Constants

- Source paths: `LCCC_PATH`, `MMC_PATH`, `CLUE_PATH`, `WIKIPEDIA_PATH`, `ZHIHUKOL_PATH`, `FINEWEB_EDU_CHS_PATH`. Active sources are defined by `PARQUET_SOURCES` (zhihu-kol INSTRUCTION/RESPONSE; fineweb `text` is commented out), `JSONL_SOURCES` (LCCC, MMC), `TEXTONLY_SOURCES` (CLUE).
- Sampling bounds: `TRUNCATION_LEN_MIN=16`, `TRUNCATION_LEN_MAX=64`, `MAX_TEXT_LEN=72`, `BAN_TEXT_LEN=96`, `MAX_NON_CHINESE_SEG_LEN=16`, `MIN_CHINESE_RATIO=0.5`.
- Split/output: `SPLIT_RATIO=0.995`, train/val/original output directories, `NUM_VAL_SHARDS=1`, `NUM_ORIGIN_SHARDS=15`, `SHUFFLE_SEED=114514`, `NUM_PROC=15`, `BATCH_SIZE=8`.
- MDS: `MDS_COMPRESSION="zstd"`, `MDS_SIZE_LIMIT=1<<28` (~256 MB/shard), columns `text_list: json`, `pinyin_list: str`, `labels: json`.
- Val augmentation constants: `DROP_VOWELS=0.2`, `DROP_LAST_VOWEL=0.3`, `VOWELS_DROPRATE_MIN=0.5`, `VOWELS_DROPRATE_MAX=1.0`, `HETERONYM_CONFUSION=0.1`, `NO_CONTEXT_PROB=0.2`, `FORWARD_RATIO=0.0`.
- Segment labels: `PAUSE_LABEL=0`, `CHINESE_LABEL=1`, `NON_CHINESE_LABEL=2` (must match `dataset.py`).

### Helper functions

#### `_get_initial_and_final(syllable)`
- Functionality: splits a pinyin syllable into (initial, final).
- Usage: internal; used by val pinyin augmentation.
- Behavior: checks compound initials `zh/ch/sh` first, then single-character initials; returns `("", syllable)` when no initial is found.

#### `_chinese_char_ratio(text)`
- Functionality: fraction of characters in `[\u4e00-\u9fff]`.
- Usage: filtering check at segment and sample level.
- Behavior: returns `0.0` for empty text.

#### `_clean_emoji(text)`
- Functionality: strips `[...]` bracket content (emoji placeholders).
- Usage: first normalization step of `preprocess_text`.

#### `_remove_segmentation_spaces(text)`
- Functionality: removes spaces between two Chinese characters (fixes pre-segmented corpora).
- Usage: second normalization step.

#### `word_to_pinyin_nested(word)`
- Functionality: converts a Chinese word into `(clean_word, nested_pinyin)` where `nested_pinyin` has exactly one entry per character of `clean_word`, and each entry is the list of possible readings with the most likely reading first.
- Usage: called by `_make_sample` for every Chinese segment.
- Behavior: obtains the best (non-heteronym) reading sequence for the whole word, then per-character heteronym readings; filters the unusual `ê` reading; promotes the best reading to front when present; drops characters with no pinyin so the character and pinyin sequences stay aligned.

#### `resolve_pinyin_list(pinyin_raw, heteronym_confusion_p=0.0)`
- Functionality: flattens per-character reading lists into a single reading each.
- Usage: used by val augmentation and (identically named) in `dataset.py` for training.
- Behavior: picks the first (most likely) reading per char; when `heteronym_confusion_p > 0` and a char has multiple readings, it randomly picks a non-first reading with that probability (training-only noise).

#### `build_val_pinyin(pinyin_flat_readings)`
- Functionality: static augmentation for the val split applied to a flat list of per-char reading lists.
- Usage: called by `_prepare_val_batch`.
- Behavior: resolves readings (with heteronym confusion), then with probability `DROP_VOWELS` replaces each syllable by its initial's first character with per-syllable probability uniform in `[VOWELS_DROPRATE_MIN, VOWELS_DROPRATE_MAX]`; with probability `DROP_LAST_VOWEL` also reduces the last syllable. Returns the augmented syllable list.

### Segmentation

#### `preprocess_text(raw_text, tokenizer)`
- Functionality: normalizes one raw text string.
- Usage: first stage of `process_raw_text`.
- Behavior: strips emoji placeholders, removes inter-Chinese spaces, converts traditional->simplified (`zhconv_rs.zhconv`, zh-hans), applies NFKC normalization, then keeps only characters in the context vocabulary plus pause punctuation (`! ? 。 : ; ,` and newline); returns the cleaned string (empty input yields empty output).

#### `_char_category(ch, tokenizer)`
- Functionality: classifies a single character.
- Behavior: returns `PAUSE_LABEL` for pause punctuation, `CHINESE_LABEL` when the char is in `chinese_vocab`, else `NON_CHINESE_LABEL`.

#### `segment_text(text, tokenizer, max_non_chinese_len=16)`
- Functionality: splits a normalized text into `(segments, labels)`.
- Usage: called by `process_raw_text`.
- Behavior: drops the whole text when its Chinese ratio is below `MIN_CHINESE_RATIO`; groups the text into maximal runs of one category; pause runs are collapsed to their first character (one PAUSE segment); Chinese runs are split by jieba into word segments; non-Chinese runs are kept only if their length does not exceed `max_non_chinese_len`.

### Sampling

#### `build_samples(segs, labels)`
- Functionality: slices the full segment list into samples.
- Usage: called by `process_raw_text`.
- Behavior: picks a random `trunc_len` in `[16, 64]`, accumulates segments until reaching it; keeps extending until a PAUSE segment preceded by a CHINESE segment is found (or, once `MAX_TEXT_LEN=72` is exceeded, until a non-Chinese segment preceded by a Chinese segment is found); if the total exceeds `BAN_TEXT_LEN=96` the window is dropped and the scan jumps past it; each produced window goes through `_make_sample`; after emitting, the scan continues after the stopping PAUSE segment.

#### `_make_sample(sample_segs, sample_labels)`
- Functionality: packs a segment window into one dataset sample.
- Behavior: computes nested pinyin for every Chinese segment (dropping segments whose word yields no pinyin); requires at least one Chinese segment; drops the sample if the final joined text's Chinese ratio is below `MIN_CHINESE_RATIO`; returns `{"text_list", "pinyin_list" (JSON string), "labels"}`.

#### `process_raw_text(raw_text, tokenizer)`
- Functionality: full raw string -> list of sample dicts.
- Behavior: runs `preprocess_text`; returns `[]` for texts of length ≤ 1; then segments and samples.

### Span selection (offline val flavor)

#### `select_span(labels, k, direction="backward")`
- Functionality: maps a 1-indexed choice `k` of Chinese segment to `(sel, end, chinese_offset)`: `prefix = segments[0:sel]`, `suffix = segments[sel:end]` (consecutive Chinese), and the pinyin slice `[chinese_offset : chinese_offset + (end - sel)]`.
- Usage: used by `_prepare_val_batch`; the training-time variant `_select_span` in `dataset.py` is identical in logic.
- Behavior: `backward` anchors at the k-th Chinese segment and extends rightward over consecutive Chinese segments; `forward` anchors at the end and extends leftward; `bidirectional` extends both ways; raises `ValueError` for unknown directions.

### Batch processors and loaders

#### `process_pure_batch(batch, tokenizer)`
- Functionality: maps a batch of `sentences` (lists of raw texts) to lists of samples.
- Behavior: for every sentence and every raw text inside it runs `process_raw_text` and accumulates `text_list`/`pinyin_list`/`labels`.

#### `process_parquet_batch(batch, tokenizer, text_columns)`
- Functionality: like `process_pure_batch` but reads the given text columns of a parquet-derived batch.
- Behavior: skips missing columns and empty strings; accumulates all produced samples.

#### `_find_files(path, suffix)`
- Functionality: recursive glob for files with a suffix under a directory.
- Behavior: returns a sorted list; `[]` when the path is missing or not a directory.

#### `load_and_process_jsonl_source(tokenizer, jsonl_path)`
- Functionality: loads and processes a JSONL corpus directory.
- Behavior: finds `*.jsonl` files; yields each line parsed as JSON — lines that parse to a list become `{"sentences": items}` (single-object lines are dropped, JSON decode errors are skipped); builds an HF dataset and maps `process_pure_batch` over it with `batched=True`, `BATCH_SIZE=8`, `NUM_PROC=15`; returns the processed dataset or `None` if no files exist.

#### `load_and_process_textonly_source(tokenizer, textonly_path)`
- Functionality: loads and processes a plain-text corpus directory.
- Behavior: finds `*.txt` files; each non-empty line becomes one sentence; maps `process_pure_batch`; returns the processed dataset or `None`.

#### `load_and_process_parquet_source(path, text_columns, tokenizer)`
- Functionality: processes a parquet corpus shard by shard.
- Behavior: finds `*.parquet` files, processes them in random order; for each shard reads only the present text columns, maps `process_parquet_batch` (each shard is released after processing to bound memory), and concatenates all processed shards; returns `None` if no shards were usable.

### Split and counts

#### `shuffled_train_test_split(ds, test_size, seed)`
- Functionality: train/val split via a full permutation.
- Behavior: permutes indices with a seeded numpy RNG; assigns the first `n_test` permuted indices to the val set; sorts both index sets so reads are sequential; returns `(train_ds, test_ds)` via `ds.select`.

#### `count_characters(ds)`
- Functionality: total number of characters in all samples.
- Behavior: maps a counting batch over the dataset in parallel and sums the lengths of all `text_list` entries.

### MDS writing

#### `_write_mds_partition(args)`
- Functionality: worker entry that writes one contiguous partition of the dataset to MDS.
- Usage: executed in a `spawn` pool by `save_train_as_mds`; `args` is `(num_proc, shard_id, dataset_disk_path, out_dir)`.
- Behavior: mmaps the original dataset from disk, takes the contiguous shard `shard_id`, and streams every sample through `MDSWriter` with zstd compression and a 256 MB size limit.

#### `save_train_as_mds(dataset, out_dir, num_proc=15)`
- Functionality: saves the training split as MDS in parallel.
- Behavior: persists the full dataset to `train_original` first (15 shards), computes contiguous boundaries across `num_proc` workers, spawns a process pool writing each partition, then merges the partition indexes with `streaming.base.util.merge_index` (`keep_local=True`) so the result is a single valid MDS directory.

### Val preparation

#### `_prepare_val_batch(batch, tokenizer)`
- Functionality: materializes one val sample into `prefix` / `suffix` / `pinyin`.
- Behavior: for each sample picks a random Chinese segment `k`; with probability `NO_CONTEXT_PROB` uses no prefix (bidirectional span), else a span in `forward`/`backward` direction with probability `FORWARD_RATIO` (0.0 by default -> almost always backward); the suffix is the consecutive Chinese span; the pinyin column is the augmented flat syllable list for the suffix, JSON-encoded.

### `main()`
- Functionality: CLI entry.
- Behavior: with `--preprocess`, builds the tokenizer, loads and processes all sources in order (JSONL, text-only, parquet), concatenates them, counts characters, splits train/val, writes the train MDS and the `val_original` Arrow dataset; with `--generate_val`, loads `val_original`, applies `_prepare_val_batch` (batched, 1 process), and saves `val` (1 shard). Raises `RuntimeError` when no data source is found; prints final row/character statistics.

## 4. `tokenizer.py` — P2CTokenizer

**Functionality:** The three-vocabulary tokenizer used across all departments:

- **chinese_vocab** — the output/target space of the post model (pure Chinese characters).
- **context_vocab** — the input space of the pre model (Chinese + punctuation + other symbols); special tokens are appended at its end.
- **pinyin_vocab** — the input space of the post model (pinyin syllables).

It also builds the pinyin->Chinese **possibility map** used to mask impossible post-model logits.

**Usage:** `P2CTokenizer.from_config("vocabs/config.yaml")`; in `preprocessor.py` (filtering), `dataset.py` (encoding), `tasks/train.py` (sizes + mask), `demo.py`, and the `export` task.

### `_read_vocab_tokens(path)`
- Functionality: reads a unigram vocab file (one token per line).
- Behavior: a line consisting of the two literal characters `\n` is converted to a real newline token; empty lines are skipped; returns the token list in file order.

### `P2CTokenizer`

#### `__init__(chinese_vocab_path, context_vocab_path, pinyin_vocab_path, special_tokens_def=None)`
- Functionality: builds the three vocabularies.
- Behavior: each vocabulary is deduplicated preserving file order (first occurrence wins the lowest id); special tokens are appended at the end of the context vocabulary with ids starting at `context_base_size`; their names become attributes on `self.spec_tokens` (e.g. `tokenizer.spec_tokens.bos_token`).

#### `from_config(config_path)` / `from_config_dict(config_dir, config_dict)`
- Functionality: classmethod builders from the `vocabs/config.yaml` layout (`{"vocabs": {chinese_vocab, context_vocab, pinyin_vocab, context_special_tokens}}`).
- Behavior: vocab paths are resolved relative to the config file's directory; `context_special_tokens` is optional (None when absent).

#### Properties
- `chinese_vocab_size`, `pinyin_vocab_size`, `context_vocab_size`, `vocab_size` (alias of context size, pre model), `proj_size` (alias of chinese size, post model output), `num_special_tokens`.

#### Encoding
- `encode_context(text)` — maps characters through context_vocab; characters absent from the vocab are silently dropped; returns id list (used for the pre model input and the prefix).
- `encode_chinese(text)` — maps characters through chinese_vocab, dropping unknown ones; used for post-model targets.
- `encode_pinyin(pinyin_list)` — maps syllables through pinyin_vocab; a syllable not in the vocab is mapped to the in-vocab syllable with the smallest edit distance (`_edit_distance`).

#### Checks and filtering
- `check_pinyin(pinyin_list)` — True iff every syllable is in the pinyin vocab.
- `filter_text(text, allow_characters=None, vocab="chinese")` — keeps only characters present in the selected vocab, plus any `allow_characters`; `vocab="context"` uses the wider context vocabulary.
- `is_chinese(ch)` — True iff `ch` is a single char in chinese_vocab.
- `check(text, vocab="chinese")` — True iff every char of `text` is in the selected vocab.

#### Decoding
- `ids_to_text(ids)` — chinese_vocab ids -> characters.
- `ids_to_context_text(ids)` — context_vocab ids -> characters, skipping special tokens (ids ≥ context_base_size).
- `decode_greedy(ids)` — alias of `ids_to_text`.

#### `_edit_distance(a, b)` (static)
- Functionality: Levenshtein distance.
- Usage: fallback mapping for unknown pinyin syllables.
- Behavior: full DP table; O(m·n).

#### `create_possibility_map()`
- Functionality: builds the `(pinyin_vocab_size, chinese_vocab_size)` bool mask; entry `(p, c)` is True iff pinyin `p` can possibly map to Chinese char `c`.
- Usage: the trainer assigns the map to `post_model.logits_mask`; the post model masks every logit whose entry is False to `-inf`.
- Behavior: for every Chinese char, obtains all heteronym readings via pypinyin (`heteronym=True`, `strict=False`, `errors="ignore"`, `Style.NORMAL`, `v_to_u=False`); the unusual `ê` reading is skipped; marks an exact full-syllable match, and also marks any 简拼 prefix of length 1–3 that is a valid simple initial present in the pinyin vocab (so typing `zh` can still reach a char whose full pinyin starts with `zh`). Logs a warning for chars with no pinyin at all.

## 5. `vocabs/` — Vocabulary Assets

- `config.yaml` — maps `vocabs.chinese_vocab`, `vocabs.context_vocab`, `vocabs.pinyin_vocab` and `vocabs.context_special_tokens` (here: `bos_token`); consumed by `P2CTokenizer.from_config`.
- `chinese_vocab.txt` — one Chinese character per line (8104 lines); defines the post-model output space and the "is Chinese" test.
- `context_vocab.txt` — one token per line (8223 lines); the pre-model input space (Chinese + punctuation + symbols).
- `pinyin_vocab.txt` — one pinyin syllable per line (443 lines); the post-model input space.

The `\n` literal in a line encodes a real newline token (see `_read_vocab_tokens`).

## 6. `dataset.py` — Data Layer for Training

**Functionality:** The training-time data representation: online sample transforms (span selection + pinyin augmentation), an NJT collate function, and a streaming dataset wrapper over the MDS training split. It also reproduces the segmentation label constants used by `preprocessor.py`.

**Usage:** Imported by `tasks/train.py` for transforms, collate, streaming datasets, and validation dataset loading.

### Constants
`PAUSE_LABEL=0`, `CHINESE_LABEL=1`, `NON_CHINESE_LABEL=2` — must match `preprocessor.py`. Pinyin-augmentation sets: `_COMP_CONSONANTS` (`zh/ch/sh`), `_ALL_INITIALS` (all simple initials).

### Pinyin augmentation helpers

#### `_get_initial_and_final(syllable)`
- Functionality: split syllable into (initial, final).
- Behavior: compound initials first, then single initials; `("", syllable)` when no initial.

#### `augment_pinyin_sequence(pinyin_list, aug_cfg)`
- Functionality: applies vowel-dropping augmentation to a syllable list.
- Usage: called by `transform_pinyin_predict_train`; configured via the `dataset.augmentation` config block.
- Behavior: with probability `drop_vowels` the whole sentence is augmented: each syllable with both an initial and a final is replaced by its initial's first character with probability uniform in `vowels_droprate` (e.g. `zhu -> z`); independently, with probability `drop_last_vowel` the last syllable is reduced the same way.

#### `resolve_pinyin_list(pinyin_raw, heteronym_confusion_p=0.0)`
- Functionality: flatten per-char reading lists (same semantics as in `preprocessor.py`).
- Behavior: picks the first reading by default; with `heteronym_confusion_p > 0` may randomly pick a non-first reading.

### Golden-sequence span choice

#### `PHI`, `_compute_base_seed_norm(text)`
- Functionality: a deterministic per-text seed in [0, 1) derived from the MD5 of the text.
- Behavior: first 8 bytes of the MD5 as a big-endian integer normalized by 2^64.

#### `_pick_golden_choice(count_chinese, text, epoch)`
- Functionality: deterministic golden-ratio span choice for the `golden-sequence` online strategy.
- Usage: called by `transform_pinyin_predict_train`; the returned value is a 1-indexed choice in `[1, count_chinese]`.
- Behavior: computes `relative = (base + epoch·φ) mod 1` where `φ` is the golden ratio; maps it onto the Chinese segment index. The same (text, epoch) always yields the same choice, spreading coverage evenly across epochs.

#### `_select_span(labels, k, direction="backward")`
- Functionality: 1-indexed Chinese-segment choice -> `(sel, end, chinese_offset)` (identical logic to `preprocessor.select_span`).
- Behavior: `backward` (extend right), `forward` (extend left), `bidirectional` (both sides); `ValueError` on unknown direction.

### Transforms

#### `transform_pinyin_predict_train(batch, tokenizer, aug_cfg=None, online_policy=None, epoch=None)`
- Functionality: converts a batch of preprocessed samples into `prefix_ids` / `postfix_ids` / `target_ids` lists (plus lengths).
- Usage: the training transform; applied inside `P2CStreamingDataset` and per-batch in the trainer.
- Behavior: parses the JSON pinyin; skips samples without Chinese; chooses span `k` via `golden-sequence` (epoch-dependent, deterministic) or `random` strategy; with `no_context_prob` probability uses an empty prefix (bidirectional span), otherwise a forward/backward span with probability `forward_prob` (default 0.5 when not configured); the prefix is `[bos_token] + encode_context(prefix_text)`; the suffix pinyin is the flattened per-char readings for the suffix span, resolved (heteronym confusion) and augmented (vowel dropping); `postfix_ids` via `encode_pinyin`, `target_ids` via `encode_chinese`. Samples without Chinese are skipped entirely.

#### `transform_pinyin_predict_val(batch, tokenizer, aug_cfg=None)`
- Functionality: converts a prepared val batch (`prefix` / `suffix` / `pinyin` columns) into ids.
- Behavior: prefix is `[bos_token] + encode_context(prefix_text)`; postfix via `encode_pinyin`; target via `encode_chinese`; no online augmentation (already materialized offline).

### Collate

#### `make_collate_fn()`
- Functionality: builds the NJT collate function.
- Usage: passed to `StreamingDataLoader` and the val `DataLoader`.
- Behavior: converts each of `prefix_ids` / `postfix_ids` / `target_ids` into a jagged nested tensor (`torch.nested.nested_tensor(..., layout=jagged)`); returns the batch dict keyed by `prefix_ids_njt`, `postfix_ids_njt`, `target_ids_njt`.

### Loading

#### `create_dataset(data_path, keep_in_memory=False)`
- Functionality: loads an HF dataset directory from disk.
- Behavior: `load_from_disk(data_path, keep_in_memory=keep_in_memory)`.

### `P2CStreamingDataset(StreamingDataset)`
- Functionality: streaming wrapper over the MDS training split that applies the online training transform per item.
- Usage: constructed by the trainer with `local`, `tokenizer`, `aug_cfg`, `online_policy`, plus streaming kwargs (shuffle, cache_limit, ...).
- Behavior: `__getitem__` wraps a raw MDS sample into a single-item batch, calls `transform_pinyin_predict_train` with `epoch=self.next_epoch - 1` (so the golden-sequence seed starts at epoch 0), and returns the first item of each result list.
