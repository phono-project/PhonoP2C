"""
preprocessor.py — segment-based pipeline (flat per-character pinyin format).

Unified preprocessing for the P2C pretraining corpus, driven by the merged
dataset configuration (see config/dataset/pretrain_v2.yaml).

Stored sample columns
---------------------
  text_list   : list[str]        — the segments
  labels      : list[int]        — per-segment property id
  pinyin_list : str (JSON)       — FLAT pinyin, one entry per Chinese character
                                   (most probable reading only), concatenated
                                   over the Chinese segments in order.

Segmentation modes
------------------
  * "word": a jieba word is one Chinese segment (the default legacy behavior).
  * "char": every Chinese character is its own segment; jieba word detection
    is still used for pronunciation disambiguation (the run is handed to
    jieba/pypinyin as a whole, then split into characters).

The most probable per-character pronunciations observed while processing are
counted, normalized into probabilities, and saved to
``characters_pronounce_frequency.json`` (a part of the tokenizer, referenced
from ``vocabs/config.yaml``).

Train output: MosaicML StreamingDataset (MDS), written in parallel by
NUM_PROC workers, then merged via ``streaming.base.util.merge_index``.

Val output: HF ``save_to_disk`` (Arrow), with prefix/suffix/pinyin materialized.
"""

import argparse
import glob
import json
import logging
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import get_context

import jieba
import numpy as np
import yaml
import zhconv_rs

import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets
from pypinyin import pinyin, Style
from tqdm import tqdm

from streaming import MDSWriter
from streaming.base.util import merge_index

from datasets_pipeline.constants import (
    CHINESE_LABEL,
    NON_CHINESE_LABEL,
    PAUSE_LABEL,
    PAUSE_PUNCT,
    _PAUSE_SET,
)
from datasets_pipeline.pinyin import build_val_pinyin
from datasets_pipeline.segments import chinese_segment_char_prefixes, select_span

jieba.setLogLevel(logging.WARNING)

# ── Defaults (overridable through the dataset config) ────────────────────────
DEFAULT_CONFIG = {
    "vocabs_config": "./vocabs/config.yaml",
    "segmentation": {"mode": "char"},
    "sources": {
        "jsonl": ["./datasets/pretrain_base/LCCC", "./datasets/pretrain_base/MMC"],
        "textonly": ["./datasets/pretrain_base/CLUE"],
        "parquet": [
            {"path": "./datasets/pretrain_base/zhihu-kol", "columns": ["INSTRUCTION", "RESPONSE"]},
        ],
    },
    "sampling": {
        "truncation_len_min": 16,
        "truncation_len_max": 64,
        "max_text_len": 72,
        "ban_text_len": 96,
        "max_non_chinese_seg_len": 16,
        "min_chinese_ratio": 0.5,
        "split_ratio": 0.995,
    },
    "output": {
        "train_dir": "./datasets/pretrain_v2/train",
        "val_dir": "./datasets/pretrain_v2/val",
        "val_original_dir": "./datasets/pretrain_v2/val_original",
        "train_original_dir": "./datasets/pretrain_v2/train_original",
        "num_val_shards": 1,
        "num_original_shards": 15,
    },
    "mds": {
        "compression": "zstd",
        "size_limit": 1 << 28,
    },
    "val_augmentation": {
        "drop_vowels": 0.2,
        "drop_last_vowel": 0.3,
        "vowels_droprate": [0.5, 1.0],
        "heteronym_confusion": 0.1,
        "no_context_prob": 0.2,
        "forward_ratio": 0.0,
    },
    "processing": {
        "num_proc": 15,
        "batch_size": 8,
        "shuffle_seed": 114514,
    },
}

_MDS_COLUMNS = {
    "text_list": "json",
    "pinyin_list": "str",
    "labels": "json",
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _chinese_char_ratio(text: str) -> float:
    return sum(1 for ch in text if re.match(r'[\u4e00-\u9fff]', ch)) / len(text) if text else 0


def _clean_emoji(text: str) -> str:
    return re.sub(r'\[[^\]]+\]', '', text)


def _remove_segmentation_spaces(segmented_text):
    return re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', segmented_text)


def word_to_pinyin_nested(word: str) -> tuple[str, list[list[str]]]:
    """Convert a Chinese *word* into (clean_word, nested_pinyin).

    ``nested_pinyin`` has exactly one entry per character in ``clean_word``;
    each entry is the list of possible readings for that character with the
    most-likely reading placed first (multi-pronunciation info retained).
    Characters with no obtainable pinyin are dropped from ``clean_word`` so
    the character sequence and pinyin sequence stay aligned.
    """
    best_matrix = pinyin(word, heteronym=False, strict=False,
                         errors='ignore', style=Style.NORMAL, v_to_u=False)
    best_flat = [item for sub in best_matrix for item in sub]

    clean_chars: list[str] = []
    nested: list[list[str]] = []
    best_idx = 0

    for ch in word:
        all_matrix = pinyin(ch, heteronym=True, strict=False,
                            errors='ignore', style=Style.NORMAL, v_to_u=False)
        sounds = []
        if all_matrix and all_matrix[0]:
            sounds = [s for s in all_matrix[0] if s != 'ê']
        if not sounds:
            continue  # drop char with no pinyin (keeps alignment)

        best = best_flat[best_idx] if best_idx < len(best_flat) else sounds[0]
        best_idx += 1

        if best in sounds and sounds[0] != best:
            sounds = [best] + [s for s in sounds if s != best]

        clean_chars.append(ch)
        nested.append(sounds)

    return "".join(clean_chars), nested


# ── Text normalization & segmentation ─────────────────────────────────────────
def preprocess_text(raw_text: str, tokenizer) -> str:
    """t2s -> NFKC -> context-vocab filter."""
    if not raw_text:
        return ""
    text = _clean_emoji(raw_text)
    text = _remove_segmentation_spaces(text)
    text = zhconv_rs.zhconv(text, "zh-hans")
    text = unicodedata.normalize("NFKC", text)
    text = tokenizer.filter_text(text, allow_characters=PAUSE_PUNCT, vocab="context")
    return text


def _char_category(ch: str, tokenizer) -> int:
    if ch in _PAUSE_SET:
        return PAUSE_LABEL
    if tokenizer.is_chinese(ch):
        return CHINESE_LABEL
    return NON_CHINESE_LABEL


def segment_text(text: str, tokenizer,
                 max_non_chinese_len: int = 16,
                 mode: str = "word",
                 min_chinese_ratio: float = 0.5):
    """Split *text* into (segments, labels, seg_pinyin).

    Rules:
      - consecutive pause punctuation -> one PAUSE segment (first char kept)
      - consecutive Chinese chars -> jieba words; each word becomes one
        CHINESE segment (``mode="word"``) or is split into single characters
        (``mode="char"``)
      - consecutive non-Chinese chars -> one NON_CHINESE segment, dropped if
        longer than ``max_non_chinese_len``

    ``seg_pinyin`` is aligned with ``segs``: the nested per-character readings
    for Chinese segments, ``None`` for the others.  In character mode the
    Chinese run is still handed to jieba/pypinyin as a whole so word context
    disambiguates pronunciation; only afterwards is it split into characters.
    """
    segs: list[str] = []
    labels: list[int] = []
    seg_pinyin: list = []

    if not text:
        return segs, labels, seg_pinyin

    if _chinese_char_ratio(text) < min_chinese_ratio:
        return segs, labels, seg_pinyin  # drop segments with too few Chinese characters

    # Group text into maximal runs of the same category.
    runs: list[tuple[int, str]] = []
    cur_cat = _char_category(text[0], tokenizer)
    cur_buf = [text[0]]
    for ch in text[1:]:
        cat = _char_category(ch, tokenizer)
        if cat == cur_cat:
            cur_buf.append(ch)
        else:
            runs.append((cur_cat, "".join(cur_buf)))
            cur_cat = cat
            cur_buf = [ch]
    runs.append((cur_cat, "".join(cur_buf)))

    for cat, run in runs:
        if cat == PAUSE_LABEL:
            segs.append(run[0])          # normalize the run to a single symbol
            labels.append(PAUSE_LABEL)
            seg_pinyin.append(None)
        elif cat == CHINESE_LABEL:
            # jieba detects words on the whole run first — even in char mode —
            # for accurate pronunciation detection.
            for word in jieba.lcut(run):
                if not word:
                    continue
                clean_word, nested = word_to_pinyin_nested(word)
                if not clean_word:
                    continue
                if mode == "word":
                    segs.append(clean_word)
                    labels.append(CHINESE_LABEL)
                    seg_pinyin.append(nested)
                else:  # char mode: one segment per character
                    for ch, sounds in zip(clean_word, nested):
                        segs.append(ch)
                        labels.append(CHINESE_LABEL)
                        seg_pinyin.append([sounds])
        else:  # NON_CHINESE_LABEL
            if len(run) <= max_non_chinese_len:
                segs.append(run)
                labels.append(NON_CHINESE_LABEL)
                seg_pinyin.append(None)

    return segs, labels, seg_pinyin


def build_samples(segs, labels, seg_pinyin, sampling_cfg: dict):
    """Slice (segs, labels, seg_pinyin) into samples.

    Yields dicts: {"text_list", "pinyin_list" (JSON str), "labels"}.
    """
    trunc_min = sampling_cfg["truncation_len_min"]
    trunc_max = sampling_cfg["truncation_len_max"]
    max_text_len = sampling_cfg["max_text_len"]
    ban_text_len = sampling_cfg["ban_text_len"]

    n = len(segs)
    i = 0
    while i < n:
        trunc_len = random.randint(trunc_min, trunc_max)

        # 1. accumulate until the total length reaches trunc_len
        j = i
        total = 0
        while j < n and total < trunc_len:
            total += len(segs[j])
            j += 1

        if total > ban_text_len:
            i = j
            continue

        exceeded = False
        dropped = False

        # 2. keep going until a PAUSE segment preceded by a CHINESE segment.
        while j < n:
            total += len(segs[j])
            if total > max_text_len:
                exceeded = True
                if total > ban_text_len:
                    dropped = True
                    break
            if exceeded:
                if labels[j] != CHINESE_LABEL and j - 1 >= i and labels[j - 1] == CHINESE_LABEL:
                    break
            else:
                if labels[j] == PAUSE_LABEL and j - 1 >= i and labels[j - 1] == CHINESE_LABEL:
                    break
            j += 1

        if dropped:
            i = j
            continue

        _emit = _make_sample(segs[i:j], labels[i:j], seg_pinyin[i:j],
                             sampling_cfg["min_chinese_ratio"])
        if _emit is not None:
            yield _emit

        # skip the pause segment (if we stopped on one)
        i = j + 1 if (j < n and labels[j] == PAUSE_LABEL) else j


def _make_sample(sample_segs, sample_labels, sample_pinyin, min_chinese_ratio):
    """Pack a sample with the FLAT per-character pinyin list."""
    if not sample_segs:
        return None

    text_list: list[str] = []
    out_labels: list[int] = []
    pinyin_flat: list[str] = []

    has_chinese = False
    for seg, lab, py in zip(sample_segs, sample_labels, sample_pinyin):
        if lab == CHINESE_LABEL:
            text_list.append(seg)
            out_labels.append(CHINESE_LABEL)
            for sounds in py:
                pinyin_flat.append(sounds[0] if sounds else "")
            has_chinese = True
        else:
            text_list.append(seg)
            out_labels.append(lab)

    if not has_chinese:
        return None

    final_text = "".join(text_list)
    if _chinese_char_ratio(final_text) < min_chinese_ratio:
        return None  # drop segments with too few Chinese characters

    return {
        "text_list": text_list,
        "pinyin_list": json.dumps(pinyin_flat, ensure_ascii=False),
        "labels": out_labels,
    }


def process_raw_text(raw_text: str, tokenizer, sampling_cfg: dict,
                     max_non_chinese_len: int, mode: str):
    """Full raw string -> list of sample dicts."""
    text = preprocess_text(raw_text, tokenizer)
    if len(text) <= 1:
        return []
    segs, labels, seg_pinyin = segment_text(
        text, tokenizer, max_non_chinese_len, mode,
        sampling_cfg["min_chinese_ratio"],
    )
    return list(build_samples(segs, labels, seg_pinyin, sampling_cfg))


# ── Batch processors ──────────────────────────────────────────────────────────
def process_pure_batch(batch, tokenizer, sampling_cfg, max_non_chinese_len, mode):
    out_text, out_py, out_labels = [], [], []
    for sentence_list in batch["sentences"]:
        for raw_text in sentence_list:
            for s in process_raw_text(raw_text, tokenizer, sampling_cfg,
                                      max_non_chinese_len, mode):
                out_text.append(s["text_list"])
                out_py.append(s["pinyin_list"])
                out_labels.append(s["labels"])
    return {"text_list": out_text, "pinyin_list": out_py, "labels": out_labels}


def process_parquet_batch(batch, tokenizer, text_columns, sampling_cfg,
                          max_non_chinese_len, mode):
    out_text, out_py, out_labels = [], [], []
    for col in text_columns:
        if col not in batch:
            continue
        for raw_text in batch[col]:
            if not raw_text:
                continue
            for s in process_raw_text(raw_text, tokenizer, sampling_cfg,
                                      max_non_chinese_len, mode):
                out_text.append(s["text_list"])
                out_py.append(s["pinyin_list"])
                out_labels.append(s["labels"])
    return {"text_list": out_text, "pinyin_list": out_py, "labels": out_labels}


# ── Character pronunciation frequencies ───────────────────────────────────────
def count_character_pinyins(ds: Dataset) -> dict[str, dict[str, float]]:
    """Count the most probable pinyin for every character over the dataset.

    Returns {char: {pinyin: probability}} normalized per character.
    """
    freq: dict[str, Counter] = defaultdict(Counter)
    for sample in tqdm(ds, desc="Counting pronunciations"):
        text_list = sample["text_list"]
        labels = sample["labels"]
        pinyin_raw = sample["pinyin_list"]
        pinyin_list = json.loads(pinyin_raw) if isinstance(pinyin_raw, str) else pinyin_raw

        cursor = 0
        for seg, lab in zip(text_list, labels):
            if lab == CHINESE_LABEL:
                for ch, py in zip(seg, pinyin_list[cursor : cursor + len(seg)]):
                    if py:
                        freq[ch][py] += 1
                cursor += len(seg)

    normalized: dict[str, dict[str, float]] = {}
    for ch, counts in freq.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        normalized[ch] = {py: c / total for py, c in counts.items()}
    return normalized


def save_character_pinyin_frequency(freq: dict, vocabs_config_path: str) -> str:
    """Write the frequency map next to the tokenizer vocab files."""
    config_dir = os.path.dirname(os.path.abspath(vocabs_config_path))
    with open(vocabs_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rel = cfg["vocabs"].get("characters_pronounce_frequency",
                            "characters_pronounce_frequency.json")
    out_path = os.path.join(config_dir, rel)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(freq, f, ensure_ascii=False, indent=2)
    return out_path


# ── Loaders ───────────────────────────────────────────────────────────────────
def _find_files(path: str, suffix: str) -> list[str]:
    suffix_glob = "*." + suffix
    if not path or not os.path.isdir(path):
        return []
    return sorted(glob.glob(os.path.join(path, "**", suffix_glob), recursive=True))


def load_and_process_jsonl_source(tokenizer, jsonl_path, sampling_cfg,
                                  max_non_chinese_len, mode, batch_size, num_proc):
    jsonl_files = _find_files(jsonl_path, "jsonl")
    print(f"[source] {jsonl_path}: {len(jsonl_files)} JSONL file(s) found")
    if not jsonl_files:
        print(f"[skip] JSONL not found: {jsonl_path}")
        return None

    def jsonl_generator():
        for json_file in jsonl_files:
            with open(json_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items = json.loads(line)
                        if isinstance(items, list):
                            yield {"sentences": items}
                    except json.JSONDecodeError:
                        continue

    ds = Dataset.from_generator(jsonl_generator)
    ds = ds.map(
        lambda batch: process_pure_batch(batch, tokenizer, sampling_cfg,
                                         max_non_chinese_len, mode),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Processing JSONL",
    )
    return ds


def load_and_process_textonly_source(tokenizer, textonly_path, sampling_cfg,
                                     max_non_chinese_len, mode, batch_size, num_proc):
    text_files = _find_files(textonly_path, "txt")
    print(f"[source] {textonly_path}: {len(text_files)} text file(s) found")
    if not text_files:
        print(f"[skip] Textfile not found: {textonly_path}")
        return None

    def text_generator():
        for text_file in text_files:
            with open(text_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield {"sentences": [line]}

    ds = Dataset.from_generator(text_generator)
    ds = ds.map(
        lambda batch: process_pure_batch(batch, tokenizer, sampling_cfg,
                                         max_non_chinese_len, mode),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Processing TextOnly",
    )
    return ds


def load_and_process_parquet_source(path, text_columns, tokenizer, sampling_cfg,
                                    max_non_chinese_len, mode, batch_size, num_proc):
    """Process one parquet source shard-by-shard."""
    files = _find_files(path, "parquet")
    if not files:
        print(f"[skip] no parquet under: {path}")
        return None

    random.shuffle(files)  # pick shards in random order
    print(f"[source] {path}: {len(files)} shard(s), columns={text_columns}")

    processed_shards = []
    for shard_path in tqdm(files, desc=f"Shards of {os.path.basename(os.path.normpath(path))}"):
        meta = pq.read_metadata(shard_path)
        available_cols = [c.name for c in meta.schema]
        present = [c for c in text_columns if c in available_cols]
        if not present:
            continue

        shard_ds = Dataset.from_parquet(
            shard_path,
            columns=present,
            keep_in_memory=False
        )

        shard_ds = shard_ds.map(
            lambda batch: process_parquet_batch(batch, tokenizer, present, sampling_cfg,
                                                max_non_chinese_len, mode),
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=shard_ds.column_names,
            desc=f"Processing {os.path.basename(shard_path)}",
        )
        processed_shards.append(shard_ds)

    if not processed_shards:
        return None
    return concatenate_datasets(processed_shards)


# ── Train / val split & writing ───────────────────────────────────────────────
def shuffled_train_test_split(ds: Dataset, test_size: float, seed: int):
    n = len(ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test = int(round(n * test_size))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])

    train_ds = ds.select(train_idx.tolist())
    test_ds = ds.select(test_idx.tolist())
    return train_ds, test_ds


def count_characters(ds: Dataset, num_proc: int) -> int:
    def _count(batch):
        return {"_nchars": [sum(len(t) for t in tl) for tl in batch["text_list"]]}

    counts = ds.map(
        _count,
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Counting characters",
    )
    return int(sum(counts["_nchars"]))


def _write_mds_partition(args):
    num_proc, shard_id, dataset_disk_path, out_dir, compression, size_limit = args
    from datasets import Dataset

    dataset = Dataset.load_from_disk(dataset_disk_path)  # mmap
    local = os.path.join(out_dir, str(shard_id))
    subset = dataset.shard(num_proc, shard_id, contiguous=True)

    with MDSWriter(
        out=local,
        columns=_MDS_COLUMNS,
        compression=compression,
        size_limit=size_limit,
    ) as writer:
        for sample in subset:
            writer.write({
                "text_list": sample["text_list"],
                "pinyin_list": sample["pinyin_list"],
                "labels": sample["labels"],
            })
    return shard_id


def save_train_as_mds(dataset: Dataset, out_dir: str, train_original_dir: str,
                      num_proc: int, num_original_shards: int,
                      compression, size_limit):
    n = len(dataset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(train_original_dir, exist_ok=True)

    dataset.save_to_disk(train_original_dir, num_shards=num_original_shards)

    boundaries = np.linspace(0, n, num_proc + 1, dtype=int)
    tasks = [
        (num_proc, i, train_original_dir, out_dir, compression, size_limit)
        for i in range(num_proc)
        if boundaries[i + 1] > boundaries[i]
    ]

    ctx = get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        for _ in tqdm(pool.imap_unordered(_write_mds_partition, tasks),
                      total=len(tasks), desc="Saving MDS partitions"):
            pass

    merge_index(out_dir, keep_local=True)


# ── Val preparation ───────────────────────────────────────────────────────────
def _prepare_val_batch(batch, tokenizer, val_cfg):
    out_prefix, out_suffix, out_pinyin = [], [], []

    no_context_prob = val_cfg.get("no_context_prob", 0.0)
    forward_ratio = val_cfg.get("forward_ratio", 0.0)

    for text_list, pinyin_json, labels in zip(
        batch["text_list"], batch["pinyin_list"], batch["labels"]
    ):
        pinyin_list = json.loads(pinyin_json) if isinstance(pinyin_json, str) else pinyin_json
        labels = list(labels)
        count_chinese = sum(1 for lab in labels if lab == CHINESE_LABEL)
        if count_chinese < 1:
            continue

        k = random.randint(1, count_chinese)
        if random.random() < no_context_prob:
            sel, end, ch_off = select_span(labels, k, "bidirectional")
            prefix_text = ""
        else:
            direction = "forward" if random.random() < forward_ratio else "backward"
            sel, end, ch_off = select_span(labels, k, direction)
            prefix_text = "".join(text_list[:sel])

        suffix_text = "".join(text_list[sel:end])

        char_prefix = chinese_segment_char_prefixes(text_list, labels)
        char_off = char_prefix[sel]
        num_chars = char_prefix[end] - char_prefix[sel]
        pinyin_slice = pinyin_list[char_off : char_off + num_chars]

        aug_syllables = build_val_pinyin(pinyin_slice, suffix_text, tokenizer, val_cfg)

        out_prefix.append(prefix_text)
        out_suffix.append(suffix_text)
        out_pinyin.append(json.dumps(aug_syllables, ensure_ascii=False))

    return {"prefix": out_prefix, "suffix": out_suffix, "pinyin": out_pinyin}


# ── Main pipeline ─────────────────────────────────────────────────────────────
def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_placeholders(value, cfg):
    """Resolve ${dataset.dataset_dir}-style placeholders in config strings."""
    if isinstance(value, str):
        dataset_dir = cfg.get("dataset_dir", "")
        value = value.replace("${dataset.dataset_dir}", dataset_dir)
        return value
    if isinstance(value, list):
        return [_resolve_placeholders(v, cfg) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, cfg) for k, v in value.items()}
    return value


def run_preprocess(cfg: dict, generate_val: bool = True):
    """Run the full preprocessing pipeline from a merged dataset config."""
    merged = _deep_merge(DEFAULT_CONFIG, cfg)
    merged = _resolve_placeholders(merged, merged)

    seg_cfg = merged["segmentation"]
    mode = seg_cfg.get("mode", "word")
    if mode not in ("char", "word"):
        raise ValueError(f"Unknown segmentation mode: {mode!r} (expected 'char' or 'word')")

    sampling = merged["sampling"]
    output = merged["output"]
    mds = merged["mds"]
    val_cfg = merged["val_augmentation"]
    proc = merged["processing"]

    vocabs_config = merged["vocabs_config"]

    from tokenizer import P2CTokenizer

    tokenizer = P2CTokenizer.from_config(vocabs_config)

    datasets_to_concat = []
    for jsonl_path in merged["sources"].get("jsonl", []):
        ds_jsonl = load_and_process_jsonl_source(
            tokenizer, jsonl_path, sampling, sampling["max_non_chinese_seg_len"],
            mode, proc["batch_size"], proc["num_proc"],
        )
        if ds_jsonl is not None:
            datasets_to_concat.append(ds_jsonl)

    for textonly_path in merged["sources"].get("textonly", []):
        ds_textonly = load_and_process_textonly_source(
            tokenizer, textonly_path, sampling, sampling["max_non_chinese_seg_len"],
            mode, proc["batch_size"], proc["num_proc"],
        )
        if ds_textonly is not None:
            datasets_to_concat.append(ds_textonly)

    for entry in merged["sources"].get("parquet", []):
        ds_src = load_and_process_parquet_source(
            entry["path"], entry["columns"], tokenizer, sampling,
            sampling["max_non_chinese_seg_len"], mode,
            proc["batch_size"], proc["num_proc"],
        )
        if ds_src is not None:
            datasets_to_concat.append(ds_src)

    if not datasets_to_concat:
        raise RuntimeError("No data sources found — nothing to process.")

    ds = concatenate_datasets(datasets_to_concat)

    # Count per-character pronunciations (normalized frequencies) and save
    # them as a part of the tokenizer.
    freq = count_character_pinyins(ds)
    freq_path = save_character_pinyin_frequency(freq, vocabs_config)
    print(f"Character pinyin frequencies -> {freq_path} ({len(freq)} chars)")

    total_chars = count_characters(ds, proc["num_proc"])

    # Train / val split
    train_ds, val_ds = shuffled_train_test_split(
        ds, test_size=1 - sampling["split_ratio"], seed=proc["shuffle_seed"]
    )

    # Train -> MDS parallel
    save_train_as_mds(
        train_ds, output["train_dir"], output["train_original_dir"],
        proc["num_proc"], output["num_original_shards"],
        mds["compression"], mds["size_limit"],
    )

    # Save original validation dataset
    val_ds.save_to_disk(output["val_original_dir"], num_shards=output["num_val_shards"])

    if generate_val:
        val_ds = Dataset.load_from_disk(output["val_original_dir"])
        val_ds_prepared = val_ds.map(
            lambda batch: _prepare_val_batch(batch, tokenizer, val_cfg),
            batched=True,
            batch_size=1024,
            num_proc=1,
            desc="Preparing val",
            remove_columns=val_ds.column_names,
        )
        val_ds_prepared.save_to_disk(output["val_dir"], num_shards=output["num_val_shards"])

    print(f"Total samples    : {len(ds) / 1e6:.1f}M rows")
    print(f"Total characters : {total_chars / 1e9:.2f}B")
    print(f"Train            : {len(train_ds) / 1e6:.1f}M rows -> {output['train_dir']} (MDS)")
    print(f"Val              : -> {output['val_dir']} (HF Datasets)")
    print(f"Segmentation mode: {mode}")


# CLI fallback (hydra is the primary entry point via main.py).
def main():
    parser = argparse.ArgumentParser(description="PhonoP2C preprocessing pipeline")
    parser.add_argument("--config", default="config/dataset/pretrain_v2.yaml",
                        help="Merged dataset/preprocessor config yaml")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run the preprocessing pipeline to generate train/val datasets.")
    parser.add_argument("--generate_val", action="store_true",
                        help="Generate validation dataset from existing preprocessed data.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.preprocess:
        run_preprocess(cfg, generate_val=args.generate_val)
    elif args.generate_val:
        from tokenizer import P2CTokenizer

        merged = _resolve_placeholders(_deep_merge(DEFAULT_CONFIG, cfg), cfg)
        tokenizer = P2CTokenizer.from_config(merged["vocabs_config"])
        val_ds = Dataset.load_from_disk(merged["output"]["val_original_dir"])
        val_ds_prepared = val_ds.map(
            lambda batch: _prepare_val_batch(batch, tokenizer, merged["val_augmentation"]),
            batched=True,
            batch_size=1024,
            num_proc=1,
            desc="Preparing val",
            remove_columns=val_ds.column_names,
        )
        val_ds_prepared.save_to_disk(merged["output"]["val_dir"],
                                     num_shards=merged["output"]["num_val_shards"])


if __name__ == "__main__":
    main()
