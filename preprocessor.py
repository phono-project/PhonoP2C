"""
preprocessor.py — segment-based pipeline.

Unified preprocessing for the P2C pretraining corpus.

Stored sample columns
---------------------
  text_list   : list[str]        — the segments
  labels      : list[int]        — per-segment property id
  pinyin_list : str (JSON)       — nested pinyin, one entry per *Chinese* segment
                                   [[[reading,...],  # char 0 of the word
                                     [reading,...]], # char 1 of the word
                                    ...]

Train output: MosaicML StreamingDataset (MDS), written in parallel by
NUM_PROC workers (index shuffled -> split -> sorted for sequential reads -> each
worker writes its own partition) then merged via ``streaming.base.util.merge_index``.

Val output: HF ``save_to_disk`` (Arrow), with prefix/suffix/pinyin materialized.
"""

import glob
import json
import os
import random
import re
import logging
import unicodedata
import argparse
from multiprocessing import get_context

import jieba
import numpy as np
import zhconv_rs

import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets
from pypinyin import pinyin, Style
from tqdm import tqdm

from streaming import MDSWriter
from streaming.base.util import merge_index

jieba.setLogLevel(logging.WARNING)

# ── Global configuration ──────────────────────────────────────────────────────
LCCC_PATH            = "./datasets/pretrain_base/LCCC"
MMC_PATH             = "./datasets/pretrain_base/MMC"
CLUE_PATH            = "./datasets/pretrain_base/CLUE"
WIKIPEDIA_PATH       = "./datasets/pretrain_base/wikipedia"
ZHIHUKOL_PATH        = "./datasets/pretrain_base/zhihu-kol"
FINEWEB_EDU_CHS_PATH = "./datasets/pretrain_base/fw_subset"

# Each parquet source: (path, [text columns to process]).
PARQUET_SOURCES = [
    (ZHIHUKOL_PATH,        ["INSTRUCTION", "RESPONSE"]),
    #(FINEWEB_EDU_CHS_PATH, ["text"]),
]

JSONL_SOURCES = [
    LCCC_PATH,
    MMC_PATH
]

TEXTONLY_SOURCES = [
    CLUE_PATH,
]

VOCABS_CONFIG        = "./vocabs/config.yaml"

TRUNCATION_LEN_MIN   = 16        # Min sample length (chars) target
TRUNCATION_LEN_MAX   = 64        # Max sample length (chars) target
MAX_TEXT_LEN         = 72        # Max text length (chars) in sample
BAN_TEXT_LEN         = 96        # Drop segments longer than this (chars)

MAX_NON_CHINESE_SEG_LEN = 16     # Non-Chinese segments longer than this are dropped
MIN_CHINESE_RATIO    = 0.5       # Minimum ratio of Chinese characters in a segment

SPLIT_RATIO          = 0.995      # Train/val split ratio

TRAIN_OUT_DIR        = "./datasets/pretrain_v1/train"
VAL_OUT_DIR          = "./datasets/pretrain_v1/val"
VAL_ORIG_OUT_DIR     = "./datasets/pretrain_v1/val_original"
TRAIN_ORIG_OUT_DIR   = "./datasets/pretrain_v1/train_original"
NUM_VAL_SHARDS       = 1
NUM_ORIGIN_SHARDS    = 15

SHUFFLE_SEED         = 114514
NUM_PROC             = 15
BATCH_SIZE           = 8

# MDS output
MDS_COMPRESSION      = "zstd"    # set to None to disable compression
MDS_SIZE_LIMIT       = 1 << 28   # ~256MB per shard

_MDS_COLUMNS = {
    "text_list": "json",
    "pinyin_list": "str",
    "labels": "json",
}

# Val augmentation parameters
DROP_VOWELS         = 0.2
DROP_LAST_VOWEL     = 0.3
VOWELS_DROPRATE_MIN = 0.5
VOWELS_DROPRATE_MAX = 1.0
HETERONYM_CONFUSION = 0.1
NO_CONTEXT_PROB     = 0.2
FORWARD_RATIO       = 0.0

# Segment labels
PAUSE_LABEL       = 0
CHINESE_LABEL     = 1
NON_CHINESE_LABEL = 2

# Pause-meaning punctuation (half-width, after NFKC).
PAUSE_PUNCT = ["!", "?", "。", "\n", ":", ";", ","]
_PAUSE_SET = set(PAUSE_PUNCT)

_COMP_CONSONANTS = {"zh", "ch", "sh"}
_SINGLE_INITIALS = {
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "y", "w", "z", "c", "s",
}


# helpers
def _get_initial_and_final(syllable: str) -> tuple[str, str]:
    if len(syllable) >= 2 and syllable[:2] in _COMP_CONSONANTS:
        return syllable[:2], syllable[2:]
    if syllable and syllable[0] in _SINGLE_INITIALS:
        return syllable[0], syllable[1:]
    return "", syllable

def _chinese_char_ratio(text: str) -> float:
    """Count the number of Chinese characters in a string."""
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
    Characters with no obtainable pinyin are dropped from ``clean_word`` so the
    character sequence and pinyin sequence stay aligned.
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


def resolve_pinyin_list(pinyin_raw: list, heteronym_confusion_p: float = 0.0) -> list[str]:
    """Flatten a list of per-char reading lists into a single reading each."""
    result = []
    for item in pinyin_raw:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            if heteronym_confusion_p > 0.0 and len(item) > 1 and random.random() < heteronym_confusion_p:
                result.append(random.choice(item[1:]))
            else:
                result.append(item[0])
    return result


def build_val_pinyin(pinyin_flat_readings: list) -> list[str]:
    """Static val augmentation on a flat list of per-char reading lists."""
    syllables = resolve_pinyin_list(pinyin_flat_readings, HETERONYM_CONFUSION)

    if random.random() < DROP_VOWELS:
        for idx in range(len(syllables)):
            initial, final = _get_initial_and_final(syllables[idx])
            if initial and final:
                drop_threshold = random.uniform(VOWELS_DROPRATE_MIN, VOWELS_DROPRATE_MAX)
                if random.random() < drop_threshold:
                    syllables[idx] = initial[0]

    if random.random() < DROP_LAST_VOWEL and len(syllables) > 0:
        last_idx = len(syllables) - 1
        last_initial, last_final = _get_initial_and_final(syllables[last_idx])
        if last_initial and last_final:
            syllables[last_idx] = last_initial[0]

    return syllables


# Segmentation
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
                 max_non_chinese_len: int = MAX_NON_CHINESE_SEG_LEN):
    """Split *text* into (segments, labels).

    Rules:
      - consecutive pause punctuation -> one PAUSE segment (first char kept)
      - consecutive Chinese chars -> jieba words, each a CHINESE segment
      - consecutive non-Chinese chars -> one NON_CHINESE segment, dropped if
        longer than ``max_non_chinese_len``
    """
    segs: list[str] = []
    labels: list[int] = []

    if not text:
        return segs, labels
    
    if _chinese_char_ratio(text) < MIN_CHINESE_RATIO:
        return segs, labels  # drop segments with too few Chinese characters

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
        elif cat == CHINESE_LABEL:
            for word in jieba.lcut(run):
                if not word:
                    continue
                segs.append(word)
                labels.append(CHINESE_LABEL)
        else:  # NON_CHINESE_LABEL
            if len(run) <= max_non_chinese_len:
                segs.append(run)
                labels.append(NON_CHINESE_LABEL)

    return segs, labels


def build_samples(segs: list[str], labels: list[int]):
    """Slice (segs, labels) into samples.

    Yields dicts: {"text_list", "pinyin_list" (JSON str), "labels"}.
    """
    n = len(segs)
    i = 0
    while i < n:
        trunc_len = random.randint(TRUNCATION_LEN_MIN, TRUNCATION_LEN_MAX)

        # 1. accumulate until the total length reaches trunc_len
        j = i
        total = 0
        while j < n and total < trunc_len:
            total += len(segs[j])
            j += 1

        if total > BAN_TEXT_LEN:
            i = j
            continue
        
        exceeded = False
        dropped = False
        
        # 2. keep going until a PAUSE segment preceded by a CHINESE segment.
        while j < n:
            total += len(segs[j])
            if total > MAX_TEXT_LEN:
                exceeded = True
                if total > BAN_TEXT_LEN:
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
        
        sample_segs = segs[i:j]
        sample_labels = labels[i:j]
        
        _emit = _make_sample(sample_segs, sample_labels)
        if _emit is not None:
            yield _emit

        # skip the pause segment (if we stopped on one)
        i = j + 1 if (j < n and labels[j] == PAUSE_LABEL) else j


def _make_sample(sample_segs: list[str], sample_labels: list[int]):
    """Compute nested pinyin for the Chinese segments and pack a sample."""
    if not sample_segs:
        return None

    text_list: list[str] = []
    out_labels: list[int] = []
    pinyin_list: list[list[list[str]]] = []

    has_chinese = False
    for seg, lab in zip(sample_segs, sample_labels):
        if lab == CHINESE_LABEL:
            clean_word, nested = word_to_pinyin_nested(seg)
            if not clean_word:
                continue  # whole word had no pinyin — drop it
            text_list.append(clean_word)
            out_labels.append(CHINESE_LABEL)
            pinyin_list.append(nested)
            has_chinese = True
        else:
            text_list.append(seg)
            out_labels.append(lab)

    if not has_chinese:
        return None
    
    final_text = "".join(text_list)
    if _chinese_char_ratio(final_text) < MIN_CHINESE_RATIO:
        return None  # drop segments with too few Chinese characters

    return {
        "text_list": text_list,
        "pinyin_list": json.dumps(pinyin_list, ensure_ascii=False),
        "labels": out_labels,
    }


def process_raw_text(raw_text: str, tokenizer):
    """Full raw string -> list of sample dicts."""
    text = preprocess_text(raw_text, tokenizer)
    if len(text) <= 1:
        return []
    segs, labels = segment_text(text, tokenizer)
    return list(build_samples(segs, labels))


# Span selection
def select_span(labels: list[int], k: int, direction: str = "backward"):
    """1-indexed Chinese-segment choice *k* -> (sel, end, chinese_offset).

    prefix = segments[0:sel]; suffix = segments[sel:end] (consecutive Chinese);
    pinyin slice = [chinese_offset : chinese_offset + (end - sel)].

    Args:
        labels: List of labels.
        k: The k-th Chinese segment (1-indexed).
        direction: Search direction, "backward" (extend to the right), 
                   "forward" (extend to the left), or "bidirectional" (extend to both sides).
    """
    chinese_positions = [i for i, l in enumerate(labels) if l == CHINESE_LABEL]
    anchor = chinese_positions[k - 1]

    if direction == "backward":
        # Backward search (rightward): anchor is the start 'sel', search right for 'end'
        sel = anchor
        end = sel
        while end < len(labels) and labels[end] == CHINESE_LABEL:
            end += 1
        chinese_offset = k - 1
        
    elif direction == "forward":
        # Forward search (leftward): anchor is the end, search left for 'sel'
        end = anchor + 1
        sel = anchor
        while sel >= 0 and labels[sel] == CHINESE_LABEL:
            sel -= 1
        sel += 1  # Adjust back to the first Chinese index
        
        # Adjust the pinyin offset based on the segment length
        length = end - sel
        chinese_offset = (k - 1) - (length - 1)
        
    elif direction == "bidirectional":
        # Bidirectional search: search both left for 'sel' and right for 'end'
        sel = anchor
        while sel >= 0 and labels[sel] == CHINESE_LABEL:
            sel -= 1
        sel += 1  # Adjust back to the first Chinese index

        end = anchor
        while end < len(labels) and labels[end] == CHINESE_LABEL:
            end += 1

        # Adjust the pinyin offset based on the distance from anchor to the start index
        chinese_offset = (k - 1) - (anchor - sel)
        
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return sel, end, chinese_offset


# ---------------------------------------------------------------------------
# Batch processors
# ---------------------------------------------------------------------------

def process_pure_batch(batch, tokenizer):
    out_text, out_py, out_labels = [], [], []
    for sentence_list in batch["sentences"]:
        for raw_text in sentence_list:
            for s in process_raw_text(raw_text, tokenizer):
                out_text.append(s["text_list"])
                out_py.append(s["pinyin_list"])
                out_labels.append(s["labels"])
    return {"text_list": out_text, "pinyin_list": out_py, "labels": out_labels}


def process_parquet_batch(batch, tokenizer, text_columns):
    out_text, out_py, out_labels = [], [], []
    for col in text_columns:
        if col not in batch:
            continue
        for raw_text in batch[col]:
            if not raw_text:
                continue
            for s in process_raw_text(raw_text, tokenizer):
                out_text.append(s["text_list"])
                out_py.append(s["pinyin_list"])
                out_labels.append(s["labels"])
    return {"text_list": out_text, "pinyin_list": out_py, "labels": out_labels}


# Loaders
def _find_files(path: str, suffix: str) -> list[str]:
    suffix_glob = "*." + suffix
    if not path or not os.path.isdir(path):
        return []
    return sorted(glob.glob(os.path.join(path, "**", suffix_glob), recursive=True))


def load_and_process_jsonl_source(tokenizer, jsonl_path):
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
        lambda batch: process_pure_batch(batch, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
        desc="Processing JSONL",
    )
    return ds

def load_and_process_textonly_source(tokenizer, textonly_path):
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
        lambda batch: process_pure_batch(batch, tokenizer),
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
        desc="Processing TextOnly",
    )
    return ds

def load_and_process_parquet_source(path: str, text_columns: list[str], tokenizer):
    """Process one parquet source shard-by-shard.

    A random not-yet-processed shard is fully loaded into memory, processed,
    then unloaded before moving on to the next.  Returns the concatenated
    processed dataset, or None if the source is empty/invalid.
    """
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
            lambda batch: process_parquet_batch(batch, tokenizer, present),
            batched=True,
            batch_size=BATCH_SIZE,
            num_proc=NUM_PROC,
            remove_columns=shard_ds.column_names,
            desc=f"Processing {os.path.basename(shard_path)}",
        )
        processed_shards.append(shard_ds)

    if not processed_shards:
        return None
    return concatenate_datasets(processed_shards)


# Train / val split, random membership, sequential read-back
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


# Character counting
def count_characters(ds: Dataset) -> int:
    def _count(batch):
        return {"_nchars": [sum(len(t) for t in tl) for tl in batch["text_list"]]}

    counts = ds.map(
        _count,
        batched=True,
        batch_size=2048,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
        desc="Counting characters",
    )
    return int(sum(counts["_nchars"]))


# MDS parallel sharded writer
def _write_mds_partition(args):
    num_proc, shard_id, dataset_disk_path, out_dir = args
    from datasets import Dataset

    dataset = Dataset.load_from_disk(dataset_disk_path)  # mmap
    local = os.path.join(out_dir, str(shard_id))
    subset = dataset.shard(num_proc, shard_id, contiguous=True)

    with MDSWriter(
        out=local,
        columns=_MDS_COLUMNS,
        compression=MDS_COMPRESSION,
        size_limit=MDS_SIZE_LIMIT,
    ) as writer:
        for sample in subset:
            writer.write({
                "text_list": sample["text_list"],
                "pinyin_list": sample["pinyin_list"],
                "labels": sample["labels"],
            })
    return shard_id


def save_train_as_mds(dataset: Dataset, out_dir: str, num_proc: int = NUM_PROC):
    n = len(dataset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(TRAIN_ORIG_OUT_DIR, exist_ok=True)

    dataset.save_to_disk(TRAIN_ORIG_OUT_DIR, num_shards=NUM_ORIGIN_SHARDS)

    try:
        boundaries = np.linspace(0, n, num_proc + 1, dtype=int)
        tasks = [
            (num_proc, i, TRAIN_ORIG_OUT_DIR, out_dir)
            for i in range(num_proc)
            if boundaries[i + 1] > boundaries[i]
        ]

        ctx = get_context("spawn")
        with ctx.Pool(len(tasks)) as pool:
            for _ in tqdm(pool.imap_unordered(_write_mds_partition, tasks),
                          total=len(tasks), desc="Saving MDS partitions"):
                pass

        merge_index(out_dir, keep_local=True)
    finally:
        pass


# Val preparation
def _prepare_val_batch(batch, tokenizer):
    out_prefix, out_suffix, out_pinyin = [], [], []

    for text_list, pinyin_json, labels in zip(
        batch["text_list"], batch["pinyin_list"], batch["labels"]
    ):
        pinyin_list = json.loads(pinyin_json) if isinstance(pinyin_json, str) else pinyin_json
        count_chinese = sum(1 for l in labels if l == CHINESE_LABEL)
        if count_chinese < 1:
            continue

        k = random.randint(1, count_chinese)
        if random.random() < NO_CONTEXT_PROB:
            sel, end, ch_off = select_span(labels, k, "bidirectional")
            prefix_text = ""
        else:
            if random.random() < FORWARD_RATIO:
                direction = "forward"
            else:
                direction = "backward"
            sel, end, ch_off = select_span(labels, k, direction)
            prefix_text = "".join(text_list[:sel])

        suffix_text = "".join(text_list[sel:end])

        # flatten per-char readings for the suffix Chinese segments
        num_suffix = end - sel
        flat_readings = []
        for seg_nested in pinyin_list[ch_off:ch_off + num_suffix]:
            flat_readings.extend(seg_nested)

        aug_syllables = build_val_pinyin(flat_readings)

        out_prefix.append(prefix_text)
        out_suffix.append(suffix_text)
        out_pinyin.append(json.dumps(aug_syllables, ensure_ascii=False))

    return {"prefix": out_prefix, "suffix": out_suffix, "pinyin": out_pinyin}


# Main pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="Run the preprocessing pipeline to generate train/val datasets.")
    parser.add_argument("--generate_val", action="store_true", help="Generate validation dataset from existing preprocessed data.")
    args = parser.parse_args()

    if args.preprocess:
        from tokenizer import P2CTokenizer

        tokenizer = P2CTokenizer.from_config(VOCABS_CONFIG)

        datasets_to_concat = []
        for jsonl_path in JSONL_SOURCES:
            ds_jsonl = load_and_process_jsonl_source(tokenizer, jsonl_path)
            if ds_jsonl is not None:
                datasets_to_concat.append(ds_jsonl)
                
        for textonly_path in TEXTONLY_SOURCES:
            ds_textonly = load_and_process_textonly_source(tokenizer, textonly_path)
            if ds_textonly is not None:
                datasets_to_concat.append(ds_textonly)

        for path, text_columns in PARQUET_SOURCES:
            ds_src = load_and_process_parquet_source(path, text_columns, tokenizer)
            if ds_src is not None:
                datasets_to_concat.append(ds_src)

        if not datasets_to_concat:
            raise RuntimeError("No data sources found — nothing to process.")

        ds = concatenate_datasets(datasets_to_concat)

        # total character count
        total_chars = count_characters(ds)

        # Train / val split
        train_ds, val_ds = shuffled_train_test_split(
            ds, test_size=1 - SPLIT_RATIO, seed=SHUFFLE_SEED
        )

        # Train -> MDS parallel
        save_train_as_mds(train_ds, TRAIN_OUT_DIR, num_proc=NUM_PROC)
        
        # Save Original Validation Dataset
        val_ds.save_to_disk(VAL_ORIG_OUT_DIR, num_shards=NUM_VAL_SHARDS)

    if args.generate_val:
        from tokenizer import P2CTokenizer
        # Val -> Arrow
        tokenizer = P2CTokenizer.from_config(VOCABS_CONFIG)
        
        val_ds = Dataset.load_from_disk(VAL_ORIG_OUT_DIR)  # Load the original validation dataset
        
        val_ds_prepared = val_ds.map(
            lambda batch: _prepare_val_batch(batch, tokenizer),
            batched=True,
            batch_size=1024,
            num_proc=1,
            desc="Preparing val",
            remove_columns=val_ds.column_names,
        )
        val_ds_prepared.save_to_disk(VAL_OUT_DIR, num_shards=NUM_VAL_SHARDS)

    print(f"Total samples    : {len(ds) / 1e6:.1f}M rows")
    print(f"Total characters : {total_chars / 1e9:.2f}B")
    print(f"Train            : {len(train_ds) / 1e6:.1f}M rows -> {TRAIN_OUT_DIR} (MDS)")
    print(f"Val              : {len(val_ds_prepared) / 1e6:.1f}M rows -> {VAL_OUT_DIR} (HF Datasets)")


if __name__ == "__main__":
    main()
