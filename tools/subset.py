"""
tools/subset.py — dataset subset extractor
Extracts a random subset of a dataset and saves it as parquet.
"""

import glob
import os

import numpy as np
from datasets import Dataset, load_from_disk

# Config
INPUT_PATH   = "./datasets/pretrain_base/fineweb"   # HF save_to_disk dir OR parquet path
OUTPUT_PATH  = "./datasets/pretrain_base/fw_subset"  # output directory (parquet)
INPUT_FORMAT = "parquet"        # "hf" (save_to_disk) or "parquet"
SUBSET_SIZE  = 0.1              # fraction in (0, 1], or an int row count
SEED         = 114514
NUM_SHARDS   = 20


def _find_parquet_files(path: str) -> list[str]:
    if os.path.isfile(path) and path.endswith(".parquet"):
        return [path]
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    return []


def load_source(path: str, fmt: str) -> Dataset:
    if fmt == "hf":
        return load_from_disk(path)
    elif fmt == "parquet":
        files = _find_parquet_files(path)
        if not files:
            raise FileNotFoundError(f"No parquet found under: {path}")
        return Dataset.from_parquet(files)
    raise ValueError(f"Unknown INPUT_FORMAT: {fmt}")


def extract_subset(ds: Dataset, subset_size, seed: int) -> Dataset:
    n = len(ds)
    if isinstance(subset_size, float):
        n_keep = int(round(n * subset_size))
    else:
        n_keep = int(subset_size)
    n_keep = max(0, min(n_keep, n))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    keep_idx = np.sort(perm[:n_keep])   # sorted -> sequential read-back
    return ds.select(keep_idx.tolist())


def save_parquet(ds: Dataset, out_dir: str, num_shards: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if num_shards <= 1:
        ds.to_parquet(os.path.join(out_dir, "data.parquet"))
        return
    for shard_id in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=shard_id, contiguous=True)
        shard.to_parquet(os.path.join(out_dir, f"data-{shard_id:05d}-of-{num_shards:05d}.parquet"))


def main():
    ds = load_source(INPUT_PATH, INPUT_FORMAT)
    subset = extract_subset(ds, SUBSET_SIZE, SEED)
    save_parquet(subset, OUTPUT_PATH, NUM_SHARDS)
    print(f"Source : {len(ds)} rows ({INPUT_PATH})")
    print(f"Subset : {len(subset)} rows -> {OUTPUT_PATH} (parquet)")


if __name__ == "__main__":
    main()
