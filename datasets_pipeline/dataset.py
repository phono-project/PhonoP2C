"""
PostfixLM dataset pipeline (flat per-character pinyin format).

Stored sample columns
---------------------
    text_list   : list[str]   — the segments (chars or words for Chinese runs)
    labels      : list[int]   — per-segment property id
    pinyin_list : str (JSON)  — FLAT pinyin, one entry per Chinese character,
                                concatenated over the Chinese segments in
                                order; non-Chinese segments have no entries.

Example (single-character mode)::

    text_list:  ["今","天","的","阳","光","真","好","呀"]
    pinyin_list: ["jin","tian","de","tian","qi","zhen","hao","ya"]
    labels:      [1,1,1,1,1,1,1,1]

In word mode the segmentation is coarser but ``pinyin_list`` stays flat.
"""

import hashlib
import json
import math
import random

import torch
from datasets import load_from_disk
from streaming import StreamingDataset

from tokenizer import P2CTokenizer

from datasets_pipeline.constants import CHINESE_LABEL
from datasets_pipeline.pinyin import augment_pinyin_sequence, sample_pinyin_list
from datasets_pipeline.segments import chinese_segment_char_prefixes, select_span

# helpers
PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _compute_base_seed_norm(text: str) -> float:
    h = hashlib.md5(text.encode("utf-8")).digest()
    val = int.from_bytes(h[:8], "big")
    return val / float(2**64)


def _pick_golden_choice(count_chinese: int, text: str, epoch: int) -> int:
    """Return a 1-indexed choice in [1, count_chinese] via a golden sequence."""
    if count_chinese <= 1:
        return 1
    base = _compute_base_seed_norm(text)
    relative = (base + float(epoch or 0) * PHI) % 1.0
    idx = int(relative * count_chinese)
    return min(idx, count_chinese - 1) + 1


def _select_suffix(text_list, pinyin_list, labels, online_cfg, epoch):
    """Choose a (sel, end, suffix_text, pinyin_slice) for one sample.

    prefix = segments[0:sel]; suffix = segments[sel:end] (consecutive Chinese).
    The flat pinyin slice is derived from the Chinese-segment character
    offsets, so it works for both character- and word-level segmentation.
    """
    strategy = online_cfg.get("strategy", "random")
    no_context_prob = online_cfg.get("no_context_prob", 0.0)
    forward_prob = online_cfg.get("forward_prob", 0.5)

    count_chinese = sum(1 for lab in labels if lab == CHINESE_LABEL)
    if strategy == "golden-sequence":
        k = _pick_golden_choice(count_chinese, "".join(text_list), epoch)
    else:  # "random"
        k = random.randint(1, count_chinese)

    if random.random() < no_context_prob:
        sel, end, ch_off = select_span(labels, k, "bidirectional")
    else:
        direction = "forward" if random.random() < forward_prob else "backward"
        sel, end, ch_off = select_span(labels, k, direction)

    suffix_text = "".join(text_list[sel:end])

    char_prefix = chinese_segment_char_prefixes(text_list, labels)
    char_off = char_prefix[sel]
    num_chars = char_prefix[end] - char_prefix[sel]
    pinyin_slice = pinyin_list[char_off : char_off + num_chars]

    assert len(pinyin_slice) == len(suffix_text), (
        f"pinyin/text misalignment: {len(pinyin_slice)} vs {len(suffix_text)}"
    )
    return sel, end, suffix_text, pinyin_slice


# Transform functions
def transform_pinyin_predict_train(batch, tokenizer: P2CTokenizer, aug_cfg=None,
                                   online_policy=None, epoch=None):
    pre_ids_list = []
    postfix_ids_list = []
    targets_ids_list = []

    aug_cfg = aug_cfg or {}
    online_cfg = online_policy if isinstance(online_policy, dict) else {}
    heteronym_confusion_p = aug_cfg.get("heteronym_confusion", 0.0)

    for text_list, pinyin_json, labels in zip(
        batch["text_list"], batch["pinyin_list"], batch["labels"]
    ):
        pinyin_list = json.loads(pinyin_json) if isinstance(pinyin_json, str) else pinyin_json
        labels = list(labels)

        count_chinese = sum(1 for lab in labels if lab == CHINESE_LABEL)
        if count_chinese < 1:
            continue

        sel, end, suffix_text, pinyin_slice = _select_suffix(
            text_list, pinyin_list, labels, online_cfg, epoch
        )

        prefix_text = "".join(text_list[:sel])

        pinyin_segments = sample_pinyin_list(
            pinyin_slice, suffix_text, tokenizer, heteronym_confusion_p
        )
        pinyin_segments = augment_pinyin_sequence(pinyin_segments, aug_cfg)

        # Decoder input: [BOS] + context prefix + target (both encoded with the
        # context vocabulary; targets use the chinese vocab only for labels).
        pre_ids = [tokenizer.spec_tokens.bos_token] \
            + tokenizer.encode_context(prefix_text) \
            + tokenizer.encode_context(suffix_text)
        postfix_ids = tokenizer.encode_pinyin(pinyin_segments)
        target_ids = tokenizer.encode_chinese(suffix_text)

        pre_ids_list.append(pre_ids)
        postfix_ids_list.append(postfix_ids)
        targets_ids_list.append(target_ids)

    return {
        "pre_ids": pre_ids_list,
        "postfix_ids": postfix_ids_list,
        "target_ids": targets_ids_list,
    }


def transform_pinyin_predict_val(batch, tokenizer: P2CTokenizer, aug_cfg=None):
    """Validation transform — prefix/suffix/pinyin were materialized offline."""
    pre_ids_list = []
    postfix_ids_list = []
    targets_ids_list = []

    for prefix_text, suffix_text, pinyin_json in zip(
        batch["prefix"], batch["suffix"], batch["pinyin"]
    ):
        pinyin_flat = json.loads(pinyin_json) if isinstance(pinyin_json, str) else pinyin_json

        pre_ids = [tokenizer.spec_tokens.bos_token] \
            + tokenizer.encode_context(prefix_text) \
            + tokenizer.encode_context(suffix_text)
        postfix_ids = tokenizer.encode_pinyin(pinyin_flat)
        target_ids = tokenizer.encode_chinese(suffix_text)

        pre_ids_list.append(pre_ids)
        postfix_ids_list.append(postfix_ids)
        targets_ids_list.append(target_ids)

    return {
        "pre_ids": pre_ids_list,
        "postfix_ids": postfix_ids_list,
        "target_ids": targets_ids_list,
    }


# Collate functions
def make_collate_fn():
    """NJT collate for efficient training.

    Returns:
        pre_ids_njt:      nested jagged tensor of decoder input ids
        postfix_ids_njt:  nested jagged tensor of postfix (pinyin) ids
        target_ids_njt:   nested jagged tensor of target (Chinese) ids
    """
    def collate_fn(batch):
        pre_ids_list = [item["pre_ids"] for item in batch]
        postfix_ids_list = [item["postfix_ids"] for item in batch]
        targets_ids_list = [item["target_ids"] for item in batch]

        def _to_njt(id_lists):
            tensor_list = [torch.tensor(ids, dtype=torch.long) for ids in id_lists]
            return torch.nested.nested_tensor(tensor_list, layout=torch.jagged)

        return {
            "pre_ids_njt": _to_njt(pre_ids_list),
            "postfix_ids_njt": _to_njt(postfix_ids_list),
            "target_ids_njt": _to_njt(targets_ids_list),
        }

    return collate_fn


def create_dataset(data_path: str, keep_in_memory=False):
    return load_from_disk(data_path, keep_in_memory=keep_in_memory)


class P2CStreamingDataset(StreamingDataset):
    def __init__(self, local, tokenizer, aug_cfg, online_policy, **kwargs):
        super().__init__(local=local, **kwargs)
        self.tokenizer = tokenizer
        self.aug_cfg = aug_cfg
        self.online_policy = online_policy

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        result = transform_pinyin_predict_train(
            {k: [v] for k, v in sample.items()},
            self.tokenizer,
            self.aug_cfg,
            self.online_policy,
            epoch=self.next_epoch - 1,  # start with 1, next_epoch -1 means start with 0
        )
        return {k: v[0] for k, v in result.items()}
