"""
PostfixLM dataset pipeline.
"""

import json
import math
import hashlib
import random

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


# Segment label ids, must match preprocessor.py!
PAUSE_LABEL       = 0
CHINESE_LABEL     = 1
NON_CHINESE_LABEL = 2


# Pinyin augmentation
_COMP_CONSONANTS = {"zh", "ch", "sh"}
_ALL_INITIALS = {
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "y", "w", "zh", "ch", "sh", "z", "c", "s",
}


def _get_initial_and_final(syllable: str) -> tuple[str, str]:
    if len(syllable) >= 2 and syllable[:2] in _COMP_CONSONANTS:
        return syllable[:2], syllable[2:]
    if syllable and syllable[0] in {i for i in _ALL_INITIALS if len(i) == 1}:
        return syllable[0], syllable[1:]
    return "", syllable


def augment_pinyin_sequence(pinyin_list: list[str], aug_cfg: dict) -> list[str]:
    drop_vowels_p   = aug_cfg.get("drop_vowels", 0.0)
    drop_last_p     = aug_cfg.get("drop_last_vowel", 0.0)
    vowels_droprate = aug_cfg.get("vowels_droprate", [0.0, 1.0])

    result = list(pinyin_list)
    should_augment_sentence = random.random() < drop_vowels_p

    if should_augment_sentence:
        for idx in range(len(result)):
            syllable = result[idx]
            initial, final = _get_initial_and_final(syllable)
            if initial and final:
                drop_threshold = random.uniform(vowels_droprate[0], vowels_droprate[1])
                if random.random() < drop_threshold:
                    result[idx] = initial[0]

    if random.random() < drop_last_p and len(result) > 0:
        last_idx = len(result) - 1
        last_syllable = result[last_idx]
        last_initial, last_final = _get_initial_and_final(last_syllable)
        if last_initial and last_final:
            result[last_idx] = last_initial[0]

    return result


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


def _select_span(labels: list[int], k: int, direction: str = "backward"):
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


# Transform functions
def transform_pinyin_predict_train(batch, tokenizer, aug_cfg=None, online_policy=None, epoch=None):
    prefix_ids_list      = []
    postfix_ids_list     = []
    targets_ids_list     = []
    prefix_lengths_list  = []
    postfix_lengths_list = []

    aug_cfg = aug_cfg or {}
    online_cfg = online_policy if isinstance(online_policy, dict) else {}
    strategy = online_cfg.get("strategy", "random")
    no_context_prob = online_cfg.get("no_context_prob", 0.0)
    forward_prob = online_cfg.get("forward_prob", 0.5)
    heteronym_confusion_p = aug_cfg.get("heteronym_confusion", 0.0)

    for text_list, pinyin_json, labels in zip(
        batch["text_list"], batch["pinyin_list"], batch["labels"]
    ):
        pinyin_list = json.loads(pinyin_json) if isinstance(pinyin_json, str) else pinyin_json
        labels = list(labels)

        count_chinese = sum(1 for l in labels if l == CHINESE_LABEL)
        if count_chinese < 1:
            continue

        if strategy == "golden-sequence":
            k = _pick_golden_choice(count_chinese, "".join(text_list), epoch)
        else:  # "random"
            k = random.randint(1, count_chinese)

        # prefix (context)
        if random.random() < no_context_prob:
            sel, end, ch_off = _select_span(labels, k, "bidirectional")
            prefix_text = ""
        else:
            if random.random() < forward_prob:
                direction = "forward"
            else:
                direction = "backward"
            sel, end, ch_off = _select_span(labels, k, direction)
            prefix_text = "".join(text_list[:sel])
        prefix_ids = [tokenizer.spec_tokens.bos_token] + tokenizer.encode_context(prefix_text)

        # suffix (target) + its pinyin
        suffix_text = "".join(text_list[sel:end])
        num_suffix = end - sel
        flat_readings = []
        for seg_nested in pinyin_list[ch_off:ch_off + num_suffix]:
            flat_readings.extend(seg_nested)

        pinyin_segments = resolve_pinyin_list(flat_readings, heteronym_confusion_p)
        pinyin_segments = augment_pinyin_sequence(pinyin_segments, aug_cfg)

        postfix_ids = tokenizer.encode_pinyin(pinyin_segments)
        target_ids  = tokenizer.encode_chinese(suffix_text)

        prefix_ids_list.append(prefix_ids)
        postfix_ids_list.append(postfix_ids)
        targets_ids_list.append(target_ids)
        prefix_lengths_list.append(len(prefix_ids))
        postfix_lengths_list.append(len(postfix_ids))

    return {
        "prefix_ids":      prefix_ids_list,
        "postfix_ids":     postfix_ids_list,
        "target_ids":      targets_ids_list,
        "prefix_lengths":  prefix_lengths_list,
        "postfix_lengths": postfix_lengths_list,
    }


def transform_pinyin_predict_val(batch, tokenizer, aug_cfg=None):
    """Validation transform — prefix/suffix/pinyin were materialized offline."""
    prefix_ids_list      = []
    postfix_ids_list     = []
    targets_ids_list     = []
    prefix_lengths_list  = []
    postfix_lengths_list = []

    for prefix_text, suffix_text, pinyin_json in zip(
        batch["prefix"], batch["suffix"], batch["pinyin"]
    ):
        pinyin_flat = json.loads(pinyin_json) if isinstance(pinyin_json, str) else pinyin_json

        prefix_ids  = [tokenizer.spec_tokens.bos_token] + tokenizer.encode_context(prefix_text)
        postfix_ids = tokenizer.encode_pinyin(pinyin_flat)
        target_ids  = tokenizer.encode_chinese(suffix_text)

        prefix_ids_list.append(prefix_ids)
        postfix_ids_list.append(postfix_ids)
        targets_ids_list.append(target_ids)
        prefix_lengths_list.append(len(prefix_ids))
        postfix_lengths_list.append(len(postfix_ids))

    return {
        "prefix_ids":      prefix_ids_list,
        "postfix_ids":     postfix_ids_list,
        "target_ids":      targets_ids_list,
        "prefix_lengths":  prefix_lengths_list,
        "postfix_lengths": postfix_lengths_list,
    }



# Collate functions
def make_collate_fn():
    """NJT collate for efficient training.

    Returns:
        prefix_ids_njt:  nested jagged tensor of prefix token ids
        postfix_ids_njt: nested jagged tensor of postfix (pinyin) ids
        target_ids_njt:  nested jagged tensor of target (Chinese) ids
    """
    def collate_fn(batch):
        prefix_ids_list  = [item["prefix_ids"] for item in batch]
        postfix_ids_list = [item["postfix_ids"] for item in batch]
        targets_ids_list = [item["target_ids"] for item in batch]

        def _to_njt(id_lists):
            tensor_list = [torch.tensor(ids, dtype=torch.long) for ids in id_lists]
            return torch.nested.nested_tensor(tensor_list, layout=torch.jagged)

        return {
            "prefix_ids_njt":  _to_njt(prefix_ids_list),
            "postfix_ids_njt": _to_njt(postfix_ids_list),
            "target_ids_njt":  _to_njt(targets_ids_list),
        }

    return collate_fn


def create_dataset(data_path: str, keep_in_memory=False):
    return load_from_disk(data_path, keep_in_memory=keep_in_memory)


from streaming import StreamingDataset, StreamingDataLoader
from typing import Any

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
            epoch=self.next_epoch - 1, # start with 1, next_epoch -1 means start with 0
        )
        return {k: v[0] for k, v in result.items()}