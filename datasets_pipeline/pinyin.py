"""Shared pinyin helpers for preprocessing and training-time transforms."""

import random
from typing import Optional

from tokenizer import P2CTokenizer

_COMP_CONSONANTS = {"zh", "ch", "sh"}
_ALL_INITIALS = {
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "y", "w", "zh", "ch", "sh", "z", "c", "s",
}
_SINGLE_INITIALS = {i for i in _ALL_INITIALS if len(i) == 1}


def get_initial_and_final(syllable: str) -> tuple[str, str]:
    """Split a pinyin syllable into (initial, final) parts.

    Syllables without an initial ("ai", "ao", "an", ...) yield ("", syllable).
    """
    if len(syllable) >= 2 and syllable[:2] in _COMP_CONSONANTS:
        return syllable[:2], syllable[2:]
    if syllable and syllable[0] in _SINGLE_INITIALS:
        return syllable[0], syllable[1:]
    return "", syllable


def sample_pinyin_list(
    pinyin_flat: list[str],
    chars: str,
    tokenizer: Optional[P2CTokenizer],
    heteronym_confusion_p: float = 0.0,
    rng: Optional[random.Random] = None,
) -> list[str]:
    """Resolve a flat per-char reading list, applying heteronym confusion.

    When heteronym confusion is hit for position j, the pronunciation is
    sampled from the tokenizer's corpus-frequency distribution for the
    character ``chars[j]`` (see ``P2CTokenizer.sample_heteronym``) instead of
    being chosen uniformly at random.
    """
    rng = rng or random
    result = list(pinyin_flat)
    if heteronym_confusion_p > 0.0 and tokenizer is not None:
        for idx in range(len(result)):
            if rng.random() < heteronym_confusion_p:
                sampled = tokenizer.sample_heteronym(chars[idx], rng=rng)
                if sampled:
                    result[idx] = sampled
    return result


def augment_pinyin_sequence(pinyin_list: list[str], aug_cfg: dict,
                            rng: Optional[random.Random] = None) -> list[str]:
    """Training-time pinyin augmentation (vowel dropping -> 简拼)."""
    rng = rng or random
    drop_vowels_p = aug_cfg.get("drop_vowels", 0.0)
    drop_last_p = aug_cfg.get("drop_last_vowel", 0.0)
    vowels_droprate = aug_cfg.get("vowels_droprate", [0.0, 1.0])

    result = list(pinyin_list)
    should_augment_sentence = rng.random() < drop_vowels_p

    if should_augment_sentence:
        for idx in range(len(result)):
            syllable = result[idx]
            initial, final = get_initial_and_final(syllable)
            if initial and final:
                drop_threshold = rng.uniform(vowels_droprate[0], vowels_droprate[1])
                if rng.random() < drop_threshold:
                    result[idx] = initial[0]

    if rng.random() < drop_last_p and len(result) > 0:
        last_idx = len(result) - 1
        last_syllable = result[last_idx]
        last_initial, last_final = get_initial_and_final(last_syllable)
        if last_initial and last_final:
            result[last_idx] = last_initial[0]

    return result


def build_val_pinyin(pinyin_flat: list[str], chars: str, tokenizer: Optional[P2CTokenizer],
                     val_cfg: dict, rng: Optional[random.Random] = None) -> list[str]:
    """Static val augmentation on a flat per-char reading list."""
    rng = rng or random

    syllables = sample_pinyin_list(
        pinyin_flat, chars, tokenizer, val_cfg.get("heteronym_confusion", 0.0), rng
    )

    drop_vowels = val_cfg.get("drop_vowels", 0.0)
    drop_last = val_cfg.get("drop_last_vowel", 0.0)
    vowels_droprate = val_cfg.get("vowels_droprate", [0.0, 1.0])

    if rng.random() < drop_vowels:
        for idx in range(len(syllables)):
            initial, final = get_initial_and_final(syllables[idx])
            if initial and final:
                drop_threshold = rng.uniform(vowels_droprate[0], vowels_droprate[1])
                if rng.random() < drop_threshold:
                    syllables[idx] = initial[0]

    if rng.random() < drop_last and len(syllables) > 0:
        last_idx = len(syllables) - 1
        last_initial, last_final = get_initial_and_final(syllables[last_idx])
        if last_initial and last_final:
            syllables[last_idx] = last_initial[0]

    return syllables
