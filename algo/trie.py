"""
Trie-based dictionary matching for Chinese P2C decoding.

Builds a nested-dict trie from a word list and finds all dictionary words
that can start at a given position given per-position character probabilities.
"""

import json
import math


def load_dictionary(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_trie(words: list[str]) -> dict:
    trie: dict = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node["#"] = True
    return trie


def load_trie(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_trie(trie: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trie, f, ensure_ascii=False)


def find_matching_words(
    trie: dict,
    candidates: list[dict[str, float]],
    start: int,
) -> list[tuple[int, str, float]]:
    """Find all dictionary words that can start at position *start*.

    Args:
        trie: nested-dict trie (loaded from JSON or built via build_trie).
        candidates: list of {char -> prob} dicts, one per position.
        start: position index to start matching from.

    Returns:
        list of (length, word_str, log_prob_sum) sorted longest-first.
    """
    results: list[tuple[int, str, float]] = []
    stack: list[tuple[dict, int, list[str], float]] = [(trie, start, [], 0.0)]

    while stack:
        node, pos, chars, log_prob = stack.pop()
        if pos >= len(candidates):
            continue
        cand_dict = candidates[pos]

        for ch, child in node.items():
            if ch == "#":
                continue
            prob = cand_dict.get(ch)
            if prob is None or prob <= 0:
                continue
            new_chars = chars + [ch]
            new_log_prob = log_prob + math.log(prob)
            if "#" in child:
                results.append((len(new_chars), "".join(new_chars), new_log_prob))
            stack.append((child, pos + 1, new_chars, new_log_prob))

    return results
