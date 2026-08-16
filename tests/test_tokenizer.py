"""Tokenizer tests: heteronym sampling, freq loading, 简拼 possibility map."""

import json

import torch


def test_frequency_loaded_from_config(tokenizer):
    freq = tokenizer.character_pinyin_frequency
    assert freq["长"] == {"chang": 0.7, "zhang": 0.3}
    assert freq["的"]["de"] == 0.8


def test_sample_heteronym_distribution(tokenizer):
    import random

    rng = random.Random(42)
    counts = {"chang": 0, "zhang": 0}
    for _ in range(5000):
        counts[tokenizer.sample_heteronym("长", rng=rng)] += 1
    total = sum(counts.values())
    assert counts["chang"] / total > 0.65
    assert counts["zhang"] / total > 0.25


def test_sample_heteronym_fallback(tokenizer_no_freq):
    # chars not in the freq map fall back to pypinyin's most probable reading
    py = tokenizer_no_freq.sample_heteronym("我")
    assert py == "wo"


def test_simplified_pinyin_zero_initial_fix(tokenizer_no_freq):
    mask = tokenizer_no_freq.create_possibility_map()
    chinese_vocab = tokenizer_no_freq._chinese_vocab
    pinyin_vocab = tokenizer_no_freq._pinyin_vocab

    # zero-initial pinyins now split to their first letter as the 简拼
    for ch, readings in [("爱", ["ai"]), ("安", ["an"]), ("奥", ["ao"])]:
        cid = chinese_vocab[ch]
        for py in readings:
            pid = pinyin_vocab[py]
            assert mask[pid, cid], f"full pinyin {py} must map to {ch}"
        # 简拼: first letter must be allowed for zero-initial syllables
        if "a" in pinyin_vocab:
            assert mask[pinyin_vocab["a"], cid], f"简拼 'a' must map to {ch}"

    # regular initials unchanged
    cid = chinese_vocab["长"]
    assert mask[pinyin_vocab["chang"], cid]
    assert mask[pinyin_vocab["zhang"], cid]
    assert mask[pinyin_vocab["c"], cid]
    assert mask[pinyin_vocab["z"], cid]


def test_encode_roundtrip(tokenizer):
    assert tokenizer.encode_chinese("今天") == [tokenizer._chinese_vocab["今"],
                                                tokenizer._chinese_vocab["天"]]
    assert tokenizer.encode_pinyin(["jin", "tian"]) == [
        tokenizer._pinyin_vocab["jin"], tokenizer._pinyin_vocab["tian"]
    ]
    ids = tokenizer.encode_context("今天")
    assert ids == [tokenizer._context_vocab["今"], tokenizer._context_vocab["天"]]
    assert tokenizer.spec_tokens.bos_token == tokenizer.context_vocab_size - 1
