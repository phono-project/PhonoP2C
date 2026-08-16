"""Dataset transform tests: flat pinyin format, span slicing, collate."""

import json


def _sample(text_list, pinyin_list, labels):
    return {
        "text_list": text_list,
        "pinyin_list": json.dumps(pinyin_list, ensure_ascii=False),
        "labels": labels,
    }


def _run_transform(tokenizer, sample, online_policy=None, aug_cfg=None, epoch=0):
    from datasets_pipeline.dataset import transform_pinyin_predict_train

    online_policy = online_policy if online_policy is not None else {
        "strategy": "golden-sequence",
        "no_context_prob": 0.0,
        "forward_prob": 1.0,
    }
    aug_cfg = aug_cfg or {"drop_vowels": 0.0, "drop_last_vowel": 0.0,
                          "vowels_droprate": [0.0, 1.0], "heteronym_confusion": 0.0}
    batch = {k: [v] for k, v in sample.items()}
    out = transform_pinyin_predict_train(batch, tokenizer, aug_cfg, online_policy, epoch=epoch)
    return {k: v[0] for k, v in out.items()}


def _expected_span(text_list, labels, epoch, count_chinese):
    from datasets_pipeline.dataset import _pick_golden_choice
    from datasets_pipeline.segments import select_span

    k = _pick_golden_choice(count_chinese, "".join(text_list), epoch)
    sel, end, _ = select_span(labels, k, "forward")
    suffix_text = "".join(text_list[sel:end])
    prefix_text = "".join(text_list[:sel])
    return k, sel, end, prefix_text, suffix_text


def test_char_mode_transform(tokenizer):
    sample = _sample(
        ["今", "天", "的", "阳", "光", "真", "好", "呀"],
        ["jin", "tian", "de", "tian", "qi", "zhen", "hao", "ya"],
        [1, 1, 1, 1, 1, 1, 1, 1],
    )
    # deterministic: golden-sequence + forward direction
    out = _run_transform(tokenizer, sample, epoch=0)

    k, sel, end, prefix_text, suffix_text = _expected_span(
        sample["text_list"], sample["labels"], 0, 8
    )
    assert 1 <= k <= 8

    assert len(out["postfix_ids"]) == len(suffix_text)
    assert len(out["target_ids"]) == len(suffix_text)
    assert len(out["pre_ids"]) == 1 + len(prefix_text) + len(suffix_text)
    # pinyin slice for the suffix
    pinyin_of_suffix = [tokenizer._id_to_pinyin[i] for i in out["postfix_ids"]]
    assert pinyin_of_suffix == json.loads(sample["pinyin_list"])[sel:end]
    assert tokenizer.ids_to_text(out["target_ids"]) == suffix_text
    assert out["pre_ids"] == [tokenizer.spec_tokens.bos_token] + \
        tokenizer.encode_context(prefix_text) + tokenizer.encode_context(suffix_text)


def test_word_mode_transform(tokenizer):
    from datasets_pipeline.segments import chinese_segment_char_prefixes

    sample = _sample(
        ["今天", "的", "阳光", "真好", "呀"],
        ["jin", "tian", "de", "tian", "qi", "zhen", "hao", "ya"],
        [1, 1, 1, 1, 1],
    )
    out = _run_transform(tokenizer, sample, epoch=0)
    k, sel, end, prefix_text, suffix_text = _expected_span(
        sample["text_list"], sample["labels"], 0, 5
    )
    char_prefix = chinese_segment_char_prefixes(sample["text_list"], sample["labels"])
    pinyin_of_suffix = [tokenizer._id_to_pinyin[i] for i in out["postfix_ids"]]
    assert pinyin_of_suffix == json.loads(sample["pinyin_list"])[char_prefix[sel]:char_prefix[end]]
    assert tokenizer.ids_to_text(out["target_ids"]) == suffix_text
    assert out["pre_ids"] == [tokenizer.spec_tokens.bos_token] + \
        tokenizer.encode_context(prefix_text) + tokenizer.encode_context(suffix_text)


def test_transform_skips_non_chinese_suffix(tokenizer):
    sample = _sample(
        ["我们", "出去", "BBQ", "怎么", "样"],
        ["wo", "men", "chu", "qu", "zen", "me", "yang"],
        [1, 1, 2, 1, 1],
    )
    # forward direction from a chinese segment never includes "BBQ"
    for epoch in range(8):
        out = _run_transform(tokenizer, sample, epoch=epoch)
        suffix = tokenizer.ids_to_text(out["target_ids"])
        assert "BBQ" not in suffix
        assert len(out["postfix_ids"]) == len(suffix) == len(out["target_ids"])
        pinyin_of_suffix = [tokenizer._id_to_pinyin[i] for i in out["postfix_ids"]]
        # pinyin slice must match the flat list region of the suffix chars
        joined = "".join(sample["text_list"])
        idx = joined.find(suffix)
        assert idx >= 0
        char_offset = sum(1 for ch in joined[:idx] if tokenizer.is_chinese(ch))
        expected = json.loads(sample["pinyin_list"])[char_offset:char_offset + len(suffix)]
        assert pinyin_of_suffix == expected


def test_heteronym_confusion_uses_frequency(tokenizer):
    import random

    sample = _sample(
        ["长", "大"],
        ["chang", "da"],
        [1, 1],
    )
    random.seed(7)
    aug_cfg = {"drop_vowels": 0.0, "drop_last_vowel": 0.0,
               "vowels_droprate": [0.0, 1.0], "heteronym_confusion": 1.0}
    # heteronym_confusion = 1.0 -> always sample via tokenizer.sample_heteronym
    # (frequency: chang 0.7 / zhang 0.3)
    seen = set()
    for _ in range(40):
        out = _run_transform(tokenizer, sample, aug_cfg=aug_cfg, epoch=0)
        py = tokenizer._id_to_pinyin[out["postfix_ids"][0]]
        assert py in ("chang", "zhang")
        seen.add(py)
    assert "chang" in seen and "zhang" in seen


def test_collate_and_streaming_dataset_interface(tokenizer):
    from datasets_pipeline.dataset import make_collate_fn

    collate = make_collate_fn()
    items = [
        {"pre_ids": [1, 2, 3], "postfix_ids": [4, 5], "target_ids": [6, 7]},
        {"pre_ids": [1, 2, 3, 4, 5], "postfix_ids": [4, 5, 6, 7], "target_ids": [8, 9, 10, 11]},
    ]
    batch = collate(items)
    assert set(batch.keys()) == {"pre_ids_njt", "postfix_ids_njt", "target_ids_njt"}
    for key in batch:
        assert batch[key].offsets().tolist() == [0, 3, 8] if key == "pre_ids_njt" \
            else batch[key].offsets().tolist() == [0, 2, 6]


def test_val_transform(tokenizer):
    from datasets_pipeline.dataset import transform_pinyin_predict_val

    batch = {
        "prefix": ["今天"],
        "suffix": ["天气"],
        "pinyin": [json.dumps(["tian", "qi"], ensure_ascii=False)],
    }
    out = transform_pinyin_predict_val(batch, tokenizer, None)
    assert out["pre_ids"][0] == [tokenizer.spec_tokens.bos_token] + \
        tokenizer.encode_context("今天") + tokenizer.encode_context("天气")
    assert out["postfix_ids"][0] == tokenizer.encode_pinyin(["tian", "qi"])
    assert tokenizer.ids_to_text(out["target_ids"][0]) == "天气"


def test_chinese_segment_char_prefixes():
    from datasets_pipeline.segments import chinese_segment_char_prefixes
    from datasets_pipeline.constants import CHINESE_LABEL

    text_list = ["今天", "的", "阳光", "BBQ", "呀"]
    labels = [CHINESE_LABEL, CHINESE_LABEL, CHINESE_LABEL, 2, CHINESE_LABEL]
    prefix = chinese_segment_char_prefixes(text_list, labels)
    assert prefix == [0, 2, 3, 5, 5, 6]
