"""Preprocessor tests: segmentation modes, flat format, freq counting, val."""

import json
import random

from datasets import Dataset


# --------------------------------------------------------------------------
# Segmentation modes
# --------------------------------------------------------------------------
def test_segment_char_mode(tokenizer_no_freq):
    from datasets_pipeline.preprocessor import segment_text
    from datasets_pipeline.constants import CHINESE_LABEL

    segs, labels, seg_pinyin = segment_text("今天天气很好", tokenizer_no_freq,
                                            mode="char")
    assert segs == ["今", "天", "天", "气", "很", "好"]
    assert labels == [CHINESE_LABEL] * 6
    assert len(seg_pinyin) == 6
    for nested in seg_pinyin:
        assert len(nested) == 1
        assert len(nested[0]) >= 1


def test_segment_word_mode(tokenizer_no_freq):
    from datasets_pipeline.preprocessor import segment_text
    from datasets_pipeline.constants import CHINESE_LABEL

    segs, labels, seg_pinyin = segment_text("今天天气很好", tokenizer_no_freq,
                                            mode="word")
    assert "".join(segs) == "今天天气很好"
    assert labels == [CHINESE_LABEL] * len(segs)
    assert all(len(py) == len(seg) for seg, py in zip(segs, seg_pinyin))


def test_char_mode_keeps_word_pronunciation(tokenizer_no_freq):
    """In char mode jieba/pypinyin still disambiguate within the run."""
    from datasets_pipeline.preprocessor import segment_text

    segs, labels, seg_pinyin = segment_text("长大以后", tokenizer_no_freq,
                                            mode="char")
    # 长 inside the word "长大" reads "zhang" (not the default "chang")
    assert segs[0] == "长"
    assert "zhang" in seg_pinyin[0][0]


def test_pause_and_non_chinese_runs(tokenizer_no_freq):
    from datasets_pipeline.preprocessor import segment_text
    from datasets_pipeline.constants import PAUSE_LABEL, NON_CHINESE_LABEL

    # note: segment_text expects NFKC-normalized text (half-width punctuation)
    segs, labels, _ = segment_text("你好朋友,world你好啊", tokenizer_no_freq,
                                   max_non_chinese_len=16, mode="char")
    assert "," in segs
    assert labels[segs.index(",")] == PAUSE_LABEL
    assert "world" in segs
    assert labels[segs.index("world")] == NON_CHINESE_LABEL


def test_long_non_chinese_run_dropped(tokenizer_no_freq):
    from datasets_pipeline.preprocessor import segment_text

    segs, labels, _ = segment_text("你好AAAAAA你好", tokenizer_no_freq,
                                   max_non_chinese_len=4, mode="char")
    assert all("A" not in s for s in segs)


# --------------------------------------------------------------------------
# Flat sample format
# --------------------------------------------------------------------------
def test_flat_sample_format(tokenizer_no_freq):
    from datasets_pipeline.preprocessor import process_raw_text
    from datasets_pipeline.constants import CHINESE_LABEL

    sampling = {
        "truncation_len_min": 8, "truncation_len_max": 16,
        "max_text_len": 24, "ban_text_len": 32,
        "min_chinese_ratio": 0.5,
    }
    samples = process_raw_text("今天天气很好，我们出去散步。", tokenizer_no_freq,
                               sampling, 16, "char")
    assert samples
    for s in samples:
        pinyin_flat = json.loads(s["pinyin_list"])
        n_chinese = sum(len(seg) for seg, lab in zip(s["text_list"], s["labels"])
                        if lab == CHINESE_LABEL)
        assert isinstance(pinyin_flat, list)
        assert all(isinstance(py, str) for py in pinyin_flat)
        assert len(pinyin_flat) == n_chinese


def test_word_mode_flat_format(tokenizer_no_freq):
    from datasets_pipeline.preprocessor import process_raw_text
    from datasets_pipeline.constants import CHINESE_LABEL

    sampling = {
        "truncation_len_min": 8, "truncation_len_max": 16,
        "max_text_len": 24, "ban_text_len": 32,
        "min_chinese_ratio": 0.5,
    }
    samples = process_raw_text("今天天气很好，我们出去散步。", tokenizer_no_freq,
                               sampling, 16, "word")
    assert samples
    for s in samples:
        pinyin_flat = json.loads(s["pinyin_list"])
        n_chinese = sum(len(seg) for seg, lab in zip(s["text_list"], s["labels"])
                        if lab == CHINESE_LABEL)
        assert len(pinyin_flat) == n_chinese


# --------------------------------------------------------------------------
# Character pronunciation frequency counting
# --------------------------------------------------------------------------
def test_count_character_pinyins(tokenizer_no_freq):
    from datasets_pipeline.preprocessor import count_character_pinyins

    rows = [
        {"text_list": ["长", "大", "的"], "labels": [1, 1, 1],
         "pinyin_list": json.dumps(["zhang", "da", "de"], ensure_ascii=False)},
        {"text_list": ["长", "江", "的"], "labels": [1, 1, 1],
         "pinyin_list": json.dumps(["chang", "jiang", "di"], ensure_ascii=False)},
    ]
    ds = Dataset.from_list(rows)
    freq = count_character_pinyins(ds)
    assert freq["长"] == {"zhang": 0.5, "chang": 0.5}
    assert freq["的"] == {"de": 0.5, "di": 0.5}
    assert freq["大"] == {"da": 1.0}


def test_count_character_pinyins_word_mode():
    from datasets_pipeline.preprocessor import count_character_pinyins

    rows = [
        {"text_list": ["长江", "BBQ", "大桥"], "labels": [1, 2, 1],
         "pinyin_list": json.dumps(["chang", "jiang", "da", "qiao"], ensure_ascii=False)},
    ]
    ds = Dataset.from_list(rows)
    freq = count_character_pinyins(ds)
    assert freq == {"长": {"chang": 1.0}, "江": {"jiang": 1.0},
                    "大": {"da": 1.0}, "桥": {"qiao": 1.0}}


def test_save_character_pinyin_frequency(tmp_path):
    from datasets_pipeline.preprocessor import save_character_pinyin_frequency

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "vocabs:\n"
        "  chinese_vocab: chinese_vocab.txt\n"
        "  context_vocab: context_vocab.txt\n"
        "  pinyin_vocab: pinyin_vocab.txt\n"
        "  characters_pronounce_frequency: characters_pronounce_frequency.json\n",
        encoding="utf-8",
    )
    out = save_character_pinyin_frequency({"长": {"chang": 0.5, "zhang": 0.5}},
                                          str(config_path))
    assert out.endswith("characters_pronounce_frequency.json")
    data = json.loads(open(out, encoding="utf-8").read())
    assert data["长"]["chang"] == 0.5


# --------------------------------------------------------------------------
# Val materialization
# --------------------------------------------------------------------------
def test_prepare_val_batch(tokenizer):
    from datasets_pipeline.preprocessor import _prepare_val_batch

    sample = {
        "text_list": ["今", "天", "的", "阳", "光", "真", "好"],
        "pinyin_list": json.dumps(["jin", "tian", "de", "tian", "qi", "zhen", "hao"],
                                  ensure_ascii=False),
        "labels": [1, 1, 1, 1, 1, 1, 1],
    }
    val_cfg = {
        "drop_vowels": 0.0, "drop_last_vowel": 0.0,
        "vowels_droprate": [0.0, 1.0], "heteronym_confusion": 0.0,
        "no_context_prob": 0.0, "forward_ratio": 0.0,
    }
    random.seed(0)
    out = _prepare_val_batch(
        {k: [v] for k, v in sample.items()}, tokenizer, val_cfg
    )
    assert len(out["prefix"]) == 1
    suffix = out["suffix"][0]
    pinyin = json.loads(out["pinyin"][0])
    assert len(pinyin) == len(suffix)
    # pinyin corresponds to the suffix region of the flat list
    joined = "".join(sample["text_list"])
    idx = joined.find(suffix)
    char_off = sum(1 for ch in joined[:idx] if tokenizer.is_chinese(ch))
    assert pinyin == json.loads(sample["pinyin_list"])[char_off : char_off + len(suffix)]
