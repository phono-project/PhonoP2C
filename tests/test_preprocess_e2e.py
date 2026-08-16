"""End-to-end preprocess test on a tiny synthetic corpus."""

import json
import os

import pytest


@pytest.fixture()
def tiny_corpus(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    lines = [
        "今天天气很好，我们出去散步。",
        "长江大桥是中国著名的桥梁建筑。",
        "学习编程需要耐心和坚持。",
        "春天来了，花儿都开了。",
    ]
    with open(src / "corpus.jsonl", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps([line], ensure_ascii=False) + "\n")
    return src


@pytest.fixture()
def e2e_config(tmp_path, tiny_corpus):
    vocabs_dir = tmp_path / "vocabs"
    vocabs_dir.mkdir()
    real_vocabs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vocabs")
    vocabs_config = vocabs_dir / "config.yaml"
    vocabs_config.write_text(
        "vocabs:\n"
        f"  chinese_vocab: {real_vocabs}/chinese_vocab.txt\n"
        f"  context_vocab: {real_vocabs}/context_vocab.txt\n"
        f"  pinyin_vocab: {real_vocabs}/pinyin_vocab.txt\n"
        "  characters_pronounce_frequency: characters_pronounce_frequency.json\n"
        "  context_special_tokens:\n"
        "    - bos_token\n",
        encoding="utf-8",
    )
    return {
        "dataset_dir": str(tmp_path / "datasets"),
        "vocabs_config": str(vocabs_config),
        "segmentation": {"mode": "char"},
        "sources": {
            "jsonl": [str(tiny_corpus)],
            "textonly": [],
            "parquet": [],
        },
        "sampling": {
            "truncation_len_min": 6,
            "truncation_len_max": 12,
            "max_text_len": 16,
            "ban_text_len": 24,
            "max_non_chinese_seg_len": 16,
            "min_chinese_ratio": 0.3,
            "split_ratio": 0.5,
        },
        "output": {
            "train_dir": "${dataset.dataset_dir}/train",
            "val_dir": "${dataset.dataset_dir}/val",
            "val_original_dir": "${dataset.dataset_dir}/val_original",
            "train_original_dir": "${dataset.dataset_dir}/train_original",
            "num_val_shards": 1,
            "num_original_shards": 1,
        },
        "mds": {"compression": None, "size_limit": 1 << 20},
        "val_augmentation": {
            "drop_vowels": 0.0,
            "drop_last_vowel": 0.0,
            "vowels_droprate": [0.0, 1.0],
            "heteronym_confusion": 0.0,
            "no_context_prob": 0.2,
            "forward_ratio": 0.0,
        },
        "processing": {"num_proc": 1, "batch_size": 2, "shuffle_seed": 114514},
    }


def test_run_preprocess_end_to_end(e2e_config, tmp_path):
    from datasets_pipeline.preprocessor import run_preprocess
    from datasets import Dataset
    from tokenizer import P2CTokenizer

    run_preprocess(e2e_config, generate_val=True)

    # train MDS index written
    train_index = tmp_path / "datasets" / "train" / "index.json"
    assert train_index.exists()
    index = json.loads(train_index.read_text(encoding="utf-8"))
    assert index["version"] == 2

    # character pronunciation frequencies saved next to the tokenizer
    freq_path = os.path.join(os.path.dirname(e2e_config["vocabs_config"]),
                             "characters_pronounce_frequency.json")
    assert os.path.exists(freq_path)
    freq = json.loads(open(freq_path, encoding="utf-8").read())
    assert "长" in freq
    for readings in freq.values():
        total = sum(readings.values())
        assert abs(total - 1.0) < 1e-6

    # tokenizer can load the frequencies through its config
    tokenizer = P2CTokenizer.from_config(e2e_config["vocabs_config"])
    assert tokenizer.character_pinyin_frequency["长"]
    assert tokenizer.sample_heteronym("长") in ("chang", "zhang")

    # val dataset materialized
    val_ds = Dataset.load_from_disk(str(tmp_path / "datasets" / "val"))
    for col in ("prefix", "suffix", "pinyin"):
        assert col in val_ds.column_names
    assert len(val_ds) > 0
    for row in val_ds:
        pinyin_flat = json.loads(row["pinyin"])
        assert len(pinyin_flat) == len(row["suffix"])

    # val transform produces aligned ids
    from datasets_pipeline.dataset import transform_pinyin_predict_val

    out = transform_pinyin_predict_val(
        {"prefix": [val_ds[0]["prefix"]],
         "suffix": [val_ds[0]["suffix"]],
         "pinyin": [val_ds[0]["pinyin"]]},
        tokenizer, None,
    )
    assert len(out["target_ids"][0]) == len(out["postfix_ids"][0])
    assert out["pre_ids"][0][0] == tokenizer.spec_tokens.bos_token


def test_preprocess_word_mode(e2e_config, tmp_path):
    from datasets_pipeline.preprocessor import run_preprocess
    from datasets import Dataset

    e2e_config = dict(e2e_config)
    e2e_config["segmentation"] = {"mode": "word"}
    e2e_config["output"] = {k: v for k, v in e2e_config["output"].items()}
    run_preprocess(e2e_config, generate_val=True)

    val_ds = Dataset.load_from_disk(str(tmp_path / "datasets" / "val_original"))
    from datasets_pipeline.constants import CHINESE_LABEL

    # in word mode some segments have more than one char but the flat pinyin
    # list still covers every chinese char
    for row in val_ds:
        pinyin_flat = json.loads(row["pinyin_list"])
        n_chinese = sum(len(seg) for seg, lab in zip(row["text_list"], row["labels"])
                        if lab == CHINESE_LABEL)
        assert len(pinyin_flat) == n_chinese
        assert any(len(seg) > 1 for seg, lab in zip(row["text_list"], row["labels"])
                   if lab == CHINESE_LABEL)
