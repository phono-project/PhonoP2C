import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.set_float32_matmul_precision("high")

VOCABS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vocabs")


@pytest.fixture(scope="session")
def tiny_cfg_dict():
    return {
        "common": {"model_dim": 32, "rope_theta": 1000.0},
        "pre_model": {
            "max_seqlen": 64, "mhsa_layers": 2, "mhsa_heads": 2,
            "attn_dim": 16, "ffn_common_dim": 32,
            "mhca_heads": 2, "mhca_attn_dim": 16,
        },
        "post_model": {
            "max_seqlen": 32, "mhsa_layers": 2, "mhsa_heads": 2,
            "attn_dim": 16, 
            "ffn_common_dim": 32,
        },
    }


@pytest.fixture(scope="session")
def tiny_vocab_sizes():
    return {"context": 300, "pinyin": 100, "chinese": 250}


@pytest.fixture()
def tiny_models(tiny_cfg_dict, tiny_vocab_sizes):
    from model.config import build_configs_from_dict
    from model.model import PhonoP2CPreModel, PhonoP2CPostModel

    torch.manual_seed(0)
    pre_cfg, post_cfg = build_configs_from_dict(tiny_cfg_dict, tiny_vocab_sizes)
    pre = PhonoP2CPreModel(pre_cfg)
    post = PhonoP2CPostModel(post_cfg)
    return pre, post, pre_cfg, post_cfg


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory):
    """Tokenizer built from the repo's real vocab files plus a temp freq json."""
    from tokenizer import P2CTokenizer

    freq_dir = tmp_path_factory.mktemp("freq")
    freq = {
        "长": {"chang": 0.7, "zhang": 0.3},
        "的": {"de": 0.8, "di": 0.2},
        "重": {"zhong": 0.6, "chong": 0.4},
    }
    freq_path = freq_dir / "characters_pronounce_frequency.json"
    freq_path.write_text(json.dumps(freq, ensure_ascii=False), encoding="utf-8")
    config_path = freq_dir / "config.yaml"
    config_path.write_text(
        "vocabs:\n"
        f"  chinese_vocab: {VOCABS_DIR}/chinese_vocab.txt\n"
        f"  context_vocab: {VOCABS_DIR}/context_vocab.txt\n"
        f"  pinyin_vocab: {VOCABS_DIR}/pinyin_vocab.txt\n"
        "  characters_pronounce_frequency: characters_pronounce_frequency.json\n"
        "  context_special_tokens:\n"
        "    - bos_token\n",
        encoding="utf-8",
    )
    return P2CTokenizer.from_config(str(config_path))


@pytest.fixture(scope="session")
def tokenizer_no_freq(tmp_path_factory):
    """Tokenizer without a frequency file (fallback path)."""
    from tokenizer import P2CTokenizer

    d = tmp_path_factory.mktemp("nofreq")
    config_path = d / "config.yaml"
    config_path.write_text(
        "vocabs:\n"
        f"  chinese_vocab: {VOCABS_DIR}/chinese_vocab.txt\n"
        f"  context_vocab: {VOCABS_DIR}/context_vocab.txt\n"
        f"  pinyin_vocab: {VOCABS_DIR}/pinyin_vocab.txt\n"
        "  context_special_tokens:\n"
        "    - bos_token\n",
        encoding="utf-8",
    )
    return P2CTokenizer.from_config(str(config_path))
