import pytest

from tools.build_pack_v2_2 import _tokens, _validate_hf_config, build_parser


def _pre_config():
    return {
        "architectures": ["PhonoP2CPreModel"],
        "model_type": "phono_p2c_pre",
        "model_dim": 16,
        "attn_dim": 8,
        "rope_theta": 1000.0,
        "max_seqlen": 16,
        "mhsa_layers": 2,
        "mhsa_heads": 2,
        "ffn_common_dim": 32,
        "vocab_size": 7,
        "proj_size": 5,
        "mhca_heads": 2,
        "mhca_attn_dim": 8,
    }


def test_hf_config_requires_expected_architecture_and_head_shapes():
    config = _pre_config()
    _validate_hf_config(config, "pre")
    config["mhca_attn_dim"] = 7
    with pytest.raises(ValueError, match="mhca_attn_dim"):
        _validate_hf_config(config, "pre")


def test_vocab_rejects_duplicates(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("a\nb\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _tokens(path, "test vocabulary")


def test_parser_keeps_segment_inputs_optional():
    parser = build_parser()
    required = [
        "--pre-model", "pre.pte", "--post-model", "post.pte",
        "--pre-config", "pre.json", "--post-config", "post.json",
        "--chinese-vocab", "chinese.txt", "--context-vocab", "context.txt",
        "--pinyin-vocab", "pinyin.txt", "--model-version", "v2_2-test",
        "--output-dir", "pack",
    ]
    args = parser.parse_args(required)
    assert args.segment_model is None
    assert args.segment_vocab is None
    assert args.context_special_token == ["bos_token"]
