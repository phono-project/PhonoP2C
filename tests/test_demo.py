"""Tests for the parameterized inference demo CLI."""

from pathlib import Path

import pytest
import torch

from demo import build_parser, load_model_from_checkpoint, resolve_runtime


def test_demo_parser_accepts_runtime_inputs():
    args = build_parser().parse_args([
        "--checkpoint",
        "/tmp/checkpoint",
        "--text",
        "上下文",
        "--pinyin",
        "pin",
        "yin",
        "--beam-size",
        "5",
        "--device",
        "cpu",
        "--dtype",
        "float32",
    ])

    assert args.checkpoint == Path("/tmp/checkpoint")
    assert args.text == "上下文"
    assert args.pinyin == ["pin", "yin"]
    assert args.beam_size == 5
    assert resolve_runtime(args.device, args.dtype) == (
        torch.device("cpu"),
        torch.float32,
    )


def test_demo_rejects_invalid_checkpoint_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="pre_model/ and post_model/"):
        load_model_from_checkpoint(tmp_path, torch.device("cpu"))
