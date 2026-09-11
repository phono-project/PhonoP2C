"""Build and validate a phono-core model package (format version 2.2)."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

MODEL_FORMAT_VERSION = "2.2"
PRE_METHODS = {"pre_model_pass1", "pre_model_pass2", "pre_model_cross_kv"}
POST_METHODS = {"post_model"}
SEGMENT_METHODS = {"forward"}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not a readable JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _tokens(path: Path, label: str) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is not a readable UTF-8 file: {path}: {exc}") from exc
    tokens = ["\n" if line == r"\n" else line for line in lines if line]
    if not tokens:
        raise ValueError(f"{label} must not be empty: {path}")
    duplicate = next((token for index, token in enumerate(tokens) if token in tokens[:index]), None)
    if duplicate is not None:
        raise ValueError(f"{label} contains duplicate token {duplicate!r}: {path}")
    return tokens


def _positive_int(config: dict[str, Any], key: str, label: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label}.{key} must be a positive integer")
    return value


def _number(config: dict[str, Any], key: str, label: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label}.{key} must be a positive number")
    return float(value)


def _validate_hf_config(config: dict[str, Any], kind: str) -> None:
    expected_type = f"phono_p2c_{kind}"
    expected_arch = f"PhonoP2C{kind.title()}Model"
    if config.get("model_type") != expected_type:
        raise ValueError(f"{kind} config model_type must be {expected_type!r}")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or expected_arch not in architectures:
        raise ValueError(f"{kind} config architectures must contain {expected_arch!r}")
    for key in ("model_dim", "attn_dim", "max_seqlen", "mhsa_layers", "mhsa_heads",
                "ffn_common_dim", "vocab_size", "proj_size"):
        _positive_int(config, key, f"{kind} config")
    if config["attn_dim"] % config["mhsa_heads"]:
        raise ValueError(f"{kind} config attn_dim must be divisible by mhsa_heads")
    _number(config, "rope_theta", f"{kind} config")
    if kind == "pre":
        for key in ("mhca_heads", "mhca_attn_dim"):
            _positive_int(config, key, "pre config")
        if config["mhca_attn_dim"] % config["mhca_heads"]:
            raise ValueError("pre config mhca_attn_dim must be divisible by mhca_heads")


def _validate_pte(path: Path, label: str, expected_methods: set[str]):
    try:
        header = path.read_bytes()[:16]
    except OSError as exc:
        raise ValueError(f"{label} is not readable: {path}: {exc}") from exc
    if len(header) < 12 or header[4:8] != b"ET12":
        raise ValueError(f"{label} is not an ExecuTorch ET12 program: {path}")
    try:
        from executorch.runtime import Runtime

        program = Runtime.get().load_program(path)
        methods = set(program.method_names)
    except Exception as exc:
        raise ValueError(f"{label} cannot be parsed by the installed ExecuTorch runtime: {exc}") from exc
    if methods != expected_methods:
        raise ValueError(
            f"{label} methods must be exactly {sorted(expected_methods)}, got {sorted(methods)}"
        )
    return program


def _shape(program, method: str, direction: str, index: int) -> list[int]:
    metadata = program._program.method_meta(method)
    getter = metadata.input_tensor_meta if direction == "input" else metadata.output_tensor_meta
    try:
        return list(getter(index).sizes())
    except Exception as exc:
        raise ValueError(f"{method} {direction} {index} must be a tensor") from exc


def _validate_program_contract(pre_program, post_program, pre: dict[str, Any],
                               post: dict[str, Any], batch_size: int) -> None:
    pre_cache = [pre["mhsa_layers"], 2, 1, pre["max_seqlen"],
                 pre["mhsa_heads"], pre["attn_dim"] // pre["mhsa_heads"]]
    if _shape(pre_program, "pre_model_pass1", "input", 0) != [1, pre["max_seqlen"]]:
        raise ValueError("pre_model_pass1 input shape disagrees with pre config")
    if _shape(pre_program, "pre_model_pass1", "input", 1) != pre_cache:
        raise ValueError("pre_model_pass1 cache shape disagrees with pre config")
    pass2_cache = pre_cache.copy()
    pass2_cache[2] = batch_size
    if _shape(pre_program, "pre_model_pass2", "input", 0) != [batch_size, 1]:
        raise ValueError("runtime batch size disagrees with pre_model_pass2")
    if _shape(pre_program, "pre_model_pass2", "input", 1) != pass2_cache:
        raise ValueError("pre_model_pass2 cache shape disagrees with config")
    cross_shape = [pre["mhsa_layers"], 2, 1, post["max_seqlen"],
                   pre["mhca_heads"], pre["mhca_attn_dim"] // pre["mhca_heads"]]
    if _shape(pre_program, "pre_model_cross_kv", "input", 0) != [1, post["max_seqlen"], post["model_dim"]]:
        raise ValueError("pre_model_cross_kv input shape disagrees with post config")
    if _shape(pre_program, "pre_model_cross_kv", "output", 0) != cross_shape:
        raise ValueError("pre_model_cross_kv output shape disagrees with pre config")
    if _shape(pre_program, "pre_model_pass2", "input", 3) != cross_shape:
        raise ValueError("pre_model_pass2 cross-KV shape is incompatible with pre_model_cross_kv")

    if _shape(post_program, "post_model", "input", 0) != [1, post["max_seqlen"]]:
        raise ValueError("post_model input shape disagrees with post config")
    if _shape(post_program, "post_model", "output", 0) != [1, post["max_seqlen"], post["model_dim"]]:
        raise ValueError("post_model hidden output shape disagrees with post config")
    candidate_ids = _shape(post_program, "post_model", "output", 1)
    candidate_mask = _shape(post_program, "post_model", "output", 2)
    pass2_candidates = _shape(pre_program, "pre_model_pass2", "input", 6)
    if candidate_ids != candidate_mask or candidate_ids[:2] != [1, post["max_seqlen"]]:
        raise ValueError("post_model candidate ID and mask outputs are incompatible")
    if len(pass2_candidates) != 1 or pass2_candidates[0] != candidate_ids[2]:
        raise ValueError("pre and post PTE candidate widths do not match")


def _validate_segment_config(config: dict[str, Any], vocab_size: int) -> None:
    if config.get("model_type") != "phono_pinyin_segment":
        raise ValueError("segment config model_type must be 'phono_pinyin_segment'")
    if "PinyinSegmentModel" not in config.get("architectures", []):
        raise ValueError("segment config architectures must contain 'PinyinSegmentModel'")
    if _positive_int(config, "vocab_size", "segment config") != vocab_size:
        raise ValueError("segment config vocab_size does not match segment vocabulary")
    for key in ("model_dim", "num_layers", "kernel_size", "ffn_dim"):
        _positive_int(config, key, "segment config")
    pad_id = config.get("pad_token_id")
    if not isinstance(pad_id, int) or not 0 <= pad_id < vocab_size:
        raise ValueError("segment config pad_token_id is outside its vocabulary")


def build_package(args: argparse.Namespace) -> Path:
    paths = {
        name: Path(getattr(args, name)).expanduser().resolve()
        for name in ("pre_model", "post_model", "pre_config", "post_config",
                     "chinese_vocab", "context_vocab", "pinyin_vocab")
    }
    pre = _read_json(paths["pre_config"], "pre config")
    post = _read_json(paths["post_config"], "post config")
    _validate_hf_config(pre, "pre")
    _validate_hf_config(post, "post")

    for key in ("model_dim", "rope_theta"):
        if pre[key] != post[key]:
            raise ValueError(f"pre and post configs disagree on {key}")

    chinese = _tokens(paths["chinese_vocab"], "Chinese vocabulary")
    context = _tokens(paths["context_vocab"], "context vocabulary")
    pinyin = _tokens(paths["pinyin_vocab"], "pinyin vocabulary")
    special_tokens = list(args.context_special_token)
    if not special_tokens or len(special_tokens) != len(set(special_tokens)):
        raise ValueError("context special tokens must be non-empty and unique")
    collision = set(context) & set(special_tokens)
    if collision:
        raise ValueError(f"context special tokens already occur in context vocabulary: {sorted(collision)}")
    expected_sizes = {
        "pre vocab_size": len(context) + len(special_tokens),
        "pre proj_size": len(chinese),
        "post vocab_size": len(pinyin),
        "post proj_size": len(chinese),
    }
    actual_sizes = {
        "pre vocab_size": pre["vocab_size"], "pre proj_size": pre["proj_size"],
        "post vocab_size": post["vocab_size"], "post proj_size": post["proj_size"],
    }
    for label, expected in expected_sizes.items():
        if actual_sizes[label] != expected:
            raise ValueError(f"{label} is {actual_sizes[label]}, but vocabulary requires {expected}")

    if not args.model_version.startswith("v2_2-"):
        raise ValueError("model version must start with 'v2_2-' to bind it to format 2.2")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.cache_dtype not in {"float32"}:
        raise ValueError("phono-core v2.2 currently requires cache dtype float32")
    pre_program = _validate_pte(paths["pre_model"], "pre model", PRE_METHODS)
    post_program = _validate_pte(paths["post_model"], "post model", POST_METHODS)
    _validate_program_contract(pre_program, post_program, pre, post, args.batch_size)

    segment_inputs = (args.segment_model, args.segment_vocab)
    if any(segment_inputs) and not all(segment_inputs):
        raise ValueError("segment model and segment vocabulary must be provided together")
    if args.segment_config and not args.segment_model:
        raise ValueError("segment config requires a segment model and vocabulary")
    segment_model = segment_vocab = segment_config = None
    if args.segment_model:
        segment_model = Path(args.segment_model).expanduser().resolve()
        segment_vocab = Path(args.segment_vocab).expanduser().resolve()
        segment_tokens = _tokens(segment_vocab, "segment vocabulary")
        if "<unk>" not in segment_tokens or "<pad>" not in segment_tokens:
            raise ValueError("segment vocabulary must contain <pad> and <unk>")
        if any(len(token.encode("utf-8")) != 1 for token in segment_tokens if not token.startswith("<")):
            raise ValueError("segment vocabulary characters must be single-byte tokens")
        segment_program = _validate_pte(segment_model, "segment model", SEGMENT_METHODS)
        if args.segment_config:
            segment_config = Path(args.segment_config).expanduser().resolve()
            _validate_segment_config(
                _read_json(segment_config, "segment config"), len(segment_tokens)
            )
        if not 3 <= args.segment_min_chars <= args.segment_max_chars:
            raise ValueError("segment limits must satisfy 3 <= min <= max")
        segment_input = _shape(segment_program, "forward", "input", 0)
        segment_output = _shape(segment_program, "forward", "output", 0)
        if segment_input[:1] != [1] or len(segment_input) != 2 or segment_output != [1, segment_input[1] - 1]:
            raise ValueError("segment PTE must map [1, L] to [1, L-1]")
        if args.segment_max_chars > segment_input[1]:
            raise ValueError("segment maximum exceeds the exported PTE input limit")

    package_config: dict[str, Any] = {
        "model_version": args.model_version,
        "model_format_version": MODEL_FORMAT_VERSION,
        "common": {"model_dim": pre["model_dim"], "rope_theta": pre["rope_theta"]},
        "pre_model": {key: pre[key] for key in
                      ("max_seqlen", "mhsa_layers", "mhsa_heads", "attn_dim", "ffn_common_dim")},
        "post_model": {
            **{key: post[key] for key in
               ("max_seqlen", "mhsa_layers", "mhsa_heads", "attn_dim", "ffn_common_dim")},
            "mhca_heads": pre["mhca_heads"],
            "mhca_attn_dim": pre["mhca_attn_dim"],
        },
        "vocabs": {
            "chinese_vocab": "vocabs/chinese_vocab.txt",
            "context_vocab": "vocabs/context_vocab.txt",
            "pinyin_vocab": "vocabs/pinyin_vocab.txt",
            "context_special_tokens": special_tokens,
        },
        "runtime": {
            "batch_size": args.batch_size, "cache_dtype": args.cache_dtype,
            "pre_model_path": "bins/pre_model.pte", "post_model_path": "bins/post_model.pte",
            "pre_pass1_method": "pre_model_pass1", "pre_pass2_method": "pre_model_pass2",
            "pre_cross_kv_method": "pre_model_cross_kv", "post_method": "post_model",
        },
    }
    if segment_model:
        package_config["segmenter"] = {
            "model_path": "bins/pinyin_segment.pte", "method": "forward",
            "char_vocab": "vocabs/pinyin_char_vocab.txt",
            "min_input_chars": args.segment_min_chars,
            "max_input_chars": args.segment_max_chars,
            "layout": "BHWC", "quantization": args.segment_quantization,
        }

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (staging / "bins").mkdir()
        (staging / "vocabs").mkdir()
        for source, relative in (
            (paths["pre_model"], "bins/pre_model.pte"),
            (paths["post_model"], "bins/post_model.pte"),
            (paths["chinese_vocab"], "vocabs/chinese_vocab.txt"),
            (paths["context_vocab"], "vocabs/context_vocab.txt"),
            (paths["pinyin_vocab"], "vocabs/pinyin_vocab.txt"),
        ):
            shutil.copy2(source, staging / relative)
        if segment_model and segment_vocab:
            shutil.copy2(segment_model, staging / "bins/pinyin_segment.pte")
            shutil.copy2(segment_vocab, staging / "vocabs/pinyin_char_vocab.txt")
        (staging / "config.json").write_text(
            json.dumps(package_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-model", required=True, help="exported pre_model PTE")
    parser.add_argument("--post-model", required=True, help="exported post_model PTE")
    parser.add_argument("--pre-config", required=True, help="PhonoP2CPreModel config.json")
    parser.add_argument("--post-config", required=True, help="PhonoP2CPostModel config.json")
    parser.add_argument("--chinese-vocab", required=True)
    parser.add_argument("--context-vocab", required=True)
    parser.add_argument("--pinyin-vocab", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-version", required=True, help="must start with v2_2-")
    parser.add_argument("--context-special-token", action="append", default=["bos_token"])
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--cache-dtype", default="float32")
    parser.add_argument("--segment-model")
    parser.add_argument("--segment-vocab")
    parser.add_argument("--segment-config", help="optional PinyinSegmentModel config.json")
    parser.add_argument("--segment-min-chars", type=int, default=3)
    parser.add_argument("--segment-max-chars", type=int, default=512)
    parser.add_argument("--segment-quantization", choices=("none", "w8a8"), default="none")
    return parser


def main(argv: list[str] | None = None) -> None:
    output = build_package(build_parser().parse_args(argv))
    print(f"Built phono-core model package {MODEL_FORMAT_VERSION}: {output}")


if __name__ == "__main__":
    main()
