"""
PhonoP2C inference command-line demo.

The inference pipeline:
  1. The pre model (causal decoder) primes its self-attn KV cache with the
     Chinese context prefix (unconditional pass).
  2. The post model (bidirectional encoder) encodes the pinyin sequence once,
     yielding the encoder hidden states and the per-position logits mask.
  3. The pre model's conditional pass (cross-attention over the encoder
     hidden states, masked logits) generates the target characters, either
     greedily or with beam search.

"""

import argparse
import time
from pathlib import Path

import torch

from model.beam_search import beam_search
from model.model import PhonoP2CPostModel, PhonoP2CPreModel
from tokenizer import P2CTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_model_from_checkpoint(checkpoint_dir: str | Path, device: torch.device):
    """Load pre and post models from a checkpoint directory.

    The directory should contain:
      pre_model/  — save_pretrained output (config.json + model.safetensors)
      post_model/ — save_pretrained output (config.json + model.safetensors)
    """
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    pre_path = checkpoint_dir / "pre_model"
    post_path = checkpoint_dir / "post_model"
    if not pre_path.is_dir() or not post_path.is_dir():
        raise FileNotFoundError(
            f"{checkpoint_dir} must contain pre_model/ and post_model/ directories"
        )

    pre_model = PhonoP2CPreModel.from_pretrained(pre_path).to(device)
    post_model = PhonoP2CPostModel.from_pretrained(post_path).to(device)

    pre_model.eval()
    post_model.eval()
    return pre_model, post_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint directory containing pre_model/ and post_model/",
    )
    parser.add_argument(
        "--vocab-config",
        type=Path,
        default=PROJECT_DIR / "vocabs" / "config.yaml",
        help="tokenizer vocabulary configuration",
    )
    parser.add_argument(
        "--text",
        default="",
        help="Chinese context prefix (defaults to an empty prefix)",
    )
    parser.add_argument(
        "--pinyin",
        nargs="+",
        required=True,
        metavar="SYLLABLE",
        help="space-separated pinyin syllables to decode",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=3,
        help="number of beam-search hypotheses (default: 3)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cpu, cuda, or cuda:1 (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", *DTYPES),
        default="auto",
        help="model dtype (default: bfloat16 on CUDA, float32 on CPU)",
    )
    return parser


def resolve_runtime(device_name: str, dtype_name: str) -> tuple[torch.device, torch.dtype]:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    if dtype_name == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = DTYPES[dtype_name]
    return device, dtype


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict_step(
    text: str,
    pinyin_list: list[str],
    pre_model: PhonoP2CPreModel,
    post_model: PhonoP2CPostModel,
    tokenizer: P2CTokenizer,
    device: torch.device,
    topk: int = 1,
):
    """Decode pinyin into Chinese given the context prefix.

    topk == 1: greedy decoding.
    topk > 1 : beam search with beam_width = topk; returns the N-best
               sentences together with per-position candidate distributions.
    """
    prefix_ids = [tokenizer.spec_tokens.bos_token] + tokenizer.encode_context(text)
    pinyin_ids = tokenizer.encode_pinyin(pinyin_list)

    if topk <= 1:
        scores, beam_ids = beam_search(pre_model, post_model, prefix_ids, pinyin_ids,
                                       beam_width=1, device=device)
        pred_ids = beam_ids[0].tolist() if beam_ids.numel() else []
        decoded = tokenizer.ids_to_text(pred_ids)
        return {
            "pred_ids": pred_ids,
            "decoded": decoded,
        }

    (scores, beam_ids), candidates = beam_search(
        pre_model, post_model, prefix_ids, pinyin_ids,
        beam_width=topk, device=device, return_candidates=True,
    )

    per_pos = []
    for dist in candidates:
        items = sorted(dist.items(), key=lambda kv: -kv[1])[:topk]
        per_pos.append({
            "ids": [i for i, _ in items],
            "chars": [tokenizer.ids_to_text([i]) for i, _ in items],
            "probs": [p for _, p in items],
        })

    return {
        "nbest": [
            {"score": score, "pred_ids": ids, "decoded": tokenizer.ids_to_text(ids)}
            for score, ids in zip(scores.tolist(), beam_ids.tolist())
        ],
        "per_pos": per_pos,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.beam_size <= 0:
        raise ValueError("--beam-size must be positive")

    device, dtype = resolve_runtime(args.device, args.dtype)
    tokenizer = P2CTokenizer.from_config(str(args.vocab_config.expanduser()))

    pre_model, post_model = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        device=device,
    )
    pre_model.to(dtype)
    post_model.to(dtype)

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Prefix: {args.text!r}")
    print(f"Pinyin: {args.pinyin}")

    synchronize(device)
    start = time.perf_counter_ns()
    result = predict_step(
        args.text,
        args.pinyin,
        pre_model, post_model, tokenizer, device,
        topk=1,
    )
    synchronize(device)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    print(f"Greedy: ids={result['pred_ids']}, decoded='{result['decoded']}'")
    print(f"Greedy time: {elapsed:.4f}s")

    if args.beam_size == 1:
        return

    synchronize(device)
    start = time.perf_counter_ns()
    beam_result = predict_step(
        args.text,
        args.pinyin,
        pre_model, post_model, tokenizer, device,
        topk=args.beam_size,
    )
    synchronize(device)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    print(f"Top-{args.beam_size} beams:")
    for rank, entry in enumerate(beam_result["nbest"], start=1):
        print(f"  [{rank}] score={entry['score']:.4f} {entry['decoded']!r}")
    print(f"Beam-search time: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
