"""
PostfixLM inference demo (new-standard encoder-decoder architecture).

The inference pipeline:
  1. The pre model (causal decoder) primes its self-attn KV cache with the
     Chinese context prefix (unconditional pass).
  2. The post model (bidirectional encoder) encodes the pinyin sequence once,
     yielding the encoder hidden states and the per-position logits mask.
  3. The pre model's conditional pass (cross-attention over the encoder
     hidden states, masked logits) generates the target characters, either
     greedily or with beam search.

The viterbi demo reuses the per-position candidate distributions collected
during beam search and applies dictionary-constrained N-best decoding.
"""

import time

import torch

from algo.trie import load_trie, find_matching_words
from algo.viterbi_dp import viterbi_nbest
from tokenizer import P2CTokenizer
from model.model import PhonoP2CPreModel, PhonoP2CPostModel
from model.beam_search import beam_search

def load_model_from_checkpoint(checkpoint_dir: str, device: torch.device):
    """Load pre and post models from a checkpoint directory.

    The directory should contain:
      pre_model/  — save_pretrained output (config.json + model.safetensors)
      post_model/ — save_pretrained output (config.json + model.safetensors)
    """
    import os

    pre_path = os.path.join(checkpoint_dir, "pre_model")
    post_path = os.path.join(checkpoint_dir, "post_model")

    pre_model = PhonoP2CPreModel.from_pretrained(pre_path).to(device)
    post_model = PhonoP2CPostModel.from_pretrained(post_path).to(device)

    pre_model.eval()
    post_model.eval()
    return pre_model, post_model


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


# Demo
if __name__ == "__main__":
    device = torch.device("cpu")
    dtype = torch.float32

    checkpoint_dir = "./checkpoints/v2_0-base-alpha03/epoch_3"
    n_best = 5

    # Load tokenizer
    tokenizer = P2CTokenizer.from_config("./vocabs/config.yaml")

    pre_model, post_model = load_model_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    pre_model.to(dtype)
    post_model.to(dtype)

    # Test inference
    text = ""
    pinyin_list = "o mei kai dan mu ji wo cao".split()  # expected "乱吗" not "乱码"

    print(f"\nPrefix: '{text}'")
    print(f"Pinyin: {pinyin_list}")

    start = time.perf_counter_ns()
    result = predict_step(
        text, pinyin_list,
        pre_model, post_model, tokenizer, device,
        topk=1,
    )
    elapsed = (time.perf_counter_ns() - start) / 1e9
    print(f"Greedy: ids={result['pred_ids']}, decoded='{result['decoded']}'")
    print(f"Time: {elapsed:.4f}s")

    # Beam search top-k
    result5 = predict_step(
        text, pinyin_list,
        pre_model, post_model, tokenizer, device,
        topk=5,
    )
    print(f"\nTop-5 beams:")
    for rank, entry in enumerate(result5["nbest"]):
        print(f"  [{rank+1}] score={entry['score']:.4f} '{entry['decoded']}'")