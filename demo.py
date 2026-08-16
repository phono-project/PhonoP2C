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

BETA_SINGLE = 0.4636
BETA_WORD = 0.4839
TRIE_PATH = "./param_search_output/dict_trie.json"
EPSILON = 0.001
N_BEST = 3

_trie_cache: dict | None = None


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
        beams = beam_search(pre_model, post_model, prefix_ids, pinyin_ids,
                            beam_width=1, device=device)
        pred_ids = beams[0][1] if beams else []
        decoded = tokenizer.ids_to_text(pred_ids)
        return {
            "pred_ids": pred_ids,
            "decoded": decoded,
        }

    beams, candidates = beam_search(
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
            for score, ids in beams
        ],
        "per_pos": per_pos,
    }


@torch.no_grad()
def predict_step_viterbi(
    text: str,
    pinyin_list: list[str],
    pre_model: PhonoP2CPreModel,
    post_model: PhonoP2CPostModel,
    tokenizer: P2CTokenizer,
    device: torch.device,
    beta_single: float = BETA_SINGLE,
    beta_word: float = BETA_WORD,
    trie_path: str = TRIE_PATH,
    epsilon: float = EPSILON,
    n_best: int = N_BEST,
    beam_width: int = 8,
) -> dict:
    """Dictionary-constrained Viterbi N-best decoding.

    Collects per-position candidates from a wide beam search, then applies
    trie word-matching + Viterbi beam-search DP.
    """
    prefix_ids = [tokenizer.spec_tokens.bos_token] + tokenizer.encode_context(text)
    pinyin_ids = tokenizer.encode_pinyin(pinyin_list)

    _, candidates = beam_search(
        pre_model, post_model, prefix_ids, pinyin_ids,
        beam_width=beam_width, device=device, return_candidates=True,
    )

    char_candidates: list[dict[str, float]] = []
    for dist in candidates:
        d: dict[str, float] = {}
        for tid, pr in dist.items():
            ch = tokenizer._id_to_chinese.get(tid, "")
            if ch:
                d[ch] = max(d.get(ch, 0.0), float(pr))
        char_candidates.append(d)

    trie = _load_trie_cached(trie_path)

    words_at: list[list[tuple[int, str, float]]] = [
        find_matching_words(trie, char_candidates, i)
        for i in range(len(char_candidates))
    ]

    best = viterbi_nbest(char_candidates, words_at, beta_single, beta_word, n_best)

    formatted = []
    for score, word_list in best:
        formatted.append({
            "score": score,
            "words": word_list,
            "text": "".join(word_list),
        })

    return {
        "nbest": formatted,
        "candidates": char_candidates,
    }


def _load_trie_cached(path: str) -> dict:
    global _trie_cache
    if _trie_cache is None:
        _trie_cache = load_trie(path)
    return _trie_cache


# Demo
if __name__ == "__main__":
    device = torch.device("cpu")
    dtype = torch.float32

    checkpoint_dir = "./checkpoints/v1_0-base/final_model"
    trie_path = "./param_search_output/dict_trie.json"
    beta_single = 0.43
    beta_word = 0.41
    n_best = 5
    epsilon = 0.0

    # Load tokenizer
    tokenizer = P2CTokenizer.from_config("./vocabs/config.yaml")

    pre_model, post_model = load_model_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    pre_model.to(dtype)
    post_model.to(dtype)

    # Test inference
    text = "这难道不会变得很"
    pinyin_list = "luan ma".split()  # expected "乱吗" not "乱码"

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

    # Viterbi dictionary decoding
    print(f"\n--- Viterbi N-Best (N={n_best}) ---")
    print(f"Beta single={beta_single:.4f}, Beta word={beta_word:.4f}, Trie={trie_path}")
    start = time.perf_counter_ns()
    vresult = predict_step_viterbi(
        text, pinyin_list,
        pre_model, post_model, tokenizer, device,
        beta_single=beta_single,
        beta_word=beta_word,
        trie_path=trie_path,
        epsilon=epsilon,
        n_best=n_best,
    )
    elapsed = (time.perf_counter_ns() - start) / 1e9
    print(f"Viterbi (N={n_best}) results ({elapsed:.4f}s):")
    for rank, entry in enumerate(vresult["nbest"]):
        print(f"  [{rank+1}] score={entry['score']:.4f} text='{entry['text']}' words={entry['words']}")
