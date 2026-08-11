"""
PostfixLM inference with KV cache.

The inference pipeline:
  1. Pre model encodes Chinese prefix (with KV cache for self-attention).
  2. Pre model projects its output to shared K/V and writes them in-place
     to the cross-attention KV cache.
  3. Post model reads cross-attention K/V from cache (all layers share).

Cache shapes:
  pre_kv_cache:       (num_layers, 2, 1, pre_max_seqlen, nheads, head_dim)
  pre_cross_kv_cache: (2, 1, pre_max_seqlen, post_nheads, post_head_dim)
"""

import time

import torch
import torch.nn.functional as F

from algo.trie import load_trie, find_matching_words
from algo.viterbi_dp import viterbi_nbest
from tokenizer import P2CTokenizer
from model.config import PreModelConfig, PostModelConfig, build_configs_from_dict
from model.model import PhonoP2CPreModel, PhonoP2CPostModel

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


def create_kv_caches(pre_model: PhonoP2CPreModel, post_model: PhonoP2CPostModel,
                     device: torch.device, batch_size: int = 1, dtype=torch.float32):
    """Allocate KV cache tensors for inference.

    Returns:
        pre_kv_cache:       (pre_num_layers, 2, B, pre_max_seqlen, pre_nheads, pre_head_dim)
        pre_cross_kv_cache: (2, B, pre_max_seqlen, post_nheads, post_head_dim)
    """
    pre_nheads = pre_model.layers[0]["mhsa"].num_heads
    pre_head_dim = pre_model.layers[0]["mhsa"].head_dim
    pre_max = pre_model.max_seqlen

    post_nheads = post_model.layers[0]["mhca"].num_heads
    post_head_dim = post_model.layers[0]["mhca"].head_dim
    print("pre_max value:", pre_max)
    print("pre_max type:", type(pre_max))
    pre_kv_cache = torch.zeros(
        (pre_model.num_layers, 2, batch_size, pre_max,
        pre_nheads, pre_head_dim),
        device=device,
        dtype=dtype
    )
    pre_cross_kv_cache = torch.zeros(
        (2, batch_size, pre_max,
        post_nheads, post_head_dim),
        device=device,
        dtype=dtype
    )
    return pre_kv_cache, pre_cross_kv_cache


@torch.no_grad()
def predict_step(
    text: str,
    pinyin_list: list[str],
    pre_model: PhonoP2CPreModel,
    post_model: PhonoP2CPostModel,
    tokenizer: P2CTokenizer,
    device: torch.device,
    pre_kv_cache: torch.Tensor = None,
    pre_cross_kv_cache: torch.Tensor = None,
    current_seqlen: int = 0,
    topk: int = 1

):

    prefix_ids = [tokenizer.spec_tokens.bos_token]  # BOS
    input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    pre_cache_pos = torch.tensor([current_seqlen], dtype=torch.long, device=device)
    _, _ = pre_model(
        input_ids=input_tensor,
        kv_cache_memory=pre_kv_cache,
        current_seqlen=pre_cache_pos,
        pre_cross_kv_cache=pre_cross_kv_cache,
        pre_cross_cache_pos=pre_cache_pos,
    )
    current_seqlen += 1

    new_prefix_ids = tokenizer.encode_context(text) if text else []
    new_len = len(new_prefix_ids)

    pre_cache_pos = torch.tensor([current_seqlen], dtype=torch.long, device=device)

    if new_len > 0:
        input_tensor = torch.tensor([new_prefix_ids], dtype=torch.long, device=device)
        _, _ = pre_model(
            input_ids=input_tensor,
            kv_cache_memory=pre_kv_cache,
            current_seqlen=pre_cache_pos,
            pre_cross_kv_cache=pre_cross_kv_cache,
            pre_cross_cache_pos=pre_cache_pos,
        )
        current_seqlen += new_len
    else:
        input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        _, _ = pre_model(
            input_ids=input_tensor,
            pre_cross_kv_cache=pre_cross_kv_cache,
            pre_cross_cache_pos=torch.tensor([0], dtype=torch.long, device=device),
        )
        current_seqlen = len(prefix_ids)

    # Encode pinyin
    postfix_ids = tokenizer.encode_pinyin(pinyin_list)
    post_tensor = torch.tensor([postfix_ids], dtype=torch.long, device=device)

    total_pre = torch.tensor([current_seqlen], dtype=torch.long, device=device)
    
    logits = post_model(
        input_ids=post_tensor,
        pre_cross_kv_cache=pre_cross_kv_cache if pre_cross_kv_cache is not None else None,
        current_seqlen=total_pre,
    )
    # logits: [1, S_post, proj_size]

    if topk == 0:
        return {
            "full_logits": logits[0],
            "current_seqlen": current_seqlen,
        }
    elif topk == 1:
        pred_ids = logits[0].argmax(dim=-1).tolist()
        decoded = tokenizer.ids_to_text(pred_ids)
        return {
            "pred_ids": pred_ids,
            "decoded": decoded,
            "current_seqlen": current_seqlen,
        }
    else:
        probs = torch.softmax(logits[0], dim=-1)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        p_log_p = torch.where(probs == 0, torch.zeros_like(probs), probs * log_probs)
        entropy = -p_log_p.sum(dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=topk, dim=-1)
        logit_topk_vals, _ = torch.topk(logits[0], k=topk, dim=-1)
        return {
            "pred_ids_per_pos": topk_idx.tolist(),
            "decoded_per_pos": [[tokenizer.ids_to_text([id_]) for id_ in pos_ids]
                                for pos_ids in topk_idx.tolist()],
            "probs_per_pos": topk_vals.tolist(),
            "logits_per_pos": logit_topk_vals.tolist(),
            "entropy_per_pos": entropy.tolist(),
            "current_seqlen": current_seqlen,
        }


@torch.no_grad()
def predict_step_viterbi(
    text: str,
    pinyin_list: list[str],
    pre_model: PhonoP2CPreModel,
    post_model: PhonoP2CPostModel,
    tokenizer: P2CTokenizer,
    device: torch.device,
    pre_kv_cache: torch.Tensor = None,
    pre_cross_kv_cache: torch.Tensor = None,
    current_seqlen: int = 0,
    beta_single: float = BETA_SINGLE,
    beta_word: float = BETA_WORD,
    trie_path: str = TRIE_PATH,
    epsilon: float = EPSILON,
    n_best: int = N_BEST,
) -> dict:
    """Run one inference step with dictionary-constrained Viterbi N-best decoding.

    Internally calls predict_step(topk=0) for full per-position logits, then
    applies trie word-matching + Viterbi beam-search DP.

    Args:
        text: Accumulated Chinese prefix text.
        pinyin_list: Pinyin syllables to decode.
        pre_model, post_model: The two sub-models.
        tokenizer: P2CTokenizer.
        device: torch device.
        pre_kv_cache, pre_cross_kv_cache: KV caches from create_kv_caches.
        current_seqlen: Current context length.
        beta_single: Prior for single-char transitions (default from hyperparam search).
        beta_word: Prior for multi-char word transitions (default from hyperparam search).
        trie_path: Path to dict_trie.json.
        epsilon: Probability threshold for candidate filtering.
        n_best: Beam width for N-best output.

    Returns:
        dict with keys: nbest, current_seqlen, candidates.
    """
    result = predict_step(
        text, pinyin_list,
        pre_model, post_model, tokenizer, device,
        pre_kv_cache, pre_cross_kv_cache, current_seqlen,
        topk=0,
    )

    full_logits = result["full_logits"].cpu()
    cur_seqlen = result["current_seqlen"]
    probs = torch.softmax(full_logits, dim=-1)

    candidates: list[dict[str, float]] = []
    for pos in range(probs.shape[0]):
        p = probs[pos]
        mask = p > epsilon
        ids = torch.where(mask)[0]
        vals = p[mask]
        if ids.numel() == 0:
            candidates.append({})
            continue

        cand_dict: dict[str, float] = {}
        for tid, pr in zip(ids.tolist(), vals.tolist()):
            ch = tokenizer._id_to_chinese.get(tid, "")
            if ch:
                cand_dict[ch] = float(pr)
        candidates.append(cand_dict)

    trie = _load_trie_cached(trie_path)

    words_at: list[list[tuple[int, str, float]]] = [
        find_matching_words(trie, candidates, i)
        for i in range(len(candidates))
    ]

    best = viterbi_nbest(candidates, words_at, beta_single, beta_word, n_best)

    formatted = []
    for score, word_list in best:
        formatted.append({
            "score": score,
            "words": word_list,
            "text": "".join(word_list),
        })

    return {
        "nbest": formatted,
        "current_seqlen": cur_seqlen,
        "candidates": candidates,
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

    # Create caches
    pre_kv, cross_kv = create_kv_caches(pre_model, post_model, device, batch_size=1, dtype=dtype)

    # Test inference
    text = "这难道不会变得很"
    pinyin_list = "luan ma".split() # expected "乱吗" not "乱码"
    
    print(f"\nPrefix: '{text}'")
    print(f"Pinyin: {pinyin_list}")

    start = time.perf_counter_ns()
    result = predict_step(
        text, pinyin_list,
        pre_model, post_model, tokenizer, device,
        pre_kv_cache=pre_kv,
        pre_cross_kv_cache=cross_kv,
        current_seqlen=0,
        topk=1
    )
    elapsed = (time.perf_counter_ns() - start) / 1e9
    print(f"Greedy: ids={result['pred_ids']}, decoded='{result['decoded']}'")
    print(f"Time: {elapsed:.4f}s")

    # Top-k
    result5 = predict_step(
        text, pinyin_list,
        pre_model, post_model, tokenizer, device,
        pre_kv_cache=pre_kv,
        pre_cross_kv_cache=cross_kv,
        current_seqlen=0,
        topk=5
    )
    print(f"\nTop-5(Pos n: [probs/logits] -> [results], entropy, [ids]):")
    for i, (ids, chars, probs, entropy, logits) in enumerate(zip(result5["pred_ids_per_pos"], result5["decoded_per_pos"], result5["probs_per_pos"], result5["entropy_per_pos"], result5["logits_per_pos"])):
        percentage_n_logits = [f"{num * 100:.1f}%/{logit:.2f}" for num, logit in zip(probs, logits)]
        print(f"  Pos {i}: {percentage_n_logits} -> {chars}, {entropy:.4f}, {ids}")

    # Viterbi dictionary decoding
    print(f"\n--- Viterbi N-Best (N={n_best}) ---")
    print(f"Beta single={beta_single:.4f}, Beta word={beta_word:.4f}, Trie={trie_path}")
    start = time.perf_counter_ns()
    vresult = predict_step_viterbi(
        text, pinyin_list,
        pre_model, post_model, tokenizer, device,
        pre_kv_cache=pre_kv,
        pre_cross_kv_cache=cross_kv,
        current_seqlen=0,
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
