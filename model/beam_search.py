"""Beam search decoding for the encoder-decoder PhonoP2C.

Decoding scheme (batched, KV-cache based; mirrors the non-NJT path):

1. Prefill the self-attn KV cache with the context prefix ``prefix_ids[:-1]``
   (no cross-attention).
2. Encode the pinyin sequence once with the post model, producing the encoder
   hidden states and the per-position logits mask.
3. Decode autoregressively: each step feeds one token (the last prefix token,
   then the previously generated target) with cross-attention over the pinyin
   encoder and the corresponding mask row; the KV cache is updated in place.

The self-attention uses *global* per-sample positions (``current_seqlen`` is a
``[B]`` tensor), while the cross-attention uses *local* positions (the pinyin
key offset is 1; the query step is ``cross_q_pos_start``) — RoPE only depends
on relative positions, so these match the training convention exactly.

``beam_search_batch`` decodes a *batch* of samples (same pinyin length) with
tensor ops only, so the full test set can be evaluated with large, efficient
kernels instead of tiny per-sample launches.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def create_pre_kv_cache(pre_model, batch_size: int, device: torch.device, dtype: torch.dtype):
    """Allocate the self-attention KV cache for the pre (decoder) model.

    Shape: [layers, 2, B, pre_max_seqlen, pre_nheads, pre_head_dim].
    """
    nheads = pre_model.layers[0]["mhsa"].num_heads
    head_dim = pre_model.layers[0]["mhsa"].head_dim
    pre_max = pre_model.max_seqlen
    return torch.zeros(
        (pre_model.num_layers, 2, batch_size, pre_max, nheads, head_dim),
        device=device,
        dtype=dtype,
    )


def _candidate_dist(lp: torch.Tensor) -> dict[int, float]:
    """Convert a CPU log-prob vector into {id: prob} over finite entries."""
    finite = torch.isfinite(lp)
    ids = torch.where(finite)[0]
    vals = lp[finite]
    if ids.numel() == 0:
        return {}
    probs = torch.exp(vals)
    return {int(i): float(p) for i, p in zip(ids.tolist(), probs.tolist())}


@torch.no_grad()
def beam_search_batch(
    pre_model,
    post_model,
    prefix_ids_batch: list[list[int]],
    pinyin_ids_batch: list[list[int]],
    beam_width: int = 3,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Batched fixed-length beam search over ``B`` samples (same pinyin length).

    Args:
        pre_model: PhonoP2CPreModel (causal decoder).
        post_model: PhonoP2CPostModel (pinyin encoder).
        prefix_ids_batch: list of ``B`` prefix id lists (BOS included).
        pinyin_ids_batch: list of ``B`` pinyin id lists (all the same length ``T``).
        beam_width: number of beams kept per sample.
        device / dtype: target device and cache/compute dtype.

    Returns:
        ``(scores, ids)`` where ``scores`` is ``[B, beam_width]`` and ``ids`` is
        ``[B, beam_width, T]`` (chinese-vocab ids), sorted descending per sample.
    """
    if device is None:
        device = next(pre_model.parameters()).device
    if dtype is None:
        dtype = next(pre_model.parameters()).dtype

    B = len(prefix_ids_batch)
    T = len(pinyin_ids_batch[0])
    prefix_lens = torch.tensor([len(p) for p in prefix_ids_batch], dtype=torch.long, device=device)  # [B]

    # Right-pad prefixes to P_max.
    P_max = int(prefix_lens.max().item())
    pad_id = prefix_ids_batch[0][0]  # the BOS token
    padded = [p + [pad_id] * (P_max - len(p)) for p in prefix_ids_batch]
    prefix_t = torch.tensor(padded, dtype=torch.long, device=device)  # [B, P_max]
    pinyin_t = torch.tensor(pinyin_ids_batch, dtype=torch.long, device=device)  # [B, T]

    # 1. Encode pinyin once (batched).
    post_hidden, post_mask = post_model(pinyin_t)  # [B, T, dim], [B, T, C]

    # 2. Prefill prefix[:-1] (right-padded, uniform position 0).
    cache = create_pre_kv_cache(pre_model, B, device, dtype)
    if P_max > 1:
        _, cache = pre_model(
            prefix_t[:, :-1], kv_cache_memory=cache,
            current_seqlen=torch.zeros(B, dtype=torch.long, device=device),
        )

    # 3. Step 0: each sample's last prefix token predicts its first target char.
    step0 = torch.tensor([p[-1] for p in prefix_ids_batch], dtype=torch.long, device=device).unsqueeze(1)  # [B, 1]
    logits, cache = pre_model(
        step0, kv_cache_memory=cache,
        current_seqlen=prefix_lens - 1,
        post_hidden=post_hidden, post_position_offset=1, cross_q_pos_start=0,
        logits_mask=post_mask[:, 0:1],
    )  # [B, 1, C]
    lp = F.log_softmax(logits[:, 0, :], dim=-1)  # [B, C]
    beam_scores, beam_ids = torch.topk(lp, k=beam_width, dim=-1)  # [B, beam_width]
    beam_ids = beam_ids.unsqueeze(-1)  # [B, beam_width, 1]

    # 4. Expand to B*beam_width and decode the remaining characters.
    cache = cache.repeat_interleave(beam_width, dim=2)
    post_hidden = post_hidden.repeat_interleave(beam_width, dim=0)
    post_mask = post_mask.repeat_interleave(beam_width, dim=0)

    b_idx = torch.arange(B, device=device)[:, None]  # [B, 1]

    for j in range(1, T):
        flat_prev = beam_ids[:, :, -1].reshape(-1, 1)  # [B*beam_width, 1]
        cur_pos = (prefix_lens - 1 + j).repeat_interleave(beam_width)  # [B*beam_width]
        logits, cache = pre_model(
            flat_prev, kv_cache_memory=cache,
            current_seqlen=cur_pos,
            post_hidden=post_hidden, post_position_offset=1, cross_q_pos_start=j,
            logits_mask=post_mask[:, j:j + 1],
        )  # [B*beam_width, 1, C]
        lp = F.log_softmax(logits[:, 0, :], dim=-1).view(B, beam_width, -1)  # [B, beam_width, C]
        topk_vals, topk_ids = torch.topk(lp, k=beam_width, dim=-1)  # [B, beam_width, beam_width]

        combined = beam_scores[:, :, None] + topk_vals  # [B, beam_width, beam_width]
        top_scores, top_idx = torch.topk(combined.reshape(B, -1), k=beam_width, dim=-1)  # [B, beam_width]
        parent = top_idx // beam_width  # [B, beam_width]
        token = top_idx % beam_width
        new_ids = topk_ids[b_idx, parent, token]  # [B, beam_width]

        parent_ids = beam_ids[b_idx, parent]  # [B, beam_width, j]
        beam_ids = torch.cat([parent_ids, new_ids.unsqueeze(-1)], dim=-1)  # [B, beam_width, j+1]
        beam_scores = top_scores

        # Re-index the KV cache by parent beam.
        global_parent = (b_idx * beam_width + parent).reshape(-1)  # [B*beam_width]
        cache = cache[:, :, global_parent]

    return beam_scores, beam_ids


@torch.no_grad()
def beam_search(
    pre_model,
    post_model,
    prefix_ids: list[int],
    pinyin_ids: list[int],
    beam_width: int = 3,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    return_candidates: bool = False,
):
    """Fixed-length beam search over the chinese vocabulary for one sample.

    Thin wrapper around :func:`beam_search_batch` for a single sample.  When
    ``return_candidates`` is set, per-position candidate distributions are
    additionally computed on CPU.

    Returns ``(scores, ids)`` (``[beam_width]`` / ``[beam_width, T]``), or
    ``((scores, ids), candidates)`` when ``return_candidates`` is set.
    """
    scores, ids = beam_search_batch(
        pre_model, post_model, [prefix_ids], [pinyin_ids],
        beam_width=beam_width, device=device, dtype=dtype,
    )
    scores = scores[0]  # [beam_width]
    ids = ids[0]        # [beam_width, T]

    if not return_candidates:
        return scores, ids

    # Recompute per-position candidates (max-pooled over beams) for the demo.
    if device is None:
        device = next(pre_model.parameters()).device
    if dtype is None:
        dtype = next(pre_model.parameters()).dtype

    P = len(prefix_ids)
    T = len(pinyin_ids)
    prefix_t = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    pinyin_t = torch.tensor([pinyin_ids], dtype=torch.long, device=device)
    post_hidden, post_mask = post_model(pinyin_t)

    cache = create_pre_kv_cache(pre_model, 1, device, dtype)
    if P > 1:
        _, cache = pre_model(prefix_t[:, :-1], kv_cache_memory=cache,
                             current_seqlen=torch.zeros(1, dtype=torch.long, device=device))

    candidates: list[dict[int, float]] = []
    logits, cache = pre_model(
        prefix_t[:, -1:], kv_cache_memory=cache,
        current_seqlen=torch.full((1,), P - 1, dtype=torch.long, device=device),
        post_hidden=post_hidden, post_position_offset=1, cross_q_pos_start=0,
        logits_mask=post_mask[:, 0:1],
    )
    candidates.append(_candidate_dist(F.log_softmax(logits[0, 0], dim=-1).cpu()))

    cache = cache.repeat(1, 1, beam_width, 1, 1, 1)
    post_hidden = post_hidden.expand(beam_width, -1, -1)
    post_mask = post_mask.expand(beam_width, -1, -1)

    for j in range(1, T):
        logits, cache = pre_model(
            ids[:, j - 1:j], kv_cache_memory=cache,
            current_seqlen=torch.full((beam_width,), P - 1 + j, dtype=torch.long, device=device),
            post_hidden=post_hidden, post_position_offset=1, cross_q_pos_start=j,
            logits_mask=post_mask[:, j:j + 1],
        )
        lp = F.log_softmax(logits[:, 0, :], dim=-1).max(dim=0).values
        candidates.append(_candidate_dist(lp.cpu()))

    return (scores, ids), candidates
