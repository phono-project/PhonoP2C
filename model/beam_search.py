"""Beam search decoding for the encoder-decoder PhonoP2C.

Encoder position ids are placed after the *full* decoder length
``L_b = len(prefix) + len(pinyin)`` so that training (NJT) and inference
(batched) cross-attention positions coincide.
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


@torch.no_grad()
def beam_search(
    pre_model,
    post_model,
    prefix_ids: list[int],
    pinyin_ids: list[int],
    beam_width: int = 3,
    device: Optional[torch.device] = None,
    return_candidates: bool = False,
):
    """Fixed-length beam search over the chinese vocabulary for one sample.

    Args:
        pre_model: PhonoP2CPreModel (causal decoder).
        post_model: PhonoP2CPostModel (pinyin encoder).
        prefix_ids: context-vocab ids of the prefix (BOS included).
        pinyin_ids: pinyin-vocab ids of the suffix.
        beam_width: number of beams kept at each step.
        device: target device; defaults to the pre model's device.
        return_candidates: also return per-position candidate distributions
            (max-pooled over beams) as ``list[dict[int, float]]``.

    Returns:
        A list of ``(logprob_sum, chinese_id_list)`` sorted by descending
        score.  When ``return_candidates`` is set, returns
        ``(beams, candidates)``.
    """
    if device is None:
        device = next(pre_model.parameters()).device
    dtype = next(pre_model.parameters()).dtype

    P = len(prefix_ids)
    T = len(pinyin_ids)
    L_full = P + T  # full decoder length; encoder positions start here

    prefix_t = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    pinyin_t = torch.tensor([pinyin_ids], dtype=torch.long, device=device)

    # Encode pinyin once (batchsize 1).
    post_hidden, post_mask = post_model(pinyin_t)  # [1, T, dim], [1, T, C]

    cache = create_pre_kv_cache(pre_model, 1, device, dtype)

    # Prime the self-attn KV cache with the prefix.
    _, cache = pre_model(
        prefix_t, kv_cache_memory=cache,
        current_seqlen=torch.zeros(1, dtype=torch.long, device=device),
    )

    # Pass over the prefix for the logits of the first target character.
    logits, cache = pre_model(
        prefix_t, kv_cache_memory=cache,
        current_seqlen=torch.zeros(1, dtype=torch.long, device=device),
        post_hidden=post_hidden,
        post_position_offset=L_full,
        logits_mask=post_mask,
    )  # logits: [1, P, C]
    lp = F.log_softmax(logits[0, P - 1], dim=-1)  # [C]
    top_vals, top_ids = _topk_finite(lp, beam_width)

    beams = []  # list of (score, ids, cache)
    for score, tid in zip(top_vals.tolist(), top_ids.tolist()):
        beams.append((score, [tid], cache))

    candidates: list[dict[int, float]] = []
    if return_candidates:
        candidates.append(_candidate_dist(lp))

    # 4. Generate the remaining characters.
    for t in range(1, T):
        if not beams:
            break
        B = len(beams)
        prev_ids = torch.tensor([[b[1][-1]] for b in beams], dtype=torch.long, device=device)
        # batch dim is dim 2: [layers, 2, B, max, H, D]
        caches = torch.cat([b[2] for b in beams], dim=2)

        logits, caches = pre_model(
            prev_ids,
            kv_cache_memory=caches,
            current_seqlen=torch.full((B,), P + t - 1, dtype=torch.long, device=device),
            post_hidden=post_hidden.expand(B, -1, -1),
            post_position_offset=L_full,
            logits_mask=post_mask.expand(B, -1, -1),
        )  # logits: [B, 1, C]
        lp = F.log_softmax(logits[:, 0, :], dim=-1)  # [B, C]
        top_vals, top_ids = torch.topk(
            lp, k=min(beam_width, lp.size(-1)), dim=-1
        )

        if return_candidates:
            candidates.append(_candidate_dist(lp.max(dim=0).values))

        step_candidates = []
        for b, (beam_score, beam_ids, _) in enumerate(beams):
            row_vals = top_vals[b]
            row_ids = top_ids[b]
            for k in range(row_vals.numel()):
                if not torch.isfinite(row_vals[k]):
                    continue  # masked-out (-inf) expansion
                new_score = beam_score + row_vals[k].item()
                new_ids = beam_ids + [row_ids[k].item()]
                cache_slice = caches[:, :, b : b + 1].contiguous()
                step_candidates.append((new_score, new_ids, cache_slice))

        step_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = step_candidates[:beam_width]

    beams.sort(key=lambda x: x[0], reverse=True)
    result = [(score, ids) for score, ids, _ in beams]
    if return_candidates:
        return result, candidates
    return result


def _candidate_dist(lp: torch.Tensor) -> dict[int, float]:
    """Convert a log-prob vector into {id: prob} over finite (allowed) entries."""
    finite = torch.isfinite(lp)
    ids = torch.where(finite)[0]
    vals = lp[finite]
    if ids.numel() == 0:
        return {}
    probs = torch.exp(vals)
    return {int(i): float(p) for i, p in zip(ids.tolist(), probs.tolist())}


def _topk_finite(lp: torch.Tensor, k: int):
    """topk over the finite (logits-mask allowed) entries only."""
    n_finite = int(torch.isfinite(lp).sum().item())
    k = min(k, n_finite)
    if k <= 0:
        return torch.empty(0, device=lp.device), torch.empty(0, dtype=torch.long, device=lp.device)
    return torch.topk(lp, k=k)
