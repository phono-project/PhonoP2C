"""Beam search tests: exhaustiveness, mask adherence, metric."""

from itertools import product

import torch
import torch.nn.functional as F


def _njt(id_lists):
    ts = [torch.tensor(ids, dtype=torch.long) for ids in id_lists]
    return torch.nested.nested_tensor(ts, layout=torch.jagged)


def _sequential_logprob(pre, post, prefix_ids, pinyin_ids, combo, device):
    """Teacher-forced sequential log-prob of a full id combo (ground truth)."""
    P = len(prefix_ids)  # full prefix length (BOS included)
    T = len(pinyin_ids)
    prefix_t = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    pinyin_t = torch.tensor([pinyin_ids], dtype=torch.long, device=device)

    post_hidden, post_mask = post(pinyin_t)  # [1, T, dim], [1, T, C]

    cache = torch.zeros(
        (pre.num_layers, 2, 1, pre.max_seqlen, pre.layers[0]["mhsa"].num_heads,
         pre.layers[0]["mhsa"].head_dim),
        device=device, dtype=torch.float32,
    )
    _, cache = pre(prefix_t[:, :-1], kv_cache_memory=cache,
                   current_seqlen=torch.zeros(1, dtype=torch.long, device=device))

    # step 0: the last prefix token predicts the first target.
    step0 = torch.tensor([[prefix_ids[-1]]], dtype=torch.long, device=device)
    logits, cache = pre(
        step0, kv_cache_memory=cache,
        current_seqlen=torch.full((1,), P - 1, dtype=torch.long, device=device),
        post_hidden=post_hidden, post_position_offset=1, cross_q_pos_start=0,
        logits_mask=post_mask[:, 0:1],
    )
    total = F.log_softmax(logits[0, 0], dim=-1)[combo[0]].item()

    for j in range(1, T):
        tok = torch.tensor([[combo[j - 1]]], dtype=torch.long, device=device)
        logits, cache = pre(
            tok, kv_cache_memory=cache,
            current_seqlen=torch.full((1,), P - 1 + j, dtype=torch.long, device=device),
            post_hidden=post_hidden, post_position_offset=1, cross_q_pos_start=j,
            logits_mask=post_mask[:, j:j + 1],
        )
        total += F.log_softmax(logits[0, 0], dim=-1)[combo[j]].item()
    return total


def test_beam_search_exhaustive_agreement(tiny_models):
    from model.beam_search import beam_search

    pre, post, _, _ = tiny_models
    device = torch.device("cpu")

    pinyin_ids = [3, 5, 8]
    prefix_ids = [1, 5, 7]

    allowed = {3: [6, 7, 103], 5: [10, 11, 105], 8: [16, 17, 108]}
    mask = torch.zeros(100, 250, dtype=torch.bool)
    for p, cs in allowed.items():
        for c in cs:
            mask[p, c] = True
    post.logits_mask = mask

    # beam width >= number of combos -> beam search is exhaustive
    combos = list(product(*allowed.values()))
    beam_width = len(combos)
    scores, beam_ids = beam_search(pre, post, prefix_ids, pinyin_ids,
                                   beam_width=beam_width, device=device)
    assert beam_ids.shape[0] == beam_width

    true_scores = []
    for combo in combos:
        true_scores.append((_sequential_logprob(pre, post, prefix_ids, pinyin_ids, combo, device), combo))
    true_scores.sort(key=lambda x: -x[0])

    for rank in range(beam_width):
        assert abs(scores[rank].item() - true_scores[rank][0]) < 1e-4
        assert beam_ids[rank].tolist() == list(true_scores[rank][1])


def test_beam_search_respects_mask(tiny_models):
    from model.beam_search import beam_search

    pre, post, _, _ = tiny_models
    device = torch.device("cpu")
    pinyin_ids = [3, 5]
    mask = torch.zeros(100, 250, dtype=torch.bool)
    mask[3, 6] = True
    mask[5, 10] = True
    post.logits_mask = mask

    scores, beam_ids = beam_search(pre, post, [1, 5, 7], pinyin_ids, beam_width=3, device=device)
    assert scores.isfinite().sum().item() == 1
    assert beam_ids[0].tolist() == [6, 10]


def test_beam_search_returns_candidates(tiny_models):
    from model.beam_search import beam_search

    pre, post, _, _ = tiny_models
    device = torch.device("cpu")
    pinyin_ids = [3, 5]
    mask = torch.zeros(100, 250, dtype=torch.bool)
    for p in pinyin_ids:
        for c in range(20):
            mask[p, c] = True
    post.logits_mask = mask

    (scores, beam_ids), candidates = beam_search(pre, post, [1, 5, 7], pinyin_ids,
                                                 beam_width=4, device=device,
                                                 return_candidates=True)
    assert beam_ids.shape == (4, 2)
    assert len(candidates) == 2
    for dist in candidates:
        assert dist
        for tid, pr in dist.items():
            assert 0.0 < pr <= 1.0


def test_beam_search_empty_prefix(tiny_models):
    """Empty context (prefix_ids = [BOS] only) must skip the prefill."""
    from model.beam_search import beam_search

    pre, post, _, _ = tiny_models
    device = torch.device("cpu")
    pinyin_ids = [3, 5]
    mask = torch.zeros(100, 250, dtype=torch.bool)
    for p in pinyin_ids:
        for c in range(20):
            mask[p, c] = True
    post.logits_mask = mask

    scores, beam_ids = beam_search(pre, post, [0], pinyin_ids, beam_width=3, device=device)
    assert beam_ids.shape == (3, 2)


def test_topk_sentence_accuracy():
    from metrics.accumulator import TopKSentenceAccuracy

    acc = TopKSentenceAccuracy(k=3)
    # correct when target is in the top-3
    acc.update([[1, 2], [1, 3], [2, 2]], [1, 3])
    # wrong
    acc.update([[1, 2], [1, 3], [2, 2]], [9, 9])
    # beyond k doesn't count
    acc.update([[1, 2], [1, 3], [2, 2], [4, 5]], [4, 5])
    assert acc.compute() == 1.0 / 3.0

    # tensor path
    acc2 = TopKSentenceAccuracy(k=3)
    acc2.update(torch.tensor([[1, 2], [1, 3], [2, 2]]), torch.tensor([1, 3]))
    acc2.update(torch.tensor([[1, 2], [1, 3], [2, 2]]), torch.tensor([9, 9]))
    assert acc2.compute() == 0.5
