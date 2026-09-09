"""Model tests: two-phase NJT decoder, position alignment, masking, batched path."""

import torch
import torch.nn.functional as F


def _njt(id_lists):
    ts = [torch.tensor(ids, dtype=torch.long) for ids in id_lists]
    return torch.nested.nested_tensor(ts, layout=torch.jagged)


def _minmax(offsets):
    lens = offsets[1:] - offsets[:-1]
    return int(lens.min().item()), int(lens.max().item())


# --------------------------------------------------------------------------
# Wrapper: two-phase NJT forward
# --------------------------------------------------------------------------
def _two_phase_batch():
    # sample 0: prefix 3 chars (ids 1,2,3), target 2 (10,20)
    # sample 1: empty prefix, target 2 (10,20)
    # sample 2: prefix 1 char (5), target 1 (12)
    full_prefix = [[0, 1, 2, 3], [0], [0, 5]]
    prefix_ids = [[0, 1, 2], [0], [0]]
    suffix_ids = [[3, 10], [0, 10], [5]]
    uncond_target = [[7, 8, 9], [-100], [11]]
    postfix_ids = [[30, 31], [30, 31], [32]]
    target_ids = [[10, 20], [10, 20], [12]]
    prefix_lens = torch.tensor([3, 0, 1])
    return full_prefix, prefix_ids, suffix_ids, uncond_target, postfix_ids, target_ids, prefix_lens


def _run_wrapper(wrapper, prefix_ids, suffix_ids, uncond_target, postfix_ids, target_ids, prefix_lens):
    prefix_njt = _njt(prefix_ids)
    suffix_njt = _njt(suffix_ids)
    uncond_njt = _njt(uncond_target)
    postfix_njt = _njt(postfix_ids)
    target_njt = _njt(target_ids)

    flat_prefix = prefix_njt.values()
    prefix_offsets = prefix_njt.offsets()
    flat_suffix = suffix_njt.values()
    suffix_offsets = suffix_njt.offsets()
    flat_uncond = uncond_njt.values()
    flat_postfix = postfix_njt.values()
    flat_target = target_njt.values()

    pl = prefix_offsets[1:] - prefix_offsets[:-1]
    sl = suffix_offsets[1:] - suffix_offsets[:-1]
    full = prefix_lens + sl

    return wrapper(
        flat_prefix, prefix_offsets, flat_suffix, suffix_offsets, flat_postfix,
        flat_uncond, flat_target, prefix_lens,
        int(pl.min()), int(pl.max()), int(sl.min()), int(sl.max()),
        int(full.min()), int(full.max()),
    )


def test_wrapper_forward_shapes(tiny_models):
    from model.wrapper import PhonoP2CTrainWrapper
    from loss import get_loss_fn

    pre, post, _, _ = tiny_models
    wrapper = PhonoP2CTrainWrapper(pre, post, get_loss_fn("ce"))

    _, prefix_ids, suffix_ids, uncond_target, postfix_ids, target_ids, prefix_lens = _two_phase_batch()
    out = _run_wrapper(wrapper, prefix_ids, suffix_ids, uncond_target, postfix_ids, target_ids, prefix_lens)

    assert out.conditional_logits.shape == (5, 250)  # total targets (2+2+1)
    assert out.loss.dim() == 0
    assert torch.allclose(out.loss, out.unconditional_loss + out.conditional_loss)

    out.loss.backward()
    assert pre.embed.weight.grad is not None
    assert post.embed.weight.grad is not None


def test_training_step_reduces_loss(tiny_models):
    from model.wrapper import PhonoP2CTrainWrapper
    from loss import get_loss_fn

    pre, post, _, _ = tiny_models
    wrapper = PhonoP2CTrainWrapper(pre, post, get_loss_fn("ce"))
    _, prefix_ids, suffix_ids, uncond_target, postfix_ids, target_ids, prefix_lens = _two_phase_batch()

    def loss_forward():
        out = _run_wrapper(wrapper, prefix_ids, suffix_ids, uncond_target, postfix_ids, target_ids, prefix_lens)
        return out.unconditional_loss, out.conditional_loss

    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-2)
    l0u, l0c = loss_forward()
    opt.zero_grad()
    (l0u + l0c).backward()
    opt.step()
    l1u, l1c = loss_forward()
    assert l1c.item() < l0c.item()
    assert l1u.item() < l0u.item()


def test_empty_prefix_handled(tiny_models):
    from model.wrapper import PhonoP2CTrainWrapper
    from loss import get_loss_fn

    pre, post, _, _ = tiny_models
    wrapper = PhonoP2CTrainWrapper(pre, post, get_loss_fn("ce"))
    # single empty-prefix sample: pass-1 physically padded, logical length 0
    out = _run_wrapper(wrapper, [[0]], [[0, 10]], [[-100]], [[30, 31]], [[10, 20]], torch.tensor([0]))
    assert torch.isfinite(out.loss)
    assert out.conditional_logits.shape == (2, 250)


# --------------------------------------------------------------------------
# Past KV packing (pass 1) + pass 2 reuse
# --------------------------------------------------------------------------
def test_past_kv_is_single_njt_with_layer_axis(tiny_models):
    pre, post, pre_cfg, _ = tiny_models
    flat_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    offsets = torch.tensor([0, 3])

    logits1, past_kv = pre(flat_ids, offsets=offsets, min_seqlen=3, max_seqlen=3)
    assert past_kv.is_nested
    assert past_kv.size(2) == pre.num_layers
    assert past_kv.values().shape == (3, pre.num_layers, 2, 2, 8)

    # pass 2 consumes it
    suffix = torch.tensor([3, 10], dtype=torch.long)
    suffix_offsets = torch.tensor([0, 2])
    post_hidden = torch.randn(2, 32)
    logits2, ret = pre(
        suffix, offsets=suffix_offsets, min_seqlen=2, max_seqlen=2,
        past_kv=past_kv, prefix_lens=torch.tensor([3]),
        min_seqlen_full=5, max_seqlen_full=5,
        post_hidden=post_hidden,
    )
    assert ret is None
    assert logits2.values().shape == (2, 250)


# --------------------------------------------------------------------------
# Cross-attention position ids: pinyin aligned with targets
# --------------------------------------------------------------------------
def test_cross_attention_position_ids():
    from model.attn import MHCALayer

    # prefix_len = 4 (context incl BOS length 5 -> pass-1 length 4)
    suffix_offsets = torch.tensor([0, 2, 4])
    post_offsets = torch.tensor([0, 2, 4])
    prefix_lens = torch.tensor([4, 3])
    q_pos, kv_pos = MHCALayer.compute_position_ids(suffix_offsets, post_offsets, prefix_lens)
    # query (suffix) positions placed after the prefix
    assert q_pos.tolist() == [4, 5, 3, 4]
    # pinyin positions aligned with the target positions (= prefix_len + 1 + local)
    assert kv_pos.tolist() == [5, 6, 4, 5]


# --------------------------------------------------------------------------
# Logits mask application (1:1 with the suffix positions)
# --------------------------------------------------------------------------
def test_logits_mask_njt(tiny_models):
    pre, post, _, _ = tiny_models
    suffix = torch.tensor([3, 10, 11], dtype=torch.long)
    suffix_offsets = torch.tensor([0, 3])
    prefix_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    prefix_offsets = torch.tensor([0, 3])
    post_hidden = torch.randn(3, 32)

    _, past_kv = pre(prefix_ids, offsets=prefix_offsets, min_seqlen=3, max_seqlen=3)
    logits_nomask, _ = pre(
        suffix, offsets=suffix_offsets, min_seqlen=3, max_seqlen=3,
        past_kv=past_kv, prefix_lens=torch.tensor([3]),
        min_seqlen_full=6, max_seqlen_full=6, post_hidden=post_hidden,
    )
    mask = torch.ones(3, 250, dtype=torch.bool)
    mask[0, 42] = False
    mask[2, 7] = False
    logits_masked, _ = pre(
        suffix, offsets=suffix_offsets, min_seqlen=3, max_seqlen=3,
        past_kv=past_kv, prefix_lens=torch.tensor([3]),
        min_seqlen_full=6, max_seqlen_full=6, post_hidden=post_hidden,
        logits_mask=mask,
    )
    flat_nomask = logits_nomask.values()
    flat_masked = logits_masked.values()
    assert flat_masked[0, 42] == float("-inf")
    assert flat_masked[2, 7] == float("-inf")
    assert flat_masked[1, 42] == flat_nomask[1, 42]
    # full-mask equals no-mask
    logits_all, _ = pre(
        suffix, offsets=suffix_offsets, min_seqlen=3, max_seqlen=3,
        past_kv=past_kv, prefix_lens=torch.tensor([3]),
        min_seqlen_full=6, max_seqlen_full=6, post_hidden=post_hidden,
        logits_mask=torch.ones(3, 250, dtype=torch.bool),
    )
    assert torch.equal(logits_all.values(), flat_nomask)


# --------------------------------------------------------------------------
# Causal self-attention correctness (NJT pass 1 vs manual dense reference)
# --------------------------------------------------------------------------
def test_njt_causal_mhsa_matches_manual(tiny_models):
    from model.attn import MHSALayer
    from model.utils import apply_rotary_pos_emb

    torch.manual_seed(1)
    layer = MHSALayer(32, 16, 2, 1000.0, 64)
    hidden = torch.randn(6, 32)
    offsets = torch.tensor([0, 2, 5, 6])

    out, kv = layer(hidden, offsets=offsets, is_causal=True, return_kv=True,
                    min_seqlen=1, max_seqlen=3)
    k_cache, v_cache = kv
    assert out.shape == (6, 32)
    assert k_cache.shape == (6, 2, 8)

    seqs = [hidden[0:2], hidden[2:5], hidden[5:6]]
    manual = []
    with torch.no_grad():
        for seq in seqs:
            S = seq.shape[0]
            q = layer.q_proj(seq).view(S, 2, 8)
            k, v = layer.kv_proj(seq).chunk(2, dim=-1)
            k = k.view(S, 2, 8)
            v = v.view(S, 2, 8)
            position_ids = torch.arange(S, dtype=torch.long)
            cos, sin = layer.rotary(position_ids)
            q = apply_rotary_pos_emb(q.unsqueeze(0), cos, sin).squeeze(0)
            k = apply_rotary_pos_emb(k.unsqueeze(0), cos, sin).squeeze(0)
            o = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                is_causal=True,
            ).squeeze(0).transpose(0, 1).reshape(S, 16)
            manual.append(layer.out_proj(o))
    manual = torch.cat(manual, dim=0)
    assert torch.allclose(out, manual, atol=1e-5)


# --------------------------------------------------------------------------
# Post model (encoder): NJT vs batched equivalence
# --------------------------------------------------------------------------
def test_post_model_njt_vs_batched(tiny_models):
    pre, post, _, _ = tiny_models
    post.logits_mask = torch.zeros(100, 250, dtype=torch.bool)
    post.logits_mask[3, 10] = True

    ids = [[3, 5], [2, 4]]
    njt = _njt(ids)
    hidden_flat, mask_flat = post(njt.values(), input_offsets=njt.offsets(),
                                  min_seqlen=2, max_seqlen=2)
    batched = torch.tensor([[3, 5], [2, 4]], dtype=torch.long)
    hidden_b, mask_b = post(batched)
    assert torch.allclose(hidden_flat[0:2], hidden_b[0], atol=1e-5)
    assert torch.allclose(hidden_flat[2:4], hidden_b[1], atol=1e-5)
    assert torch.equal(mask_flat[0:2], mask_b[0])
    assert torch.equal(mask_flat[2:4], mask_b[1])

    post.enable_sparse_logits()
    _, candidate_ids, candidate_mask = post(batched)
    assert candidate_ids.shape[-1] == 1
    assert candidate_ids[0, 0, 0] == 10
    assert candidate_mask[0, 0, 0]
    assert not candidate_mask[0, 1, 0]


# --------------------------------------------------------------------------
# Batched path: prefill + conditional decode
# --------------------------------------------------------------------------
def test_batched_prefill_decode(tiny_models):
    pre, post, pre_cfg, _ = tiny_models
    B = 2
    prefix = torch.tensor([[0, 1, 2], [0, 5, 9]], dtype=torch.long)  # full prefix (with BOS)
    pinyin = torch.tensor([[30, 31], [32, 33]], dtype=torch.long)

    post_hidden, post_mask = post(pinyin)

    # prefill prefix_ids[:-1] (2 tokens each)
    cache = torch.zeros((pre.num_layers, 2, B, pre.max_seqlen, 2, 8), dtype=torch.float32)
    logits1, cache1 = pre(prefix[:, :-1], kv_cache_memory=cache,
                          current_seqlen=torch.zeros(B, dtype=torch.long))
    assert logits1.shape == (B, 2, 250)

    # decode: last prefix token -> first target, mask row 0
    next_tok = prefix[:, -1:].contiguous()
    mask = torch.ones(B, 1, 250, dtype=torch.bool)
    mask[:, :, 99] = False
    dense_cache = cache1.clone()
    logits2, _ = pre(
        next_tok, kv_cache_memory=dense_cache,
        current_seqlen=torch.tensor([2, 2], dtype=torch.long),
        post_hidden=post_hidden, post_position_offset=3,
        logits_mask=mask,
    )
    assert logits2[0, 0, 99] == float("-inf")
    assert torch.isfinite(logits2[0, 0, 100])

    shared_hidden = post_hidden[:1].expand(B, -1, -1)
    direct_logits, _ = pre(
        next_tok, kv_cache_memory=cache1.clone(),
        current_seqlen=torch.tensor([2, 2], dtype=torch.long),
        post_hidden=shared_hidden, post_position_offset=3,
    )
    cross_kv = torch.stack([
        torch.stack(layer["mhca"].project_kv(post_hidden[:1])) for layer in pre.layers
    ])
    cached_logits, _ = pre(
        next_tok, kv_cache_memory=cache1.clone(),
        current_seqlen=torch.tensor([2, 2], dtype=torch.long),
        cross_kv=cross_kv, post_position_offset=3,
    )
    assert torch.allclose(cached_logits, direct_logits, atol=1e-6)

    candidate_ids = torch.tensor([5, 99, 100], dtype=torch.long)
    candidate_mask = torch.tensor([True, False, True])
    sparse_logits, _ = pre(
        next_tok, kv_cache_memory=cache1,
        current_seqlen=torch.tensor([2, 2], dtype=torch.long),
        post_hidden=post_hidden, post_position_offset=3,
        logits_candidate_ids=candidate_ids,
        logits_candidate_mask=candidate_mask,
    )
    assert sparse_logits.shape == (B, 1, 3)
    assert torch.allclose(sparse_logits[..., [0, 2]], logits2[..., [5, 100]])
    assert sparse_logits[0, 0, 1] == float("-inf")


# --------------------------------------------------------------------------
# KV-cache update routing: standard ops (default) == custom op (export)
# --------------------------------------------------------------------------
def test_kv_cache_standard_equals_custom_op(tiny_models):
    pre, post, pre_cfg, _ = tiny_models
    B = 2
    prefix = torch.tensor([[1, 5, 7, 9], [1, 3, 4, 12]], dtype=torch.long)

    def run(use_custom_ops):
        cache = torch.zeros((pre.num_layers, 2, B, pre.max_seqlen, 2, 8), dtype=torch.float32)
        logits, updated = pre(
            prefix, kv_cache_memory=cache,
            current_seqlen=torch.zeros(B, dtype=torch.long),
            use_custom_ops=use_custom_ops,
        )
        return logits, updated

    logits_std, cache_std = run(False)
    logits_op, cache_op = run(True)
    assert torch.allclose(cache_std, cache_op, atol=1e-6)
    assert torch.allclose(logits_std, logits_op, atol=1e-6)


def test_batched_prefill_can_skip_logits(tiny_models):
    pre, _, pre_cfg, _ = tiny_models
    cache = torch.zeros((pre_cfg.mhsa_layers, 2, 1, pre_cfg.max_seqlen,
                         pre_cfg.mhsa_heads, pre_cfg.attn_dim // pre_cfg.mhsa_heads))
    input_ids = torch.tensor([[1, 5, 7]], dtype=torch.long)

    def reject_projection(*_args):
        raise AssertionError("prefill must not execute lm_proj")

    hook = pre.lm_proj.register_forward_hook(reject_projection)
    try:
        updated = pre(
            input_ids,
            kv_cache_memory=cache,
            current_seqlen=torch.zeros(1, dtype=torch.long),
            return_logits=False,
        )
    finally:
        hook.remove()

    assert updated.shape == cache.shape
    assert torch.count_nonzero(updated[:, :, :, :input_ids.shape[1]]) > 0


def test_export_with_custom_ops(tiny_models):
    from torch.export import Dim

    pre, post, pre_cfg, post_cfg = tiny_models
    for m in (pre, post):
        for p in m.parameters():
            p.requires_grad = False
    pre.eval()

    B = 1
    cache = torch.zeros((pre_cfg.mhsa_layers, 2, B, pre_cfg.max_seqlen,
                         pre_cfg.mhsa_heads, pre_cfg.attn_dim // pre_cfg.mhsa_heads))
    dummy_ids = torch.randint(0, pre_cfg.vocab_size, (B, 4), dtype=torch.long)
    dummy_pos = torch.tensor([2], dtype=torch.long)
    kwargs = {"input_ids": dummy_ids, "kv_cache_memory": cache,
              "current_seqlen": dummy_pos, "use_custom_ops": True}
    dyn = {"input_ids": {1: Dim("s", min=1, max=pre_cfg.max_seqlen)},
           "kv_cache_memory": None, "current_seqlen": None, "use_custom_ops": None}
    exported = torch.export.export(pre, args=(), kwargs=kwargs, dynamic_shapes=dyn).module()
    logits, updated = exported(**kwargs)
    assert logits.shape == (B, 4, pre_cfg.proj_size)
    assert updated.shape == cache.shape


# --------------------------------------------------------------------------
# Pass-2 self-attn == manual causal attention over [prefix, suffix] (B=1)
# --------------------------------------------------------------------------
def test_pass2_self_attn_matches_reference(tiny_models):
    from model.attn import MHSALayer
    from model.utils import apply_rotary_pos_emb

    torch.manual_seed(2)
    layer = MHSALayer(32, 16, 2, 1000.0, 64)
    prefix_hidden = torch.randn(3, 32)
    suffix_hidden = torch.randn(2, 32)

    # pass 1 over prefix
    prefix_offsets = torch.tensor([0, 3])
    _, (prefix_k, prefix_v) = layer(
        prefix_hidden, offsets=prefix_offsets, is_causal=True, return_kv=True,
        min_seqlen=3, max_seqlen=3,
    )
    # pack past_kv (single layer)
    kv_all = torch.stack([torch.stack([prefix_k, prefix_v], dim=1)], dim=1)  # [3, 1, 2, H, D]
    past_kv = torch.nested.nested_tensor_from_jagged(kv_all, prefix_offsets, min_seqlen=3, max_seqlen=3)

    # pass 2 over suffix
    suffix_offsets = torch.tensor([0, 2])
    out, _ = layer(
        suffix_hidden, offsets=suffix_offsets, is_causal=True, past_kv=past_kv, layer_idx=0,
        prefix_lens=torch.tensor([3]), min_seqlen=2, max_seqlen=2,
        min_seqlen_full=5, max_seqlen_full=5,
    )
    assert out.shape == (2, 32)

    # manual reference
    with torch.no_grad():
        q = layer.q_proj(suffix_hidden).view(2, 2, 8)
        k, v = layer.kv_proj(suffix_hidden).chunk(2, dim=-1)
        k = k.view(2, 2, 8); v = v.view(2, 2, 8)

        # suffix global positions = 3 + local
        q_pos = torch.tensor([3, 4])
        cos_q, sin_q = layer.rotary(q_pos)
        q = apply_rotary_pos_emb(q.unsqueeze(0), cos_q, sin_q).squeeze(0)
        k = apply_rotary_pos_emb(k.unsqueeze(0), cos_q, sin_q).squeeze(0)

        # prefix K RoPE (local 0..2)
        prefix_local = torch.tensor([0, 1, 2])
        cos_p, sin_p = layer.rotary(prefix_local)
        pk = apply_rotary_pos_emb(prefix_k.unsqueeze(0), cos_p, sin_p).squeeze(0)

        full_k = torch.cat([pk, k], dim=0)  # [5, H, D]
        full_v = torch.cat([prefix_v, v], dim=0)
        full_q = torch.cat([torch.zeros_like(pk), q], dim=0)

        o = F.scaled_dot_product_attention(
            full_q.transpose(0, 1).unsqueeze(0),
            full_k.transpose(0, 1).unsqueeze(0),
            full_v.transpose(0, 1).unsqueeze(0),
            is_causal=True,
        ).squeeze(0).transpose(0, 1)  # [5, H, D]
        suffix_out = layer.out_proj(o[3:5].reshape(2, 16))

    assert torch.allclose(out, suffix_out, atol=1e-5)
