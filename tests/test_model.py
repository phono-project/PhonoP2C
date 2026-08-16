"""Model tests: two-pass NJT decoder, encoder, masking, batched path."""

import torch
import torch.nn.functional as F


def _njt(id_lists):
    ts = [torch.tensor(ids, dtype=torch.long) for ids in id_lists]
    return torch.nested.nested_tensor(ts, layout=torch.jagged)


def _minmax(offsets):
    lens = offsets[1:] - offsets[:-1]
    return int(lens.min().item()), int(lens.max().item())


# --------------------------------------------------------------------------
# Config migration
# --------------------------------------------------------------------------
def test_config_migration(tiny_cfg_dict, tiny_vocab_sizes):
    from model.config import build_configs_from_dict

    pre_cfg, post_cfg = build_configs_from_dict(tiny_cfg_dict, tiny_vocab_sizes)
    # vocab wiring: decoder input = context, decoder output = chinese,
    # encoder input = pinyin, encoder mask = chinese
    assert pre_cfg.vocab_size == 300
    assert pre_cfg.proj_size == 250
    assert post_cfg.vocab_size == 100
    assert post_cfg.proj_size == 250
    # mhca_* live under post_model but configure the decoder cross-attn
    assert pre_cfg.mhca_heads == tiny_cfg_dict["post_model"]["mhca_heads"]
    assert pre_cfg.mhca_attn_dim == tiny_cfg_dict["post_model"]["mhca_attn_dim"]
    assert pre_cfg.post_max_seqlen == tiny_cfg_dict["post_model"]["max_seqlen"]


# --------------------------------------------------------------------------
# Wrapper: NJT two-pass forward
# --------------------------------------------------------------------------
def test_wrapper_forward_shapes(tiny_models):
    from model.wrapper import PhonoP2CTrainWrapper
    from loss import get_loss_fn

    pre, post, _, _ = tiny_models
    wrapper = PhonoP2CTrainWrapper(pre, post, get_loss_fn("ce"))

    pre_ids = [[1, 5, 7, 9, 11], [1, 3, 4, 12], [1, 8, 6]]
    postfix_ids = [[3, 5], [2, 4, 6], [1]]
    target_ids = [[10, 20], [30, 40, 50], [60]]

    pre_njt = _njt(pre_ids)
    postfix_njt = _njt(postfix_ids)
    target_njt = _njt(target_ids)

    flat_pre = pre_njt.values()
    pre_offsets = pre_njt.offsets()
    flat_postfix = postfix_njt.values()
    postfix_offsets = postfix_njt.offsets()
    flat_target = target_njt.values()

    min_pre, max_pre = _minmax(pre_offsets)
    min_post, max_post = _minmax(postfix_offsets)

    out = wrapper(
        flat_pre, pre_offsets, flat_postfix, postfix_offsets, flat_target,
        min_pre, max_pre, min_post, max_post,
    )

    assert out.conditional_logits.shape == (6, 250)
    assert out.unconditional_logits.shape == (6, 250)
    assert out.loss.dim() == 0
    assert torch.allclose(out.loss, out.unconditional_loss + out.conditional_loss)

    out.loss.backward()
    assert pre.embed.weight.grad is not None
    assert post.embed.weight.grad is not None


def test_past_kv_is_single_njt_with_layer_axis(tiny_models):
    pre, post, pre_cfg, _ = tiny_models
    flat_ids = torch.tensor([1, 5, 7, 9, 11], dtype=torch.long)
    offsets = torch.tensor([0, 5])

    logits1, past_kv = pre(flat_ids, offsets=offsets, min_seqlen=5, max_seqlen=5)
    # single nested jagged tensor with a layer axis: [B, j1, L, 2, H, D]
    assert past_kv.is_nested
    assert past_kv.dim() == 6
    assert past_kv.values().shape == (5, pre.num_layers, 2, 2, 8)
    assert past_kv.size(2) == pre.num_layers

    # pass 2 consumes it directly
    post_hidden = torch.randn(3, 32)
    post_offsets = torch.tensor([0, 3])
    logits2, ret = pre(
        flat_ids, offsets=offsets, min_seqlen=5, max_seqlen=5,
        past_kv=past_kv,
        post_hidden=post_hidden, post_offsets=post_offsets,
        min_seqlen_post=3, max_seqlen_post=3,
    )
    assert ret is None
    assert logits2.values().shape == (5, 250)


def test_training_step_reduces_loss(tiny_models):
    from model.wrapper import PhonoP2CTrainWrapper
    from loss import get_loss_fn

    pre, post, _, _ = tiny_models
    wrapper = PhonoP2CTrainWrapper(pre, post, get_loss_fn("ce"))

    pre_ids = [[1, 5, 7, 9, 11], [1, 3, 4, 12], [1, 8, 6]]
    postfix_ids = [[3, 5], [2, 4, 6], [1]]
    target_ids = [[10, 20], [30, 40, 50], [60]]

    pre_njt = _njt(pre_ids)
    postfix_njt = _njt(postfix_ids)
    target_njt = _njt(target_ids)

    flat_pre = pre_njt.values()
    pre_offsets = pre_njt.offsets()
    flat_postfix = postfix_njt.values()
    postfix_offsets = postfix_njt.offsets()
    flat_target = target_njt.values()
    min_pre, max_pre = _minmax(pre_offsets)
    min_post, max_post = _minmax(postfix_offsets)

    def loss_forward():
        out = wrapper(
            flat_pre, pre_offsets, flat_postfix, postfix_offsets, flat_target,
            min_pre, max_pre, min_post, max_post,
        )
        return out.unconditional_loss, out.conditional_loss

    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-2)
    l0u, l0c = loss_forward()
    opt.zero_grad()
    (l0u + l0c).backward()
    opt.step()
    l1u, l1c = loss_forward()
    assert l1c.item() < l0c.item()
    assert l1u.item() < l0u.item()


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

    # Manual per-sequence dense computation with the same projections.
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
# Pass-2 self-attention uses the cached first-pass K/V
# --------------------------------------------------------------------------
def test_pass2_uses_past_kv(tiny_models):
    pre, post, _, _ = tiny_models
    flat_ids = torch.tensor([1, 5, 7, 9, 11], dtype=torch.long)
    offsets = torch.tensor([0, 5])

    logits1, past_kv = pre(flat_ids, offsets=offsets, min_seqlen=5, max_seqlen=5)
    assert past_kv.is_nested
    assert past_kv.size(2) == pre.num_layers

    post_hidden = torch.randn(3, 32)
    post_offsets = torch.tensor([0, 3])
    logits2, ret = pre(
        flat_ids, offsets=offsets, min_seqlen=5, max_seqlen=5,
        past_kv=past_kv,
        post_hidden=post_hidden, post_offsets=post_offsets,
        min_seqlen_post=3, max_seqlen_post=3,
    )
    assert ret is None
    assert logits2.values().shape == (5, 250)


# --------------------------------------------------------------------------
# Logits mask application (NJT)
# --------------------------------------------------------------------------
def test_logits_mask_njt(tiny_models):
    pre, post, _, _ = tiny_models
    flat_ids = torch.tensor([1, 5, 7, 9, 11], dtype=torch.long)
    offsets = torch.tensor([0, 5])
    post_offsets = torch.tensor([0, 3])
    post_hidden = torch.randn(3, 32)

    _, past_kv = pre(flat_ids, offsets=offsets, min_seqlen=5, max_seqlen=5)
    logits_nomask, _ = pre(
        flat_ids, offsets=offsets, min_seqlen=5, max_seqlen=5,
        past_kv=past_kv, post_hidden=post_hidden, post_offsets=post_offsets,
        min_seqlen_post=3, max_seqlen_post=3,
    )
    mask = torch.ones(3, 250, dtype=torch.bool)
    mask[0, 42] = False
    mask[2, 7] = False
    logits_masked, _ = pre(
        flat_ids, offsets=offsets, min_seqlen=5, max_seqlen=5,
        past_kv=past_kv, post_hidden=post_hidden, post_offsets=post_offsets,
        min_seqlen_post=3, max_seqlen_post=3, logits_mask=mask,
    )

    flat_nomask = logits_nomask.values()
    flat_masked = logits_masked.values()
    # maskable logits positions for one sample: [L - T - 1, L - 1) = [1, 4)
    # row 0 -> position 1, row 2 -> position 3
    for pos in (0, 4):
        assert torch.equal(flat_masked[pos], flat_nomask[pos])
    assert flat_masked[1, 42] == float("-inf")
    assert flat_masked[3, 7] == float("-inf")
    assert torch.equal(flat_masked[1, 43:], flat_nomask[1, 43:])
    # unmasked columns unchanged
    assert flat_masked[1, 41] == flat_nomask[1, 41]
    # full-mask case equals no-mask case
    logits_all, _ = pre(
        flat_ids, offsets=offsets, min_seqlen=5, max_seqlen=5,
        past_kv=past_kv, post_hidden=post_hidden, post_offsets=post_offsets,
        min_seqlen_post=3, max_seqlen_post=3,
        logits_mask=torch.ones(3, 250, dtype=torch.bool),
    )
    assert torch.equal(logits_all.values(), flat_nomask)


# --------------------------------------------------------------------------
# Post model (encoder): NJT vs batched equivalence, mask output
# --------------------------------------------------------------------------
def test_post_model_njt_vs_batched(tiny_models):
    pre, post, _, _ = tiny_models
    post.logits_mask = torch.zeros(100, 250, dtype=torch.bool)
    post.logits_mask[3, 10] = True

    # equal-length sequences so batched rows are not padded
    ids = [[3, 5], [2, 4]]
    njt = _njt(ids)
    hidden_flat, mask_flat = post(njt.values(), input_offsets=njt.offsets(),
                                  min_seqlen=2, max_seqlen=2)
    assert hidden_flat.shape == (4, 32)
    assert mask_flat.shape == (4, 250)
    assert mask_flat[0, 10].item() is True
    assert mask_flat[1, 10].item() is False

    batched = torch.tensor([[3, 5], [2, 4]], dtype=torch.long)
    hidden_b, mask_b = post(batched)
    assert torch.allclose(hidden_flat[0:2], hidden_b[0, 0:2], atol=1e-5)
    assert torch.allclose(hidden_flat[2:4], hidden_b[1, 0:2], atol=1e-5)
    assert torch.equal(mask_flat[0:2], mask_b[0, 0:2])
    assert torch.equal(mask_flat[2:4], mask_b[1, 0:2])


# --------------------------------------------------------------------------
# Cross-attention position ids: encoder positions after the decoder
# --------------------------------------------------------------------------
def test_cross_attention_position_ids():
    from model.attn import MHCALayer

    pre_offsets = torch.tensor([0, 5, 9])
    post_offsets = torch.tensor([0, 3, 6])
    q_pos, kv_pos = MHCALayer.compute_position_ids(pre_offsets, post_offsets)
    assert q_pos.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3]
    # encoder positions placed after the decoder sequence length (5 and 4)
    assert kv_pos.tolist() == [5, 6, 7, 4, 5, 6]


# --------------------------------------------------------------------------
# Batched path: pass 1 cache update + pass 2 with cross-attn + masking
# --------------------------------------------------------------------------
def test_batched_two_pass(tiny_models):
    pre, post, pre_cfg, _ = tiny_models
    B, P, T = 2, 4, 3
    prefix = torch.tensor([[1, 5, 7, 9], [1, 3, 4, 12]], dtype=torch.long)
    pinyin = torch.tensor([[3, 5, 2], [2, 4, 6]], dtype=torch.long)

    post_hidden, post_mask = post(pinyin)

    cache = torch.zeros((pre.num_layers, 2, B, pre.max_seqlen, 2, 8), dtype=torch.float32)
    logits1, cache1 = pre(prefix, kv_cache_memory=cache,
                          current_seqlen=torch.zeros(B, dtype=torch.long))
    assert logits1.shape == (B, P, 250)
    assert cache1.shape == cache.shape
    assert not torch.equal(cache1, cache)

    mask = torch.ones(B, T, 250, dtype=torch.bool)
    mask[:, :, 99] = False
    logits2, cache2 = pre(
        prefix, kv_cache_memory=cache1,
        current_seqlen=torch.zeros(B, dtype=torch.long),
        post_hidden=post_hidden, post_position_offset=P + T,
        logits_mask=mask,
    )
    assert logits2.shape == (B, P, 250)
    # maskable positions: [L - T - 1, L - 1) = [3, 7) -> this chunk covers
    # global positions 0..3, so position 3 is masked by mask row 0
    assert logits2[:, 3, 99].isfinite().sum() == 0
    assert logits2[:, 3, 100].isfinite().all()
    assert torch.isfinite(logits2[:, :3]).all()

    # generation step: single token at pos P (global 4 -> mask row 1)
    next_tok = torch.tensor([[10], [20]], dtype=torch.long)
    logits3, _ = pre(
        next_tok, kv_cache_memory=cache2,
        current_seqlen=torch.full((B,), P, dtype=torch.long),
        post_hidden=post_hidden, post_position_offset=P + T,
        logits_mask=mask,
    )
    assert logits3[0, 0, 99] == float("-inf")
    assert torch.isfinite(logits3[0, 0, 100])


# --------------------------------------------------------------------------
# Batched incremental generation ≡ full-sequence recompute (conditional K/V)
# --------------------------------------------------------------------------
def test_batched_incremental_matches_full_recompute(tiny_models):
    pre, post, _, _ = tiny_models
    B, P, T = 1, 4, 3
    prefix = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)
    pinyin = torch.tensor([[3, 5, 2]], dtype=torch.long)
    post_hidden, post_mask = post(pinyin)
    L_full = P + T
    mask = torch.ones(B, T, 250, dtype=torch.bool)

    # --- incremental path ---
    cache = torch.zeros((pre.num_layers, 2, B, pre.max_seqlen, 2, 8), dtype=torch.float32)
    _, cache = pre(prefix, kv_cache_memory=cache,
                   current_seqlen=torch.zeros(B, dtype=torch.long))
    logits0, cache = pre(
        prefix, kv_cache_memory=cache,
        current_seqlen=torch.zeros(B, dtype=torch.long),
        post_hidden=post_hidden, post_position_offset=L_full, logits_mask=mask,
    )
    first_lp = logits0[0, P - 1]

    step1, cache = pre(
        torch.tensor([[10]], dtype=torch.long), kv_cache_memory=cache,
        current_seqlen=torch.full((B,), P, dtype=torch.long),
        post_hidden=post_hidden, post_position_offset=L_full, logits_mask=mask,
    )
    step1_lp = step1[0, 0]

    # --- full recompute path ---
    full_ids = torch.cat([prefix, torch.tensor([[10, 20]], dtype=torch.long)], dim=1)
    logits_full, _ = pre(full_ids, post_hidden=post_hidden,
                         post_position_offset=L_full, logits_mask=mask)
    assert torch.allclose(first_lp, logits_full[0, P - 1], atol=1e-5)
    assert torch.allclose(step1_lp, logits_full[0, P], atol=1e-5)

    step2, _ = pre(
        torch.tensor([[20]], dtype=torch.long), kv_cache_memory=cache,
        current_seqlen=torch.full((B,), P + 1, dtype=torch.long),
        post_hidden=post_hidden, post_position_offset=L_full, logits_mask=mask,
    )
    assert torch.allclose(step2[0, 0], logits_full[0, P + 1], atol=1e-5)


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

    assert torch.equal(cache_std, cache_op)
    assert torch.allclose(logits_std, logits_op, atol=1e-6)
    # the cache was actually written
    assert not torch.equal(cache_std, torch.zeros_like(cache_std))


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
# gather_target_logits alignment
# --------------------------------------------------------------------------
def test_gather_target_logits():
    from model.utils import gather_target_logits, make_target_logits_positions

    pre_offsets = torch.tensor([0, 6, 10])
    tgt_offsets = torch.tensor([0, 4, 6])
    pos = make_target_logits_positions(pre_offsets, tgt_offsets)
    # sample 0: L=6, T=4 -> [1, 5); sample 1: L=4, T=2 -> [7, 9)
    assert pos.tolist() == [1, 2, 3, 4, 7, 8]

    flat_logits = torch.randn(10, 5)
    gathered = gather_target_logits(flat_logits, pre_offsets, tgt_offsets)
    assert gathered.shape == (6, 5)
    assert torch.equal(gathered[0], flat_logits[1])
    assert torch.equal(gathered[3], flat_logits[4])
    assert torch.equal(gathered[4], flat_logits[7])
    assert torch.equal(gathered[5], flat_logits[8])
