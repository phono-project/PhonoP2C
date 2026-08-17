"""Attention layers for the encoder-decoder PhonoP2C.

MHSALayer — multi-head self-attention with RoPE.  Supports:
    * NJT (jagged) path: pass 1 (prefix) computes pre-RoPE K/V; pass 2
      (suffix) runs causal self-attention over ``[prefix K/V, suffix K/V]``,
    * batched path with an in-place-updated full KV cache (inference/export),
    * plain batched path.

MHCALayer — cross-attention of the causal decoder (pre model) over the
pinyin encoder's (post model) hidden states.  The cross KV projector lives
here and projects the *encoder* hidden states to K/V.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_bs,
    interleave_prefix_suffix,
    make_full_offsets,
    make_local_position_ids,
    make_logical_prefix_positions,
    make_prefix_suffix_positions,
    make_suffix_global_positions,
    offsets_from_lens,
    scaled_dot_product_attention_njt,
)
from model.custom_ops import update_mhsa_kv


class MHSALayer(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding."""

    def __init__(self, model_dim: int, attn_dim: int, num_heads: int, rope_theta: float, max_seqlen: int):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.max_seqlen = max_seqlen

        self.q_proj = nn.Linear(model_dim, attn_dim, bias=False)
        self.kv_proj = nn.Linear(model_dim, attn_dim * 2, bias=False)
        self.out_proj = nn.Linear(attn_dim, model_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta)

    def _apply_rope_flat(self, q, k, position_ids):
        """RoPE for [T, H, D] tensors with flat [T] position ids."""
        cos, sin = self.rotary(position_ids)
        q = apply_rotary_pos_emb(q.unsqueeze(0), cos, sin)
        k = apply_rotary_pos_emb(k.unsqueeze(0), cos, sin)
        return q.squeeze(0), k.squeeze(0)

    def forward(self, hidden, offsets=None, is_causal=True, kv_cache_full=None,
                cache_pos=None, layer_idx=None, position_ids=None, past_kv=None,
                return_kv=False, min_seqlen=None, max_seqlen=None,
                use_custom_ops=False, prefix_lens=None,
                min_seqlen_full=None, max_seqlen_full=None):
        """Self-attention forward.

        Args:
            hidden: [total_tokens, dim] (NJT) or [B, S, dim] (batched).
            offsets: [B+1] if NJT path, else None (pass-1: prefix offsets,
                pass-2: suffix offsets).
            is_causal: passed to SDPA.
            kv_cache_full: pre-allocated cache [layers, 2, B, max_seqlen, H, D].
            cache_pos: [B] start position for this chunk (batched cache path).
            layer_idx: layer index (batched cache / NJT pass-2).
            position_ids: precomputed position ids for the query (NJT); pass-2
                uses the suffix *global* positions.
            past_kv: single nested jagged tensor [B, j1_physical_prefix, L, 2, H, D]
                holding the (physically padded) prefix pre-RoPE K/V (NJT pass 2).
            prefix_lens: [B] logical prefix lengths (0 for empty-prefix samples);
                used to strip the pad token and offset the suffix positions.
            return_kv: return the pre-RoPE (k, v) of this pass (NJT pass 1).
            min_seqlen / max_seqlen: precomputed Python ints (NJT suffix).
            min_seqlen_full / max_seqlen_full: precomputed Python ints for the
                combined prefix+suffix sequence (NJT pass 2).
            use_custom_ops: use the ``phono::update_mhsa_kv`` torch.library op
                for the in-place cache update (ExecuTorch export only).  The
                default routes to the standard PyTorch implementation, which
                runs on any device (incl. CUDA eval).

        Returns:
            (out, kv) where kv is (k, v) when return_kv, the updated cache
            tensor in the batched cache path, or None otherwise.
        """
        if offsets is not None:   # (NJT path)
            flat_tokens = hidden
            total_tokens = flat_tokens.shape[0]
            q = self.q_proj(flat_tokens)
            q = q.view(total_tokens, self.num_heads, self.head_dim)

            if past_kv is not None:
                # ---- pass 2: suffix Q over [prefix K/V, suffix K/V] causal ----
                physical_prefix_offsets = past_kv.offsets()
                logical_prefix_pos = make_logical_prefix_positions(physical_prefix_offsets, prefix_lens)

                kv_values = past_kv.values()  # [total_physical_prefix, L, 2, H, D]
                prefix_k = kv_values[logical_prefix_pos, layer_idx, 0]  # [total_logical_prefix, H, D]
                prefix_v = kv_values[logical_prefix_pos, layer_idx, 1].contiguous()

                k, v = self.kv_proj(flat_tokens).chunk(2, dim=-1)
                k = k.view(total_tokens, self.num_heads, self.head_dim).contiguous()
                v = v.view(total_tokens, self.num_heads, self.head_dim).contiguous()

                if position_ids is None:
                    position_ids = make_suffix_global_positions(prefix_lens, offsets)
                q, k = self._apply_rope_flat(q, k, position_ids)

                # RoPE the cached prefix K with its local positions.
                prefix_local = make_local_position_ids(offsets_from_lens(prefix_lens))
                cos_p, sin_p = self.rotary(prefix_local)
                prefix_k = apply_rotary_pos_emb(prefix_k.unsqueeze(0), cos_p, sin_p).squeeze(0)

                full_offsets = make_full_offsets(prefix_lens, offsets)
                prefix_pos, suffix_pos = make_prefix_suffix_positions(prefix_lens, offsets, full_offsets)

                q_full = interleave_prefix_suffix(torch.zeros_like(prefix_k), q, prefix_pos, suffix_pos)
                k_full = interleave_prefix_suffix(prefix_k, k, prefix_pos, suffix_pos)
                v_full = interleave_prefix_suffix(prefix_v, v, prefix_pos, suffix_pos)

                if min_seqlen_full is None or max_seqlen_full is None:
                    full_lens = full_offsets[1:] - full_offsets[:-1]
                    min_seqlen_full = full_lens.min()
                    max_seqlen_full = full_lens.max()

                q_nt = torch.nested.nested_tensor_from_jagged(
                    q_full, full_offsets, min_seqlen=min_seqlen_full, max_seqlen=max_seqlen_full
                )
                k_nt = torch.nested.nested_tensor_from_jagged(
                    k_full, full_offsets, min_seqlen=min_seqlen_full, max_seqlen=max_seqlen_full
                )
                v_nt = torch.nested.nested_tensor_from_jagged(
                    v_full, full_offsets, min_seqlen=min_seqlen_full, max_seqlen=max_seqlen_full
                )
                q_nt = q_nt.transpose(1, 2)
                k_nt = k_nt.transpose(1, 2)
                v_nt = v_nt.transpose(1, 2)

                out_nt = scaled_dot_product_attention_njt(q_nt, k_nt, v_nt, is_causal=True)
                out_flat = out_nt.transpose(1, 2).values()  # [total_pre, H, D]
                suffix_out = out_flat[suffix_pos]           # [total_tokens, H, D]
                out = self.out_proj(suffix_out.reshape(total_tokens, self.attn_dim))
                return out, None

            # ---- pass 1 (or encoder bidirectional) ----
            if position_ids is None:
                position_ids = make_local_position_ids(offsets)

            k, v = self.kv_proj(flat_tokens).chunk(2, dim=-1)
            k = k.view(total_tokens, self.num_heads, self.head_dim).contiguous()
            v = v.view(total_tokens, self.num_heads, self.head_dim).contiguous()

            pre_rope_k = k
            q, k = self._apply_rope_flat(q, k, position_ids)

            if min_seqlen is None or max_seqlen is None:
                seq_lens = offsets[1:] - offsets[:-1]
                min_seqlen = seq_lens.min()
                max_seqlen = seq_lens.max()
            q_nt = torch.nested.nested_tensor_from_jagged(q, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen)
            k_nt = torch.nested.nested_tensor_from_jagged(k, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen)
            v_nt = torch.nested.nested_tensor_from_jagged(v, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen)
            q_nt = q_nt.transpose(1, 2)
            k_nt = k_nt.transpose(1, 2)
            v_nt = v_nt.transpose(1, 2)

            out_nt = scaled_dot_product_attention_njt(q_nt, k_nt, v_nt, is_causal=is_causal)
            out_flat = out_nt.transpose(1, 2).values()
            out_flat = out_flat.reshape(total_tokens, self.attn_dim)
            out = self.out_proj(out_flat)
            if return_kv:
                return out, (pre_rope_k, v)
            return out, None

        else:   # (Batched path)
            B, S = hidden.shape[:2]
            q = self.q_proj(hidden)
            q = q.view(B, S, self.num_heads, self.head_dim)

            if kv_cache_full is not None and cache_pos is not None and layer_idx is not None:
                k, v = self.kv_proj(hidden).chunk(2, dim=-1)
                k = k.view(B, S, self.num_heads, self.head_dim).contiguous()
                v = v.view(B, S, self.num_heads, self.head_dim).contiguous()

                if use_custom_ops:
                    # Export path: uniform scalar start position via the custom op.
                    pos = cache_pos[0].item()
                    total_kv = pos + S
                    torch._check(pos >= 0)
                    torch._check(total_kv <= self.max_seqlen)
                    kv_cache_full = update_mhsa_kv(kv_cache_full, k, v, pos, layer_idx)
                    k_cache = kv_cache_full[layer_idx, 0]
                    v_cache = kv_cache_full[layer_idx, 1]
                    valid_k = torch.narrow(k_cache, dim=1, start=0, length=total_kv)
                    valid_v = torch.narrow(v_cache, dim=1, start=0, length=total_kv)
                    B_c, total_kv, _, _ = valid_k.shape

                    position_ids = torch.arange(total_kv, device=hidden.device, dtype=torch.long).repeat(B_c)
                    cos, sin = self.rotary(position_ids)

                    q_pos = torch.arange(pos, pos + S, device=hidden.device, dtype=torch.long)
                    q_cos, q_sin = self.rotary(q_pos)
                    q = apply_rotary_pos_emb(q, q_cos, q_sin)

                    valid_k_flat = valid_k.reshape(B_c * total_kv, self.num_heads, self.head_dim)
                    valid_k_flat = apply_rotary_pos_emb(valid_k_flat.unsqueeze(0), cos, sin)
                    valid_k = valid_k_flat.squeeze(0).reshape(B_c, total_kv, self.num_heads, self.head_dim)

                    q = q.transpose(1, 2)
                    valid_k = valid_k.transpose(1, 2)
                    valid_v = valid_v.transpose(1, 2)

                    torch._check(position_ids.shape[0] > 0)

                    q_pos = torch.arange(S, device=hidden.device) + cache_pos[0]
                    k_pos = torch.arange(total_kv, device=hidden.device)
                    attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

                    out = F.scaled_dot_product_attention(
                        q, valid_k, valid_v, attn_mask=attn_mask, is_causal=False
                    )
                    out = out.transpose(1, 2).reshape(B, S, self.attn_dim)
                    return self.out_proj(out), kv_cache_full

                # Standard path: per-sample start positions (batched decode).
                pos = cache_pos  # [B] int tensor (per-sample)
                if pos.numel() == 1:
                    pos = pos.expand(B)

                # Write K/V at per-sample positions [pos_b, pos_b + S).
                b_idx = torch.arange(B, device=hidden.device)
                s_idx = torch.arange(S, device=hidden.device)
                positions = pos[:, None] + s_idx[None, :]  # [B, S]
                kv_cache_full = kv_cache_full.clone()
                kv_cache_full[layer_idx, 0, b_idx[:, None], positions, :, :] = k
                kv_cache_full[layer_idx, 1, b_idx[:, None], positions, :, :] = v

                k_cache = kv_cache_full[layer_idx, 0]  # [B, max_seqlen, H, D]
                v_cache = kv_cache_full[layer_idx, 1]

                # RoPE the full (pre-RoPE) K window with local positions [0, max_seqlen).
                k_pos = torch.arange(self.max_seqlen, device=hidden.device, dtype=torch.long)
                position_ids = k_pos.repeat(B)
                cos, sin = self.rotary(position_ids)
                k_flat = k_cache.reshape(B * self.max_seqlen, self.num_heads, self.head_dim)
                k_rope = apply_rotary_pos_emb(k_flat.unsqueeze(0), cos, sin)
                k_rope = k_rope.squeeze(0).reshape(B, self.max_seqlen, self.num_heads, self.head_dim)

                # RoPE the query at per-sample positions [B, S].
                cos_q, sin_q = self.rotary(positions.reshape(-1))
                cos_q = cos_q.reshape(B, S, self.head_dim)
                sin_q = sin_q.reshape(B, S, self.head_dim)
                q = apply_rotary_pos_emb_bs(q, cos_q, sin_q)

                q = q.transpose(1, 2)  # [B, H, S, D]
                k_rope = k_rope.transpose(1, 2)
                v_cache = v_cache.transpose(1, 2)

                # Causal mask: sample b's query at q_pos attends to k positions <= q_pos.
                attn_mask = k_pos[None, None, :] <= positions[:, :, None]  # [B, S, max_seqlen]
                out = F.scaled_dot_product_attention(
                    q, k_rope, v_cache, attn_mask=attn_mask.unsqueeze(1), is_causal=False
                )
                out = out.transpose(1, 2).reshape(B, S, self.attn_dim)
                return self.out_proj(out), kv_cache_full

            # (Plain batched path)
            k, v = self.kv_proj(hidden).chunk(2, dim=-1)
            k = k.view(B, S, self.num_heads, self.head_dim).contiguous()
            v = v.view(B, S, self.num_heads, self.head_dim).contiguous()

            pre_rope_k = k
            position_ids = torch.arange(S, device=hidden.device, dtype=torch.long)
            cos, sin = self.rotary(position_ids)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            out = out.transpose(1, 2).reshape(B, S, self.attn_dim)
            if return_kv:
                return self.out_proj(out), (pre_rope_k, v)
            return self.out_proj(out), None


class MHCALayer(nn.Module):
    """Cross-attention: decoder (pre model) queries over encoder hidden states.

    The encoder hidden states are projected to K/V through the *cross KV
    projector* (``kv_proj``), which is skipped entirely in the first
    (unconditional) decoder pass.

    RoPE: query positions are the decoder's suffix positions (placed after the
    prefix); key positions are the pinyin positions *aligned* with the target
    positions (``prefix_len + 1 + local``).
    """

    def __init__(self, model_dim: int, mhca_attn_dim: int, num_heads: int, rope_theta: float):
        super().__init__()
        assert mhca_attn_dim % num_heads == 0
        self.model_dim = model_dim
        self.mhca_attn_dim = mhca_attn_dim
        self.num_heads = num_heads
        self.head_dim = mhca_attn_dim // num_heads

        self.q_proj = nn.Linear(model_dim, mhca_attn_dim, bias=False)
        # Cross KV projector: encoder hidden -> K, V (intermediate dim = mhca_attn_dim)
        self.kv_proj = nn.Linear(model_dim, mhca_attn_dim * 2, bias=False)
        self.out_proj = nn.Linear(mhca_attn_dim, model_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta)

    @staticmethod
    def compute_position_ids(suffix_offsets, post_offsets, prefix_lens):
        """Position ids for the NJT cross-attention path.

        Query positions = decoder suffix positions = ``prefix_len + local``.
        Key positions = pinyin positions *aligned* with the target positions:
        ``prefix_len + 1 + local``.
        """
        suffix_lens = suffix_offsets[1:] - suffix_offsets[:-1]
        post_lens = post_offsets[1:] - post_offsets[:-1]

        q_position_ids = make_local_position_ids(suffix_offsets) + prefix_lens.repeat_interleave(suffix_lens)
        kv_position_ids = make_local_position_ids(post_offsets) + (prefix_lens + 1).repeat_interleave(post_lens)
        return q_position_ids, kv_position_ids

    def forward(self, hidden, enc_hidden=None, offsets=None, post_offsets=None,
                q_position_ids=None, kv_position_ids=None,
                min_seqlen_q=None, max_seqlen_q=None,
                min_seqlen_kv=None, max_seqlen_kv=None,
                q_pos_start=0, kv_pos_offset=0):
        """Cross-attention forward.

        Args:
            hidden: [total_q, model_dim] (NJT) or [B, S_q, model_dim] (batched).
            enc_hidden: encoder hidden states, [total_kv, model_dim] (NJT) or
                [B, T, model_dim] (batched).
            offsets: [B+1] decoder (query) NJT offsets, or None.
            post_offsets: [B+1] encoder (KV) NJT offsets, or None.
            q_position_ids / kv_position_ids: precomputed (NJT path).
            min_seqlen_q / max_seqlen_q: NJT query bounds.
            min_seqlen_kv / max_seqlen_kv: NJT KV bounds.
            q_pos_start: first query position (batched cache path).
            kv_pos_offset: first encoder key position (batched path; the full
                decoder length so encoder positions sit after the decoder).
        """
        if offsets is not None:   # (NJT path)
            flat_q = hidden
            total_q = flat_q.shape[0]
            q = self.q_proj(flat_q)
            q = q.view(total_q, self.num_heads, self.head_dim)

            cos_q, sin_q = self.rotary(q_position_ids)
            q = apply_rotary_pos_emb(q.unsqueeze(0), cos_q, sin_q)
            q = q.squeeze(0)

            total_kv = enc_hidden.shape[0]
            k, v = self.kv_proj(enc_hidden).chunk(2, dim=-1)
            k = k.view(total_kv, self.num_heads, self.head_dim).contiguous()
            v = v.view(total_kv, self.num_heads, self.head_dim).contiguous()

            cos_kv, sin_kv = self.rotary(kv_position_ids)
            k = apply_rotary_pos_emb(k.unsqueeze(0), cos_kv, sin_kv)
            k = k.squeeze(0)

            if min_seqlen_q is None or max_seqlen_q is None:
                seq_lens_q = offsets[1:] - offsets[:-1]
                min_sl_q = seq_lens_q.min()
                max_sl_q = seq_lens_q.max()
            else:
                min_sl_q = min_seqlen_q
                max_sl_q = max_seqlen_q
            q_nt = torch.nested.nested_tensor_from_jagged(q, offsets, min_seqlen=min_sl_q, max_seqlen=max_sl_q)
            q_nt = q_nt.transpose(1, 2)

            if min_seqlen_kv is None or max_seqlen_kv is None:
                seq_lens_kv = post_offsets[1:] - post_offsets[:-1]
                min_sl_kv = seq_lens_kv.min()
                max_sl_kv = seq_lens_kv.max()
            else:
                min_sl_kv = min_seqlen_kv
                max_sl_kv = max_seqlen_kv
            k_nt = torch.nested.nested_tensor_from_jagged(k, post_offsets, min_seqlen=min_sl_kv, max_seqlen=max_sl_kv)
            v_nt = torch.nested.nested_tensor_from_jagged(v, post_offsets, min_seqlen=min_sl_kv, max_seqlen=max_sl_kv)
            k_nt = k_nt.transpose(1, 2)
            v_nt = v_nt.transpose(1, 2)

            out_nt = scaled_dot_product_attention_njt(q_nt, k_nt, v_nt, is_causal=False)
            out_flat = out_nt.transpose(1, 2).values()
            out_flat = out_flat.reshape(total_q, self.mhca_attn_dim)
            return self.out_proj(out_flat)

        else:   # (Batched path)
            B, S_q = hidden.shape[:2]
            T = enc_hidden.shape[1]

            q = self.q_proj(hidden)
            q = q.view(B, S_q, self.num_heads, self.head_dim)

            q_pos = torch.arange(S_q, device=hidden.device, dtype=torch.long) + q_pos_start
            q_cos, q_sin = self.rotary(q_pos)
            q = apply_rotary_pos_emb(q, q_cos, q_sin)

            k, v = self.kv_proj(enc_hidden).chunk(2, dim=-1)
            k = k.view(B, T, self.num_heads, self.head_dim).contiguous()
            v = v.view(B, T, self.num_heads, self.head_dim).contiguous()

            kv_pos_ids = torch.arange(T, device=hidden.device, dtype=torch.long) + kv_pos_offset
            cos_kv, sin_kv = self.rotary(kv_pos_ids)
            k = apply_rotary_pos_emb(k, cos_kv, sin_kv)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            out = out.transpose(1, 2).reshape(B, S_q, self.mhca_attn_dim)
            return self.out_proj(out)
