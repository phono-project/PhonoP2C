"""Multi-head self-attention and cross-attention layers for PostfixLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import RotaryEmbedding, apply_rotary_pos_emb, make_local_position_ids
from model.custom_ops import update_mhsa_kv

class MHSALayer(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding.
    Supports NJT (Nested Jagged Tensor) and batched-dense forward paths.
    """

    def __init__(self, model_dim: int, attn_dim: int, num_heads: int, rope_theta: float, max_seqlen: int):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.max_seqlen = max_seqlen

        self.qkv_proj = nn.Linear(model_dim, attn_dim * 3, bias=False)
        self.out_proj = nn.Linear(attn_dim, model_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta)

    def _apply_rope(self, q, k, position_ids):
        cos, sin = self.rotary(position_ids)
        q = apply_rotary_pos_emb(q.unsqueeze(0), cos, sin)
        k = apply_rotary_pos_emb(k.unsqueeze(0), cos, sin)
        return q.squeeze(0), k.squeeze(0)

    def forward(self, hidden, offsets=None, is_causal=True, kv_cache=None, kv_cache_full=None,
                cache_pos=None, layer_idx=None, position_ids=None,
                min_seqlen=None, max_seqlen=None):
        """Self-attention forward.

        Args:
            hidden: [total_tokens, dim] (NJT) or [B, S, dim] (batched).
            offsets: [B+1] if NJT path, else None.
            is_causal: passed to SDPA.
            kv_cache: (k_cache, v_cache) pre-allocated tensors for KV cache
                      update. Each shaped [B, max_seqlen, num_heads, head_dim].
            cache_pos: [B] start position for this chunk.
            position_ids: optional precomputed local position ids for the NJT
                path (see make_local_position_ids).
            min_seqlen: precomputed Python int or None. The minimum sequence
                length across the batch for the NJT offsets. When provided,
                this avoids deriving it from tensor data inside the compiled
                forward, keeping the graph free of data-dependent breaks.
            max_seqlen: precomputed Python int or None. The maximum sequence
                length across the batch for the NJT offsets.

        Returns:
            Output tensor in the same shape as hidden.
        """
        if offsets is not None:   # (NJT path)

            flat_tokens = hidden
            if position_ids is None:
                position_ids = make_local_position_ids(offsets)

            total_tokens = flat_tokens.shape[0]
            q, k, v = self.qkv_proj(flat_tokens).chunk(3, dim=-1)
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_heads, self.head_dim)
            v = v.view(total_tokens, self.num_heads, self.head_dim)
            
            # Flat RoPE
            q, k = self._apply_rope(q, k, position_ids)

            # Build NJTs for SDPA
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

            out_nt = F.scaled_dot_product_attention(q_nt, k_nt, v_nt, is_causal=is_causal)
            out_flat = out_nt.transpose(1, 2).values()
            out_flat = out_flat.reshape(total_tokens, self.attn_dim)
            return self.out_proj(out_flat)

        else:   # (Batched path)
            # hidden: [B, S, dim]
            B, S = hidden.shape[:2]

            q, k, v = self.qkv_proj(hidden).chunk(3, dim=-1)
            q = q.view(B, S, self.num_heads, self.head_dim)
            k = k.view(B, S, self.num_heads, self.head_dim)
            v = v.view(B, S, self.num_heads, self.head_dim)

            # If kv_cache_full is provided, write to the full cache.
            if kv_cache_full is not None and cache_pos is not None and layer_idx is not None:
                pos = cache_pos[0].item() 
                total_kv = pos + S
                torch._check(pos >= 0)
                torch._check(total_kv <= self.max_seqlen)
                kv_cache_full = update_mhsa_kv(kv_cache_full, k, v, pos, layer_idx)
                k_cache = kv_cache_full[layer_idx, 0]
                v_cache = kv_cache_full[layer_idx, 1]
                # Slice valid portion
                valid_k = torch.narrow(k_cache, dim=1, start=0, length=pos + S)
                valid_v = torch.narrow(v_cache, dim=1, start=0, length=pos + S)
                B_c, total_kv, _, _ = valid_k.shape

                # Flatten RoPE
                position_ids = torch.arange(total_kv, device=hidden.device, dtype=torch.long).repeat(B_c)
                cos, sin = self.rotary(position_ids)

                # q RoPE
                q_pos = torch.arange(pos, pos + S, device=hidden.device, dtype=torch.long)
                q_cos, q_sin = self.rotary(q_pos)
                q = apply_rotary_pos_emb(q, q_cos, q_sin)

                # k RoPE
                valid_k_flat = valid_k.reshape(B_c * total_kv, self.num_heads, self.head_dim)
                valid_k_flat = apply_rotary_pos_emb(
                    valid_k_flat.unsqueeze(0), cos, sin
                )
                valid_k = valid_k_flat.squeeze(0).reshape(B_c, total_kv, self.num_heads, self.head_dim)

                q = q.transpose(1, 2)
                valid_k = valid_k.transpose(1, 2)
                valid_v = valid_v.transpose(1, 2)

                torch._check(position_ids.shape[0] > 0)
                
                q_pos = torch.arange(S, device=hidden.device) + cache_pos[0]
                k_pos = torch.arange(total_kv, device=hidden.device)
                # attn_mask shape: [S, total_kv]
                attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

                out = F.scaled_dot_product_attention(
                    q, valid_k, valid_v, attn_mask=attn_mask, is_causal=False
                )

                out = out.transpose(1, 2).reshape(B, S, self.attn_dim)
                return self.out_proj(out)

            # (Std batched path)
            position_ids = torch.arange(S, device=hidden.device, dtype=torch.long)
            cos, sin = self.rotary(position_ids)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            out = out.transpose(1, 2).reshape(B, S, self.attn_dim)
            return self.out_proj(out)


class MHCALayer(nn.Module):
    """Multi-head cross-attention layer.

    Queries come from the post model's hidden states; keys and values come
    directly from the pre model's shared KV projection.

    Two paths:
        NJT: pre_K, pre_V are passed directly.
        Batched: reads pre-KV from cache where pre model writes in-place.
    """

    def __init__(self, model_dim: int, mhca_attn_dim: int, num_heads: int, rope_theta: float, max_seqlen: int):
        super().__init__()
        assert mhca_attn_dim % num_heads == 0
        self.model_dim = model_dim
        self.mhca_attn_dim = mhca_attn_dim
        self.num_heads = num_heads
        self.head_dim = mhca_attn_dim // num_heads
        self.max_seqlen = max_seqlen

        self.q_proj = nn.Linear(model_dim, mhca_attn_dim, bias=False)
        self.out_proj = nn.Linear(mhca_attn_dim, model_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta)

    @staticmethod
    def compute_position_ids(offsets: torch.Tensor, pre_offsets: torch.Tensor):
        """Precompute query/key position ids for the NJT cross-attention path.
        Returns:
            (q_position_ids, kv_position_ids)
        """
        pre_seq_lens = pre_offsets[1:] - pre_offsets[:-1]   # [B]
        q_seq_lens   = offsets[1:] - offsets[:-1]           # [B]
        q_offset     = torch.repeat_interleave(pre_seq_lens, q_seq_lens)
        q_position_ids  = make_local_position_ids(offsets) + q_offset
        kv_position_ids = make_local_position_ids(pre_offsets)
        return q_position_ids, kv_position_ids

    def forward(self, hidden, pre_K=None, pre_V=None, offsets=None, pre_offsets=None,
                pre_kv_cache=None, cache_pos=None,
                q_position_ids=None, kv_position_ids=None,
                min_seqlen_q=None, max_seqlen_q=None,
                min_seqlen_kv=None, max_seqlen_kv=None):
        """Cross-attention forward.

        Two paths:
          NJT path: uses pre_K, pre_V directly from pre model.
          Batched path: reads from pre_kv_cache that pre model filled.

        Args:
            hidden: [total_tokens, model_dim] (NJT) or [B, S_q, model_dim] (batched).
            pre_K: [total_pre_tokens, heads, head_dim] (NJT). Pre-projected keys.
            pre_V: same shape as pre_K. Pre-projected values.
            offsets: [B+1] query NJT offsets, or None.
            pre_offsets: [B+1] KV NJT offsets, or None.
            pre_kv_cache: (k_cache, v_cache) tuple for batched path.
                Each [B, pre_max_seqlen, num_heads, head_dim]. K stored pre-RoPE.
            cache_pos: [B] — total pre context length (batched path).
            q_position_ids / kv_position_ids: precomputed for NJT path.
            min_seqlen_q / max_seqlen_q: NJT path.
            min_seqlen_kv / max_seqlen_kv: NJT path.

        Returns:
            Output tensor same shape as hidden.
        """
        if offsets is not None:   # (NJT path)
            flat_q = hidden
            total_q = flat_q.shape[0]
            q = self.q_proj(flat_q)
            q = q.view(total_q, self.num_heads, self.head_dim)

            if q_position_ids is None or kv_position_ids is None:
                q_position_ids, kv_position_ids = self.compute_position_ids(offsets, pre_offsets)

            cos_q, sin_q = self.rotary(q_position_ids)
            q = apply_rotary_pos_emb(q.unsqueeze(0), cos_q, sin_q)
            q = q.squeeze(0)

            k = pre_K
            v = pre_V

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
                seq_lens_kv = pre_offsets[1:] - pre_offsets[:-1]
                min_sl_kv = seq_lens_kv.min()
                max_sl_kv = seq_lens_kv.max()
            else:
                min_sl_kv = min_seqlen_kv
                max_sl_kv = max_seqlen_kv
            k_nt = torch.nested.nested_tensor_from_jagged(k, pre_offsets, min_seqlen=min_sl_kv, max_seqlen=max_sl_kv)
            v_nt = torch.nested.nested_tensor_from_jagged(v, pre_offsets, min_seqlen=min_sl_kv, max_seqlen=max_sl_kv)
            k_nt = k_nt.transpose(1, 2)
            v_nt = v_nt.transpose(1, 2)

            out_nt = F.scaled_dot_product_attention(q_nt, k_nt, v_nt, is_causal=False)
            out_flat = out_nt.transpose(1, 2).values()
            out_flat = out_flat.reshape(total_q, self.mhca_attn_dim)
            return self.out_proj(out_flat)

        else:   # (Batched path)
            # Reads from cache
            B, S_q = hidden.shape[:2]

            q = self.q_proj(hidden)
            q = q.view(B, S_q, self.num_heads, self.head_dim)

            k_cache, v_cache = pre_kv_cache
            total_kv = cache_pos[0].item()
            torch._check(total_kv >= 0)
            torch._check(total_kv <= self.max_seqlen)

            q_pos = torch.arange(total_kv, total_kv + S_q, device=hidden.device, dtype=torch.long)
            q_cos, q_sin = self.rotary(q_pos)
            q = apply_rotary_pos_emb(q, q_cos, q_sin)

            valid_k = torch.narrow(k_cache, dim=1, start=0, length=total_kv)
            valid_v = torch.narrow(v_cache, dim=1, start=0, length=total_kv)
            B_c = valid_k.shape[0]

            kv_pos_ids = torch.arange(total_kv, device=hidden.device, dtype=torch.long).repeat(B_c)
            cos_kv, sin_kv = self.rotary(kv_pos_ids)
            valid_k_flat = valid_k.reshape(B_c * total_kv, self.num_heads, self.head_dim)
            valid_k = apply_rotary_pos_emb(valid_k_flat.unsqueeze(0), cos_kv, sin_kv)
            valid_k = valid_k.squeeze(0).reshape(B_c, total_kv, self.num_heads, self.head_dim)

            q = q.transpose(1, 2)
            valid_k = valid_k.transpose(1, 2)
            valid_v = valid_v.transpose(1, 2)
            
            torch._check(cos_kv.shape[0] > 0)
            out = F.scaled_dot_product_attention(q, valid_k, valid_v, is_causal=False)
            out = out.transpose(1, 2).reshape(B, S_q, self.mhca_attn_dim)
            return self.out_proj(out)