"""Shared helpers: RoPE, position ids, two-phase sequence utilities, SDPA shims.

The decoder is a single causal sequence ``[BOS, prefix..., suffix...]`` split
into two training phases:

  * pass 1 (unconditional): ``prefix = full_prefix[:-1]`` (the context minus
    its last token), predicting the context one token ahead;
  * pass 2 (conditional): ``suffix = full_prefix[-1:] + target[:-1]`` (the last
    context token + the first ``T-1`` target tokens), reusing pass-1's
    self-attn K/V, predicting the ``T`` target tokens with cross-attention over
    the pinyin encoder.

The pinyin (encoder) RoPE positions are *aligned* with the target positions:
target j sits at ``prefix_len + 1 + j``, so the encoder keys use the same
positions.  ``prefix_len`` is the *logical* prefix length (pass-1 length);
empty-prefix samples have ``prefix_len == 0`` (their pass-1 input is padded to
length 1 with a pad token that is ignored).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer(
            "inv_freq",
            1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        )

    def forward(self, position_ids: torch.Tensor):
        freqs = torch.outer(position_ids.float(), self.inv_freq)  # [T, dim/2]
        freqs = torch.cat((freqs, freqs), dim=-1)                 # [T, dim]
        return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # v: [..., seq_len, heads, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(2).to(v.dtype)  # [1, seq_len, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2).to(v.dtype)
    embed = (v * cos) + (rotate_half(v) * sin)
    return embed


def apply_rotary_pos_emb_bs(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """RoPE for a batched [B, S, H, D] tensor with per-position [B, S, D] freqs."""
    cos = cos.unsqueeze(2).to(v.dtype)  # [B, S, 1, D]
    sin = sin.unsqueeze(2).to(v.dtype)
    return (v * cos) + (rotate_half(v) * sin)


def make_local_position_ids(offsets: torch.Tensor) -> torch.Tensor:
    """Per-sequence local position ids for a jagged layout.

    offsets: [B+1]; returns [total_tokens] with 0..len-1 inside each sequence.
    """
    total_tokens = offsets[-1]
    global_ids = torch.arange(total_tokens, device=offsets.device)

    seq_starts = torch.repeat_interleave(
        offsets[:-1],
        offsets[1:] - offsets[:-1]
    )
    return global_ids - seq_starts


def offsets_from_lens(lens: torch.Tensor) -> torch.Tensor:
    """[B] sequence lengths -> [B+1] offsets."""
    return torch.cat([
        torch.zeros(1, dtype=lens.dtype, device=lens.device),
        torch.cumsum(lens, dim=0),
    ])


def _disable_cuda_sdp_backends_for_cpu():
    # On CPU-only machines the jagged SDPA backend-selection code checks the
    # CUDA backends and may raise "no CUDA-capable device"; disable them.
    if not torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)


def scaled_dot_product_attention_njt(q_nt, k_nt, v_nt, is_causal):
    """SDPA for nested jagged tensors with a CPU causal fallback.

    On CUDA the native jagged SDPA kernel handles ``is_causal=True``.  On CPU
    (tests / no-GPU machines) no jagged causal backend exists, so we fall back
    to per-sequence dense SDPA (skipping empty sequences).  The fallback is
    only exercised outside ``torch.compile``.
    """
    _disable_cuda_sdp_backends_for_cpu()
    if q_nt.is_cuda or not is_causal or torch.compiler.is_compiling():
        return F.scaled_dot_product_attention(q_nt, k_nt, v_nt, is_causal=is_causal)
    outs = []
    for q, k, v in zip(q_nt.unbind(), k_nt.unbind(), v_nt.unbind()):
        if q.size(-2) == 0:
            outs.append(q)
        else:
            outs.append(F.scaled_dot_product_attention(q, k, v, is_causal=True))
    return torch.nested.nested_tensor(outs, layout=torch.jagged)


def make_full_offsets(prefix_lens, suffix_offsets):
    """Offsets of the combined (logical prefix ++ suffix) sequence."""
    suffix_lens = suffix_offsets[1:] - suffix_offsets[:-1]
    return offsets_from_lens(prefix_lens + suffix_lens)


def make_prefix_suffix_positions(prefix_lens, suffix_offsets, full_offsets):
    """Flat indices of prefix and suffix tokens in the combined sequence.

    Returns ``(prefix_pos, suffix_pos)`` into the interleaved combined layout.
    """
    suffix_lens = suffix_offsets[1:] - suffix_offsets[:-1]
    full_starts = full_offsets[:-1]

    prefix_pos = full_starts.repeat_interleave(prefix_lens) + make_local_position_ids(offsets_from_lens(prefix_lens))
    suffix_pos = (full_starts + prefix_lens).repeat_interleave(suffix_lens) + make_local_position_ids(suffix_offsets)
    return prefix_pos, suffix_pos


def interleave_prefix_suffix(prefix_flat, suffix_flat, prefix_pos, suffix_pos):
    """Scatter prefix and suffix flat tensors into one combined flat tensor."""
    total = prefix_flat.shape[0] + suffix_flat.shape[0]
    out = torch.zeros(
        (total,) + tuple(prefix_flat.shape[1:]),
        dtype=prefix_flat.dtype,
        device=prefix_flat.device,
    )
    out = out.index_copy(0, prefix_pos, prefix_flat)
    out = out.index_copy(0, suffix_pos, suffix_flat)
    return out


def make_suffix_global_positions(prefix_lens, suffix_offsets):
    """Global decoder positions of the suffix tokens (placed after the prefix)."""
    suffix_lens = suffix_offsets[1:] - suffix_offsets[:-1]
    return make_local_position_ids(suffix_offsets) + prefix_lens.repeat_interleave(suffix_lens)


def make_logical_prefix_positions(physical_prefix_offsets, prefix_lens):
    """Flat indices of the *logical* prefix tokens inside the physical prefix.

    Empty-prefix samples were padded to length 1; their logical length is 0 so
    they contribute no positions (the pad token is dropped).
    """
    logical_local = make_local_position_ids(offsets_from_lens(prefix_lens))
    return physical_prefix_offsets[:-1].repeat_interleave(prefix_lens) + logical_local


def apply_logits_mask(flat_logits, flat_mask):
    """Mask flat [N, C] logits with an aligned [N, C] bool mask (True = allowed)."""
    return flat_logits.masked_fill(~flat_mask, float("-inf"))


def apply_logits_mask_batched(logits, mask):
    """Mask dense [B, S, C] logits with an aligned [B, S, C] bool mask."""
    return logits.masked_fill(~mask, float("-inf"))
