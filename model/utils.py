"""Shared helpers: RoPE, position ids, target-logits alignment, SDPA shims."""

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
    to per-sequence dense SDPA.  The fallback is only exercised outside
    ``torch.compile`` (training compiles the CUDA native path).
    """
    _disable_cuda_sdp_backends_for_cpu()
    if q_nt.is_cuda or not is_causal or torch.compiler.is_compiling():
        return F.scaled_dot_product_attention(q_nt, k_nt, v_nt, is_causal=is_causal)
    outs = []
    for q, k, v in zip(q_nt.unbind(), k_nt.unbind(), v_nt.unbind()):
        outs.append(F.scaled_dot_product_attention(q, k, v, is_causal=True))
    return torch.nested.nested_tensor(outs, layout=torch.jagged)


def make_target_logits_positions(pre_offsets, target_offsets):
    """Flat indices of the decoder logits positions that carry labels.

    The decoder input layout is ``[BOS + context prefix] + target``.  With
    next-token prediction, target token j of sample b (input position
    ``P_b + j``) is predicted by the logits at position ``P_b + j - 1``,
    i.e. ``L_b - T_b - 1 + j`` where ``L_b`` is the sample's decoder length
    and ``T_b`` the target (== pinyin) length.

    Returns an ascending index tensor ``[total_target]`` into the flat logits.
    """
    pre_lens = pre_offsets[1:] - pre_offsets[:-1]
    tgt_lens = target_offsets[1:] - target_offsets[:-1]
    starts = pre_offsets[:-1] + (pre_lens - tgt_lens - 1)
    local = make_local_position_ids(target_offsets)
    return starts.repeat_interleave(tgt_lens) + local


def gather_target_logits(flat_logits, pre_offsets, target_offsets):
    """Slice the flat decoder logits down to target-aligned rows.

    Returns ``[total_target, C]`` aligned with the target ids order.
    """
    pos = make_target_logits_positions(pre_offsets, target_offsets)
    return flat_logits[pos]


def apply_logits_mask(flat_logits, pre_offsets, target_offsets, flat_mask):
    """Mask decoder logits with per-position allowed-class masks (in place).

    flat_logits: [total_pre, C]; flat_mask: [total_target, C] bool (True =
    allowed).  Rows of flat_logits that do not carry labels are untouched.
    The in-place update avoids materializing a second full [total_pre, C]
    copy of the logits.
    """
    pos = make_target_logits_positions(pre_offsets, target_offsets)
    masked = flat_logits[pos].masked_fill(~flat_mask, float("-inf"))
    flat_logits.index_put_((pos,), masked)
    return flat_logits


def apply_logits_mask_batched(logits, cache_pos, post_position_offset, mask):
    """Mask dense decoder logits with the encoder-provided mask.

    logits: [B, S, C]; mask: [B, T, C] bool (True = allowed).
    cache_pos: scalar tensor [1] holding the cache start position of this
        chunk (0 for a plain batched pass).
    post_position_offset: the full decoder length ``L_b``; mask row j applies
        to logits position ``L_b - T - 1 + j``.
    """
    B, S, C = logits.shape
    T = mask.shape[1]
    pos = cache_pos[0].item()
    torch._check(pos >= 0)
    start = post_position_offset - T - 1
    global_pos = torch.arange(S, device=logits.device, dtype=torch.long) + pos
    row = global_pos - start
    valid = (row >= 0) & (row < T)
    rows = row.clamp(0, T - 1)
    gathered = mask[:, rows, :]  # [B, S, C]
    full_mask = torch.where(valid.view(1, -1, 1), gathered, torch.ones_like(gathered))
    return logits.masked_fill(~full_mask, float("-inf"))
