import torch
import torch.nn as nn


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
    total_tokens = offsets[-1]
    global_ids = torch.arange(total_tokens, device=offsets.device)
    
    seq_starts = torch.repeat_interleave(
        offsets[:-1],
        offsets[1:] - offsets[:-1]
    )
    return global_ids - seq_starts