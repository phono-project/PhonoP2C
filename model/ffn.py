import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.up_proj   = nn.Linear(in_dim, hidden_dim,  bias=False)
        self.gate_proj = nn.Linear(in_dim, hidden_dim,  bias=False)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))