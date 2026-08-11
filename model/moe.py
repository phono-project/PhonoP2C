import torch
import torch.nn as nn

from model.ffn import SwiGLU
    

class MoE_EC_FFN(nn.Module):
    """Expert Choice FFN with one shared expert and N routed experts."""

    def __init__(self, dim: int, common_dim: int, num_experts: int, choice: int, expert_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.choice = choice
        self.dim = dim

        self.affinity = nn.Linear(dim, num_experts, bias=False)
        self.expert_common = SwiGLU(dim, common_dim, dim)

        # Stacked expert weights for batched matmul; shape: [num_experts, dim, expert_dim]
        self.experts_up_proj   = nn.Parameter(torch.empty(num_experts, dim, expert_dim))
        self.experts_gate_proj = nn.Parameter(torch.empty(num_experts, dim, expert_dim))
        # shape: [num_experts, expert_dim, dim]
        self.experts_down_proj = nn.Parameter(torch.empty(num_experts, expert_dim, dim))
        
        for w in [self.experts_up_proj, self.experts_gate_proj, self.experts_down_proj]:
            for i in range(self.num_experts):
                nn.init.xavier_uniform_(w[i])

    def _batched_swiglu(self, x):
        # x: [num_experts, expert_capacity, dim]
        up   = torch.bmm(x, self.experts_up_proj)              # [num_experts, expert_capacity, expert_dim]
        gate = torch.bmm(x, self.experts_gate_proj)            # [num_experts, expert_capacity, expert_dim]
        hidden = nn.functional.silu(gate) * up                 # [num_experts, expert_capacity, expert_dim]
        return torch.bmm(hidden, self.experts_down_proj)       # [num_experts, expert_capacity, dim]

    def forward(self, hidden, offsets=None):
        # offsets: [B+1] if NJT path, else None
        # hidden (NJT path):   [total_tokens, dim]
        # hidden (dense path): [B, S, dim]
        if offsets is not None:
            # flat_tokens: [total_tokens, dim]
            flat_tokens = hidden
        else:
            B, S, D = hidden.shape
            # flat_tokens: [B*S, dim]
            flat_tokens = hidden.reshape(B * S, D)

        total_tokens, _ = flat_tokens.shape

        # Expert capacity
        expert_capacity = max(1, (total_tokens * self.choice) // self.num_experts)

        # Gate scores: [total_tokens, num_experts]
        scores = self.affinity(flat_tokens)

        # Expert Choice: each expert selects top-C tokens
        # scores.T: [num_experts, total_tokens]
        # topk_indices: [num_experts, expert_capacity]
        topk_vals, topk_indices = torch.topk(scores.T, k=expert_capacity, dim=-1)

        # Gather selected tokens: [num_experts, expert_capacity, dim]
        selected = flat_tokens[topk_indices]

        # Expert computation — all experts in one batched matmul
        expert_out = self._batched_swiglu(selected)      # [num_experts, expert_capacity, dim]
        common_out = self.expert_common(flat_tokens)     # [total_tokens, dim]
        
        # Combine weights via softmax over top-C per expert
        combine_weights = torch.sigmoid(topk_vals).unsqueeze(0).to(common_out.dtype)
        # [num_experts, expert_capacity, dim]
        weighted_expert_out = (expert_out * combine_weights.unsqueeze(-1)).reshape(-1, self.dim)

        # out-of-place
        routed_out = torch.zeros_like(common_out).index_add(
            0,
            topk_indices.reshape(-1),
            weighted_expert_out,
        )
        output = common_out + routed_out

        if offsets is not None:
            # return flat values: [total_tokens, dim]
            return output
        else:
            # return: [B, S, dim]
            return output.reshape(B, S, self.dim)