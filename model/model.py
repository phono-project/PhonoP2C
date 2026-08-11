"""
PostfixLM hybrid model: PhonoP2CPreModel + PhonoP2CPostModel.

  ┌──────────────────────┐    pre_K / pre_V    ┌──────────────────────┐
  │  PhonoP2CPreModel    │─────────────────────│  PhonoP2CPostModel   │
  │  (causal)            │   cross-attention   │  (bidirectional)     │
  │  Chinese context     │     (shared KV)     │  Pinyin -> Chinese   │
  │  encoder + KV proj   │                     │  decoder             │
  └──────────────────────┘                     └──────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from model.config import PreModelConfig, PostModelConfig
from model.attn import MHSALayer, MHCALayer
from model.ffn import SwiGLU
from model.moe import MoE_EC_FFN
from model.utils import make_local_position_ids
from model.custom_ops import kv_cache_write, update_cross_kv, update_mhsa_kv

# PhonoP2CPreModel
class PhonoP2CPreModel(PreTrainedModel):
    config_class = PreModelConfig
    base_model_prefix = "phono_p2c_pre"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MHSALayer"]

    def __init__(self, config: PreModelConfig):
        super().__init__(config)
        dim = config.model_dim

        self.embed = nn.Embedding(config.vocab_size, dim)
        self.num_layers = config.mhsa_layers
        self.max_seqlen = config.max_seqlen

        self.layers = nn.ModuleList([])
        for _ in range(config.mhsa_layers):
            self.layers.append(
                nn.ModuleDict({
                    "mhsa": MHSALayer(dim, config.attn_dim, config.mhsa_heads, config.rope_theta, config.max_seqlen),
                    "ffn": SwiGLU(dim, config.ffn_common_dim, dim),
                    "norm1": nn.RMSNorm(dim),
                    "norm2": nn.RMSNorm(dim),
                })
            )
        self.final_norm = nn.RMSNorm(dim)

        self.cross_attn_heads = config.cross_attn_heads
        self.mhca_attn_dim = config.mhca_attn_dim
        self.kv_proj = nn.Linear(dim, config.mhca_attn_dim * 2, bias=False)

        self.post_init()

    def forward(self, input_ids, offsets=None, kv_cache_memory=None, current_seqlen=None,
                min_seqlen=None, max_seqlen=None,
                pre_cross_kv_cache=None, pre_cross_cache_pos=None):
        using_cache = kv_cache_memory is not None and current_seqlen is not None

        if offsets is not None:
            flat_ids = input_ids
            hidden = self.embed(flat_ids)
            position_ids = make_local_position_ids(offsets)
        else:
            hidden = self.embed(input_ids)
            position_ids = None

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden
            hidden = layer["norm1"](hidden)

            if offsets is not None:
                hidden = layer["mhsa"](
                    hidden, offsets=offsets, is_causal=True, position_ids=position_ids,
                    min_seqlen=min_seqlen, max_seqlen=max_seqlen,
                )
            elif using_cache:
                hidden = layer["mhsa"](
                    hidden, is_causal=True,
                    kv_cache_full=kv_cache_memory,
                    cache_pos=current_seqlen,
                    layer_idx=layer_idx,
                )
            else:
                hidden = layer["mhsa"](hidden, is_causal=True)

            hidden = hidden + residual

            residual = hidden
            hidden = layer["norm2"](hidden)
            hidden = layer["ffn"](hidden)
            hidden = hidden + residual

        hidden = self.final_norm(hidden)

        pre_K, pre_V = self.kv_proj(hidden).chunk(2, dim=-1)
        head_dim = self.mhca_attn_dim // self.cross_attn_heads

        if offsets is not None:
            total_tokens = hidden.shape[0]
            pre_K = pre_K.view(total_tokens, self.cross_attn_heads, head_dim)
            pre_V = pre_V.view(total_tokens, self.cross_attn_heads, head_dim)
        else:
            S = hidden.shape[1]
            pre_K = pre_K.view(hidden.shape[0], S, self.cross_attn_heads, head_dim)
            pre_V = pre_V.view(hidden.shape[0], S, self.cross_attn_heads, head_dim)

            if pre_cross_kv_cache is not None and pre_cross_cache_pos is not None:
                pos = pre_cross_cache_pos[0].item()
                torch._check(pos >= 0)
                torch._check(pos + S <= self.max_seqlen)
                updated = update_cross_kv(pre_cross_kv_cache, pre_K, pre_V, pos)
                return updated

        return hidden, pre_K, pre_V


# PhonoP2CPostModel
class PhonoP2CPostModel(PreTrainedModel):
    config_class = PostModelConfig
    base_model_prefix = "phono_p2c_post"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MHSALayer", "MHCALayer", "MoE_EC_FFN"]

    def __init__(self, config: PostModelConfig):
        super().__init__(config)
        dim = config.model_dim

        self.embed = nn.Embedding(config.vocab_size, dim)
        self.use_moe = config.use_moe_ffn
        self.pre_max_seqlen = config.pre_max_seqlen
        self.max_seqlen = config.max_seqlen

        self.layers = nn.ModuleList([])
        for _ in range(config.mhsa_layers):
            layer_mods = {
                "mhsa": MHSALayer(dim, config.attn_dim, config.mhsa_heads, config.rope_theta, config.max_seqlen),
                "mhca": MHCALayer(dim, config.mhca_attn_dim, config.mhca_heads, config.rope_theta, config.pre_max_seqlen),
                "norm1": nn.RMSNorm(dim),
                "norm2": nn.RMSNorm(dim),
                "norm3": nn.RMSNorm(dim),
            }
            if config.use_moe_ffn:
                layer_mods["ffn"] = MoE_EC_FFN(
                    dim, config.ffn_common_dim,
                    config.ffn_num_experts, config.ffn_choice, config.ffn_expert_dim,
                )
            else:
                layer_mods["ffn"] = SwiGLU(dim, config.ffn_common_dim, dim)
            self.layers.append(nn.ModuleDict(layer_mods))

        self.final_norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, config.proj_size, bias=False)
        
        self.register_buffer("logits_mask", torch.ones((config.vocab_size, config.proj_size), dtype=torch.bool))

        self.post_init()

    def forward(self, input_ids, input_offsets=None, pre_K=None, pre_V=None,
                pre_offsets=None, pre_cross_kv_cache=None, current_seqlen=None,
                min_seqlen=None, max_seqlen=None,
                min_seqlen_pre=None, max_seqlen_pre=None,
                return_last_hidden=False):
        using_cache = pre_cross_kv_cache is not None and current_seqlen is not None

        if input_offsets is not None:
            flat_ids = input_ids
            hidden = self.embed(flat_ids)

            self_position_ids = make_local_position_ids(input_offsets)
            cross_q_position_ids, cross_kv_position_ids = MHCALayer.compute_position_ids(
                input_offsets, pre_offsets
            )
        else:
            hidden = self.embed(input_ids)
            self_position_ids = None
            cross_q_position_ids = None
            cross_kv_position_ids = None

        for layer_idx, layer in enumerate(self.layers):
            # Bidirectional self-attention
            residual = hidden
            hidden = layer["norm1"](hidden)
            hidden = layer["mhsa"](
                hidden, offsets=input_offsets, is_causal=False, position_ids=self_position_ids,
                min_seqlen=min_seqlen, max_seqlen=max_seqlen,
            )
            hidden = hidden + residual

            # Cross-attention shared KV across all layers
            residual = hidden
            hidden = layer["norm2"](hidden)
            hidden = layer["mhca"](
                hidden, pre_K=pre_K, pre_V=pre_V,
                offsets=input_offsets,
                pre_offsets=pre_offsets,
                pre_kv_cache=(
                    (pre_cross_kv_cache[0], pre_cross_kv_cache[1])
                    if using_cache else None
                ),
                cache_pos=current_seqlen if using_cache else None,
                q_position_ids=cross_q_position_ids,
                kv_position_ids=cross_kv_position_ids,
                min_seqlen_q=min_seqlen, max_seqlen_q=max_seqlen,
                min_seqlen_kv=min_seqlen_pre, max_seqlen_kv=max_seqlen_pre,
            )
            hidden = hidden + residual

            # FFN
            residual = hidden
            hidden = layer["norm3"](hidden)
            if self.use_moe:
                hidden = layer["ffn"](hidden, offsets=input_offsets)
            else:
                hidden = layer["ffn"](hidden)
            hidden = hidden + residual

        hidden = self.final_norm(hidden)

        if input_offsets is not None:
            flat_logits = self.lm_head(hidden)

            per_pos_mask = self.logits_mask[flat_ids]
            flat_logits = flat_logits.masked_fill(~per_pos_mask, float('-inf'))

            if min_seqlen is None or max_seqlen is None:
                seq_lens = input_offsets[1:] - input_offsets[:-1]
                min_seqlen = seq_lens.min()
                max_seqlen = seq_lens.max()
            logits_njt = torch.nested.nested_tensor_from_jagged(
                flat_logits, input_offsets,
                min_seqlen=min_seqlen, max_seqlen=max_seqlen,
            )
            if return_last_hidden:
                return logits_njt, hidden
            return logits_njt
        else:
            logits = self.lm_head(hidden)

            per_pos_mask = self.logits_mask[input_ids]
            logits = logits.masked_fill(~per_pos_mask, float('-inf'))

            if return_last_hidden:
                return logits, hidden
            return logits