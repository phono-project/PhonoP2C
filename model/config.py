from dataclasses import dataclass, field
from typing import Optional

from transformers import PretrainedConfig


class PreModelConfig(PretrainedConfig):
    """Configuration for PhonoP2CPreModel."""
    model_type = "phono_p2c_pre"

    def __init__(
        self,
        model_dim: int = 512,
        attn_dim: int = 512,
        rope_theta: float = 1000.0,
        max_seqlen: int = 64,
        mhsa_layers: int = 8,
        mhsa_heads: int = 8,
        ffn_common_dim: int = 2048,
        vocab_size: int = 10000,
        mhca_attn_dim: int = 512,
        cross_attn_heads: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.attn_dim = attn_dim
        self.rope_theta = rope_theta
        self.max_seqlen = max_seqlen
        self.mhsa_layers = mhsa_layers
        self.mhsa_heads = mhsa_heads
        self.ffn_common_dim = ffn_common_dim
        self.vocab_size = vocab_size
        self.mhca_attn_dim = mhca_attn_dim
        self.cross_attn_heads = cross_attn_heads


class PostModelConfig(PretrainedConfig):
    """Configuration for PhonoP2CPostModel."""
    model_type = "phono_p2c_post"

    def __init__(
        self,
        model_dim: int = 512,
        attn_dim: int = 512,
        rope_theta: float = 1000.0,
        pre_max_seqlen: float = 64,
        max_seqlen: int = 32,
        mhsa_layers: int = 8,
        mhsa_heads: int = 8,
        use_moe_ffn: bool = False,
        ffn_common_dim: int = 2048,
        ffn_num_experts: int = 0,
        ffn_choice: int = 0,
        ffn_expert_dim: int = 0,
        vocab_size: int = 500,
        proj_size: int = 10000,
        mhca_heads: int = 8,
        mhca_attn_dim: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.attn_dim = attn_dim
        self.rope_theta = rope_theta
        self.pre_max_seqlen = pre_max_seqlen
        self.max_seqlen = max_seqlen
        self.mhsa_layers = mhsa_layers
        self.mhsa_heads = mhsa_heads
        self.use_moe_ffn = use_moe_ffn
        self.ffn_common_dim = ffn_common_dim
        self.ffn_num_experts = ffn_num_experts
        self.ffn_choice = ffn_choice
        self.ffn_expert_dim = ffn_expert_dim
        self.vocab_size = vocab_size
        self.proj_size = proj_size
        self.mhca_heads = mhca_heads
        self.mhca_attn_dim = mhca_attn_dim


# Helper function to build configs from the nested YAML dict
def build_configs_from_dict(d: dict, vocab_sizes: dict) -> tuple[PreModelConfig, PostModelConfig]:
    common = d.get("common", {})
    pre_d  = d.get("pre_model", {})
    post_d = d.get("post_model", {})

    pre_cfg = PreModelConfig(
        model_dim=common.get("model_dim", 512),
        attn_dim=pre_d.get("attn_dim", common.get("model_dim", 512)),
        rope_theta=common.get("rope_theta", 1000.0),
        max_seqlen=pre_d.get("max_seqlen", 64),
        mhsa_layers=pre_d.get("mhsa_layers", 8),
        mhsa_heads=pre_d.get("mhsa_heads", 8),
        ffn_common_dim=pre_d.get("ffn_common_dim", 2048),
        vocab_size=vocab_sizes["context"],
        mhca_attn_dim=post_d.get("mhca_attn_dim", common.get("model_dim", 512)),
        cross_attn_heads=post_d.get("mhca_heads", post_d.get("mhsa_heads", 8)),
    )
    post_cfg = PostModelConfig(
        model_dim=common.get("model_dim", 512),
        attn_dim=post_d.get("attn_dim", common.get("model_dim", 512)),
        rope_theta=common.get("rope_theta", 1000.0),
        pre_max_seqlen=pre_d.get("max_seqlen", 64),
        max_seqlen=post_d.get("max_seqlen", 32),
        mhsa_layers=post_d.get("mhsa_layers", 8),
        mhsa_heads=post_d.get("mhsa_heads", 8),
        use_moe_ffn=post_d.get("use_moe_ffn", False),
        ffn_common_dim=post_d.get("ffn_common_dim", 2048),
        ffn_num_experts=post_d.get("ffn_num_experts", 0),
        ffn_choice=post_d.get("ffn_choice", 0),
        ffn_expert_dim=post_d.get("ffn_expert_dim", 0),
        vocab_size=vocab_sizes["pinyin"],
        proj_size=vocab_sizes["chinese"],
        mhca_heads=post_d.get("mhca_heads", post_d.get("mhsa_heads", 8)),
        mhca_attn_dim=post_d.get("mhca_attn_dim", common.get("model_dim", 512)),
    )
    return pre_cfg, post_cfg
