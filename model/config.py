"""Model configuration for the new-standard encoder-decoder PhonoP2C.

Architecture overview
---------------------
    PhonoP2CPreModel  (causal decoder)
        Reads the Chinese sequence (context prefix + target, teacher-forced
        during training), performs causal self-attention and — in the second
        pass — cross-attention over the pinyin encoder's hidden states.
        Produces logits over the *chinese* vocabulary via ``lm_proj``.

    PhonoP2CPostModel (bidirectional encoder)
        Encodes the pinyin sequence and yields its hidden states plus a
        per-position logits mask (pinyin -> possible Chinese chars).  It does
        not output logits.

"""

from transformers import PretrainedConfig


class PreModelConfig(PretrainedConfig):
    """Configuration for PhonoP2CPreModel (causal decoder)."""
    model_type = "phono_p2c_pre"

    def __init__(
        self,
        model_dim: int = 768,
        attn_dim: int = 256,
        rope_theta: float = 1000.0,
        max_seqlen: int = 128,
        mhsa_layers: int = 8,
        mhsa_heads: int = 4,
        ffn_common_dim: int = 4096,
        vocab_size: int = 10000,
        proj_size: int = 10000,
        mhca_heads: int = 4,
        mhca_attn_dim: int = 256,
        post_max_seqlen: int = 32,
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
        # vocab_size: decoder input embedding (context vocab).
        self.vocab_size = vocab_size
        # proj_size: decoder output logits (chinese vocab).
        self.proj_size = proj_size
        # Cross-attention over the post (encoder) hidden states.  These values
        # live under the post_model section of the YAML config.
        self.mhca_heads = mhca_heads
        self.mhca_attn_dim = mhca_attn_dim
        # Encoder sequence length bound (post model max_seqlen).
        self.post_max_seqlen = post_max_seqlen


class PostModelConfig(PretrainedConfig):
    """Configuration for PhonoP2CPostModel (bidirectional pinyin encoder)."""
    model_type = "phono_p2c_post"

    def __init__(
        self,
        model_dim: int = 768,
        attn_dim: int = 256,
        rope_theta: float = 1000.0,
        max_seqlen: int = 32,
        mhsa_layers: int = 12,
        mhsa_heads: int = 4,
        use_moe_ffn: bool = False,
        ffn_common_dim: int = 4096,
        ffn_num_experts: int = 0,
        ffn_choice: int = 0,
        ffn_expert_dim: int = 0,
        vocab_size: int = 500,
        proj_size: int = 10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.attn_dim = attn_dim
        self.rope_theta = rope_theta
        self.max_seqlen = max_seqlen
        self.mhsa_layers = mhsa_layers
        self.mhsa_heads = mhsa_heads
        self.use_moe_ffn = use_moe_ffn
        self.ffn_common_dim = ffn_common_dim
        self.ffn_num_experts = ffn_num_experts
        self.ffn_choice = ffn_choice
        self.ffn_expert_dim = ffn_expert_dim
        # vocab_size: encoder input embedding (pinyin vocab).
        self.vocab_size = vocab_size
        # proj_size: chinese vocab size; shapes the logits mask buffer
        # (pinyin_vocab, chinese_vocab) returned alongside the hidden states.
        self.proj_size = proj_size


def build_configs_from_dict(d: dict, vocab_sizes: dict) -> tuple[PreModelConfig, PostModelConfig]:
    common = d.get("common", {})
    pre_d = d.get("pre_model", {})
    post_d = d.get("post_model", {})

    pre_cfg = PreModelConfig(
        model_dim=common.get("model_dim", 768),
        attn_dim=pre_d.get("attn_dim", common.get("model_dim", 768)),
        rope_theta=common.get("rope_theta", 1000.0),
        max_seqlen=pre_d.get("max_seqlen", 128),
        mhsa_layers=pre_d.get("mhsa_layers", 8),
        mhsa_heads=pre_d.get("mhsa_heads", 4),
        ffn_common_dim=pre_d.get("ffn_common_dim", 4096),
        vocab_size=vocab_sizes["context"],
        proj_size=vocab_sizes["chinese"],
        mhca_heads=post_d.get("mhca_heads", post_d.get("mhsa_heads", 4)),
        mhca_attn_dim=post_d.get("mhca_attn_dim", common.get("model_dim", 768)),
        post_max_seqlen=post_d.get("max_seqlen", 32),
    )
    post_cfg = PostModelConfig(
        model_dim=common.get("model_dim", 768),
        attn_dim=post_d.get("attn_dim", common.get("model_dim", 768)),
        rope_theta=common.get("rope_theta", 1000.0),
        max_seqlen=post_d.get("max_seqlen", 32),
        mhsa_layers=post_d.get("mhsa_layers", 12),
        mhsa_heads=post_d.get("mhsa_heads", 4),
        use_moe_ffn=post_d.get("use_moe_ffn", False),
        ffn_common_dim=post_d.get("ffn_common_dim", 4096),
        ffn_num_experts=post_d.get("ffn_num_experts", 0),
        ffn_choice=post_d.get("ffn_choice", 0),
        ffn_expert_dim=post_d.get("ffn_expert_dim", 0),
        vocab_size=vocab_sizes["pinyin"],
        proj_size=vocab_sizes["chinese"],
    )
    return pre_cfg, post_cfg
