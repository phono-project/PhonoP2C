"""
New-standard encoder-decoder PhonoP2C model.

    ┌──────────────────────┐        hidden + mask        ┌──────────────────────┐
    │  PhonoP2CPostModel   │─────────────────────────────│  PhonoP2CPreModel    │
    │  (bidirectional)     │   cross-attention (K/V)     │  (causal decoder)    │
    │  pinyin encoder      │                             │  Chinese -> logits   │
    └──────────────────────┘                             └──────────────────────┘

The pre model is the *decoder*: a causal LM over the sequence
``[BOS, prefix..., target...]`` split into two phases.

Two-pass training semantics (NJT path, used in train + validation):

    pass 1 (unconditional, no post hidden): input = ``full_prefix[:-1]`` (the
        context minus its last token), predicting the context one token ahead.
        Returns the unconditional logits together with the pre-RoPE self-attn
        K/V packed into a single nested jagged tensor with a layer axis
        ([B, j1, L, 2, H, D]).
    pass 2 (conditional, post hidden + past K/V + logits mask): input =
        ``full_prefix[-1:] + target[:-1]`` (the last context token + the first
        T-1 targets), running causal self-attn over ``[prefix K/V, suffix K/V]``
        then cross-attention over the pinyin encoder hidden states.  Logits are
        masked 1:1 with the pinyin-derived mask.

The pinyin (encoder) RoPE positions are aligned with the target positions:
target j sits at ``prefix_len + 1 + j``.

The non-NJT (batched) path is the incremental decode/prefill used for
inference/export: pass 1 prefill updates the self-attn KV cache in place;
pass 2 decode reads the cache and cross-attends.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import PreTrainedModel

from model.config import PreModelConfig, PostModelConfig
from model.attn import MHSALayer, MHCALayer
from model.ffn import SwiGLU
from model.moe import MoE_EC_FFN
from model.utils import (
    make_local_position_ids,
    make_suffix_global_positions,
    apply_logits_mask,
    apply_logits_mask_batched,
)


def _to_njt(flat: torch.Tensor, offsets: torch.Tensor, min_seqlen, max_seqlen):
    if min_seqlen is None or max_seqlen is None:
        seq_lens = offsets[1:] - offsets[:-1]
        min_seqlen = seq_lens.min()
        max_seqlen = seq_lens.max()
    return torch.nested.nested_tensor_from_jagged(
        flat, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen
    )


def _layer_checkpoint(fn, *args):
    """Checkpointed layer block (non-reentrant).

    ``determinism_check="none"`` is required: the default metadata extractor
    reads ``tensor.shape`` on saved tensors, which nested jagged tensors do
    not support.  The layers contain no dropout / RNG, so the determinism
    check is not needed.
    """
    return torch.utils.checkpoint.checkpoint(
        fn, *args, use_reentrant=False, determinism_check="none"
    )


# PhonoP2CPreModel (causal decoder)
class PhonoP2CPreModel(PreTrainedModel):
    config_class = PreModelConfig
    base_model_prefix = "phono_p2c_pre"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MHSALayer", "MHCALayer"]

    def __init__(self, config: PreModelConfig):
        super().__init__(config)
        dim = config.model_dim

        self.embed = nn.Embedding(config.vocab_size, dim)
        self.num_layers = config.mhsa_layers
        self.max_seqlen = config.max_seqlen
        self.post_max_seqlen = config.post_max_seqlen
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList([])
        for _ in range(config.mhsa_layers):
            self.layers.append(
                nn.ModuleDict({
                    "mhsa": MHSALayer(dim, config.attn_dim, config.mhsa_heads, config.rope_theta, config.max_seqlen),
                    "mhca": MHCALayer(dim, config.mhca_attn_dim, config.mhca_heads, config.rope_theta),
                    "ffn": SwiGLU(dim, config.ffn_common_dim, dim),
                    "norm1": nn.RMSNorm(dim),
                    "norm2": nn.RMSNorm(dim),
                    "norm3": nn.RMSNorm(dim),
                })
            )
        self.final_norm = nn.RMSNorm(dim)
        self.lm_proj = nn.Linear(dim, config.proj_size, bias=False)

        self.post_init()

    def _checkpoint_pass1_layer(self, layer, hidden, offsets, position_ids, min_seqlen, max_seqlen):
        """Unconditional decoder layer block (pass 1), checkpointable."""
        residual = hidden
        hidden = layer["norm1"](hidden)
        hidden, kv = layer["mhsa"](
            hidden, offsets=offsets, is_causal=True,
            position_ids=position_ids, return_kv=True,
            min_seqlen=min_seqlen, max_seqlen=max_seqlen,
        )
        hidden = hidden + residual

        residual = hidden
        hidden = layer["norm3"](hidden)
        hidden = layer["ffn"](hidden)
        hidden = hidden + residual
        return hidden, kv

    def _checkpoint_pass2_layer(self, layer, hidden, offsets, position_ids, past_kv,
                                layer_idx, prefix_lens, min_seqlen_full, max_seqlen_full,
                                post_hidden, q_pos_ids, kv_pos_ids, min_seqlen, max_seqlen):
        """Conditional (cross-attended) decoder layer block (pass 2), checkpointable."""
        residual = hidden
        hidden = layer["norm1"](hidden)
        hidden, _ = layer["mhsa"](
            hidden, offsets=offsets, is_causal=True,
            position_ids=position_ids, past_kv=past_kv, layer_idx=layer_idx,
            prefix_lens=prefix_lens,
            min_seqlen=min_seqlen, max_seqlen=max_seqlen,
            min_seqlen_full=min_seqlen_full, max_seqlen_full=max_seqlen_full,
        )
        hidden = hidden + residual

        residual = hidden
        hidden = layer["norm2"](hidden)
        hidden = layer["mhca"](
            hidden, enc_hidden=post_hidden,
            offsets=offsets, post_offsets=offsets,
            q_position_ids=q_pos_ids, kv_position_ids=kv_pos_ids,
            min_seqlen_q=min_seqlen, max_seqlen_q=max_seqlen,
            min_seqlen_kv=min_seqlen, max_seqlen_kv=max_seqlen,
        )
        hidden = hidden + residual

        residual = hidden
        hidden = layer["norm3"](hidden)
        hidden = layer["ffn"](hidden)
        hidden = hidden + residual
        return hidden

    def forward(self, input_ids, offsets=None, min_seqlen=None, max_seqlen=None,
                kv_cache_memory=None, current_seqlen=None,
                past_kv=None, prefix_lens=None,
                min_seqlen_full=None, max_seqlen_full=None,
                post_hidden=None, logits_mask=None,
                post_position_offset=None,
                cross_q_pos_start=None,
                use_custom_ops=False):
        """Decoder forward.

        NJT path (``offsets`` given):
            * pass 1 (``post_hidden=None``): ``offsets`` are the (physically
              padded) prefix offsets; returns ``(logits_njt, past_kv)`` where
              ``past_kv`` is a nested jagged tensor ``[B, j1, L, 2, H, D]``.
            * pass 2 (``post_hidden`` + ``past_kv`` + ``prefix_lens`` given):
              ``offsets`` are the suffix offsets; returns ``(logits_njt, None)``
              with the logits masked 1:1 by ``logits_mask``.

        Batched path (``offsets=None``): incremental decode/prefill with the
            self-attn KV cache (``kv_cache_memory`` + ``current_seqlen``, the
            latter a per-sample ``[B]`` start-position tensor).  The
            cross-attention uses *local* positions: ``cross_q_pos_start`` is the
            current suffix step (0..T-1) and ``post_position_offset`` the pinyin
            key offset (1, since pinyin is aligned with the target that is one
            position ahead of the query).

        ``use_custom_ops`` routes the in-place KV-cache update through the
        ``phono::update_mhsa_kv`` torch.library op (ExecuTorch export only);
        the default (False) uses standard PyTorch ops that run on any device.
        """
        using_cross = post_hidden is not None
        using_cache = kv_cache_memory is not None and current_seqlen is not None

        if using_cross and post_position_offset is None and offsets is None:
            raise ValueError("Batched pass 2 requires post_position_offset (pinyin key offset).")

        if offsets is not None:
            # --------------------------- NJT path ---------------------------
            flat_ids = input_ids
            hidden = self.embed(flat_ids)

            if using_cross:
                if past_kv is None:
                    raise ValueError("NJT pass 2 requires past_kv (first-pass self-attn KV).")
                if prefix_lens is None:
                    raise ValueError("NJT pass 2 requires prefix_lens (logical prefix lengths).")
                position_ids = make_suffix_global_positions(prefix_lens, offsets)
                q_pos_ids, kv_pos_ids = MHCALayer.compute_position_ids(offsets, offsets, prefix_lens)
            else:
                position_ids = make_local_position_ids(offsets)

            past_k_list = []
            past_v_list = []
            for layer_idx, layer in enumerate(self.layers):
                if using_cross:
                    if self.gradient_checkpointing:
                        hidden = _layer_checkpoint(
                            self._checkpoint_pass2_layer, layer, hidden,
                            offsets, position_ids, past_kv, layer_idx,
                            prefix_lens, min_seqlen_full, max_seqlen_full,
                            post_hidden, q_pos_ids, kv_pos_ids, min_seqlen, max_seqlen,
                        )
                    else:
                        hidden = self._checkpoint_pass2_layer(
                            layer, hidden, offsets, position_ids, past_kv, layer_idx,
                            prefix_lens, min_seqlen_full, max_seqlen_full,
                            post_hidden, q_pos_ids, kv_pos_ids, min_seqlen, max_seqlen,
                        )
                else:
                    if self.gradient_checkpointing:
                        hidden, kv = _layer_checkpoint(
                            self._checkpoint_pass1_layer, layer, hidden,
                            offsets, position_ids, min_seqlen, max_seqlen,
                        )
                    else:
                        hidden, kv = self._checkpoint_pass1_layer(
                            layer, hidden, offsets, position_ids, min_seqlen, max_seqlen
                        )
                    past_k_list.append(kv[0])
                    past_v_list.append(kv[1])

            hidden = self.final_norm(hidden)
            flat_logits = self.lm_proj(hidden)

            if using_cross and logits_mask is not None:
                flat_logits = apply_logits_mask(flat_logits, logits_mask)

            logits_njt = _to_njt(flat_logits, offsets, min_seqlen, max_seqlen)
            if using_cross:
                return logits_njt, None

            # Pack the per-layer K/V into a single nested jagged tensor with a
            # layer axis: [B, j1, L, 2, H, D].
            kv_all = torch.stack(
                [torch.stack([k, v], dim=1) for k, v in zip(past_k_list, past_v_list)],
                dim=1,
            )  # [total_tokens, L, 2, H, D]
            past_kv = torch.nested.nested_tensor_from_jagged(
                kv_all, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen,
            )
            return logits_njt, past_kv

        else:
            # -------------------------- Batched path ------------------------
            hidden = self.embed(input_ids)
            B, S = hidden.shape[:2]

            # Cross-attention uses local positions: the current suffix step
            # (query) and the pinyin key offset (1).
            cross_q_start = cross_q_pos_start if (using_cross and cross_q_pos_start is not None) else 0
            kv_pos_offset = post_position_offset if using_cross else None

            plain_past_kv = []
            for layer_idx, layer in enumerate(self.layers):
                residual = hidden
                hidden = layer["norm1"](hidden)
                if using_cache:
                    hidden, kv_cache_memory = layer["mhsa"](
                        hidden, is_causal=True,
                        kv_cache_full=kv_cache_memory,
                        cache_pos=current_seqlen,
                        layer_idx=layer_idx,
                        use_custom_ops=use_custom_ops,
                    )
                else:
                    hidden, kv = layer["mhsa"](
                        hidden, is_causal=True, return_kv=not using_cross,
                    )
                    if not using_cross:
                        plain_past_kv.append(kv)
                hidden = hidden + residual

                if using_cross:
                    residual = hidden
                    hidden = layer["norm2"](hidden)
                    hidden = layer["mhca"](
                        hidden, enc_hidden=post_hidden,
                        q_pos_start=cross_q_start,
                        kv_pos_offset=kv_pos_offset,
                    )
                    hidden = hidden + residual

                residual = hidden
                hidden = layer["norm3"](hidden)
                hidden = layer["ffn"](hidden)
                hidden = hidden + residual

            hidden = self.final_norm(hidden)
            logits = self.lm_proj(hidden)

            if using_cross and logits_mask is not None:
                logits = apply_logits_mask_batched(logits, logits_mask)

            if using_cache:
                return logits, kv_cache_memory
            if not using_cross:
                return logits, plain_past_kv
            return logits, None


# PhonoP2CPostModel (bidirectional pinyin encoder)
class PhonoP2CPostModel(PreTrainedModel):
    config_class = PostModelConfig
    base_model_prefix = "phono_p2c_post"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MHSALayer", "MoE_EC_FFN"]

    def __init__(self, config: PostModelConfig):
        super().__init__(config)
        dim = config.model_dim

        self.embed = nn.Embedding(config.vocab_size, dim)
        self.use_moe = config.use_moe_ffn
        self.max_seqlen = config.max_seqlen
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList([])
        for _ in range(config.mhsa_layers):
            layer_mods = {
                "mhsa": MHSALayer(dim, config.attn_dim, config.mhsa_heads, config.rope_theta, config.max_seqlen),
                "norm1": nn.RMSNorm(dim),
                "norm2": nn.RMSNorm(dim),
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

        # pinyin -> possible chinese chars possibility map.  Replaced by the
        # trainer with the tokenizer-built mask.
        self.register_buffer(
            "logits_mask", torch.ones((config.vocab_size, config.proj_size), dtype=torch.bool)
        )

        self.post_init()

    def _checkpoint_encoder_layer(self, layer, hidden, input_offsets, position_ids, min_seqlen, max_seqlen):
        """Bidirectional encoder layer block (NJT path), checkpointable."""
        residual = hidden
        hidden = layer["norm1"](hidden)
        hidden, _ = layer["mhsa"](
            hidden, offsets=input_offsets, is_causal=False,
            position_ids=position_ids,
            min_seqlen=min_seqlen, max_seqlen=max_seqlen,
        )
        hidden = hidden + residual

        residual = hidden
        hidden = layer["norm2"](hidden)
        if self.use_moe:
            hidden = layer["ffn"](hidden, offsets=input_offsets)
        else:
            hidden = layer["ffn"](hidden)
        hidden = hidden + residual
        return hidden

    def forward(self, input_ids, input_offsets=None, min_seqlen=None, max_seqlen=None):
        """Encode a pinyin sequence; return hidden states and the logits mask.

        NJT path: returns (hidden [total_tokens, model_dim], mask [total_tokens, proj_size]).
        Batched path: returns (hidden [B, S, model_dim], mask [B, S, proj_size]).
        """
        if input_offsets is not None:
            flat_ids = input_ids
            hidden = self.embed(flat_ids)
            position_ids = make_local_position_ids(input_offsets)

            for layer in self.layers:
                if self.gradient_checkpointing:
                    hidden = _layer_checkpoint(
                        self._checkpoint_encoder_layer, layer, hidden,
                        input_offsets, position_ids, min_seqlen, max_seqlen,
                    )
                else:
                    hidden = self._checkpoint_encoder_layer(
                        layer, hidden, input_offsets, position_ids, min_seqlen, max_seqlen
                    )

            hidden = self.final_norm(hidden)
            mask = self.logits_mask[flat_ids]
            return hidden, mask

        else:
            hidden = self.embed(input_ids)

            for layer in self.layers:
                residual = hidden
                hidden = layer["norm1"](hidden)
                hidden, _ = layer["mhsa"](hidden, is_causal=False)
                hidden = hidden + residual

                residual = hidden
                hidden = layer["norm2"](hidden)
                hidden = layer["ffn"](hidden)
                hidden = hidden + residual

            hidden = self.final_norm(hidden)
            mask = self.logits_mask[input_ids]
            return hidden, mask
