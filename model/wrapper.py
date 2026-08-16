"""Two-pass training wrapper with internal loss computation.

The wrapper packs the whole (NJT) forward pass — unconditional decoder pass,
pinyin encoder pass, and conditional cross-attended decoder pass — and
computes the two losses (unconditional / conditional) plus their total
*internally*.  Keeping the loss inside the compiled region lets ``torch.compile``
fuse the target-logit gather and loss reduction with the decoder output, so
the full-sequence logits never need to be materialized as a dense tensor.

Because it bundles a loss function (and is therefore no longer a pure model),
it lives in its own module rather than in ``model.model``.
"""

from typing import NamedTuple

import torch
import torch.nn as nn

from model.utils import gather_target_logits


class TrainOutput(NamedTuple):
    """Result of one wrapper forward step.

    ``unconditional_logits`` / ``conditional_logits`` are the target-aligned
    (flat ``[total_target, chinese_vocab]``) logits, kept for metrics.
    """

    loss: torch.Tensor
    unconditional_loss: torch.Tensor
    conditional_loss: torch.Tensor
    unconditional_logits: torch.Tensor
    conditional_logits: torch.Tensor


class PhonoP2CTrainWrapper(nn.Module):
    """Wrap the two-pass forward pass and compute the joint loss internally.

    Pass 1: unconditional decoder logits + cached self-attn K/V.
    Pass 2: pinyin encoder hidden states + mask, then conditional (cross-
    attended, masked) decoder logits.  ``loss_fn`` maps ``(logits, targets)``
    to a scalar; it is applied to both passes and the results are summed.
    torch.compile is applied to this wrapper; the pre and post models stay
    individually saveable.
    """

    def __init__(self, pre_model, post_model, loss_fn):
        super().__init__()
        self.pre_model = pre_model
        self.post_model = post_model
        self.loss_fn = loss_fn

    def forward(self, flat_pre_ids, pre_offsets, flat_postfix_ids, postfix_offsets,
                flat_target_ids, min_sl_pre=None, max_sl_pre=None,
                min_sl_post=None, max_sl_post=None):
        # Pass 1: unconditional decoder forward.
        logits_uncond, past_kv = self.pre_model(
            flat_pre_ids, offsets=pre_offsets,
            min_seqlen=min_sl_pre, max_seqlen=max_sl_pre,
        )

        # Encoder: pinyin -> hidden states + logits mask.
        post_hidden, post_mask = self.post_model(
            flat_postfix_ids, input_offsets=postfix_offsets,
            min_seqlen=min_sl_post, max_seqlen=max_sl_post,
        )

        # Pass 2: conditional decoder forward (cross-attention + masked logits).
        logits_cond, _ = self.pre_model(
            flat_pre_ids, offsets=pre_offsets,
            min_seqlen=min_sl_pre, max_seqlen=max_sl_pre,
            past_kv=past_kv,
            post_hidden=post_hidden, post_offsets=postfix_offsets,
            min_seqlen_post=min_sl_post, max_seqlen_post=max_sl_post,
            logits_mask=post_mask,
        )

        flat_uncond = gather_target_logits(logits_uncond.values(), pre_offsets, postfix_offsets)
        flat_cond = gather_target_logits(logits_cond.values(), pre_offsets, postfix_offsets)

        loss_uncond = self.loss_fn(flat_uncond, flat_target_ids)
        loss_cond = self.loss_fn(flat_cond, flat_target_ids)
        loss = loss_uncond + loss_cond

        return TrainOutput(loss, loss_uncond, loss_cond, flat_uncond, flat_cond)
