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


class TrainOutput(NamedTuple):
    """Result of one wrapper forward step.

    ``conditional_logits`` are the flat ``[total_suffix, chinese_vocab]``
    conditional logits, kept for metrics.
    """

    loss: torch.Tensor
    unconditional_loss: torch.Tensor
    conditional_loss: torch.Tensor
    conditional_logits: torch.Tensor


class PhonoP2CTrainWrapper(nn.Module):
    """Wrap the two-pass forward pass and compute the joint loss internally.

    Pass 1 (unconditional): prefix input -> unconditional logits + cached
    self-attn K/V.
    Pass 2 (conditional): pinyin encoder -> hidden + mask, then the suffix
    input -> cross-attended, masked conditional logits.

    ``loss_fn`` maps ``(logits, targets)`` to a scalar; it is applied to both
    passes and the results are summed.  torch.compile is applied to this
    wrapper; the pre and post models stay individually saveable.
    """

    def __init__(self, pre_model, post_model, loss_fn):
        super().__init__()
        self.pre_model = pre_model
        self.post_model = post_model
        self.loss_fn = loss_fn

    def _loss(self, logits, targets):
        # Guard the all-ignored case (e.g. a batch of empty prefixes) that
        # plain CrossEntropyLoss would turn into a NaN mean.
        if targets.numel() == 0 or bool((targets == -100).all()):
            return logits.sum() * 0.0
        return self.loss_fn(logits, targets)

    def forward(self, flat_prefix_ids, prefix_offsets, flat_suffix_ids, suffix_offsets,
                flat_postfix_ids, flat_uncond_target, flat_target_ids, prefix_lens,
                min_sl_prefix, max_sl_prefix, min_sl_suffix, max_sl_suffix,
                min_sl_full, max_sl_full):
        # Pass 1: unconditional decoder forward over the (padded) prefix.
        logits_uncond, past_kv = self.pre_model(
            flat_prefix_ids, offsets=prefix_offsets,
            min_seqlen=min_sl_prefix, max_seqlen=max_sl_prefix,
        )

        # Encoder: pinyin -> hidden states + logits mask.
        post_hidden, post_mask = self.post_model(
            flat_postfix_ids, input_offsets=suffix_offsets,
            min_seqlen=min_sl_suffix, max_seqlen=max_sl_suffix,
        )

        # Pass 2: conditional decoder forward (cross-attention + masked logits).
        logits_cond, _ = self.pre_model(
            flat_suffix_ids, offsets=suffix_offsets,
            min_seqlen=min_sl_suffix, max_seqlen=max_sl_suffix,
            past_kv=past_kv, prefix_lens=prefix_lens,
            min_seqlen_full=min_sl_full, max_seqlen_full=max_sl_full,
            post_hidden=post_hidden, logits_mask=post_mask,
        )

        flat_uncond = logits_uncond.values()
        flat_cond = logits_cond.values()

        loss_uncond = self._loss(flat_uncond, flat_uncond_target)
        loss_cond = self._loss(flat_cond, flat_target_ids)
        loss = loss_uncond + loss_cond

        return TrainOutput(loss, loss_uncond, loss_cond, flat_cond)
