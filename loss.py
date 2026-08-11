"""
Loss functions for PhonoP2C training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the rare class(es).  Scalar or tensor of
               shape [num_classes].  Default 0.25 (per the original paper).
        gamma: Focusing parameter.  Larger values reduce the relative loss for
               well-classified examples.  Default 2.0.
        ignore_index: Target value to ignore.  Default -100.
        reduction: 'mean' or 'sum'.  Default 'mean'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            logits:  [N, C] unnormalized class scores.
            targets: [N]   ground-truth class indices.
        Returns:
            Scalar loss tensor (mean or sum depending on *reduction*).
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )

        # mask out ignored positions
        mask = (targets != self.ignore_index).float()

        # p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)

        # (1 - p_t)^gamma -------------------------
        focal_weight = (1.0 - p_t) ** self.gamma

        alpha_t = self.alpha

        focal_loss = alpha_t * focal_weight * ce_loss

        # mask and reduce
        focal_loss = focal_loss * mask

        if self.reduction == "mean":
            return focal_loss.sum() / mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing that is aware of logits_mask.

    A naive label-smoothing implementation spreads the epsilon probability
    mass uniformly across *all* classes.  In this codebase, however, logits
    arrive here with impossible classes already set to -inf by the model's
    logits_mask (see PhonoP2CPostModel: False entries in the pinyin->Chinese
    possibility map get their logits set to -inf before the loss ever sees
    them).  Smoothing over every class would therefore assign non-zero
    target probability to classes with -log_prob == +inf, sending the loss
    to +inf.

    Instead, for every row we:
        Determine the set of "valid" classes by checking which logits are
        finite (i.e. which classes logits_mask actually allows).
        Put (1 - epsilon) probability mass on the target class.
        Spread the remaining epsilon mass uniformly over the *other* valid
        classes only -- never onto masked-out (-inf) classes.
        If a row has only one valid class (the target itself, i.e. the
        answer is unambiguous given logits_mask), there is nothing else to
        smooth into, so epsilon is treated as 0 for that row and it falls
        back to plain one-hot cross-entropy.  This avoids both a
        divide-by-zero and a degenerate target distribution.

    Args:
        epsilon:      Label smoothing factor in [0, 1).  Default 0.1.
        ignore_index: Target value to ignore. Default -100.
        reduction:    'mean' or 'sum'.  Default 'mean'.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            logits:  [N, C] unnormalized class scores.  Entries for classes
                     forbidden by logits_mask are expected to be -inf.
            targets: [N]   ground-truth class indices.
        Returns:
            Scalar loss tensor (mean or sum depending on *reduction*).
        """
        # mask out ignored positions
        valid_row = (targets != self.ignore_index).float()
        # clamp
        safe_targets = targets.clamp(min=0)

        log_probs = F.log_softmax(logits, dim=-1)

        # which classes are reachable for each row, per logits_mask
        class_mask = torch.isfinite(logits)
        num_valid = class_mask.sum(dim=-1)

        # zero out the -inf entries before summing so we never compute
        # 0 * (-inf) = nan when weighting by a zero smoothing probability
        log_probs_safe = torch.where(class_mask, log_probs, torch.zeros_like(log_probs))
        sum_log_probs = log_probs_safe.sum(dim=-1)

        target_log_prob = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

        # sum of log-probs over valid classes, excluding the target itself
        other_log_probs_sum = sum_log_probs - target_log_prob

        # rows whose target is the only valid class get epsilon
        has_other_classes = num_valid > 1
        row_epsilon = torch.where(
            has_other_classes,
            torch.full_like(target_log_prob, self.epsilon),
            torch.zeros_like(target_log_prob),
        )
        # avoid divide-by-zero for singleton rows
        smooth_denom = (num_valid - 1).clamp(min=1).float()

        smoothed_loss = (
            -(1.0 - row_epsilon) * target_log_prob
            - (row_epsilon / smooth_denom) * other_log_probs_sum
        )

        # mask and reduce
        smoothed_loss = smoothed_loss * valid_row

        if self.reduction == "mean":
            return smoothed_loss.sum() / valid_row.sum().clamp(min=1)
        elif self.reduction == "sum":
            return smoothed_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


def get_loss_fn(
    loss_type: str = "ce",
    ignore_index: int = -100,
    focal_loss_alpha: float = 0.25,
    focal_loss_gamma: float = 2.0,
    label_smoothing_epsilon: float = None,
) -> nn.Module:
    """
    Factory that returns a loss module based on the given type.

    Args:
        loss_type:          "ce" for cross-entropy, "focal" for focal loss.
        ignore_index:       Target value to ignore.
        focal_loss_alpha:   Alpha parameter for focal loss (used only when
                            loss_type == "focal").
        focal_loss_gamma:   Gamma parameter for focal loss (used only when
                            loss_type == "focal").
        label_smoothing_epsilon: If set to a truthy value and loss_type ==
                            "ce", enables mask-aware label smoothing with
                            this epsilon (see LabelSmoothingCrossEntropy).
                            Ignored for other loss types.

    Returns:
        A ``torch.nn.Module`` compatible with ``loss_fn(logits, targets)``.
    """
    loss_type = loss_type.lower().strip()

    if loss_type == "ce":
        if label_smoothing_epsilon:
            return LabelSmoothingCrossEntropy(
                epsilon=label_smoothing_epsilon,
                ignore_index=ignore_index,
                reduction="mean",
            )
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    elif loss_type == "focal":
        return FocalLoss(
            alpha=focal_loss_alpha,
            gamma=focal_loss_gamma,
            ignore_index=ignore_index,
            reduction="mean",
        )

    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'.  Expected one of: 'ce', 'focal'."
        )