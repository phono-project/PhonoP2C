"""Training loss functions."""

from loss.loss import FocalLoss, LabelSmoothingCrossEntropy, get_loss_fn

__all__ = ["FocalLoss", "LabelSmoothingCrossEntropy", "get_loss_fn"]
