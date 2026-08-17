"""
Metric accumulators.

Metrics:
- ACC        : per-token top-1 accuracy
- Top3/5-ACC : per-token top-k accuracy
- S-ACC      : sentence-level accuracy
- ECE        : Top-K Adaptive ECE (netcal.metrics.confidence.ACE, quantile binning)

NOTE ON MEMORY:
Adaptive (quantile) binning needs the full distribution of confidences to
determine bin edges, so it cannot be computed as a running scalar the way
equal-width ECE could. ACC / Top-k-ACC / S-ACC remain O(1) running scalars,
but the ECE component now accumulates two small float arrays
(per-token confidence, per-token top-k-correct flag) across the whole
validation set.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from netcal.metrics.confidence import ACE
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This module requires the `netcal` package. Install it with "
        "`pip install netcal`."
    ) from e


class MetricsAccumulator:
    """
    Usage:

        acc = MetricsAccumulator(ece_bins=15, ece_top_k=1)
        for batch in val_loader:
            logits, targets, offsets = model(batch)
            acc.update(logits, targets, offsets)
        results = acc.compute()   # dict with acc, top3_acc, top5_acc, s_acc, ece
        acc.reset()
    """

    def __init__(self, ece_bins: int = 15, ece_top_k: int = 1):
        """
        Args:
            ece_bins:  number of adaptive (equal-frequency) bins for ECE.
            ece_top_k: k used to define the ECE "confidence" and "correct"
                       signals. confidence = sum of top-k softmax probs,
                       correct = target found among the top-k predictions.
                       ece_top_k=1 reproduces the classic Argmax ECE
                       definition, just with adaptive instead of
                       equal-width binning.
        """
        self.ece_bins = ece_bins
        self.ece_top_k = ece_top_k
        self.reset()

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_offsets: torch.Tensor,
    ) -> None:
        """Ingest one batch.

        Args:
            logits:         [N, C]  raw logits for every target position (any device).
            targets:        [N]     ground-truth class indices.
            target_offsets: [B+1]   cumulative sentence boundaries.
        """
        if targets.numel() == 0:
            return

        probs = logits.softmax(dim=-1)                   # [N, C]
        confidences, preds = probs.max(dim=-1)           # [N], [N]
        correct = (preds == targets)                     # [N] bool

        # ACC
        n = int(targets.numel())
        self._total_tokens += n
        self._correct_top1 += int(correct.sum().item())

        # Top3 / Top5-ACC
        _, top3_idx = probs.topk(3, dim=-1)              # [N, 3]
        top3_correct = (top3_idx == targets.unsqueeze(-1)).any(dim=-1)
        self._correct_top3 += int(top3_correct.sum().item())

        k5 = min(5, probs.size(-1))
        _, top5_idx = probs.topk(k5, dim=-1)              # [N, k5]
        top5_correct = (top5_idx == targets.unsqueeze(-1)).any(dim=-1)
        self._correct_top5 += int(top5_correct.sum().item())

        # S-ACC
        correct_cpu = correct.cpu()
        offsets_cpu = target_offsets.cpu()
        B = int(offsets_cpu.numel()) - 1
        self._total_sentences += B
        for i in range(B):
            s = int(offsets_cpu[i].item())
            e = int(offsets_cpu[i + 1].item())
            if correct_cpu[s:e].all():
                self._correct_sentences += 1

        # Top-K Adaptive ECE
        k = min(self.ece_top_k, probs.size(-1))
        topk_probs, topk_idx = probs.topk(k, dim=-1)      # [N, k]
        ece_confidence = topk_probs.sum(dim=-1)           # [N]
        ece_correct = (topk_idx == targets.unsqueeze(-1)).any(dim=-1)  # [N] bool

        self._ece_confidences.append(ece_confidence.detach().to("cpu", torch.float64))
        self._ece_correct.append(ece_correct.detach().to("cpu", torch.float64))

    def compute(self) -> dict[str, float]:
        """Return all accumulated metrics as a dict."""
        n = self._total_tokens
        s = self._total_sentences

        acc = (self._correct_top1 / n) if n > 0 else 0.0
        top3 = (self._correct_top3 / n) if n > 0 else 0.0
        top5 = (self._correct_top5 / n) if n > 0 else 0.0
        s_acc = (self._correct_sentences / s) if s > 0 else 0.0

        # Top-K Adaptive ECE
        ece = 0.0
        if n > 0 and len(self._ece_confidences) > 0:
            confidences = torch.cat(self._ece_confidences).numpy().astype(np.float64)
            matched = torch.cat(self._ece_correct).numpy().astype(np.float64)

            ace = ACE(bins=self.ece_bins, detection=True)
            ece = float(ace.measure(confidences, matched))

        return {
            "acc": acc,
            "top3_acc": top3,
            "top5_acc": top5,
            "s_acc": s_acc,
            "ece": ece,
        }

    def reset(self) -> None:
        """Clear all running statistics."""
        self._total_tokens = 0
        self._correct_top1 = 0
        self._correct_top3 = 0
        self._correct_top5 = 0
        self._total_sentences = 0
        self._correct_sentences = 0
        self._ece_confidences: list[torch.Tensor] = []
        self._ece_correct: list[torch.Tensor] = []

    def __len__(self) -> int:
        """Number of tokens accumulated (proxy for batches seen)."""
        return self._total_tokens


class TopKSentenceAccuracy:
    """Top-K sentence accuracy from beam-search hypotheses.

    A sentence counts as correct when its target id sequence appears among
    the top-K decoded hypotheses.

    Usage::

        acc = TopKSentenceAccuracy(k=3)
        for beam_ids, target in ...:
            acc.update(beam_ids, target)  # tensors: [k, T] and [T]
        top3_s_acc = acc.compute()
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.reset()

    def update(self, beams, target) -> None:
        """Ingest one sample.

        Args:
            beams: the (up to k) decoded id sequences, ``[k, T]`` int tensor
                (or a nested list of ids).
            target: the ground-truth id sequence, ``[T]`` int tensor (or list).
        """
        self._total_sentences += 1
        if isinstance(beams, torch.Tensor):
            beams_t = beams
            target_t = target if isinstance(target, torch.Tensor) else torch.tensor(target, device=beams.device)
            correct = bool((beams_t[: self.k] == target_t.unsqueeze(0)).all(dim=-1).any())
        else:
            correct = any(list(hyp) == list(target) for hyp in beams[: self.k])
        if correct:
            self._correct_sentences += 1

    def update_batch(self, beams, targets) -> None:
        """Ingest a batch of samples in a single device-side reduction.

        Args:
            beams: ``[B, k, T]`` int tensor of decoded hypotheses.
            targets: ``[B, T]`` int tensor of ground-truth sequences.
        """
        correct = (beams[:, : self.k] == targets.unsqueeze(1)).all(dim=-1).any(dim=-1)
        self._correct_sentences += int(correct.sum().item())
        self._total_sentences += int(beams.shape[0])

    def compute(self) -> float:
        if self._total_sentences == 0:
            return 0.0
        return self._correct_sentences / self._total_sentences

    def reset(self) -> None:
        self._total_sentences = 0
        self._correct_sentences = 0
