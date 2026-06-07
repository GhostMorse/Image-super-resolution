"""Segmentation losses for artifact-mask prediction.

The model outputs probabilities (sigmoid), so the combined losses default to
``with_logits=False``. Dice and Jaccard (IoU) are soft, differentiable region
overlaps; combining them with BCE balances pixel-wise and region-wise signals.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _overlap_terms(probs: torch.Tensor, target: torch.Tensor):
    dims = tuple(range(1, probs.dim()))
    intersection = torch.sum(probs * target, dim=dims)
    totals = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
    return intersection, totals


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    if reduction in (None, "none"):
        return values
    raise ValueError(f"Unknown reduction: {reduction}")


class DiceLoss(nn.Module):
    """Soft Dice loss: ``1 - 2|A∩B| / (|A| + |B|)``."""

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", with_logits: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.with_logits = with_logits

    def forward(self, logits: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        true_labels = true_labels.to(dtype=logits.dtype)
        probs = torch.sigmoid(logits) if self.with_logits else logits
        intersection, totals = _overlap_terms(probs, true_labels)
        dice = (2.0 * intersection + self.eps) / (totals + self.eps)
        return _reduce(1 - dice, self.reduction)


class JaccardLoss(nn.Module):
    """Soft Jaccard / IoU loss: ``1 - |A∩B| / |A∪B|``."""

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", with_logits: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.with_logits = with_logits

    def forward(self, logits: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        true_labels = true_labels.to(dtype=logits.dtype)
        probs = torch.sigmoid(logits) if self.with_logits else logits
        intersection, totals = _overlap_terms(probs, true_labels)
        union = totals - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return _reduce(1 - iou, self.reduction)


class CombinedBCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and Dice."""

    def __init__(self, dice_weight: float = 0.2, bce_weight: float = 0.8, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(reduction=reduction, with_logits=False)
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, true_labels) + self.bce_weight * self.bce(logits, true_labels)


class CombinedBCEJaccardLoss(nn.Module):
    """Weighted sum of BCE and Jaccard (expects probabilities, not logits)."""

    def __init__(self, bce_weight: float = 0.4, jaccard_weight: float = 0.6) -> None:
        super().__init__()
        self.reduction = "mean"
        self.bce_weight = bce_weight
        self.jaccard_weight = jaccard_weight
        self.bce = nn.BCELoss(reduction="mean")
        self.jaccard = JaccardLoss(reduction="mean", with_logits=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=pred.dtype)
        return self.bce_weight * self.bce(pred, target) + self.jaccard_weight * self.jaccard(pred, target)
