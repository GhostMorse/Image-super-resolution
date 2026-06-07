"""Tests for the segmentation losses."""

import torch

from artifacts.losses import CombinedBCEDiceLoss, CombinedBCEJaccardLoss, DiceLoss, JaccardLoss


def test_dice_zero_for_perfect_overlap():
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    assert DiceLoss(with_logits=False)(target, target).item() < 1e-3


def test_jaccard_zero_for_perfect_overlap():
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    assert JaccardLoss(with_logits=False)(target, target).item() < 1e-3


def test_dice_jaccard_in_unit_range():
    pred = torch.rand(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    for loss in (DiceLoss(with_logits=False), JaccardLoss(with_logits=False)):
        v = loss(pred, target).item()
        assert 0.0 <= v <= 1.0


def test_combined_bce_jaccard_finite_on_probabilities():
    pred = torch.rand(2, 1, 16, 16)  # probabilities in [0, 1]
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    v = CombinedBCEJaccardLoss(bce_weight=0.25, jaccard_weight=0.75)(pred, target)
    assert torch.isfinite(v) and v.item() >= 0


def test_combined_bce_dice_finite_on_logits():
    logits = torch.randn(2, 1, 16, 16)  # raw logits
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    v = CombinedBCEDiceLoss()(logits, target)
    assert torch.isfinite(v)
