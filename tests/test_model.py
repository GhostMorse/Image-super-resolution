"""Output-shape tests for the artifact-segmentation U-Net."""

import torch

from artifacts.model import MyModel


def _batch(c_img=3, h=64, w=64):
    return {"img": torch.rand(1, 3, h, w), "gt": torch.rand(1, 3, h, w)}


def test_forward_without_gt():
    model = MyModel(num_blocks=2, start_filters=8, use_gt=False).eval()
    with torch.no_grad():
        out = model(_batch())
    assert out.shape == (1, 1, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0  # sigmoid


def test_forward_with_gt():
    model = MyModel(num_blocks=2, start_filters=8, use_gt=True).eval()
    with torch.no_grad():
        out = model(_batch())
    assert out.shape == (1, 1, 64, 64)


def test_forward_unbatched_input():
    # A 3-D (C, H, W) input is handled by an internal unsqueeze/squeeze.
    model = MyModel(num_blocks=2, start_filters=8, use_gt=False).eval()
    with torch.no_grad():
        out = model({"img": torch.rand(3, 64, 64), "gt": torch.rand(3, 64, 64)})
    assert out.shape == (1, 64, 64)


def test_threshold_field_present():
    model = MyModel()
    assert hasattr(model, "threshold")
    model.threshold = 0.6  # must remain settable
    assert model.threshold == 0.6


def test_save_and_load_weights(tmp_path):
    model = MyModel(num_blocks=2, start_filters=8, use_gt=True)
    path = tmp_path / "w.pth"
    model.save_weights(str(path))
    clone = MyModel(num_blocks=2, start_filters=8, use_gt=True)
    clone.load_weights(str(path), device="cpu")
    for p, q in zip(model.parameters(), clone.parameters()):
        assert torch.allclose(p, q)
