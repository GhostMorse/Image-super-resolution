"""Inference and visualisation helpers for a trained artifact segmenter.

These use the course-provided ``useful_utils`` (image IO / display) and
``eval_metric.iou``; add those modules to the project root (see the README).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eval_metric import iou
from useful_utils import read_image, show_images

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def mask2img(mask: torch.Tensor) -> np.ndarray:
    """Convert a ``(C, H, W)`` float mask/tensor in ``[0, 1]`` to a BGR uint8 image."""
    img = mask.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_test_batch(test_img_path: str, device) -> dict:
    """Load ``<name>.png`` and its ``<name>@gt.png`` into a model input dict."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        _NORMALIZE,
    ])
    img = transform(Image.open(test_img_path).convert("RGB")).to(device)
    gt = transform(Image.open(test_img_path[:-4] + "@gt.png").convert("RGB")).to(device)
    return {"img": img, "gt": gt}


def get_mask(mask_path: str, device) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])
    return transform(Image.open(mask_path).convert("L")).to(device)


def visualize_probability_mask(model, image_path: str, device=None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pred = model(get_test_batch(image_path, device))
    show_images([read_image(image_path), mask2img(pred)], ["Test image", "Artifact mask (prob)"])


def visualize_binary_mask(model, image_path: str, device=None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pred = (model(get_test_batch(image_path, device)) >= model.threshold).float()
    iou_val = iou(pred, get_mask(image_path[:-4] + "@mask.png", device))
    show_images([read_image(image_path), mask2img(pred), read_image(image_path[:-4] + "@mask.png")],
                ["Test image", f"Predicted mask | IoU {iou_val:.3f}", "GT mask"])
