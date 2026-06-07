"""Visualise a trained model's artifact mask on a test image.

Expects ``<image>.png`` plus ``<image>@gt.png`` (reference) and, for the binary
overlay, ``<image>@mask.png`` (ground-truth mask). Requires the provided
``useful_utils.py`` and ``eval_metric.py`` in the project root.

    python predict.py --weights my_model.pth --image test_image.png --use-gt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from artifacts.inference import visualize_binary_mask, visualize_probability_mask
from artifacts.model import MyModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise predicted artifact masks.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--start-filters", type=int, default=32)
    p.add_argument("--use-gt", action="store_true")
    p.add_argument("--binary", action="store_true", help="overlay the thresholded mask + IoU vs GT")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model = MyModel(num_blocks=args.num_blocks, start_filters=args.start_filters, use_gt=args.use_gt).to(device)
    model.load_weights(str(args.weights), device)
    model.eval()

    if args.binary:
        visualize_binary_mask(model, str(args.image), device)
    else:
        visualize_probability_mask(model, str(args.image), device)


if __name__ == "__main__":
    main()
