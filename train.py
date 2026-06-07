"""Train the artifact-detection model.

Requires the course-provided ``artifact_dataset.py`` (``create_dataloader``) and
``eval_metric.py`` (``iou``) in the project root, plus the dataset on disk
(``labels.csv`` with columns ``sr_fn``, ``mask_fn``, ``gt_fn`` and a boolean
``has_artifact``).

    python train.py --dataset-dir /path/to/SR-task/train --use-gt --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingLR

from artifact_dataset import create_dataloader  # provided course module
from eval_metric import iou                      # provided course module
from artifacts.engine import train_model
from artifacts.losses import CombinedBCEJaccardLoss
from artifacts.model import MyModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_augs(resize=(1024, 768), crop=(700, 900)):
    extra = {"image1": "image", "mask": "mask"}
    train = A.Compose([
        A.RandomCrop(height=crop[0], width=crop[1], p=0.3),
        A.Resize(height=resize[0], width=resize[1]),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ], additional_targets=extra)
    val = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ], additional_targets=extra)
    return train, val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the SR artifact segmenter.")
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None, help="defaults to <dataset-dir>/labels.csv")
    p.add_argument("--output", type=Path, default=Path("my_model.pth"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--start-filters", type=int, default=32)
    p.add_argument("--use-gt", action="store_true")
    p.add_argument("--no-init", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    labels = args.labels or (args.dataset_dir / "labels.csv")

    train_augs, val_augs = build_augs()
    train_loader, val_loader = create_dataloader(
        args.dataset_dir, labels, batch_size=args.batch_size, val_size=args.val_size,
        random_state=42, num_workers=args.num_workers, train_augs=train_augs, val_augs=val_augs,
    )

    model = MyModel(num_blocks=args.num_blocks, start_filters=args.start_filters,
                    use_gt=args.use_gt, init=not args.no_init).to(device)
    criterion = CombinedBCEJaccardLoss(bce_weight=0.25, jaccard_weight=0.75).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_model(model, args.epochs, train_loader, val_loader, device, criterion, optimizer,
                iou_fn=iou, best_model_path=str(args.output), scheduler=scheduler)
    print(f"Best checkpoint: {args.output} | threshold: {model.threshold:.2f}")


if __name__ == "__main__":
    main()
