"""Training loop for the artifact segmenter.

Each epoch trains, then evaluates and searches a probability threshold that
maximises a combined score ``0.75 * IoU + 0.25 * F1`` on the validation set
(F1 is image-level "has artifact"). The chosen threshold is stored on the model
and the best checkpoint is saved.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm


def train_model(model, num_epochs: int, train_loader, val_loader, device, criterion, optimizer,
                iou_fn: Callable, best_model_path: str, scheduler=None):
    """Train ``model`` and save the best checkpoint by combined IoU/F1 score.

    :param iou_fn: Callable ``iou_fn(pred_mask, target_mask) -> tensor`` (e.g. the
        course ``eval_metric.iou``).
    """
    reduction_type = getattr(criterion, "reduction", "mean")
    f1_metric = BinaryF1Score(threshold=0.5).to("cpu")
    best_combined_metric = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, total_items = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            bsz = batch["img"].shape[0]
            total_items += bsz

            pred_mask = model({"img": batch["img"], "gt": batch["gt"]})
            loss = criterion(pred_mask, batch["mask"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * (bsz if reduction_type == "mean" else 1)

        avg_train_loss = total_train_loss / total_items

        model.eval()
        all_probs, all_masks, all_has_art = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                probs = model({"img": batch["img"], "gt": batch["gt"]})
                all_probs.append(probs.cpu())
                all_masks.append(batch["mask"].cpu())
                all_has_art.append(batch["has_artifact"].float().cpu())

        all_probs = torch.cat(all_probs)
        all_masks = torch.cat(all_masks)
        all_has_art = torch.cat(all_has_art)

        best_threshold, best_score, best_iou, best_f1 = 0.5, -np.inf, 0.0, 0.0
        for thr in np.arange(0.1, 0.9, 0.05):
            bin_mask = (all_probs >= thr).float()
            pred_has_art = (bin_mask.sum(dim=(1, 2, 3)) > 0).float()
            f1_metric.reset()
            f1_metric.update(pred_has_art, all_has_art)
            f1 = f1_metric.compute().item()
            iou = float(iou_fn(bin_mask, all_masks))
            score = 0.75 * iou + 0.25 * f1
            if score > best_score:
                best_score, best_threshold, best_iou, best_f1 = score, thr, iou, f1

        model.threshold = best_threshold
        print(f"Epoch {epoch + 1} | train {avg_train_loss:.4f} | "
              f"val IoU {best_iou:.4f} | val F1 {best_f1:.4f} | thr {best_threshold:.2f}")

        if best_score > best_combined_metric:
            best_combined_metric = best_score
            model.save_weights(best_model_path)
            print(f"  new best combined {best_score:.4f} -> saved {best_model_path}")

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(best_score if scheduler.mode == "max" else avg_train_loss)
            else:
                scheduler.step()

    print(f"Done. Best combined metric = {best_combined_metric:.4f}")
    return model
