# Super-Resolution: Interpolation & Artifact Detection

Two components from a super-resolution study, in PyTorch / NumPy:

1. **Classical interpolation** — a from-scratch, fully vectorised 1-D resampler
   with nearest-neighbour, linear and Catmull-Rom **cubic** kernels.
2. **SR artifact detection** — a U-Net that segments the visual artifacts that
   learned super-resolution models can introduce, given the SR output and the
   reference image.

The assignment also evaluates SR models (e.g. SwinIR) with PSNR / SSIM / LPIPS;
see [Findings](#findings) for that comparison.

## Components

### 1. Interpolation kernels (`interpolation/`)

`interpolate(signal, scale_factor, kernel)` resamples a 1-D signal by taking,
for each output position, a kernel-weighted average of the input samples. The
three kernels reproduce SciPy's `nearest` / `linear` interpolation and the
standard Catmull-Rom cubic, with no Python loops.

### 2. Artifact-detection model (`artifacts/`)

The dataset consists of quadruples `(img, mask, gt, has_artifact)`:

- `img` — an image processed by a super-resolution model,
- `gt` — the clean reference image,
- `mask` — binary mask of the artifact region,
- `has_artifact` — image-level boolean (mask is all-zero when `False`).

`MyModel` is an encoder-decoder (U-Net) with skip connections. When `use_gt=True`
the reference image is concatenated to the input (6 channels), letting the
network compare the SR output against the reference. Training uses a combined
**BCE + Jaccard** loss, a cosine LR schedule, and a per-epoch search for the
probability `threshold` that maximises `0.75 * IoU + 0.25 * F1` on validation.

> Constraints honoured from the task: no pretrained backbones, the `MyModel`
> class name / `save_weights` / `load_weights` / `threshold` interface is kept.

## Project structure

```
super-resolution/
├── interpolation/
│   └── kernels.py          # interpolate + nearest / linear / cubic kernels
├── artifacts/
│   ├── model.py            # DoubleConv, MyEncoder, MyDecoder(Block), MyModel
│   ├── losses.py           # Dice, Jaccard, BCE+Dice, BCE+Jaccard
│   ├── engine.py           # train_model (threshold search, best-by IoU/F1)
│   └── inference.py        # mask2img, visualise probability / binary masks
├── train.py                # training entry point
├── predict.py              # visualise predictions on a test image
├── notebooks/
│   └── artifact_detection_pipeline.ipynb
└── tests/                  # interpolation (graded checks), losses, model
```

## External files to add

This repository contains the model and training code. A few course-provided
files and the dataset are **not** included — drop them in as follows:

| File | Put it in | Provides |
| --- | --- | --- |
| `useful_utils.py` | project root | `read_image`, `show_images`, `preprocess_image`, … |
| `eval_metric.py` | project root | `iou(pred_mask, gt_mask)` |
| `artifact_dataset.py` | project root | `create_dataloader(...)` |

(These come from the assignment's `additional_files.zip`.) `create_dataloader`
is expected to have the signature

```python
create_dataloader(dataset_dir, labels_path, batch_size, val_size,
                  random_state, num_workers, train_augs, val_augs)
        -> (train_loader, val_loader)
```

where each batch is a dict with `img` `(B,3,H,W)`, `gt` `(B,3,H,W)`,
`mask` `(B,1,H,W)` and `has_artifact` `(B,)`. The **dataset** itself lives
elsewhere on disk; its `labels.csv` has columns `sr_fn`, `mask_fn`, `gt_fn` and a
boolean `has_artifact`. Point `--dataset-dir` at the folder containing
`labels.csv` and the images.

## Installation

```bash
git clone https://github.com/GhostMorse/super-resolution.git
cd super-resolution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# train (needs the external files above + the dataset)
python train.py --dataset-dir /path/to/SR-task/train --use-gt --epochs 20

# visualise the predicted mask on a test image (expects <img>@gt.png, <img>@mask.png)
python predict.py --weights my_model.pth --image test_image.png --use-gt --binary
```

`notebooks/artifact_detection_pipeline.ipynb` runs the whole thing end to end
(build loaders → train → threshold search → save weights → visualise).

## Implementation notes

- **Vectorised interpolation:** weights are built as a `(new_len, orig_len)`
  matrix and applied with a single matmul; rows are renormalised so the kernel
  need not sum to one exactly.
- **`use_gt` input:** concatenating the reference image gives the model an
  explicit signal for "what the SR output should have looked like".
- **Threshold is part of the model:** the best validation threshold is stored on
  `model.threshold` so inference returns a binary mask directly.
- **Loss on probabilities vs logits:** `MyModel` ends in a sigmoid, so the
  combined losses default to `with_logits=False`.

## Findings

**Interpolation vs learned SR.** Comparing nearest-neighbour, bicubic and SwinIR:
nearest-neighbour copies original pixels, producing blocky "staircase" edges —
worse PSNR/SSIM, but its untouched pixels can score a *better* (lower) LPIPS.
Bicubic averages neighbouring pixels into smooth gradients — better PSNR/SSIM,
but its invented values are not necessarily perceptually ideal, so LPIPS suffers.
SwinIR behaves like an "improved middle ground": smooth yet sharp, generally the
best perceptual quality. The takeaway is that pixel-wise metrics (PSNR/SSIM) and
perceptual metrics (LPIPS) can disagree, which is exactly why learned SR can look
good while occasionally hallucinating artifacts — motivating part 2.

**Artifact detector.** A small U-Net learns to localise artifact regions from
`(img, gt)` pairs. Observations and next steps: the model tends to be
over-confident, so a better-calibrated loss (or label smoothing) would help; and
the network is under-capacity with not-very-diverse features, so a wider/deeper
encoder (or stronger augmentation) is the natural next improvement. After
training, record the achieved IoU / F1 here.

## Tests

```bash
pytest
```

Covers the interpolation kernels (the assignment's exact graded checks), the
Dice / Jaccard / combined losses, and the model's output shapes, `threshold`
field and weight save/load. All tests run on CPU and need no external files.

## License

Released under the [MIT License](LICENSE).
