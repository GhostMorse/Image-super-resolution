"""Super-resolution artifact detection (segmentation).

``model`` and ``losses`` are import-light (PyTorch only). ``engine`` also needs
``torchmetrics``; ``inference`` needs the course-provided ``useful_utils`` /
``eval_metric`` modules (added to the project root).
"""

from .losses import CombinedBCEDiceLoss, CombinedBCEJaccardLoss, DiceLoss, JaccardLoss
from .model import MyModel

__all__ = ["MyModel", "DiceLoss", "JaccardLoss", "CombinedBCEDiceLoss", "CombinedBCEJaccardLoss"]
