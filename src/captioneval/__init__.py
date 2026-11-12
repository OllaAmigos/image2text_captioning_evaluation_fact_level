"""i2t-caption-eval: fact-based precision/recall evaluation of image captions."""
from .metrics import (
    IMPORTANCE_WEIGHTS,
    VERDICT_SCORES,
    weighted_score,
    caption_precision,
    image_recall,
    f1,
    is_good,
    bootstrap_ci,
)

__all__ = [
    "IMPORTANCE_WEIGHTS",
    "VERDICT_SCORES",
    "weighted_score",
    "caption_precision",
    "image_recall",
    "f1",
    "is_good",
    "bootstrap_ci",
]
__version__ = "0.1.0"
