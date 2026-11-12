"""Weighted fact-level Precision / Recall for captions.

Precision = how much of what the caption *claims* is supported by the image.
Recall    = how much of what the image *contains* is covered by the caption.

Both are the same formula over different fact lists:

    score = sum_i(S_i * W_i) / sum_i(W_i)

S_i - verdict score (full=1, partial=0.5, wrong/missing=0)
W_i - importance weight (critical=5, important=3, minor=1)
"""
from __future__ import annotations

import random
import statistics
from typing import Iterable, Sequence

IMPORTANCE_WEIGHTS = {"critical": 5, "important": 3, "minor": 1}

VERDICT_SCORES = {
    "full": 1.0,
    "partial": 0.5,
    "wrong": 0.0,
    "contradicted": 0.0,
    "missing": 0.0,
}

# Verdicts that mean "the caption says something that is not true".
ERROR_VERDICTS = {"wrong", "contradicted"}

FactVerdict = tuple[str, str]  # (verdict, importance)


def weighted_score(items: Iterable[FactVerdict]) -> float | None:
    """Importance-weighted mean of verdict scores. None if there is nothing to score."""
    num = den = 0.0
    for verdict, importance in items:
        w = IMPORTANCE_WEIGHTS[importance]
        num += VERDICT_SCORES[verdict] * w
        den += w
    return num / den if den else None


def caption_precision(caption_fact_verdicts: Sequence[FactVerdict]) -> float | None:
    """Facts extracted from the caption, judged against the image."""
    return weighted_score(caption_fact_verdicts)


def image_recall(image_fact_verdicts: Sequence[FactVerdict]) -> float | None:
    """Reference facts extracted from the image, judged against the caption."""
    return weighted_score(image_fact_verdicts)


def f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def has_critical_error(caption_fact_verdicts: Sequence[FactVerdict]) -> bool:
    return any(v in ERROR_VERDICTS and imp == "critical" for v, imp in caption_fact_verdicts)


def is_good(
    p: float | None,
    r: float | None,
    critical_error: bool,
    p_min: float = 0.85,
    r_min: float = 0.50,
) -> bool:
    """GOOD / BAD rule used for dataset-level reporting.

    A caption is GOOD if both metrics clear their thresholds AND it has no
    wrong critical fact. Thresholds are *working* thresholds for analysis,
    not the acceptance gates (see configs/gates.yaml).
    """
    if p is None or r is None:
        return False
    return p >= p_min and r >= r_min and not critical_error


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)
    )
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def critical_recall(verdicts: Sequence[str]) -> float | None:
    """Unweighted recall over critical facts only (used for rewrites)."""
    if not verdicts:
        return None
    return sum(VERDICT_SCORES[v] for v in verdicts) / len(verdicts)
