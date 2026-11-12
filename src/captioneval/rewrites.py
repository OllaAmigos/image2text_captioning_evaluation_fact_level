"""Helpers for evaluating caption rewrites (long -> medium / short / shortest)."""
from __future__ import annotations

import re
import unicodedata
from typing import Sequence

from .metrics import critical_recall

_LAT = re.compile(r"[A-Za-z]")
_CYR = re.compile(r"[\u0400-\u04FF]")
_WORD = re.compile(r"\w+", re.UNICODE)


def rewrite_critical_recall(verdicts: Sequence[str]) -> float | None:
    """Main rewrite metric: recall over the *critical* facts of the source caption.

    Verdicts per critical fact: full (1) / partial (0.5) / missing (0).
    Counting over all facts is misleading because ~half of them are minor and
    over-sliced; compression is supposed to drop those.
    """
    return critical_recall(verdicts)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"\s+", " ", s).strip(" \"'«»“”.,;:!?…")


def text_preservation(quotes: Sequence[str], rewrite: str) -> float | None:
    """Share of on-image text quotes that survive verbatim in the rewrite."""
    if not quotes:
        return None
    rw = _norm(rewrite)
    return sum(_norm(q) in rw for q in quotes) / len(quotes)


def mixed_script_tokens(text: str) -> list[str]:
    """Tokens that glue Latin and Cyrillic letters, e.g. a Russian suffix on an English stem.

    A cheap detector for a real generation defect: the model switches alphabet
    in the middle of a word.
    """
    return [t for t in _WORD.findall(text) if _LAT.search(t) and _CYR.search(t)]


def ellipsis_truncated(rewrite: str) -> bool:
    """Rewrite cuts a quote or sentence off with an ellipsis."""
    return "..." in rewrite or "…" in rewrite
