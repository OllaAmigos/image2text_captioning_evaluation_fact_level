"""Atomic-fact schema and adequacy gates for caption -> facts extraction.

Before any caption is scored, we check that the LLM's decomposition of the
caption into facts is itself trustworthy. A row passes if all gates hold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STOPWORDS = {
    # en
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with", "is", "are",
    "was", "were", "it", "its", "this", "that", "for", "by", "as", "from",
    # ru
    "и", "в", "во", "на", "с", "со", "к", "по", "из", "у", "о", "об", "от", "до",
    "за", "для", "что", "это", "как", "а", "но", "или", "не",
}

_TOKEN = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass
class Fact:
    id: str
    text: str
    importance: str  # critical | important | minor
    category: str = "other"
    span: str | None = None  # verbatim fragment of the caption this fact came from
    quoted_text: str | None = None  # text-on-image quote, if the fact is about lettering
    hedged: bool = False  # caption said "probably / likely"


@dataclass
class GateReport:
    parsed: bool
    span_ok_share: float = 0.0
    content_coverage: float = 0.0
    text_consistent: bool = True
    near_dup_share: float = 0.0
    main_object_critical: bool = True
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def content_words(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text) if t.lower() not in STOPWORDS]


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def validate_caption_facts(
    caption: str,
    facts: list[Fact] | None,
    *,
    has_text_flag: bool | None = None,
    span_min: float = 0.98,
    coverage_min: float = 0.90,
    dup_max: float = 0.05,
    dup_jaccard: float = 0.8,
) -> GateReport:
    """Run the six adequacy gates on one caption -> facts decomposition.

    1. output parsed            4. text-on-image flag consistent, quotes found verbatim
    2. >= 98% spans in caption  5. near-duplicates <= 5% of facts
    3. content-word coverage >= 0.90   6. main object covered by a critical fact
    """
    if not facts:
        return GateReport(parsed=False, failures=["unparsed_or_empty"])

    rep = GateReport(parsed=True)
    cap_l = caption.lower()

    # 2. spans
    with_span = [f for f in facts if f.span]
    rep.span_ok_share = (
        sum(f.span.lower() in cap_l for f in with_span) / len(with_span) if with_span else 0.0
    )
    if rep.span_ok_share < span_min:
        rep.failures.append(f"span_ok_share={rep.span_ok_share:.3f}")

    # 3. coverage of the caption's content words by the facts' texts
    cap_words = set(content_words(caption))
    fact_words = set(w for f in facts for w in content_words(f.text))
    rep.content_coverage = len(cap_words & fact_words) / len(cap_words) if cap_words else 1.0
    if rep.content_coverage < coverage_min:
        rep.failures.append(f"content_coverage={rep.content_coverage:.3f}")

    # 4. text-on-image consistency
    quotes = [f.quoted_text for f in facts if f.quoted_text]
    quotes_found = all(q.lower() in cap_l for q in quotes)
    flag_ok = has_text_flag is None or has_text_flag == bool(quotes)
    rep.text_consistent = quotes_found and flag_ok
    if not rep.text_consistent:
        rep.failures.append("text_flag_or_quotes_inconsistent")

    # 5. near duplicates
    sets = [set(content_words(f.text)) for f in facts]
    dup_idx = {
        j
        for i in range(len(sets))
        for j in range(i + 1, len(sets))
        if _jaccard(sets[i], sets[j]) >= dup_jaccard
    }
    rep.near_dup_share = len(dup_idx) / len(facts)
    if rep.near_dup_share > dup_max:
        rep.failures.append(f"near_dup_share={rep.near_dup_share:.3f}")

    # 6. at least one critical fact (proxy for "main object is covered")
    rep.main_object_critical = any(f.importance == "critical" for f in facts)
    if not rep.main_object_critical:
        rep.failures.append("no_critical_fact")

    return rep
