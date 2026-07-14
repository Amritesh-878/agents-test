from __future__ import annotations

import unicodedata
from typing import Literal, Sequence

from pydantic import BaseModel, Field

Tier = Literal["high", "medium", "low"]
CaseMetricStatus = Literal["scorable", "refusal", "quote_not_indexed"]

# Frozen for cross-run comparability — do NOT tune mid-initiative (see TASK-026).
HIGH_SCORE_THRESHOLD = 0.60
MEDIUM_SCORE_THRESHOLD = 0.45
HIGH_GAP_THRESHOLD = 0.05
DEFAULT_RECALL_K = 5


def _strip_punctuation(value: str) -> str:
    return "".join(
        " " if unicodedata.category(char)[0] in {"P", "S"} else char for char in value
    )


def normalize_for_containment(value: str) -> str:
    return " ".join(_strip_punctuation(value).casefold().split())


def quote_in_text(quote: str, text: str) -> bool:
    normalized_quote = normalize_for_containment(quote)
    if not normalized_quote:
        return False
    return normalized_quote in normalize_for_containment(text)


def quote_in_any(quote: str, texts: Sequence[str]) -> bool:
    return any(quote_in_text(quote, text) for text in texts)


def first_hit_rank(retrieved_texts: Sequence[str], quotes: Sequence[str]) -> int | None:
    for index, text in enumerate(retrieved_texts):
        if any(quote_in_text(quote, text) for quote in quotes):
            return index + 1
    return None


class QuoteCoverage(BaseModel):
    quote: str
    found_in_retrieved: bool
    found_in_indexed: bool


def quote_coverage(
    quotes: Sequence[str],
    retrieved_texts: Sequence[str],
    indexed_texts: Sequence[str],
) -> list[QuoteCoverage]:
    return [
        QuoteCoverage(
            quote=quote,
            found_in_retrieved=quote_in_any(quote, retrieved_texts),
            found_in_indexed=quote_in_any(quote, indexed_texts),
        )
        for quote in quotes
    ]


def classify_case_status(
    quotes: Sequence[str],
    indexed_texts: Sequence[str],
    *,
    is_refusal: bool,
) -> CaseMetricStatus:
    if is_refusal or not quotes:
        return "refusal"
    if any(quote_in_any(quote, indexed_texts) for quote in quotes):
        return "scorable"
    return "quote_not_indexed"


def is_hit_at_k(status: CaseMetricStatus, rank: int | None, k: int) -> bool:
    return status == "scorable" and rank is not None and rank <= k


def provisional_top_and_gap(scores: Sequence[float | None]) -> tuple[float | None, float | None]:
    ranked = sorted((score for score in scores if score is not None), reverse=True)
    if not ranked:
        return None, None
    if len(ranked) == 1:
        return ranked[0], ranked[0]
    return ranked[0], round(ranked[0] - ranked[1], 6)


def provisional_grade(scores: Sequence[float | None]) -> Tier:
    top, gap = provisional_top_and_gap(scores)
    if top is None or gap is None:
        return "low"
    if top >= HIGH_SCORE_THRESHOLD and gap >= HIGH_GAP_THRESHOLD:
        return "high"
    if top >= MEDIUM_SCORE_THRESHOLD:
        return "medium"
    return "low"


class CaseRetrievalMetrics(BaseModel):
    status: CaseMetricStatus
    first_hit_rank: int | None = None
    hit: bool
    k: int
    tier: Tier
    top_score: float | None = None
    score_gap: float | None = None
    quote_coverage: list[QuoteCoverage] = Field(default_factory=list)


def compute_case_metrics(
    *,
    quotes: Sequence[str],
    retrieved_texts: Sequence[str],
    indexed_texts: Sequence[str],
    scores: Sequence[float | None],
    is_refusal: bool,
    k: int = DEFAULT_RECALL_K,
) -> CaseRetrievalMetrics:
    status = classify_case_status(quotes, indexed_texts, is_refusal=is_refusal)
    rank = first_hit_rank(retrieved_texts, quotes)
    top_score, score_gap = provisional_top_and_gap(scores)
    return CaseRetrievalMetrics(
        status=status,
        first_hit_rank=rank,
        hit=is_hit_at_k(status, rank, k),
        k=k,
        tier=provisional_grade(scores),
        top_score=top_score,
        score_gap=score_gap,
        quote_coverage=quote_coverage(quotes, retrieved_texts, indexed_texts),
    )


class AggregateRetrievalMetrics(BaseModel):
    k: int
    total_cases: int
    scorable_case_count: int
    hit_count: int
    recall_at_k: float
    mrr: float
    refusal_case_count: int
    quote_not_indexed_case_count: int
    retrieved_beyond_k_count: int
    tier_distribution: dict[str, int] = Field(default_factory=dict)
    low_tier_rate: float


def tier_distribution(tiers: Sequence[Tier]) -> dict[str, int]:
    counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for tier in tiers:
        counts[tier] += 1
    return counts


def aggregate_metrics(
    case_metrics: Sequence[CaseRetrievalMetrics],
    *,
    k: int = DEFAULT_RECALL_K,
) -> AggregateRetrievalMetrics:
    scorable = [metric for metric in case_metrics if metric.status == "scorable"]
    hits = [metric for metric in scorable if is_hit_at_k(metric.status, metric.first_hit_rank, k)]
    reciprocal_ranks = [
        1.0 / metric.first_hit_rank for metric in hits if metric.first_hit_rank is not None
    ]
    recall = len(hits) / len(scorable) if scorable else 0.0
    mrr = sum(reciprocal_ranks) / len(scorable) if scorable else 0.0
    retrieved_beyond_k = sum(
        1
        for metric in scorable
        if metric.first_hit_rank is not None and metric.first_hit_rank > k
    )
    distribution = tier_distribution([metric.tier for metric in case_metrics])
    low_rate = distribution["low"] / len(case_metrics) if case_metrics else 0.0
    return AggregateRetrievalMetrics(
        k=k,
        total_cases=len(case_metrics),
        scorable_case_count=len(scorable),
        hit_count=len(hits),
        recall_at_k=round(recall, 6),
        mrr=round(mrr, 6),
        refusal_case_count=sum(1 for metric in case_metrics if metric.status == "refusal"),
        quote_not_indexed_case_count=sum(
            1 for metric in case_metrics if metric.status == "quote_not_indexed"
        ),
        retrieved_beyond_k_count=retrieved_beyond_k,
        tier_distribution=distribution,
        low_tier_rate=round(low_rate, 6),
    )
