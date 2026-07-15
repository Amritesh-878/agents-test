from __future__ import annotations

from typing import Literal, Sequence

from scripts.retrieval import RetrievedChunk

RouterTier = Literal["high", "medium", "low"]

ROUTER_HIGH_SCORE = 0.55
ROUTER_MEDIUM_SCORE = 0.35
ROUTER_HIGH_GAP = 0.10


def _top_and_gap(values: Sequence[float]) -> tuple[float, float] | tuple[None, None]:
    ranked = sorted(values, reverse=True)
    if not ranked:
        return None, None
    if len(ranked) == 1:
        return ranked[0], ranked[0]
    return ranked[0], round(ranked[0] - ranked[1], 6)


def grade_retrieval(chunks: Sequence[RetrievedChunk]) -> RouterTier:
    if not chunks:
        return "low"
    rerank_scores = [chunk.rerank_score for chunk in chunks if chunk.rerank_score is not None]
    cosine_scores = [chunk.score for chunk in chunks if chunk.score is not None]
    signal = rerank_scores or cosine_scores
    if not signal:
        return "medium"
    top, gap = _top_and_gap(signal)
    if top is None or gap is None:
        return "medium"
    if top >= ROUTER_HIGH_SCORE and gap >= ROUTER_HIGH_GAP:
        return "high"
    if top >= ROUTER_MEDIUM_SCORE:
        return "medium"
    return "low"
