from __future__ import annotations

import pytest

from scripts.retrieval import RetrievedChunk
from scripts.utils.retrieval_grade import (
    ROUTER_HIGH_GAP,
    ROUTER_HIGH_SCORE,
    ROUTER_MEDIUM_SCORE,
    grade_retrieval,
)


def _chunk(
    *,
    rerank_score: float | None = None,
    score: float | None = None,
    chunk_type: str = "class_context",
    chunk_id: str = "c",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        rank=1,
        rerank_score=rerank_score,
        score=score,
        student_id="2302",
        student_name="Bhagyashree",
        text="t",
    )


def test_router_default_thresholds() -> None:
    assert ROUTER_HIGH_SCORE == 0.55
    assert ROUTER_MEDIUM_SCORE == 0.35
    assert ROUTER_HIGH_GAP == 0.10


def test_empty_retrieval_is_low() -> None:
    assert grade_retrieval([]) == "low"


@pytest.mark.parametrize(
    "scores,expected",
    [
        ([0.9, 0.5], "high"),
        ([0.9, 0.85], "medium"),
        ([0.5, 0.45], "medium"),
        ([0.3, 0.1], "low"),
        ([0.6], "high"),
    ],
)
def test_rerank_path_boundaries(scores: list[float], expected: str) -> None:
    chunks = [_chunk(rerank_score=s, chunk_id=f"c{i}") for i, s in enumerate(scores)]
    assert grade_retrieval(chunks) == expected


@pytest.mark.parametrize(
    "scores,expected",
    [
        ([0.9, 0.4], "high"),
        ([0.5, 0.4], "medium"),
        ([0.3], "low"),
    ],
)
def test_cosine_fallback_path(scores: list[float], expected: str) -> None:
    chunks = [_chunk(score=s, chunk_id=f"c{i}") for i, s in enumerate(scores)]
    assert grade_retrieval(chunks) == expected


def test_rerank_signal_wins_over_cosine() -> None:
    chunks = [
        _chunk(rerank_score=0.1, score=0.99, chunk_id="c1"),
        _chunk(rerank_score=0.05, score=0.98, chunk_id="c2"),
    ]
    assert grade_retrieval(chunks) == "low"


def test_scoreless_but_nonempty_soft_lands_to_medium() -> None:
    chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2")]
    assert grade_retrieval(chunks) == "medium"
