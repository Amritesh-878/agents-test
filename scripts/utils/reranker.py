from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Protocol, Sequence

from pydantic import BaseModel

from scripts.models.pipeline import SearchResult

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER = "crossencoder"


class RerankedCandidate(BaseModel):
    result: SearchResult
    rerank_score: float | None = None


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> list[RerankedCandidate]:
        ...


class NoOpReranker:
    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> list[RerankedCandidate]:
        return [RerankedCandidate(result=candidate) for candidate in candidates]


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> list[RerankedCandidate]:
        ordered = list(candidates)
        if len(ordered) <= 1:
            return [RerankedCandidate(result=candidate) for candidate in ordered]
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, candidate.text) for candidate in ordered])
        order = sorted(range(len(ordered)), key=lambda index: scores[index], reverse=True)
        return [
            RerankedCandidate(result=ordered[index], rerank_score=_sigmoid(float(scores[index])))
            for index in order
        ]


@lru_cache(maxsize=1)
def _crossencoder_singleton() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def make_reranker(name: str = DEFAULT_RERANKER) -> Reranker:
    if name == "none":
        return NoOpReranker()
    if name == "crossencoder":
        return _crossencoder_singleton()
    raise ValueError(f"Unknown reranker: {name!r}")
