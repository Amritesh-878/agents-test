from __future__ import annotations

from typing import Any, Protocol, Sequence

from scripts.models.pipeline import SearchResult

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER = "crossencoder"


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> list[SearchResult]:
        ...


class NoOpReranker:
    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> list[SearchResult]:
        return list(candidates)


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> list[SearchResult]:
        ordered = list(candidates)
        if len(ordered) <= 1:
            return ordered
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, candidate.text) for candidate in ordered])
        order = sorted(range(len(ordered)), key=lambda index: scores[index], reverse=True)
        return [ordered[index] for index in order]


def make_reranker(name: str = DEFAULT_RERANKER) -> Reranker:
    if name == "none":
        return NoOpReranker()
    if name == "crossencoder":
        return CrossEncoderReranker()
    raise ValueError(f"Unknown reranker: {name!r}")
