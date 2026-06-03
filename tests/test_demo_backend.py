from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

from scripts.chat import PromptMessage
from scripts.demo_backend import answer_for_student, student_summary, top_score
from scripts.models.pipeline import SearchResult
from scripts.retrieval import QueryEmbedder, RetrievalResult, RetrievedChunk


class _FakeArray:
    def __init__(self, data: list[float]) -> None:
        self._data = data

    def tolist(self) -> list[float]:
        return self._data


def make_embedder() -> QueryEmbedder:
    embedder = QueryEmbedder("dummy-model")
    embedder._model = SimpleNamespace(encode=lambda query: _FakeArray([0.1, 0.2]))
    return embedder


class FakeStore:
    def __init__(
        self,
        *,
        search_results: list[SearchResult] | None = None,
        student_chunks: list[SearchResult] | None = None,
    ) -> None:
        self._search_results = search_results or []
        self._student_chunks = student_chunks or []
        self.search_calls: list[tuple[str, int, list[str]]] = []

    def search(
        self,
        query_embedding: list[float],
        student_id: str,
        top_k: int = 5,
        chunk_types: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        self.search_calls.append((student_id, top_k, list(chunk_types or [])))
        return self._search_results

    def get_student_chunks(self, student_id: str) -> list[SearchResult]:
        return self._student_chunks


class FakeChatBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[Sequence[PromptMessage], str]] = []

    def generate(self, *, messages: Sequence[PromptMessage], model: str) -> str:
        self.calls.append((messages, model))
        return "grounded answer"


def make_search_result(
    *,
    chunk_id: str = "c1",
    student_id: str = "2302",
    student_name: str = "Bhagyashree",
    class_name: str = "Economics.02",
    text: str = "supply function intercept beta",
    distance: float = 0.3,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        student_id=student_id,
        student_name=student_name,
        class_name=class_name,
        chunk_type="class_context",
        text=text,
        distance=distance,
        start_time=10.0,
        end_time=20.0,
        speaker="Nisha",
    )


# --- student_summary ---


def test_student_summary_counts_chunks_and_dedups_classes() -> None:
    chunks = [
        make_search_result(chunk_id="a", class_name="Economics.02"),
        make_search_result(chunk_id="b", class_name="Economics.02"),
        make_search_result(chunk_id="c", class_name="Math.01"),
    ]
    store = FakeStore(student_chunks=chunks)

    summary = student_summary(store, "2302")

    assert summary.chunk_count == 3
    assert summary.class_names == ["Economics.02", "Math.01"]
    assert summary.student_name == "Bhagyashree"


def test_student_summary_falls_back_to_id_when_empty() -> None:
    store = FakeStore(student_chunks=[])
    summary = student_summary(store, "9999")
    assert summary.student_name == "9999"
    assert summary.chunk_count == 0
    assert summary.class_names == []


# --- top_score ---


def make_chunk(score: float | None, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{rank}",
        chunk_type="class_context",
        rank=rank,
        score=score,
        student_id="2302",
        student_name="Bhagyashree",
        text="text",
    )


def make_result(chunks: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(
        context_string="ctx",
        embedding_model="dummy",
        query="q",
        result_count=len(chunks),
        retrieved_chunks=chunks,
        student_id="2302",
        top_k=5,
    )


def test_top_score_picks_maximum() -> None:
    result = make_result([make_chunk(0.5, 1), make_chunk(0.71, 2), make_chunk(0.4, 3)])
    assert top_score(result) == 0.71


def test_top_score_none_when_no_scores() -> None:
    assert top_score(make_result([])) is None


# --- answer_for_student ---


def test_answer_for_student_groq_path_is_scoped_and_grounded() -> None:
    store = FakeStore(search_results=[make_search_result()])
    backend = FakeChatBackend()

    turn = answer_for_student(
        student_id="2302",
        student_name="Bhagyashree",
        question="What is the supply function?",
        store=store,
        embedder=make_embedder(),
        chat_backend=backend,
        db_url="postgresql://localhost/db",
    )

    assert turn.answer == "grounded answer"
    assert turn.answer_source == "groq"
    assert turn.model is not None
    assert turn.retrieval_result.result_count == 1
    assert len(backend.calls) == 1
    # retrieval was scoped to the selected student
    assert store.search_calls[0][0] == "2302"


def test_answer_for_student_fallback_skips_llm_when_no_context() -> None:
    store = FakeStore(search_results=[])
    backend = FakeChatBackend()

    turn = answer_for_student(
        student_id="2302",
        student_name="Bhagyashree",
        question="What did we cover?",
        store=store,
        embedder=make_embedder(),
        chat_backend=backend,
        db_url="postgresql://localhost/db",
    )

    assert turn.answer_source == "fallback"
    assert turn.model is None
    assert backend.calls == []  # never invents an answer via the LLM
    assert "Bhagyashree" in turn.answer


def test_answer_for_student_turn_index_follows_history() -> None:
    store = FakeStore(search_results=[make_search_result()])
    backend = FakeChatBackend()
    args: dict[str, Any] = dict(
        student_id="2302",
        student_name="Bhagyashree",
        store=store,
        embedder=make_embedder(),
        chat_backend=backend,
        db_url="postgresql://localhost/db",
    )

    first = answer_for_student(question="q1", history_turns=[], **args)
    second = answer_for_student(question="q2", history_turns=[first], **args)

    assert first.turn_index == 1
    assert second.turn_index == 2
