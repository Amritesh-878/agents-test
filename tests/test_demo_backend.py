from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

from scripts.chat import PromptMessage
from scripts.demo_backend import answer_for_student, student_summary, top_score
from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL
from scripts.models.pipeline import SearchResult
from scripts.retrieval import QueryEmbedder, RetrievalResult, RetrievedChunk


class _FakeArray:
    def __init__(self, data: list[float]) -> None:
        self._data = data

    def tolist(self) -> list[float]:
        return self._data


def make_embedder() -> QueryEmbedder:
    embedder = QueryEmbedder(DEFAULT_EMBEDDING_MODEL)
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
        self.search_calls: list[tuple[str, int, list[str], str | None]] = []

    def search(
        self,
        query_embedding: list[float],
        student_id: str,
        top_k: int = 5,
        chunk_types: Sequence[str] | None = None,
        class_name: str | None = None,
    ) -> list[SearchResult]:
        self.search_calls.append((student_id, top_k, list(chunk_types or []), class_name))
        return self._search_results

    def search_lexical(
        self,
        query_text: str,
        *,
        student_id: str,
        chunk_types: Sequence[str] | None = None,
        limit: int = 25,
        class_name: str | None = None,
    ) -> list[SearchResult]:
        return []

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


def test_answer_for_student_scopes_self_referential_to_spoken() -> None:
    # "What did I say" must retrieve only the student's own spoken chunks, so the teacher's
    # class_context can't outrank and get mis-attributed to the student (eval Finding A).
    store = FakeStore(search_results=[make_search_result()])
    backend = FakeChatBackend()
    answer_for_student(
        student_id="2302",
        student_name="Bhagyashree",
        question="What did I say about determinants today?",
        store=store,
        embedder=make_embedder(),
        chat_backend=backend,
        db_url="postgresql://localhost/db",
    )
    assert store.search_calls[0][2] == ["spoken", "chat"]


def test_answer_for_student_leaves_class_questions_unfiltered() -> None:
    store = FakeStore(search_results=[make_search_result()])
    backend = FakeChatBackend()
    answer_for_student(
        student_id="2302",
        student_name="Bhagyashree",
        question="What did we cover in class today?",
        store=store,
        embedder=make_embedder(),
        chat_backend=backend,
        db_url="postgresql://localhost/db",
    )
    assert store.search_calls[0][2] == []


def test_answer_for_student_widens_top_k_for_general_questions() -> None:
    from scripts.chat import GENERAL_QUESTION_TOP_K

    store = FakeStore(search_results=[make_search_result()])
    backend = FakeChatBackend()
    turn = answer_for_student(
        student_id="2302",
        student_name="Bhagyashree",
        question="What did the teacher ask us to do on the worksheet?",
        store=store,
        embedder=make_embedder(),
        chat_backend=backend,
        db_url="postgresql://localhost/db",
        top_k=5,
    )
    assert turn.retrieval_result.top_k == GENERAL_QUESTION_TOP_K


def test_answer_for_student_keeps_top_k_tight_for_self_referential() -> None:
    store = FakeStore(search_results=[make_search_result()])
    backend = FakeChatBackend()
    turn = answer_for_student(
        student_id="2302",
        student_name="Bhagyashree",
        question="What did I say about determinants today?",
        store=store,
        embedder=make_embedder(),
        chat_backend=backend,
        db_url="postgresql://localhost/db",
        top_k=5,
    )
    assert turn.retrieval_result.top_k == 5


def test_answer_for_student_scopes_to_selected_class() -> None:
    # Picking one session restricts retrieval to that class_name (per-session scoping).
    store = FakeStore(search_results=[make_search_result()])
    backend = FakeChatBackend()
    answer_for_student(
        student_id="2302",
        student_name="Bhagyashree",
        question="What did we cover?",
        store=store,
        embedder=make_embedder(),
        chat_backend=backend,
        db_url="postgresql://localhost/db",
        class_name="Economics.02_AY2025-26_ Supply Function_16 April",
    )
    assert store.search_calls[0][3] == "Economics.02_AY2025-26_ Supply Function_16 April"


def test_answer_for_student_all_sessions_is_unfiltered() -> None:
    # No class_name (the "All sessions" default) leaves retrieval unscoped — current behavior.
    store = FakeStore(search_results=[make_search_result()])
    backend = FakeChatBackend()
    answer_for_student(
        student_id="2302",
        student_name="Bhagyashree",
        question="What did we cover?",
        store=store,
        embedder=make_embedder(),
        chat_backend=backend,
        db_url="postgresql://localhost/db",
    )
    assert store.search_calls[0][3] is None


def test_session_display_label_parses_topic_and_date() -> None:
    from scripts.demo_backend import session_display_label

    assert (
        session_display_label("Economics.02_AY2025-26_ Supply Function_16 April")
        == "Supply Function — 16 April"
    )
    assert (
        session_display_label(
            "Economics.02_AY2025-26_Determinants of Supply Last Part_13 April_s1"
        )
        == "Determinants of Supply Last Part — 13 April (s1)"
    )
    assert session_display_label(
        "Math.01_A _AY2025-26_Linear Equation Scaffolding Time and Work_05_08 Apr"
    ).startswith("Linear Equation Scaffolding Time and Work")
    # Unparseable input falls back to the raw value rather than raising.
    assert session_display_label("freeform-name") == "freeform-name"


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
