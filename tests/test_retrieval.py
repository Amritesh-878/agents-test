from __future__ import annotations

import sys
import types
from typing import Any, Sequence

import pytest

from scripts.chat import ChatArgs, RetrievalBackend
from scripts.models.pipeline import SearchResult
from scripts.retrieval import (
    QueryEmbedder,
    RetrievalError,
    RetrievalResult,
    format_retrieved_chunk,
    retrieve_from_pgvector,
    search_result_to_chunk,
)
from scripts.utils.reranker import NoOpReranker


class _FakeArray:
    def __init__(self, data: list[float]) -> None:
        self._data = data

    def tolist(self) -> list[float]:
        return self._data


def test_query_embedder_loads_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = {"init": 0, "encode": 0}

    class FakeModel:
        def __init__(self, name: str) -> None:
            counters["init"] += 1

        def encode(self, text: str) -> _FakeArray:
            counters["encode"] += 1
            return _FakeArray([0.1, 0.2])

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    embedder = QueryEmbedder("dummy-model")
    assert embedder.encode("hello") == [0.1, 0.2]
    assert embedder.encode("world") == [0.1, 0.2]

    assert counters["init"] == 1  # model loaded from disk only once
    assert counters["encode"] == 2


class _CountingStore:
    def __init__(self) -> None:
        self.closed = False

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def search_lexical(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def close(self) -> None:
        self.closed = True


def test_retrieval_backend_reuses_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    connect_calls = {"n": 0}
    store = _CountingStore()

    def fake_connect(db_url: str) -> _CountingStore:
        connect_calls["n"] += 1
        return store

    monkeypatch.setattr("scripts.utils.pg_store.connect_pg_store", fake_connect)

    embedder = QueryEmbedder("dummy-model")
    embedder._model = types.SimpleNamespace(encode=lambda text: _FakeArray([0.1]))

    backend = RetrievalBackend(embedder=embedder)
    args = ChatArgs(db_url="postgresql://localhost/db", student_id="2302", student_name="B")

    backend.retrieve(args, "first question")
    backend.retrieve(args, "second question")

    assert connect_calls["n"] == 1  # connection opened once, reused across turns

    backend.close()
    assert store.closed is True


def test_retrieval_backend_does_not_close_injected_store() -> None:
    store = _CountingStore()
    embedder = QueryEmbedder("dummy-model")
    embedder._model = types.SimpleNamespace(encode=lambda text: _FakeArray([0.1]))

    backend = RetrievalBackend(embedder=embedder, store=store)
    backend.retrieve(
        ChatArgs(db_url="postgresql://localhost/db", student_id="2302", student_name="B"),
        "q",
    )
    backend.close()

    assert store.closed is False  # caller owns an injected store


def _result(
    chunk_type: str,
    speaker: str | None,
    metadata: dict[str, Any] | None = None,
    text: str = "...",
) -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        student_id="2302",
        student_name="Bhagyashree",
        class_name="Eco",
        chunk_type=chunk_type,
        text=text,
        distance=0.1,
        speaker=speaker,
        metadata=metadata or {},
    )


def test_class_context_speaker_not_attributed_to_student() -> None:
    # class_context / missed have no stored speaker (teacher narration); it must NOT be
    # labeled with the student's name, or the LLM quotes the teacher back as the student.
    assert search_result_to_chunk(_result("class_context", None), 1).source_speaker == "teacher"
    assert search_result_to_chunk(_result("missed", None), 1).source_speaker == "teacher"


def test_spoken_speaker_preserved_else_student_name() -> None:
    assert search_result_to_chunk(_result("spoken", "Bhagyashree"), 1).source_speaker == "Bhagyashree"
    # spoken with no stored speaker still falls back to the student (their own words)
    assert search_result_to_chunk(_result("spoken", None), 1).source_speaker == "Bhagyashree"


def test_material_chunk_surfaces_provenance_and_speaker() -> None:
    # Material chunks carry speaker="material" (never labeled "teacher") and their source
    # filename must flow from metadata into the chunk so the prompt can attribute it.
    result = _result("material", "material", metadata={"source_file": "supply_deck.pptx"})
    chunk = search_result_to_chunk(result, 1)
    assert chunk.chunk_type == "material"
    assert chunk.source_speaker == "material"
    assert chunk.source_file == "supply_deck.pptx"


def test_format_retrieved_chunk_shows_material_source_file() -> None:
    result = _result(
        "material",
        "material",
        metadata={"source_file": "supply_deck.pptx"},
        text="determinants shift the supply curve",
    )
    rendered = format_retrieved_chunk(search_result_to_chunk(result, 1))
    assert "type=material" in rendered
    assert "source=supply_deck.pptx" in rendered


def test_format_retrieved_chunk_omits_source_when_absent() -> None:
    rendered = format_retrieved_chunk(search_result_to_chunk(_result("spoken", "Bhagyashree"), 1))
    assert "source=" not in rendered


class _FixedEmbedder:
    def encode(self, query: str) -> list[float]:
        return [0.1, 0.2]


class _HybridStore:
    def __init__(
        self, dense: list[SearchResult], lexical: list[SearchResult]
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self.dense_chunk_types: list[str] | None = None
        self.lexical_chunk_types: list[str] | None = None

    def search(
        self,
        query_embedding: list[float],
        student_id: str,
        top_k: int = 5,
        chunk_types: Any = None,
        class_name: str | None = None,
    ) -> list[SearchResult]:
        self.dense_chunk_types = list(chunk_types or [])
        return self._dense

    def search_lexical(
        self,
        query_text: str,
        *,
        student_id: str,
        chunk_types: Any = None,
        limit: int = 25,
        class_name: str | None = None,
    ) -> list[SearchResult]:
        self.lexical_chunk_types = list(chunk_types or [])
        return self._lexical


def _hit(chunk_id: str, distance: float | None, text: str = "...") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        student_id="2302",
        student_name="Bhagyashree",
        class_name="Eco",
        chunk_type="spoken",
        text=text,
        distance=distance,
    )


def _retrieve(store: _HybridStore, chunk_types: list[str] | None = None) -> RetrievalResult:
    return retrieve_from_pgvector(
        student_id="2302",
        query="q",
        top_k=5,
        chunk_types=chunk_types,  # type: ignore[arg-type]
        db_url="postgresql://localhost/db",
        store=store,
        embedder=_FixedEmbedder(),  # type: ignore[arg-type]
        reranker=NoOpReranker(),
    )


class _FakeReranker:
    def __init__(self, order: list[str]) -> None:
        self._order = order

    def rerank(self, query: str, candidates: Sequence[SearchResult]) -> list[SearchResult]:
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        promoted = [by_id[chunk_id] for chunk_id in self._order if chunk_id in by_id]
        listed = set(self._order)
        remainder = [c for c in candidates if c.chunk_id not in listed]
        return promoted + remainder


def test_reranker_reorders_pool_and_truncates_to_final_top_k() -> None:
    store = _HybridStore([_hit("d1", 0.2), _hit("d2", 0.3), _hit("d3", 0.4)], [])
    result = retrieve_from_pgvector(
        student_id="2302",
        query="q",
        top_k=2,
        db_url="postgresql://localhost/db",
        store=store,
        embedder=_FixedEmbedder(),  # type: ignore[arg-type]
        reranker=_FakeReranker(["d3", "d1", "d2"]),
    )
    assert [chunk.chunk_id for chunk in result.retrieved_chunks] == ["d3", "d1"]


def test_reranker_output_is_subset_of_fused_pool() -> None:
    store = _HybridStore([_hit("d1", 0.2), _hit("d2", 0.3)], [_hit("lex", None)])
    result = retrieve_from_pgvector(
        student_id="2302",
        query="q",
        top_k=5,
        db_url="postgresql://localhost/db",
        store=store,
        embedder=_FixedEmbedder(),  # type: ignore[arg-type]
        reranker=_FakeReranker(["lex", "d2", "d1"]),
    )
    ids = {chunk.chunk_id for chunk in result.retrieved_chunks}
    assert ids <= {"d1", "d2", "lex"}
    assert ids == {"d1", "d2", "lex"}


def test_cross_student_row_in_pool_raises_even_when_sliced_out() -> None:
    leaked = SearchResult(
        chunk_id="leaked",
        student_id="9999",
        student_name="Someone Else",
        class_name="Eco",
        chunk_type="spoken",
        text="...",
        distance=0.9,
    )
    store = _HybridStore([_hit("good", 0.1), leaked], [])
    with pytest.raises(RetrievalError, match="Cross-student leakage"):
        retrieve_from_pgvector(
            student_id="2302",
            query="q",
            top_k=1,
            db_url="postgresql://localhost/db",
            store=store,
            embedder=_FixedEmbedder(),  # type: ignore[arg-type]
            reranker=NoOpReranker(),
        )


def test_hybrid_surfaces_lexical_only_hit() -> None:
    dense = [_hit("d1", 0.2), _hit("shared", 0.5)]
    lexical = [_hit("shared", None), _hit("lex_only", None)]
    result = _retrieve(_HybridStore(dense, lexical))
    ids = {chunk.chunk_id for chunk in result.retrieved_chunks}
    assert "lex_only" in ids
    assert ids == {"d1", "shared", "lex_only"}


def test_hybrid_lexical_only_hit_has_no_fake_score() -> None:
    dense = [_hit("d1", 0.2)]
    lexical = [_hit("lex_only", None)]
    result = _retrieve(_HybridStore(dense, lexical))
    by_id = {chunk.chunk_id: chunk for chunk in result.retrieved_chunks}
    assert by_id["lex_only"].score is None
    assert by_id["lex_only"].distance is None
    assert by_id["d1"].score is not None


def test_hybrid_scopes_both_arms_to_same_chunk_types() -> None:
    store = _HybridStore([], [])
    _retrieve(store, chunk_types=["spoken", "chat"])
    assert store.dense_chunk_types == ["spoken", "chat"]
    assert store.lexical_chunk_types == ["spoken", "chat"]


def _hit_with_model(chunk_id: str, model: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        student_id="2302",
        student_name="Bhagyashree",
        class_name="Eco",
        chunk_type="spoken",
        text="...",
        distance=0.2,
        metadata={"embedding_model": model},
    )


def test_retrieve_raises_on_embedding_model_mismatch() -> None:
    store = _HybridStore([_hit_with_model("d1", "paraphrase-multilingual-MiniLM-L12-v2")], [])
    with pytest.raises(RetrievalError, match="Embedding-model mismatch"):
        _retrieve(store)


def test_retrieve_passes_when_stored_model_matches_query() -> None:
    from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL

    store = _HybridStore([_hit_with_model("d1", DEFAULT_EMBEDDING_MODEL)], [])
    result = _retrieve(store)
    assert [chunk.chunk_id for chunk in result.retrieved_chunks] == ["d1"]


def test_retrieve_treats_unstamped_chunk_as_default_model() -> None:
    dense = [
        SearchResult(
            chunk_id="legacy",
            student_id="2302",
            student_name="Bhagyashree",
            class_name="Eco",
            chunk_type="spoken",
            text="...",
            distance=0.2,
        )
    ]
    result = _retrieve(_HybridStore(dense, []))
    assert [chunk.chunk_id for chunk in result.retrieved_chunks] == ["legacy"]
