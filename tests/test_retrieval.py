from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from scripts.chat import ChatArgs, RetrievalBackend
from scripts.models.pipeline import SearchResult
from scripts.retrieval import QueryEmbedder, search_result_to_chunk


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


def _result(chunk_type: str, speaker: str | None) -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        student_id="2302",
        student_name="Bhagyashree",
        class_name="Eco",
        chunk_type=chunk_type,
        text="...",
        distance=0.1,
        speaker=speaker,
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
