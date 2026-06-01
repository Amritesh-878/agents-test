from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from scripts.chat import ChatArgs, RetrievalBackend
from scripts.retrieval import QueryEmbedder


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
