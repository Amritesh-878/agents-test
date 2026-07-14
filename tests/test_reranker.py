from __future__ import annotations

import sys
import types

import pytest

from scripts.models.pipeline import SearchResult
from scripts.utils.reranker import CrossEncoderReranker, NoOpReranker, make_reranker


def _result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        student_id="2302",
        student_name="Bhagyashree",
        class_name="Eco",
        chunk_type="spoken",
        text=text,
        distance=0.1,
    )


def test_noop_reranker_preserves_order() -> None:
    candidates = [_result("a", "x"), _result("b", "y"), _result("c", "z")]
    assert [c.chunk_id for c in NoOpReranker().rerank("q", candidates)] == ["a", "b", "c"]


def test_make_reranker_selects_implementation() -> None:
    assert isinstance(make_reranker("none"), NoOpReranker)
    assert isinstance(make_reranker("crossencoder"), CrossEncoderReranker)
    with pytest.raises(ValueError, match="Unknown reranker"):
        make_reranker("bogus")


def test_make_reranker_crossencoder_is_a_singleton() -> None:
    assert make_reranker("crossencoder") is make_reranker("crossencoder")


def test_singleton_reranker_loads_model_once_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.utils.reranker import _crossencoder_singleton

    _crossencoder_singleton.cache_clear()
    counters = {"init": 0, "predict": 0}
    _install_fake_cross_encoder(monkeypatch, counters)

    first = make_reranker("crossencoder")
    second = make_reranker("crossencoder")
    assert first is second

    candidates = [_result("a", "xxx"), _result("b", "xxxxxxx")]
    first.rerank("q1", candidates)
    second.rerank("q2", candidates)
    assert counters["init"] == 1


def _install_fake_cross_encoder(
    monkeypatch: pytest.MonkeyPatch, counters: dict[str, int]
) -> None:
    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            counters["init"] += 1

        def predict(self, pairs: list[tuple[str, str]]) -> list[int]:
            counters["predict"] += 1
            return [len(text) for _query, text in pairs]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_cross_encoder_reorders_by_score_and_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = {"init": 0, "predict": 0}
    _install_fake_cross_encoder(monkeypatch, counters)

    reranker = CrossEncoderReranker("fake-model")
    candidates = [_result("a", "xxx"), _result("b", "xxxxxxx"), _result("c", "xxxxx")]

    first = reranker.rerank("q1", candidates)
    assert [c.chunk_id for c in first] == ["b", "c", "a"]
    assert {c.chunk_id for c in first} == {"a", "b", "c"}

    reranker.rerank("q2", candidates)
    assert counters["init"] == 1
    assert counters["predict"] == 2


def test_cross_encoder_short_circuits_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("the cross-encoder must not load for a 0/1-candidate pool")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = CrossEncoderReranker("fake-model")
    assert reranker.rerank("q", []) == []
    single = [_result("only", "text")]
    assert [c.chunk_id for c in reranker.rerank("q", single)] == ["only"]
