from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from scripts.chat import PromptMessage, RetrievalBackend, is_self_referential_question
from scripts.compare_models import (
    CaseComparison,
    CompareArgs,
    ComparisonService,
    ModelCaseResult,
    build_comparison_markdown,
    classify_case,
)
from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL
from scripts.evaluate import ConceptGroupResult, load_eval_dataset
from scripts.models.pipeline import SearchResult
from scripts.retrieval import QueryEmbedder

GENERAL_GOLDEN_IDS = {
    "saisha-general-what-covered",
    "saisha-general-teacher-instruction",
    "bhagyashree-general-determinants",
}


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


class FakeBackend:
    def __init__(self, answer: str = "The class covered the supply function.") -> None:
        self._answer = answer
        self.calls: list[tuple[Sequence[PromptMessage], str]] = []

    def generate(self, *, messages: Sequence[PromptMessage], model: str) -> str:
        self.calls.append((messages, model))
        return self._answer


def make_search_result(text: str = "supply function intercept beta") -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        student_id="2302",
        student_name="Bhagyashree",
        class_name="Economics.02",
        chunk_type="class_context",
        text=text,
        distance=0.3,
        start_time=10.0,
        end_time=20.0,
        speaker="Nisha",
    )


def write_eval_file(tmp_path: Path, cases: list[dict]) -> Path:
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps({"description": "test", "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )
    return eval_file


def test_classify_case_covers_all_transitions() -> None:
    assert classify_case(True, False) == "REGRESSION"
    assert classify_case(False, True) == "IMPROVEMENT"
    assert classify_case(True, True) == "UNCHANGED"
    assert classify_case(False, False) == "UNCHANGED"


def _model_result(model: str, passed: bool) -> ModelCaseResult:
    return ModelCaseResult(
        model=model,
        answer="a",
        answer_source="groq",
        passed=passed,
        answer_mode_matched=passed,
        concept_group_results=[ConceptGroupResult(alternatives=["x"], matched=passed)],
    )


def test_markdown_reports_summary_and_classifications() -> None:
    from scripts.compare_models import ComparisonSummary

    comparison = CaseComparison(
        case_id="c1",
        student_name="B",
        question="q",
        expected_answer_mode="grounded_answer",
        classification="REGRESSION",
        retrieval_result_count=3,
        baseline=_model_result("old", True),
        candidate=_model_result("new", False),
    )
    summary = ComparisonSummary(
        baseline_model="old",
        candidate_model="new",
        dataset_path="data/eval_qa.json",
        total_cases=1,
        regression_count=1,
        improvement_count=0,
        unchanged_count=0,
        baseline_passed=1,
        candidate_passed=0,
        regression_case_ids=["c1"],
        case_comparisons=[comparison],
    )
    md = build_comparison_markdown(summary)
    assert "REGRESSION: 1" in md
    assert "c1" in md
    assert "REGRESSION" in md


def test_compare_case_retrieves_once_and_scores_both_models(tmp_path: Path) -> None:
    eval_file = write_eval_file(
        tmp_path,
        [
            {
                "case_id": "c1",
                "student_id": "2302",
                "student_name": "Bhagyashree",
                "question": "What did we cover in class today?",
                "expected_answer_mode": "grounded_answer",
                "required_concept_groups": [["supply"]],
            }
        ],
    )
    store = FakeStore(
        search_results=[make_search_result()],
        student_chunks=[make_search_result()],
    )
    backend = FakeBackend()
    args = CompareArgs(
        baseline_model="old-model",
        candidate_model="new-model",
        eval_file=eval_file,
        db_url="postgresql://localhost/db",
        output_dir=tmp_path / "out",
    )
    service = ComparisonService(
        args,
        llm_backend=backend,
        store=store,
        retrieval_backend=RetrievalBackend(store=store, embedder=make_embedder()),
    )

    comparison = service.compare_case(service.selected_cases[0])

    assert len(store.search_calls) == 1
    assert len(backend.calls) == 2
    assert [model for _, model in backend.calls] == ["old-model", "new-model"]
    assert comparison.baseline.model == "old-model"
    assert comparison.candidate.model == "new-model"
    assert comparison.classification == "UNCHANGED"
    assert comparison.retrieval_result_count == 1


def test_compare_case_zero_retrieval_uses_shared_fallback_no_generate(tmp_path: Path) -> None:
    eval_file = write_eval_file(
        tmp_path,
        [
            {
                "case_id": "refusal",
                "student_id": "2302",
                "student_name": "Bhagyashree",
                "question": "What did I say about the Balance of Payments?",
                "expected_answer_mode": "insufficient_evidence",
            }
        ],
    )
    store = FakeStore(search_results=[], student_chunks=[])
    backend = FakeBackend()
    args = CompareArgs(
        baseline_model="old-model",
        candidate_model="new-model",
        eval_file=eval_file,
        db_url="postgresql://localhost/db",
        output_dir=tmp_path / "out",
    )
    service = ComparisonService(
        args,
        llm_backend=backend,
        store=store,
        retrieval_backend=RetrievalBackend(store=store, embedder=make_embedder()),
    )

    comparison = service.compare_case(service.selected_cases[0])

    assert backend.calls == []
    assert comparison.baseline.answer == comparison.candidate.answer
    assert comparison.classification == "UNCHANGED"


def test_general_golden_cases_load_and_validate() -> None:
    dataset = load_eval_dataset(Path("data/eval_qa.json"))
    ids = {case.case_id for case in dataset.cases}
    assert GENERAL_GOLDEN_IDS <= ids

    for case in dataset.cases:
        if case.case_id in GENERAL_GOLDEN_IDS:
            assert not is_self_referential_question(case.question)
            assert case.expected_answer_mode == "grounded_answer"
            assert case.required_concept_groups
            assert not case.chunk_types
