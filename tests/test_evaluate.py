from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest

from scripts.chat import RetrievalBackend
from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL
from scripts.evaluate import BaselineService, EvalCase, EvalDataset, EvaluationArgs, load_eval_dataset
from scripts.models.pipeline import SearchResult
from scripts.retrieval import QueryEmbedder

EVAL_QA_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_qa.json"

CONCEPT_CASE_ID = "bhagyashree-concept-determinants-supply-link"
REFUSAL_CASE_ID = "bhagyashree-concept-not-in-source-refusal"


def load_dataset() -> EvalDataset:
    return load_eval_dataset(EVAL_QA_PATH)


def case_by_id(dataset: EvalDataset, case_id: str) -> EvalCase:
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise AssertionError(f"case_id not found in eval_qa.json: {case_id}")


# --- dataset validates ---


def test_eval_qa_json_validates_under_dataset() -> None:
    dataset = load_dataset()
    assert isinstance(dataset, EvalDataset)
    assert dataset.cases, "eval dataset must have at least one case"


def test_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in load_dataset().cases]
    assert len(case_ids) == len(set(case_ids))


# --- new concept + refusal cases (TASK-021) ---


def test_concept_case_grounds_on_material_chunks() -> None:
    case = case_by_id(load_dataset(), CONCEPT_CASE_ID)
    # The concept question the teacher broke on must ground on 'material' and expect
    # a grounded (not refusal) answer.
    assert case.expected_answer_mode == "grounded_answer"
    assert "material" in case.expected_chunk_types
    # Relationship-level concept groups: a determinant shifts/affects the supply function.
    assert case.required_concept_groups, "concept case needs required_concept_groups"
    flattened = [phrase.casefold() for group in case.required_concept_groups for phrase in group]
    assert any("determinant" in phrase for phrase in flattened)
    assert any("supply" in phrase for phrase in flattened)


def test_concept_case_is_general_not_self_referential() -> None:
    from scripts.chat import is_self_referential_question, select_retrieval_chunk_types

    case = case_by_id(load_dataset(), CONCEPT_CASE_ID)
    # A concept question must route UNFILTERED (so material is retrievable), never scoped
    # to the student's own spoken+chat.
    assert not is_self_referential_question(case.question)
    assert select_retrieval_chunk_types(case.question, ()) == []


def test_not_in_source_case_expects_refusal() -> None:
    case = case_by_id(load_dataset(), REFUSAL_CASE_ID)
    # Guards the TASK-020 no-world-knowledge rule: a known concept absent from every
    # source must be declined, so no grounded concepts are required.
    assert case.expected_answer_mode == "insufficient_evidence"
    assert case.required_concept_groups == []


# --- regression guards remain ---


def test_dataset_keeps_a_self_referential_case() -> None:
    dataset = load_dataset()
    from scripts.chat import is_self_referential_question

    self_referential = [
        case
        for case in dataset.cases
        if case.expected_answer_mode == "grounded_answer"
        and is_self_referential_question(case.question)
    ]
    assert self_referential, "at least one self-referential recall case must remain"
    # Self-referential cases stay scoped to the student's own words (no material bleed).
    for case in self_referential:
        assert set(case.chunk_types) <= {"spoken", "chat"}
        assert "material" not in case.chunk_types


def test_dataset_keeps_a_wrong_subject_refusal_case() -> None:
    dataset = load_dataset()
    refusals = [
        case for case in dataset.cases if case.expected_answer_mode == "insufficient_evidence"
    ]
    # Both the pre-existing wrong-subject refusal and the new not-in-source concept refusal.
    assert len(refusals) >= 2
    assert any(case.case_id == "saisha-wrong-subject-refusal" for case in refusals)


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

    def search(
        self,
        query_embedding: list[float],
        student_id: str,
        top_k: int = 5,
        chunk_types: Sequence[str] | None = None,
        class_name: str | None = None,
    ) -> list[SearchResult]:
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


def make_result(text: str, distance: float, chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        student_id="2302",
        student_name="Bhagyashree",
        class_name="Economics.02",
        chunk_type="class_context",
        text=text,
        distance=distance,
    )


def write_eval_file(tmp_path: Path, cases: list[dict[str, object]]) -> Path:
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps({"description": "test", "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )
    return eval_file


def boom(*args: object, **kwargs: object) -> object:
    raise AssertionError("the LLM backend must never be touched in --baseline mode")


DETERMINANTS_TEXT = (
    "taxes, then technology, price of related goods, complementary goods, determinant"
)

BASELINE_CASES: list[dict[str, object]] = [
    {
        "case_id": "hit",
        "student_id": "2302",
        "student_name": "Bhagyashree",
        "question": "What did the class say about complementary goods?",
        "expected_answer_mode": "grounded_answer",
        "required_concept_groups": [["complementary"]],
        "evidence_quotes": ["complementary goods, determinant"],
    },
    {
        "case_id": "not-indexed",
        "student_id": "2302",
        "student_name": "Bhagyashree",
        "question": "How is GDP calculated in national income accounting?",
        "expected_answer_mode": "grounded_answer",
        "evidence_quotes": ["gross domestic product"],
    },
    {
        "case_id": "refusal",
        "student_id": "2302",
        "student_name": "Bhagyashree",
        "question": "What did I say about the Balance of Payments?",
        "expected_answer_mode": "insufficient_evidence",
    },
]


def make_baseline_store() -> FakeStore:
    indexed = [
        make_result(DETERMINANTS_TEXT, 0.3, "c1"),
        make_result("the supply function and its intercept", 0.5, "c2"),
    ]
    return FakeStore(search_results=indexed, student_chunks=indexed)


def test_baseline_service_scores_cases_without_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("scripts.evaluate.GroqChatBackend", boom)
    monkeypatch.setattr("scripts.evaluate.load_groq_api_key", boom)

    eval_file = write_eval_file(tmp_path, BASELINE_CASES)
    store = make_baseline_store()
    args = EvaluationArgs(
        eval_file=eval_file,
        db_url="postgresql://localhost/db",
        output_dir=tmp_path / "out",
        top_k=5,
    )
    service = BaselineService(
        args,
        store=store,
        retrieval_backend=RetrievalBackend(store=store, embedder=make_embedder()),
    )

    snapshot = service.run()
    by_id = {result.case_id: result for result in snapshot.case_results}

    assert by_id["hit"].metrics.status == "scorable"
    assert by_id["hit"].metrics.hit
    assert by_id["hit"].metrics.first_hit_rank == 1
    assert by_id["not-indexed"].metrics.status == "quote_not_indexed"
    assert not by_id["not-indexed"].metrics.hit
    assert by_id["refusal"].metrics.status == "refusal"

    aggregate = snapshot.retrieval_aggregate
    assert aggregate.scorable_case_count == 1
    assert aggregate.hit_count == 1
    assert aggregate.recall_at_k == 1.0
    assert aggregate.mrr == 1.0
    assert aggregate.quote_not_indexed_case_count == 1
    assert aggregate.refusal_case_count == 1
    assert snapshot.quote_not_indexed_case_ids == ["not-indexed"]


def test_baseline_snapshot_files_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("scripts.evaluate.GroqChatBackend", boom)

    eval_file = write_eval_file(tmp_path, BASELINE_CASES[:1])
    store = make_baseline_store()
    args = EvaluationArgs(
        eval_file=eval_file,
        db_url="postgresql://localhost/db",
        output_dir=tmp_path / "out",
        top_k=5,
    )
    BaselineService(
        args,
        store=store,
        retrieval_backend=RetrievalBackend(store=store, embedder=make_embedder()),
    ).run()

    assert (tmp_path / "out" / "baseline_snapshot.json").is_file()
    assert (tmp_path / "out" / "baseline_snapshot.md").is_file()


def test_main_baseline_flag_routes_retrieval_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.evaluate import main

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("scripts.evaluate.EvaluationService", boom)
    monkeypatch.setattr("scripts.evaluate.GroqChatBackend", boom)
    store = make_baseline_store()
    monkeypatch.setattr("scripts.evaluate.connect_pg_store", lambda db_url: store)
    monkeypatch.setattr("scripts.chat.QueryEmbedder", lambda *a, **k: make_embedder())

    eval_file = write_eval_file(tmp_path, BASELINE_CASES)
    main(
        [
            "--baseline",
            "--eval-file",
            str(eval_file),
            "--db-url",
            "postgresql://localhost/db",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert (tmp_path / "out" / "baseline_snapshot.json").is_file()
