from __future__ import annotations

from pathlib import Path

from scripts.evaluate import EvalCase, EvalDataset, load_eval_dataset

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
