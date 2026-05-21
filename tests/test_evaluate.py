from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate import (
    AnswerCheckResult,
    CaseEvaluationResult,
    ConceptGroupResult,
    EvalCase,
    EvaluationArgs,
    EvaluationService,
    EvaluationSummary,
    RetrievalCheckResult,
    build_summary_markdown,
    classify_failure_stage,
    evaluate_answer_expectations,
    evaluate_retrieval_expectations,
)
from scripts.retrieval import RetrievedChunk
from scripts.utils.chunker import SourceSegmentReference


def build_chunk(
    *,
    chunk_id: str = "student-a:missed:0001",
    chunk_type: str = "missed",
    source_segment_ids: list[str] | None = None,
    trust_flags: list[str] | None = None,
) -> RetrievedChunk:
    segment_ids = source_segment_ids or ["seg-0001"]
    return RetrievedChunk(
        approximate=True,
        attendance_accuracy="estimated",
        attendance_estimated=True,
        attendance_source_mode="duration_only_estimated",
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        distance=0.2,
        duration_seconds=12.0,
        end=12.0,
        participant_kind="student",
        rank=1,
        score=0.833333,
        source_manual_review_required=True,
        source_mapped_student="Teacher",
        source_mapping_confidence="low",
        source_segment_count=1,
        source_segment_ids=segment_ids,
        source_segment_indices=[1],
        source_segment_refs=[
            SourceSegmentReference(
                end=12.0,
                segment_id=segment_ids[0],
                segment_index=1,
                source_speaker="SPEAKER_00",
                start=0.0,
                text="Teacher assigned the worksheet and thread reply.",
            )
        ],
        source_speaker="SPEAKER_00",
        start=0.0,
        student_email="student@example.com",
        student_id="student-a",
        student_manual_review_required=True,
        student_mapped_speaker=None,
        student_mapping_confidence=None,
        student_name="Student A",
        text="Teacher assigned the worksheet and thread reply.",
        trust_flags=trust_flags or ["attendance_estimated", "source_mapping_confidence=low"],
    )


def build_case(*, answer_mode: str = "grounded_answer") -> EvalCase:
    return EvalCase(
        case_id="case-001",
        student_id="student-a",
        student_name="Student A",
        question="What did I miss?",
        expected_answer_mode=answer_mode,
        chunk_types=["missed"],
        expected_source_segment_ids=["seg-0001"],
        expected_chunk_types=["missed"],
        expected_trust_flags=["attendance_estimated"],
        required_concept_groups=[["worksheet"], ["thread"]] if answer_mode == "grounded_answer" else [],
        require_trust_acknowledgement=True,
    )


def test_retrieval_expectations_identify_chunking_failure() -> None:
    case = build_case()

    retrieval_checks = evaluate_retrieval_expectations(
        case,
        indexed_chunks=[],
        retrieved_chunks=[],
    )

    assert retrieval_checks.indexed_expected_evidence_present is False
    assert classify_failure_stage(
        retrieval_checks,
        AnswerCheckResult(
            answer="",
            answer_mode_matched=False,
            answer_source="fallback",
            concept_group_results=[],
            forbidden_phrase_hits=[],
            insufficient_evidence_acknowledged=False,
            trust_acknowledged=False,
            session_trace_path="output/evaluation/chat_sessions/case-001.json",
        ),
    ) == "chunking"


def test_retrieval_expectations_identify_retrieval_failure() -> None:
    case = build_case()
    indexed_chunk = build_chunk()
    retrieved_chunk = build_chunk(chunk_id="student-a:missed:0002", source_segment_ids=["seg-9999"])

    retrieval_checks = evaluate_retrieval_expectations(
        case,
        indexed_chunks=[indexed_chunk],
        retrieved_chunks=[retrieved_chunk],
    )

    assert retrieval_checks.indexed_expected_evidence_present is True
    assert retrieval_checks.retrieved_expected_evidence_present is False
    assert classify_failure_stage(
        retrieval_checks,
        AnswerCheckResult(
            answer="Teacher mentioned something else.",
            answer_mode_matched=False,
            answer_source="groq",
            concept_group_results=[],
            forbidden_phrase_hits=[],
            insufficient_evidence_acknowledged=False,
            trust_acknowledged=True,
            session_trace_path="output/evaluation/chat_sessions/case-001.json",
        ),
    ) == "retrieval"


def test_answer_expectations_accept_insufficient_evidence_with_trust_acknowledgement() -> None:
    case = build_case(answer_mode="insufficient_evidence")

    answer_checks = evaluate_answer_expectations(
        case,
        "I do not have enough evidence to answer that reliably because the retrieved context is estimated and low confidence.",
        "groq",
        Path("output/evaluation/chat_sessions/case-001.json"),
    )

    assert answer_checks.answer_mode_matched is True
    assert answer_checks.trust_acknowledged is True
    assert answer_checks.insufficient_evidence_acknowledged is True


def test_summary_markdown_and_failed_case_artifact_are_written(tmp_path: Path) -> None:
    case = build_case()
    result = CaseEvaluationResult(
        case=case,
        passed=False,
        failure_stage="chat_generation",
        failure_reasons=["Answer did not cover all required grounded concepts."],
        retrieval_checks=RetrievalCheckResult(
            indexed_chunk_count=1,
            indexed_chunk_types=["missed"],
            indexed_expected_chunk_ids_found=[],
            indexed_expected_source_segment_ids_found=["seg-0001"],
            indexed_expected_trust_flags_found=["attendance_estimated"],
            indexed_expected_evidence_present=True,
            indexed_expected_chunk_types_present=True,
            retrieval_result_count=1,
            retrieved_chunk_ids=["student-a:missed:0001"],
            retrieved_chunk_types=["missed"],
            retrieved_expected_chunk_ids_found=[],
            retrieved_expected_source_segment_ids_found=["seg-0001"],
            retrieved_expected_trust_flags_found=["attendance_estimated"],
            retrieved_expected_evidence_present=True,
            retrieved_expected_chunk_types_present=True,
            retrieved_trust_flags=["attendance_estimated"],
            provenance_risk_detected=True,
        ),
        answer_checks=AnswerCheckResult(
            answer="Short answer.",
            answer_mode_matched=False,
            answer_source="groq",
            concept_group_results=[
                ConceptGroupResult(alternatives=["worksheet"], matched=False),
                ConceptGroupResult(alternatives=["thread"], matched=False),
            ],
            forbidden_phrase_hits=[],
            insufficient_evidence_acknowledged=False,
            trust_acknowledged=True,
            session_trace_path="output/evaluation/chat_sessions/case-001.json",
        ),
    )
    summary = EvaluationSummary(
        dataset_path="data/eval_qa.json",
        total_cases=1,
        passed_cases=0,
        failed_cases=1,
        selected_case_ids=[case.case_id],
        failure_stage_counts={"chat_generation": 1},
        case_results=[result],
    )

    service = object.__new__(EvaluationService)
    service.args = EvaluationArgs(output_dir=tmp_path)
    service.write_case_artifact(result)
    service.write_summary(summary)

    failed_case_path = tmp_path / "failed_cases" / "case-001.json"
    assert failed_case_path.exists()
    failed_payload = json.loads(failed_case_path.read_text(encoding="utf-8"))
    assert failed_payload["failure_stage"] == "chat_generation"

    summary_markdown = build_summary_markdown(summary)
    assert "| case-001 | Student A | no | chat_generation |" in summary_markdown
    assert "chat_generation: 1" in summary_markdown