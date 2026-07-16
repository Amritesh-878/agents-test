from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, Field

from scripts.chat import (
    DEFAULT_GROQ_MODEL,
    ChatArgs,
    ChatService,
    GroqChatBackend,
    RetrievalBackend,
    SupportsRetrieve,
    iso_timestamp,
    load_groq_api_key,
    utc_now,
)
from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL
from scripts.retrieval import RetrievedChunk, search_result_to_chunk
from scripts.utils.chunker import ChunkType
from scripts.utils.pg_store import PgVectorStore, connect_pg_store
from scripts.utils.reranker import DEFAULT_RERANKER
from scripts.utils.retrieval_grade import RouterTier, grade_retrieval
from scripts.utils.retrieval_metrics import (
    AggregateRetrievalMetrics,
    CaseRetrievalMetrics,
    aggregate_metrics,
    compute_case_metrics,
    tier_distribution,
)

INSUFFICIENT_EVIDENCE_HINTS = (
    "do not have enough",
    "don't have enough",
    "insufficient evidence",
    "not enough evidence",
    "cannot answer that reliably",
    "can't answer that reliably",
    "cannot determine",
    "can't determine",
    "context is insufficient",
)

TRUST_ACKNOWLEDGEMENT_HINTS = (
    "estimated",
    "low-confidence",
    "low confidence",
    "approximate",
    "uncertain",
    "manual review",
    "not reliable",
    "cannot answer that reliably",
    "can't answer that reliably",
)


class EvalCase(BaseModel):
    case_id: str
    student_id: str
    student_name: str
    question: str
    expected_answer_mode: Literal["grounded_answer", "insufficient_evidence"]
    chunk_types: list[ChunkType] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_chunk_types: list[ChunkType] = Field(default_factory=list)
    expected_source_segment_ids: list[str] = Field(default_factory=list)
    expected_trust_flags: list[str] = Field(default_factory=list)
    required_concept_groups: list[list[str]] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    require_trust_acknowledgement: bool = False
    evidence_quotes: list[str] = Field(default_factory=list)
    source_artifacts: list[str] = Field(default_factory=list)
    notes: str | None = None


class EvalDataset(BaseModel):
    description: str
    generated_from: list[str] = Field(default_factory=list)
    cases: list[EvalCase]


class EvaluationArgs(BaseModel):
    eval_file: Path = Path("data/eval_qa.json")
    db_url: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    groq_model: str = DEFAULT_GROQ_MODEL
    output_dir: Path = Path("output/evaluation")
    case_ids: list[str] = Field(default_factory=list)
    top_k: int = 5
    max_history_turns: int = 0
    fail_on_case_failure: bool = False
    baseline: bool = False
    label: str = ""


class ConceptGroupResult(BaseModel):
    alternatives: list[str]
    matched: bool
    matched_phrase: str | None = None


class RetrievalCheckResult(BaseModel):
    indexed_chunk_count: int
    indexed_chunk_types: list[str] = Field(default_factory=list)
    indexed_expected_chunk_ids_found: list[str] = Field(default_factory=list)
    indexed_expected_source_segment_ids_found: list[str] = Field(default_factory=list)
    indexed_expected_trust_flags_found: list[str] = Field(default_factory=list)
    indexed_expected_evidence_present: bool
    indexed_expected_chunk_types_present: bool
    retrieval_result_count: int
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_types: list[str] = Field(default_factory=list)
    retrieved_expected_chunk_ids_found: list[str] = Field(default_factory=list)
    retrieved_expected_source_segment_ids_found: list[str] = Field(default_factory=list)
    retrieved_expected_trust_flags_found: list[str] = Field(default_factory=list)
    retrieved_expected_evidence_present: bool
    retrieved_expected_chunk_types_present: bool
    retrieved_trust_flags: list[str] = Field(default_factory=list)
    provenance_risk_detected: bool


class AnswerCheckResult(BaseModel):
    answer: str
    answer_mode_matched: bool
    answer_source: Literal["fallback", "groq"]
    concept_group_results: list[ConceptGroupResult] = Field(default_factory=list)
    forbidden_phrase_hits: list[str] = Field(default_factory=list)
    insufficient_evidence_acknowledged: bool
    trust_acknowledged: bool
    session_trace_path: str


class CaseEvaluationResult(BaseModel):
    case: EvalCase
    passed: bool
    failure_stage: Literal[
        "none",
        "chunking",
        "retrieval",
        "chat_generation",
        "upstream_low_confidence_provenance",
    ]
    failure_reasons: list[str] = Field(default_factory=list)
    retrieval_checks: RetrievalCheckResult
    retrieval_metrics: CaseRetrievalMetrics
    router_grade: RouterTier
    answer_checks: AnswerCheckResult


class EvaluationSummary(BaseModel):
    dataset_path: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    selected_case_ids: list[str] = Field(default_factory=list)
    failure_stage_counts: dict[str, int] = Field(default_factory=dict)
    retrieval_aggregate: AggregateRetrievalMetrics
    router_tier_distribution: dict[str, int] = Field(default_factory=dict)
    router_low_tier_rate: float = 0.0
    quote_not_indexed_case_ids: list[str] = Field(default_factory=list)
    case_results: list[CaseEvaluationResult] = Field(default_factory=list)


def parse_args(argv: Sequence[str] | None = None) -> EvaluationArgs:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline against a golden QA set."
    )
    parser.add_argument("--eval-file", default="data/eval_qa.json")
    parser.add_argument("--db-url", default="", dest="db_url")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, dest="embedding_model")
    parser.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL, dest="groq_model")
    parser.add_argument("--output-dir", default="output/evaluation", dest="output_dir")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument("--max-history-turns", type=int, default=0, dest="max_history_turns")
    parser.add_argument("--fail-on-case-fail", action="store_true", dest="fail_on_case_failure")
    parser.add_argument("--baseline", action="store_true", dest="baseline")
    parser.add_argument("--label", default="", dest="label")
    namespace = parser.parse_args(argv)
    from dotenv import load_dotenv
    import os
    load_dotenv()
    db_url = namespace.db_url or os.getenv("DATABASE_URL", "")
    return EvaluationArgs(
        eval_file=Path(namespace.eval_file),
        db_url=db_url,
        embedding_model=namespace.embedding_model,
        groq_model=namespace.groq_model,
        output_dir=Path(namespace.output_dir),
        case_ids=list(namespace.case_ids or []),
        top_k=namespace.top_k,
        max_history_turns=namespace.max_history_turns,
        fail_on_case_failure=namespace.fail_on_case_failure,
        baseline=namespace.baseline,
        label=namespace.label,
    )


def validate_inputs(args: EvaluationArgs) -> None:
    if not args.eval_file.exists() or not args.eval_file.is_file():
        raise ValueError(f"Evaluation dataset was not found: {args.eval_file}")
    if args.output_dir.exists() and args.output_dir.is_file():
        raise ValueError(f"Output path must be a directory: {args.output_dir}")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if args.max_history_turns < 0:
        raise ValueError("max_history_turns must be zero or greater.")


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def unique_strings(values: Sequence[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduplicated.append(value)
        seen.add(value)
    return deduplicated


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)


def collect_chunk_types(chunks: Sequence[RetrievedChunk]) -> list[str]:
    return unique_strings([str(chunk.chunk_type) for chunk in chunks])


def collect_chunk_ids(chunks: Sequence[RetrievedChunk]) -> list[str]:
    return [chunk.chunk_id for chunk in chunks]


def collect_source_segment_ids(chunks: Sequence[RetrievedChunk]) -> list[str]:
    segment_ids: list[str] = []
    for chunk in chunks:
        segment_ids.extend(chunk.source_segment_ids)
    return unique_strings(segment_ids)


def collect_trust_flags(chunks: Sequence[RetrievedChunk]) -> list[str]:
    trust_flags: list[str] = []
    for chunk in chunks:
        trust_flags.extend(chunk.trust_flags)
    return unique_strings(trust_flags)


def all_expected_found(expected_values: Sequence[str], found_values: Sequence[str]) -> bool:
    if not expected_values:
        return True
    found_set = set(found_values)
    return all(value in found_set for value in expected_values)


def all_expected_chunk_types_present(expected_values: Sequence[ChunkType], found_values: Sequence[str]) -> bool:
    if not expected_values:
        return True
    found_set = set(found_values)
    return all(chunk_type in found_set for chunk_type in expected_values)


def match_concept_group(answer: str, alternatives: Sequence[str]) -> ConceptGroupResult:
    for alternative in alternatives:
        if contains_phrase(answer, alternative):
            return ConceptGroupResult(
                alternatives=list(alternatives),
                matched=True,
                matched_phrase=alternative,
            )
    return ConceptGroupResult(alternatives=list(alternatives), matched=False)


def answer_acknowledges_insufficient_evidence(answer: str) -> bool:
    return any(contains_phrase(answer, phrase) for phrase in INSUFFICIENT_EVIDENCE_HINTS)


def answer_acknowledges_trust(answer: str) -> bool:
    return any(contains_phrase(answer, phrase) for phrase in TRUST_ACKNOWLEDGEMENT_HINTS)


def load_eval_dataset(path: Path) -> EvalDataset:
    return EvalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def load_indexed_chunks(store: Any, case: EvalCase) -> list[RetrievedChunk]:
    raw_results = store.get_student_chunks(case.student_id)
    if case.chunk_types:
        raw_results = [r for r in raw_results if r.chunk_type in case.chunk_types]
    return [search_result_to_chunk(r, i + 1) for i, r in enumerate(raw_results)]


def case_retrieval_metrics(
    case: EvalCase,
    indexed_chunks: Sequence[RetrievedChunk],
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    k: int,
) -> CaseRetrievalMetrics:
    return compute_case_metrics(
        quotes=case.evidence_quotes,
        retrieved_texts=[chunk.text for chunk in retrieved_chunks],
        indexed_texts=[chunk.text for chunk in indexed_chunks],
        scores=[chunk.score for chunk in retrieved_chunks],
        is_refusal=case.expected_answer_mode == "insufficient_evidence",
        k=k,
    )


def router_tier_summary(grades: Sequence[RouterTier]) -> tuple[dict[str, int], float]:
    distribution = tier_distribution(list(grades))
    low_rate = round(distribution["low"] / len(grades), 6) if grades else 0.0
    return distribution, low_rate


def build_router_tier_lines(distribution: dict[str, int], low_tier_rate: float) -> list[str]:
    return [
        f"- Router-tier distribution: {distribution}",
        f"- Router low-tier fire-rate: {low_tier_rate}",
    ]


def select_cases(dataset: EvalDataset, case_ids: Sequence[str]) -> list[EvalCase]:
    if not case_ids:
        return list(dataset.cases)
    selected_ids = {case_id.casefold() for case_id in case_ids}
    selected_cases = [case for case in dataset.cases if case.case_id.casefold() in selected_ids]
    found_case_ids = {case.case_id.casefold() for case in selected_cases}
    missing_case_ids = [case_id for case_id in case_ids if case_id.casefold() not in found_case_ids]
    if missing_case_ids:
        raise ValueError(f"Requested case ids were not found in the dataset: {missing_case_ids}")
    return selected_cases


def evaluate_retrieval_expectations(
    case: EvalCase,
    indexed_chunks: Sequence[RetrievedChunk],
    retrieved_chunks: Sequence[RetrievedChunk],
) -> RetrievalCheckResult:
    indexed_chunk_ids = collect_chunk_ids(indexed_chunks)
    indexed_chunk_types = collect_chunk_types(indexed_chunks)
    indexed_source_segment_ids = collect_source_segment_ids(indexed_chunks)
    indexed_trust_flags = collect_trust_flags(indexed_chunks)
    retrieved_chunk_ids = collect_chunk_ids(retrieved_chunks)
    retrieved_chunk_types = collect_chunk_types(retrieved_chunks)
    retrieved_source_segment_ids = collect_source_segment_ids(retrieved_chunks)
    retrieved_trust_flags = collect_trust_flags(retrieved_chunks)

    indexed_expected_chunk_ids_found = [
        chunk_id for chunk_id in case.expected_chunk_ids if chunk_id in set(indexed_chunk_ids)
    ]
    indexed_expected_source_segment_ids_found = [
        segment_id
        for segment_id in case.expected_source_segment_ids
        if segment_id in set(indexed_source_segment_ids)
    ]
    indexed_expected_trust_flags_found = [
        trust_flag for trust_flag in case.expected_trust_flags if trust_flag in set(indexed_trust_flags)
    ]
    retrieved_expected_chunk_ids_found = [
        chunk_id for chunk_id in case.expected_chunk_ids if chunk_id in set(retrieved_chunk_ids)
    ]
    retrieved_expected_source_segment_ids_found = [
        segment_id
        for segment_id in case.expected_source_segment_ids
        if segment_id in set(retrieved_source_segment_ids)
    ]
    retrieved_expected_trust_flags_found = [
        trust_flag for trust_flag in case.expected_trust_flags if trust_flag in set(retrieved_trust_flags)
    ]

    indexed_expected_chunk_types_present = all_expected_chunk_types_present(
        case.expected_chunk_types,
        indexed_chunk_types,
    )
    retrieved_expected_chunk_types_present = all_expected_chunk_types_present(
        case.expected_chunk_types,
        retrieved_chunk_types,
    )
    indexed_expected_evidence_present = all(
        [
            all_expected_found(case.expected_chunk_ids, indexed_expected_chunk_ids_found),
            all_expected_found(
                case.expected_source_segment_ids,
                indexed_expected_source_segment_ids_found,
            ),
            all_expected_found(case.expected_trust_flags, indexed_expected_trust_flags_found),
            indexed_expected_chunk_types_present,
        ]
    )
    retrieved_expected_evidence_present = all(
        [
            all_expected_found(case.expected_chunk_ids, retrieved_expected_chunk_ids_found),
            all_expected_found(
                case.expected_source_segment_ids,
                retrieved_expected_source_segment_ids_found,
            ),
            all_expected_found(case.expected_trust_flags, retrieved_expected_trust_flags_found),
            retrieved_expected_chunk_types_present,
        ]
    )

    return RetrievalCheckResult(
        indexed_chunk_count=len(indexed_chunks),
        indexed_chunk_types=indexed_chunk_types,
        indexed_expected_chunk_ids_found=indexed_expected_chunk_ids_found,
        indexed_expected_source_segment_ids_found=indexed_expected_source_segment_ids_found,
        indexed_expected_trust_flags_found=indexed_expected_trust_flags_found,
        indexed_expected_evidence_present=indexed_expected_evidence_present,
        indexed_expected_chunk_types_present=indexed_expected_chunk_types_present,
        retrieval_result_count=len(retrieved_chunks),
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_chunk_types=retrieved_chunk_types,
        retrieved_expected_chunk_ids_found=retrieved_expected_chunk_ids_found,
        retrieved_expected_source_segment_ids_found=retrieved_expected_source_segment_ids_found,
        retrieved_expected_trust_flags_found=retrieved_expected_trust_flags_found,
        retrieved_expected_evidence_present=retrieved_expected_evidence_present,
        retrieved_expected_chunk_types_present=retrieved_expected_chunk_types_present,
        retrieved_trust_flags=retrieved_trust_flags,
        provenance_risk_detected=bool(retrieved_trust_flags),
    )


def evaluate_answer_expectations(
    case: EvalCase,
    answer: str,
    answer_source: Literal["fallback", "groq"],
    session_trace_path: Path,
) -> AnswerCheckResult:
    concept_group_results = [
        match_concept_group(answer, concept_group) for concept_group in case.required_concept_groups
    ]
    forbidden_phrase_hits = [
        forbidden_phrase
        for forbidden_phrase in case.forbidden_phrases
        if contains_phrase(answer, forbidden_phrase)
    ]
    insufficient_evidence_acknowledged = answer_acknowledges_insufficient_evidence(answer)
    trust_acknowledged = answer_acknowledges_trust(answer)
    if not case.require_trust_acknowledgement:
        trust_acknowledged = True

    if case.expected_answer_mode == "insufficient_evidence":
        answer_mode_matched = insufficient_evidence_acknowledged
    else:
        answer_mode_matched = all(result.matched for result in concept_group_results)

    return AnswerCheckResult(
        answer=answer,
        answer_mode_matched=answer_mode_matched,
        answer_source=answer_source,
        concept_group_results=concept_group_results,
        forbidden_phrase_hits=forbidden_phrase_hits,
        insufficient_evidence_acknowledged=insufficient_evidence_acknowledged,
        trust_acknowledged=trust_acknowledged,
        session_trace_path=str(session_trace_path),
    )


def build_failure_reasons(
    case: EvalCase,
    retrieval_checks: RetrievalCheckResult,
    answer_checks: AnswerCheckResult,
) -> list[str]:
    reasons: list[str] = []
    if not retrieval_checks.indexed_expected_evidence_present:
        reasons.append("Expected evidence is missing from the indexed student chunk universe.")
    if not retrieval_checks.retrieved_expected_evidence_present:
        reasons.append("Retrieved top-k chunks did not contain the expected evidence for this query.")
    if not answer_checks.answer_mode_matched:
        if case.expected_answer_mode == "insufficient_evidence":
            reasons.append("Answer did not clearly acknowledge insufficient evidence.")
        else:
            reasons.append("Answer did not cover all required grounded concepts.")
    if answer_checks.forbidden_phrase_hits:
        reasons.append(f"Answer used forbidden phrases: {answer_checks.forbidden_phrase_hits}")
    if case.require_trust_acknowledgement and not answer_checks.trust_acknowledged:
        reasons.append("Answer did not acknowledge estimated or low-confidence provenance.")
    return reasons


def classify_failure_stage(
    retrieval_checks: RetrievalCheckResult,
    answer_checks: AnswerCheckResult,
) -> Literal[
    "none",
    "chunking",
    "retrieval",
    "chat_generation",
    "upstream_low_confidence_provenance",
]:
    if (
        retrieval_checks.indexed_expected_evidence_present
        and retrieval_checks.retrieved_expected_evidence_present
        and answer_checks.answer_mode_matched
        and answer_checks.trust_acknowledged
        and not answer_checks.forbidden_phrase_hits
    ):
        return "none"
    if not retrieval_checks.indexed_expected_evidence_present:
        return "chunking"
    if not retrieval_checks.retrieved_expected_evidence_present:
        return "retrieval"
    if retrieval_checks.provenance_risk_detected and not answer_checks.trust_acknowledged:
        return "upstream_low_confidence_provenance"
    return "chat_generation"


def write_json(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def build_retrieval_metrics_lines(
    aggregate: AggregateRetrievalMetrics,
    quote_not_indexed_case_ids: Sequence[str],
) -> list[str]:
    lines = [
        f"- recall@{aggregate.k}: {aggregate.recall_at_k}",
        f"- MRR: {aggregate.mrr}",
        f"- Scorable cases: {aggregate.scorable_case_count} / {aggregate.total_cases}",
        f"- Retrieved beyond k (rerank opportunity): {aggregate.retrieved_beyond_k_count}",
        f"- Refusal cases (excluded from recall/MRR): {aggregate.refusal_case_count}",
        f"- quote_not_indexed (excluded, ingest gap): {aggregate.quote_not_indexed_case_count}",
        f"- Tier distribution: {aggregate.tier_distribution}",
        f"- Low-tier rate: {aggregate.low_tier_rate}",
    ]
    if quote_not_indexed_case_ids:
        lines.append(f"- quote_not_indexed case ids: {', '.join(quote_not_indexed_case_ids)}")
    return lines


def build_summary_markdown(summary: EvaluationSummary) -> str:
    lines = [
        "# RAG Evaluation Summary",
        "",
        f"- Dataset: {summary.dataset_path}",
        f"- Total cases: {summary.total_cases}",
        f"- Passed: {summary.passed_cases}",
        f"- Failed: {summary.failed_cases}",
        "",
        "## Failure Stage Counts",
        "",
    ]
    if summary.failure_stage_counts:
        for stage, count in summary.failure_stage_counts.items():
            lines.append(f"- {stage}: {count}")
    else:
        lines.append("- none: 0")

    lines.extend(["", "## Retrieval Metrics (quote-containment)", ""])
    lines.extend(
        build_retrieval_metrics_lines(
            summary.retrieval_aggregate, summary.quote_not_indexed_case_ids
        )
    )

    lines.extend(["", "## Router Tiers (confidence routing)", ""])
    lines.extend(
        build_router_tier_lines(summary.router_tier_distribution, summary.router_low_tier_rate)
    )

    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| Case | Student | Passed | Failure Stage | Status | Tier | Router | First-hit rank | Hit@k |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in summary.case_results:
        status_text = "yes" if result.passed else "no"
        metric = result.retrieval_metrics
        rank_text = "n/a" if metric.first_hit_rank is None else str(metric.first_hit_rank)
        hit_text = "yes" if metric.hit else "no"
        lines.append(
            f"| {result.case.case_id} | {result.case.student_name} | {status_text} | "
            f"{result.failure_stage} | {metric.status} | {metric.tier} | {result.router_grade} | "
            f"{rank_text} | {hit_text} |"
        )
    return "\n".join(lines)


class EvaluationService:
    def __init__(
        self,
        args: EvaluationArgs,
        *,
        llm_backend: GroqChatBackend | None = None,
        store: PgVectorStore | None = None,
    ) -> None:
        self.args = args
        self.dataset = load_eval_dataset(args.eval_file)
        self.selected_cases = select_cases(self.dataset, args.case_ids)
        api_key = load_groq_api_key()
        self.llm_backend = llm_backend or GroqChatBackend(api_key)
        self.store: PgVectorStore = store or connect_pg_store(args.db_url)

    def load_indexed_chunks(self, case: EvalCase) -> list[RetrievedChunk]:
        return load_indexed_chunks(self.store, case)

    def run_case(self, case: EvalCase) -> CaseEvaluationResult:
        chat_args = ChatArgs(
            db_url=self.args.db_url,
            chunk_types=list(case.chunk_types),
            embedding_model=self.args.embedding_model,
            groq_model=self.args.groq_model,
            max_history_turns=self.args.max_history_turns,
            question=case.question,
            save_session_dir=self.args.output_dir / "chat_sessions",
            student_id=case.student_id,
            student_name=case.student_name,
            top_k=self.args.top_k,
        )
        chat_service = ChatService(
            chat_args,
            llm_backend=self.llm_backend,
            session_id=case.case_id,
        )
        turn_record = chat_service.ask_question(case.question)
        indexed_chunks = self.load_indexed_chunks(case)
        retrieved_chunks = turn_record.retrieval_result.retrieved_chunks
        retrieval_checks = evaluate_retrieval_expectations(case, indexed_chunks, retrieved_chunks)
        retrieval_metrics = case_retrieval_metrics(
            case, indexed_chunks, retrieved_chunks, k=self.args.top_k
        )
        answer_checks = evaluate_answer_expectations(
            case,
            turn_record.answer,
            turn_record.answer_source,
            chat_service.session_path,
        )
        failure_reasons = build_failure_reasons(case, retrieval_checks, answer_checks)
        failure_stage = classify_failure_stage(retrieval_checks, answer_checks)
        passed = not failure_reasons and failure_stage == "none"
        return CaseEvaluationResult(
            case=case,
            passed=passed,
            failure_stage=failure_stage,
            failure_reasons=failure_reasons,
            retrieval_checks=retrieval_checks,
            retrieval_metrics=retrieval_metrics,
            router_grade=turn_record.grade,
            answer_checks=answer_checks,
        )

    def write_case_artifact(self, result: CaseEvaluationResult) -> None:
        case_artifact_path = self.args.output_dir / "cases" / f"{result.case.case_id}.json"
        write_json(case_artifact_path, result)
        if result.passed:
            return
        failed_case_path = self.args.output_dir / "failed_cases" / f"{result.case.case_id}.json"
        write_json(failed_case_path, result)

    def write_summary(self, summary: EvaluationSummary) -> None:
        write_json(self.args.output_dir / "summary.json", summary)
        summary_markdown = build_summary_markdown(summary)
        summary_path = self.args.output_dir / "summary.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_markdown, encoding="utf-8")

    def run(self) -> EvaluationSummary:
        print(f"Evaluating {len(self.selected_cases)} case(s) from {self.args.eval_file}...")
        case_results: list[CaseEvaluationResult] = []
        for case in self.selected_cases:
            print(f"- Evaluating {case.case_id} for {case.student_name}...")
            case_result = self.run_case(case)
            self.write_case_artifact(case_result)
            case_results.append(case_result)

        failure_stage_counts = Counter(
            case_result.failure_stage for case_result in case_results if case_result.failure_stage != "none"
        )
        router_distribution, router_low_rate = router_tier_summary(
            [case_result.router_grade for case_result in case_results]
        )
        summary = EvaluationSummary(
            dataset_path=str(self.args.eval_file),
            total_cases=len(case_results),
            passed_cases=sum(1 for case_result in case_results if case_result.passed),
            failed_cases=sum(1 for case_result in case_results if not case_result.passed),
            selected_case_ids=[case.case_id for case in self.selected_cases],
            failure_stage_counts=dict(failure_stage_counts),
            retrieval_aggregate=aggregate_metrics(
                [case_result.retrieval_metrics for case_result in case_results], k=self.args.top_k
            ),
            router_tier_distribution=router_distribution,
            router_low_tier_rate=router_low_rate,
            quote_not_indexed_case_ids=[
                case_result.case.case_id
                for case_result in case_results
                if case_result.retrieval_metrics.status == "quote_not_indexed"
            ],
            case_results=case_results,
        )
        self.write_summary(summary)
        return summary


class CaseBaselineResult(BaseModel):
    case_id: str
    student_id: str
    student_name: str
    question: str
    expected_answer_mode: Literal["grounded_answer", "insufficient_evidence"]
    evidence_quotes: list[str] = Field(default_factory=list)
    retrieval_result_count: int
    indexed_chunk_count: int
    metrics: CaseRetrievalMetrics
    router_grade: RouterTier


class BaselineSnapshot(BaseModel):
    dataset_path: str
    embedding_model: str
    top_k: int
    captured_at: str
    dataset_sha256: str
    reranker: str
    store_total_rows: int
    selected_case_ids: list[str] = Field(default_factory=list)
    retrieval_aggregate: AggregateRetrievalMetrics
    router_tier_distribution: dict[str, int] = Field(default_factory=dict)
    router_low_tier_rate: float = 0.0
    quote_not_indexed_case_ids: list[str] = Field(default_factory=list)
    case_results: list[CaseBaselineResult] = Field(default_factory=list)


def build_baseline_markdown(snapshot: BaselineSnapshot) -> str:
    lines = [
        "# Retrieval Baseline Snapshot (TASK-026)",
        "",
        "Retrieval-only capture — no Groq. Transcribe these numbers into docs/PROGRESS.md.",
        "",
        f"- Dataset: {snapshot.dataset_path}",
        f"- Dataset sha256: {snapshot.dataset_sha256}",
        f"- Embedding model: {snapshot.embedding_model}",
        f"- Reranker: {snapshot.reranker}",
        f"- Captured at: {snapshot.captured_at}",
        f"- Store total rows: {snapshot.store_total_rows}",
        f"- top_k: {snapshot.top_k}",
    ]
    lines.extend(
        build_retrieval_metrics_lines(
            snapshot.retrieval_aggregate, snapshot.quote_not_indexed_case_ids
        )
    )
    lines.extend(
        build_router_tier_lines(snapshot.router_tier_distribution, snapshot.router_low_tier_rate)
    )
    lines.extend(
        [
            "",
            "## Per-case",
            "",
            "| Case | Status | Tier | First-hit rank | Hit@k | Top score | Retrieved | Indexed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in snapshot.case_results:
        metric = result.metrics
        rank_text = "n/a" if metric.first_hit_rank is None else str(metric.first_hit_rank)
        top_text = "n/a" if metric.top_score is None else f"{metric.top_score:.4f}"
        hit_text = "yes" if metric.hit else "no"
        lines.append(
            f"| {result.case_id} | {metric.status} | {metric.tier} | {rank_text} | {hit_text} | "
            f"{top_text} | {result.retrieval_result_count} | {result.indexed_chunk_count} |"
        )
    return "\n".join(lines)


class BaselineService:
    def __init__(
        self,
        args: EvaluationArgs,
        *,
        store: Any = None,
        retrieval_backend: SupportsRetrieve | None = None,
        now_provider: Callable[[], datetime] = utc_now,
    ) -> None:
        self.args = args
        self.now_provider = now_provider
        self.dataset = load_eval_dataset(args.eval_file)
        self.selected_cases = select_cases(self.dataset, args.case_ids)
        self.store: Any = store if store is not None else connect_pg_store(args.db_url)
        self.retrieval_backend: SupportsRetrieve = retrieval_backend or RetrievalBackend(
            store=self.store
        )

    def _chat_args(self, case: EvalCase) -> ChatArgs:
        return ChatArgs(
            db_url=self.args.db_url,
            chunk_types=list(case.chunk_types),
            embedding_model=self.args.embedding_model,
            max_history_turns=0,
            question=case.question,
            student_id=case.student_id,
            student_name=case.student_name,
            top_k=self.args.top_k,
        )

    def run_case(self, case: EvalCase) -> CaseBaselineResult:
        retrieval_result = self.retrieval_backend.retrieve(self._chat_args(case), case.question)
        indexed_chunks = load_indexed_chunks(self.store, case)
        metrics = case_retrieval_metrics(
            case, indexed_chunks, retrieval_result.retrieved_chunks, k=self.args.top_k
        )
        return CaseBaselineResult(
            case_id=case.case_id,
            student_id=case.student_id,
            student_name=case.student_name,
            question=case.question,
            expected_answer_mode=case.expected_answer_mode,
            evidence_quotes=list(case.evidence_quotes),
            retrieval_result_count=retrieval_result.result_count,
            indexed_chunk_count=len(indexed_chunks),
            metrics=metrics,
            router_grade=grade_retrieval(retrieval_result.retrieved_chunks),
        )

    def run(self) -> BaselineSnapshot:
        print(
            f"Capturing retrieval baseline for {len(self.selected_cases)} case(s) "
            "(retrieval-only, no Groq)..."
        )
        case_results: list[CaseBaselineResult] = []
        for case in self.selected_cases:
            print(f"- Retrieving {case.case_id} for {case.student_name}...")
            case_results.append(self.run_case(case))

        router_distribution, router_low_rate = router_tier_summary(
            [result.router_grade for result in case_results]
        )
        snapshot = BaselineSnapshot(
            dataset_path=str(self.args.eval_file),
            embedding_model=self.args.embedding_model,
            top_k=self.args.top_k,
            captured_at=iso_timestamp(self.now_provider()),
            dataset_sha256=hashlib.sha256(self.args.eval_file.read_bytes()).hexdigest(),
            reranker=DEFAULT_RERANKER,
            store_total_rows=self.store.count_chunks(),
            selected_case_ids=[case.case_id for case in self.selected_cases],
            retrieval_aggregate=aggregate_metrics(
                [result.metrics for result in case_results], k=self.args.top_k
            ),
            router_tier_distribution=router_distribution,
            router_low_tier_rate=router_low_rate,
            quote_not_indexed_case_ids=[
                result.case_id
                for result in case_results
                if result.metrics.status == "quote_not_indexed"
            ],
            case_results=case_results,
        )
        self.write_snapshot(snapshot)
        return snapshot

    def write_snapshot(self, snapshot: BaselineSnapshot) -> None:
        suffix = f"_{self.args.label}" if self.args.label else ""
        write_json(self.args.output_dir / f"baseline_snapshot{suffix}.json", snapshot)
        markdown_path = self.args.output_dir / f"baseline_snapshot{suffix}.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(build_baseline_markdown(snapshot), encoding="utf-8")


def run_baseline(args: EvaluationArgs) -> BaselineSnapshot:
    snapshot = BaselineService(args).run()
    aggregate = snapshot.retrieval_aggregate
    suffix = f"_{args.label}" if args.label else ""
    json_path = args.output_dir / f"baseline_snapshot{suffix}.json"
    markdown_path = args.output_dir / f"baseline_snapshot{suffix}.md"
    print(
        "\n".join(
            [
                "Baseline capture complete (retrieval-only, no Groq).",
                f"recall@{aggregate.k}: {aggregate.recall_at_k}",
                f"MRR: {aggregate.mrr}",
                f"Scorable cases: {aggregate.scorable_case_count} / {aggregate.total_cases}",
                f"Tier distribution: {aggregate.tier_distribution}",
                f"Low-tier rate: {aggregate.low_tier_rate}",
                f"Router-tier distribution: {snapshot.router_tier_distribution}",
                f"Router low-tier fire-rate: {snapshot.router_low_tier_rate}",
                f"quote_not_indexed: {aggregate.quote_not_indexed_case_count} "
                f"{snapshot.quote_not_indexed_case_ids}",
                f"Snapshot JSON: {json_path}",
                f"Snapshot Markdown: {markdown_path}",
            ]
        )
    )
    return snapshot


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_inputs(args)
    if args.baseline:
        run_baseline(args)
        return
    service = EvaluationService(args)
    summary = service.run()
    print(
        "\n".join(
            [
                "Evaluation complete.",
                f"Passed cases: {summary.passed_cases}",
                f"Failed cases: {summary.failed_cases}",
                f"Summary JSON: {args.output_dir / 'summary.json'}",
                f"Summary Markdown: {args.output_dir / 'summary.md'}",
            ]
        )
    )
    if args.fail_on_case_failure and summary.failed_cases > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()