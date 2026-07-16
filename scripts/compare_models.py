
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Literal, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from scripts.chat import (
    DEFAULT_GROQ_MODEL,
    ChatArgs,
    GroqChatBackend,
    RetrievalBackend,
    SupportsGenerate,
    SupportsRetrieve,
    build_empty_context_answer,
    build_prompt_messages,
    load_groq_api_key,
)
from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL
from scripts.evaluate import (
    ConceptGroupResult,
    EvalCase,
    build_failure_reasons,
    evaluate_answer_expectations,
    evaluate_retrieval_expectations,
    load_eval_dataset,
    load_indexed_chunks,
    select_cases,
)
from scripts.utils.pg_store import connect_pg_store

Classification = Literal["REGRESSION", "IMPROVEMENT", "UNCHANGED"]


class CompareArgs(BaseModel):
    baseline_model: str
    candidate_model: str
    eval_file: Path = Path("data/eval_qa.json")
    db_url: str = ""
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    output_dir: Path = Path("output/model_comparison")
    case_ids: list[str] = Field(default_factory=list)
    top_k: int = 5
    max_history_turns: int = 0


class ModelCaseResult(BaseModel):
    model: str
    answer: str
    answer_source: Literal["fallback", "groq"]
    passed: bool
    answer_mode_matched: bool
    concept_group_results: list[ConceptGroupResult] = Field(default_factory=list)
    forbidden_phrase_hits: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


class CaseComparison(BaseModel):
    case_id: str
    student_name: str
    question: str
    expected_answer_mode: Literal["grounded_answer", "insufficient_evidence"]
    classification: Classification
    retrieval_result_count: int
    baseline: ModelCaseResult
    candidate: ModelCaseResult


class ComparisonSummary(BaseModel):
    baseline_model: str
    candidate_model: str
    dataset_path: str
    total_cases: int
    regression_count: int
    improvement_count: int
    unchanged_count: int
    baseline_passed: int
    candidate_passed: int
    regression_case_ids: list[str] = Field(default_factory=list)
    improvement_case_ids: list[str] = Field(default_factory=list)
    case_comparisons: list[CaseComparison] = Field(default_factory=list)


def classify_case(passed_baseline: bool, passed_candidate: bool) -> Classification:
    if passed_baseline and not passed_candidate:
        return "REGRESSION"
    if not passed_baseline and passed_candidate:
        return "IMPROVEMENT"
    return "UNCHANGED"


def parse_args(argv: Sequence[str] | None = None) -> CompareArgs:
    parser = argparse.ArgumentParser(
        description="A/B-compare two Groq generation models over the golden eval set."
    )
    parser.add_argument("--baseline-model", required=True, dest="baseline_model")
    parser.add_argument("--candidate-model", default=DEFAULT_GROQ_MODEL, dest="candidate_model")
    parser.add_argument("--eval-file", default="data/eval_qa.json")
    parser.add_argument("--db-url", default="", dest="db_url")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, dest="embedding_model")
    parser.add_argument("--output-dir", default="output/model_comparison", dest="output_dir")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument("--max-history-turns", type=int, default=0, dest="max_history_turns")
    namespace = parser.parse_args(argv)
    load_dotenv()
    db_url = namespace.db_url or os.getenv("DATABASE_URL", "")
    return CompareArgs(
        baseline_model=namespace.baseline_model,
        candidate_model=namespace.candidate_model,
        eval_file=Path(namespace.eval_file),
        db_url=db_url,
        embedding_model=namespace.embedding_model,
        output_dir=Path(namespace.output_dir),
        case_ids=list(namespace.case_ids or []),
        top_k=namespace.top_k,
        max_history_turns=namespace.max_history_turns,
    )


def validate_inputs(args: CompareArgs) -> None:
    if not args.baseline_model.strip():
        raise ValueError("--baseline-model must not be empty.")
    if not args.candidate_model.strip():
        raise ValueError("--candidate-model must not be empty.")
    if not args.eval_file.exists() or not args.eval_file.is_file():
        raise ValueError(f"Evaluation dataset was not found: {args.eval_file}")
    if args.output_dir.exists() and args.output_dir.is_file():
        raise ValueError(f"Output path must be a directory: {args.output_dir}")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if args.max_history_turns < 0:
        raise ValueError("max_history_turns must be zero or greater.")


def write_json(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _matched_concepts(results: Sequence[ConceptGroupResult]) -> list[str]:
    return [r.matched_phrase for r in results if r.matched and r.matched_phrase is not None]


def _unmatched_concepts(results: Sequence[ConceptGroupResult]) -> list[list[str]]:
    return [r.alternatives for r in results if not r.matched]


def build_comparison_markdown(summary: ComparisonSummary) -> str:
    lines = [
        "# A/B Model Comparison",
        "",
        f"- Baseline model: `{summary.baseline_model}`",
        f"- Candidate model: `{summary.candidate_model}`",
        f"- Dataset: {summary.dataset_path}",
        f"- Total cases: {summary.total_cases}",
        f"- Baseline passed: {summary.baseline_passed} / {summary.total_cases}",
        f"- Candidate passed: {summary.candidate_passed} / {summary.total_cases}",
        "",
        "## Summary",
        "",
        f"- REGRESSION: {summary.regression_count}"
        + (f"  ({', '.join(summary.regression_case_ids)})" if summary.regression_case_ids else ""),
        f"- IMPROVEMENT: {summary.improvement_count}"
        + (f"  ({', '.join(summary.improvement_case_ids)})" if summary.improvement_case_ids else ""),
        f"- UNCHANGED: {summary.unchanged_count}",
        "",
        "## Cases",
        "",
        "| Case | Student | Classification | Baseline | Candidate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for comparison in summary.case_comparisons:
        baseline_status = "pass" if comparison.baseline.passed else "fail"
        candidate_status = "pass" if comparison.candidate.passed else "fail"
        lines.append(
            f"| {comparison.case_id} | {comparison.student_name} | "
            f"{comparison.classification} | {baseline_status} | {candidate_status} |"
        )

    lines.extend(["", "## Case Detail", ""])
    for comparison in summary.case_comparisons:
        lines.extend(
            [
                f"### {comparison.case_id} — {comparison.classification}",
                "",
                f"- Question: {comparison.question}",
                f"- Expected mode: {comparison.expected_answer_mode}",
                f"- Retrieved chunks (shared): {comparison.retrieval_result_count}",
                "",
                f"**Baseline (`{comparison.baseline.model}`, {'pass' if comparison.baseline.passed else 'fail'})**"
                f" — matched: {_matched_concepts(comparison.baseline.concept_group_results) or 'n/a'};"
                f" missing: {_unmatched_concepts(comparison.baseline.concept_group_results) or 'none'};"
                f" forbidden hits: {comparison.baseline.forbidden_phrase_hits or 'none'}",
                "",
                f"> {comparison.baseline.answer}",
                "",
                f"**Candidate (`{comparison.candidate.model}`, {'pass' if comparison.candidate.passed else 'fail'})**"
                f" — matched: {_matched_concepts(comparison.candidate.concept_group_results) or 'n/a'};"
                f" missing: {_unmatched_concepts(comparison.candidate.concept_group_results) or 'none'};"
                f" forbidden hits: {comparison.candidate.forbidden_phrase_hits or 'none'}",
                "",
                f"> {comparison.candidate.answer}",
                "",
            ]
        )
    return "\n".join(lines)


class ComparisonService:
    def __init__(
        self,
        args: CompareArgs,
        *,
        llm_backend: SupportsGenerate | None = None,
        store: Any = None,
        retrieval_backend: SupportsRetrieve | None = None,
    ) -> None:
        self.args = args
        self.dataset = load_eval_dataset(args.eval_file)
        self.selected_cases = select_cases(self.dataset, args.case_ids)
        self.llm_backend: SupportsGenerate = llm_backend or GroqChatBackend(load_groq_api_key())
        self.store: Any = store if store is not None else connect_pg_store(args.db_url)
        self.retrieval_backend: SupportsRetrieve = retrieval_backend or RetrievalBackend(
            store=self.store
        )

    def _chat_args(self, case: EvalCase) -> ChatArgs:
        return ChatArgs(
            db_url=self.args.db_url,
            chunk_types=list(case.chunk_types),
            embedding_model=self.args.embedding_model,
            max_history_turns=self.args.max_history_turns,
            question=case.question,
            student_id=case.student_id,
            student_name=case.student_name,
            top_k=self.args.top_k,
        )

    def _score_model(
        self,
        case: EvalCase,
        model: str,
        answer: str,
        answer_source: Literal["fallback", "groq"],
        retrieval_checks: Any,
        session_trace_path: Path,
    ) -> ModelCaseResult:
        answer_checks = evaluate_answer_expectations(case, answer, answer_source, session_trace_path)
        failure_reasons = build_failure_reasons(case, retrieval_checks, answer_checks)
        return ModelCaseResult(
            model=model,
            answer=answer,
            answer_source=answer_source,
            passed=not failure_reasons,
            answer_mode_matched=answer_checks.answer_mode_matched,
            concept_group_results=answer_checks.concept_group_results,
            forbidden_phrase_hits=answer_checks.forbidden_phrase_hits,
            failure_reasons=failure_reasons,
        )

    def compare_case(self, case: EvalCase) -> CaseComparison:
        chat_args = self._chat_args(case)
        retrieval_result = self.retrieval_backend.retrieve(chat_args, case.question)
        indexed_chunks = load_indexed_chunks(self.store, case)
        retrieval_checks = evaluate_retrieval_expectations(
            case, indexed_chunks, retrieval_result.retrieved_chunks
        )
        session_trace_path = self.args.output_dir / "cases" / f"{case.case_id}.json"

        if retrieval_result.result_count == 0:
            fallback = build_empty_context_answer(case.student_name, retrieval_result)
            baseline = self._score_model(
                case, self.args.baseline_model, fallback, "fallback", retrieval_checks, session_trace_path
            )
            candidate = self._score_model(
                case, self.args.candidate_model, fallback, "fallback", retrieval_checks, session_trace_path
            )
        else:
            prompt_messages = build_prompt_messages(
                student_id=case.student_id,
                student_name=case.student_name,
                question=case.question,
                retrieval_result=retrieval_result,
                history_turns=[],
                max_history_turns=self.args.max_history_turns,
            )
            baseline_answer = self.llm_backend.generate(
                messages=prompt_messages, model=self.args.baseline_model
            )
            candidate_answer = self.llm_backend.generate(
                messages=prompt_messages, model=self.args.candidate_model
            )
            baseline = self._score_model(
                case, self.args.baseline_model, baseline_answer, "groq", retrieval_checks, session_trace_path
            )
            candidate = self._score_model(
                case, self.args.candidate_model, candidate_answer, "groq", retrieval_checks, session_trace_path
            )

        return CaseComparison(
            case_id=case.case_id,
            student_name=case.student_name,
            question=case.question,
            expected_answer_mode=case.expected_answer_mode,
            classification=classify_case(baseline.passed, candidate.passed),
            retrieval_result_count=retrieval_result.result_count,
            baseline=baseline,
            candidate=candidate,
        )

    def run(self) -> ComparisonSummary:
        print(
            f"Comparing baseline={self.args.baseline_model} vs "
            f"candidate={self.args.candidate_model} over {len(self.selected_cases)} case(s)..."
        )
        comparisons: list[CaseComparison] = []
        for case in self.selected_cases:
            print(f"- Comparing {case.case_id} for {case.student_name}...")
            comparison = self.compare_case(case)
            write_json(self.args.output_dir / "cases" / f"{case.case_id}.json", comparison)
            comparisons.append(comparison)

        regressions = [c.case_id for c in comparisons if c.classification == "REGRESSION"]
        improvements = [c.case_id for c in comparisons if c.classification == "IMPROVEMENT"]
        summary = ComparisonSummary(
            baseline_model=self.args.baseline_model,
            candidate_model=self.args.candidate_model,
            dataset_path=str(self.args.eval_file),
            total_cases=len(comparisons),
            regression_count=len(regressions),
            improvement_count=len(improvements),
            unchanged_count=sum(1 for c in comparisons if c.classification == "UNCHANGED"),
            baseline_passed=sum(1 for c in comparisons if c.baseline.passed),
            candidate_passed=sum(1 for c in comparisons if c.candidate.passed),
            regression_case_ids=regressions,
            improvement_case_ids=improvements,
            case_comparisons=comparisons,
        )
        self.write_summary(summary)
        return summary

    def write_summary(self, summary: ComparisonSummary) -> None:
        write_json(self.args.output_dir / "comparison.json", summary)
        markdown_path = self.args.output_dir / "comparison.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(build_comparison_markdown(summary), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_inputs(args)
    summary = ComparisonService(args).run()
    print(
        "\n".join(
            [
                "Comparison complete.",
                f"Baseline passed: {summary.baseline_passed} / {summary.total_cases}",
                f"Candidate passed: {summary.candidate_passed} / {summary.total_cases}",
                f"REGRESSION: {summary.regression_count} {summary.regression_case_ids}",
                f"IMPROVEMENT: {summary.improvement_count} {summary.improvement_case_ids}",
                f"UNCHANGED: {summary.unchanged_count}",
                f"Comparison JSON: {args.output_dir / 'comparison.json'}",
                f"Comparison Markdown: {args.output_dir / 'comparison.md'}",
            ]
        )
    )
    if summary.regression_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
