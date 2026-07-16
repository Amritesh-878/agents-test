from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from scripts.demo_backend import section_of_class, students_by_section
from scripts.utils.db_url import resolve_db_url

logger = logging.getLogger(__name__)

PresenceMode = Literal["absent", "audio", "chat-only", "attendance-only", "materials-only", "unknown"]

GRADES = ("high", "medium", "low")

_CHUNK_TYPES = ("spoken", "chat", "class_context", "material", "missed")


class SessionMetrics(BaseModel):
    class_name: str
    section: str
    presence_mode: PresenceMode
    chunk_counts: dict[str, int] = Field(default_factory=dict)
    has_materials: bool = False


class StudentMetrics(BaseModel):
    student_id: str
    student_name: str
    sessions: list[SessionMetrics] = Field(default_factory=list)
    sessions_on_record: int = 0
    sessions_by_mode: dict[str, int] = Field(default_factory=dict)
    spoken_chunk_total: int = 0
    chat_chunk_total: int = 0


class TierRates(BaseModel):
    total: int = 0
    counts_by_grade: dict[str, int] = Field(default_factory=dict)
    rates_by_grade: dict[str, float] = Field(default_factory=dict)
    counts_by_answer_source: dict[str, int] = Field(default_factory=dict)
    rates_by_answer_source: dict[str, float] = Field(default_factory=dict)
    low_tier_rate: float = 0.0


def derive_presence_mode(chunk_counts: dict[str, int], has_absent_status: bool) -> PresenceMode:
    if has_absent_status:
        return "absent"
    if chunk_counts.get("spoken", 0) > 0:
        return "audio"
    if chunk_counts.get("chat", 0) > 0:
        return "chat-only"
    if chunk_counts.get("class_context", 0) > 0:
        return "attendance-only"
    if chunk_counts.get("material", 0) > 0:
        return "materials-only"
    return "unknown"


def session_metrics_from_chunks(class_name: str, chunks: Sequence[Any]) -> SessionMetrics:
    counts: Counter[str] = Counter()
    has_absent_status = False
    for chunk in chunks:
        counts[chunk.chunk_type] += 1
        metadata = getattr(chunk, "metadata", None) or {}
        if metadata.get("status") == "absent":
            has_absent_status = True
    chunk_counts = {ctype: counts.get(ctype, 0) for ctype in _CHUNK_TYPES if counts.get(ctype, 0)}
    return SessionMetrics(
        class_name=class_name,
        section=section_of_class(class_name),
        presence_mode=derive_presence_mode(chunk_counts, has_absent_status),
        chunk_counts=chunk_counts,
        has_materials=chunk_counts.get("material", 0) > 0,
    )


def build_student_metrics(
    student_id: str, student_name: str, chunks: Sequence[Any]
) -> StudentMetrics:
    by_class: dict[str, list[Any]] = {}
    for chunk in chunks:
        if not chunk.class_name:
            continue
        by_class.setdefault(chunk.class_name, []).append(chunk)

    sessions = [
        session_metrics_from_chunks(class_name, class_chunks)
        for class_name, class_chunks in sorted(by_class.items())
    ]
    resolved_name = student_name
    if not resolved_name:
        named = [c.student_name for c in chunks if getattr(c, "student_name", "")]
        resolved_name = named[0] if named else student_id

    modes: Counter[str] = Counter(s.presence_mode for s in sessions)
    return StudentMetrics(
        student_id=student_id,
        student_name=resolved_name,
        sessions=sessions,
        sessions_on_record=len(sessions),
        sessions_by_mode=dict(sorted(modes.items())),
        spoken_chunk_total=sum(s.chunk_counts.get("spoken", 0) for s in sessions),
        chat_chunk_total=sum(s.chunk_counts.get("chat", 0) for s in sessions),
    )


def student_metrics(store: Any, student_id: str, student_name: str = "") -> StudentMetrics:
    return build_student_metrics(student_id, student_name, store.get_student_chunks(student_id))


def section_metrics(store: Any, section: str) -> list[StudentMetrics]:
    grouped = students_by_section(store.list_student_class_pairs())
    results: list[StudentMetrics] = []
    for student_id, student_name in grouped.get(section, []):
        metrics = student_metrics(store, student_id, student_name)
        scoped = [s for s in metrics.sessions if s.section == section]
        modes: Counter[str] = Counter(s.presence_mode for s in scoped)
        results.append(
            metrics.model_copy(
                update={
                    "sessions": scoped,
                    "sessions_on_record": len(scoped),
                    "sessions_by_mode": dict(sorted(modes.items())),
                    "spoken_chunk_total": sum(s.chunk_counts.get("spoken", 0) for s in scoped),
                    "chat_chunk_total": sum(s.chunk_counts.get("chat", 0) for s in scoped),
                }
            )
        )
    return results


def rates_from_counts(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: round(value / total, 4) for key, value in counts.items()}


def build_tier_rates(rows: Sequence[tuple[str, str, int]]) -> TierRates:
    by_grade: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    for grade, answer_source, count in rows:
        by_grade[grade] += count
        by_source[answer_source] += count
    total = sum(by_grade.values())
    counts_by_grade = {g: by_grade.get(g, 0) for g in GRADES}
    for grade, count in by_grade.items():
        if grade not in counts_by_grade:
            counts_by_grade[grade] = count
    rates_by_grade = rates_from_counts(counts_by_grade, total)
    counts_by_source = dict(sorted(by_source.items()))
    return TierRates(
        total=total,
        counts_by_grade=counts_by_grade,
        rates_by_grade=rates_by_grade,
        counts_by_answer_source=counts_by_source,
        rates_by_answer_source=rates_from_counts(counts_by_source, total),
        low_tier_rate=rates_by_grade.get("low", 0.0),
    )


def tier_rates(
    store: Any,
    *,
    student_id: str | None = None,
    section: str | None = None,
    since: datetime | None = None,
) -> TierRates:
    student_ids: list[str] | None = None
    if student_id is not None:
        student_ids = [student_id]
    elif section is not None:
        grouped = students_by_section(store.list_student_class_pairs())
        student_ids = [sid for sid, _ in grouped.get(section, [])]
    rows = store.fetch_query_stats(student_ids=student_ids, since=since)
    return build_tier_rates(rows)


def render_students_markdown(students: Sequence[StudentMetrics]) -> str:
    lines = [
        "| student_id | student_name | session | section | presence_mode | spoken | chat | materials |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for student in students:
        if not student.sessions:
            lines.append(
                f"| {student.student_id} | {student.student_name} | (no sessions on record) "
                f"| — | — | 0 | 0 | no |"
            )
            continue
        for session in student.sessions:
            lines.append(
                f"| {student.student_id} | {student.student_name} | {session.class_name} "
                f"| {session.section} | {session.presence_mode} "
                f"| {session.chunk_counts.get('spoken', 0)} "
                f"| {session.chunk_counts.get('chat', 0)} "
                f"| {'yes' if session.has_materials else 'no'} |"
            )
    return "\n".join(lines)


def render_tier_markdown(rates: TierRates) -> str:
    lines = [
        f"Answered turns: {rates.total}",
        "",
        "| grade | count | rate |",
        "| --- | --- | --- |",
    ]
    for grade, count in rates.counts_by_grade.items():
        lines.append(f"| {grade} | {count} | {rates.rates_by_grade.get(grade, 0.0):.4f} |")
    lines.append("")
    lines.append("| answer_source | count | rate |")
    lines.append("| --- | --- | --- |")
    for source, count in rates.counts_by_answer_source.items():
        lines.append(f"| {source} | {count} | {rates.rates_by_answer_source.get(source, 0.0):.4f} |")
    return "\n".join(lines)


def students_to_json(students: Sequence[StudentMetrics]) -> str:
    return json.dumps([s.model_dump(mode="json") for s in students], indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.metrics",
        description="Descriptive per-student metrics and query-tier telemetry (no AI verdicts).",
    )
    parser.add_argument("--student", dest="student", default=None)
    parser.add_argument("--section", dest="section", default=None)
    parser.add_argument("--tier-report", dest="tier_report", action="store_true")
    parser.add_argument("--since", dest="since", default=None)
    parser.add_argument("--db-url", dest="db_url", default=None)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", dest="as_json", action="store_true")
    output.add_argument("--markdown", dest="as_markdown", action="store_true")
    return parser


def parse_since(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"--since must be an ISO timestamp, got {value!r}.") from exc


def run_cli(
    args: argparse.Namespace,
    *,
    store: Any,
    output_fn: Any = print,
) -> int:
    since = parse_since(args.since)
    if args.tier_report:
        rates = tier_rates(store, student_id=args.student, section=args.section, since=since)
        if args.as_json:
            output_fn(json.dumps(rates.model_dump(mode="json"), indent=2))
        else:
            output_fn(render_tier_markdown(rates))
        return 0

    if args.student:
        students = [student_metrics(store, args.student)]
    elif args.section:
        students = section_metrics(store, args.section)
    else:
        return 2

    if args.as_json:
        output_fn(students_to_json(students))
    else:
        output_fn(render_students_markdown(students))
    return 0


def main(argv: Sequence[str] | None = None, *, store: Any = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.tier_report and not args.student and not args.section:
        parser.error("Pass --student, --section, or --tier-report.")
    resolved_store = store
    if resolved_store is None:
        from scripts.utils.pg_store import connect_pg_store

        resolved_store = connect_pg_store(resolve_db_url(args.db_url))
    try:
        return run_cli(args, store=resolved_store)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
