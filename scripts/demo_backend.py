
from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from pydantic import BaseModel, Field

from scripts.chat import (
    DEFAULT_GROQ_MODEL,
    ChatTurnRecord,
    SupportsGenerate,
    answer_turn,
    is_class_overview_question,
    resolve_student_classes,
    select_retrieval_chunk_types,
    utc_now,
)
from scripts.retrieval import QueryEmbedder, RetrievalResult, retrieve_from_pgvector
from scripts.utils.class_date import CLASS_DATE_RE

logger = logging.getLogger(__name__)

OVERVIEW_TOP_K = 12

_SESSION_SUFFIX_RE = re.compile(r"_s(\d+)$", re.IGNORECASE)
_SUBJECT_TOKEN_RE = re.compile(r"^[A-Za-z]+\.\d+$")
_AY_TOKEN_RE = re.compile(r"^AY\d{4}-\d{2}$", re.IGNORECASE)
_AY_SPLIT_RE = re.compile(r"_\s*AY\d{2,4}-\d{2}", re.IGNORECASE)


def section_of_class(class_name: str) -> str:
    head = _AY_SPLIT_RE.split(class_name)[0]
    section = " ".join(head.replace("_", " ").split())
    return section or class_name


def students_by_section(
    pairs: Sequence[tuple[str, str, str]],
) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, dict[str, str]] = {}
    for student_id, student_name, class_name in pairs:
        grouped.setdefault(section_of_class(class_name), {})[student_id] = student_name
    return {
        section: sorted(students.items(), key=lambda item: (item[1].lower(), item[0]))
        for section, students in sorted(grouped.items())
    }


def session_display_label(class_name: str) -> str:
    raw = class_name
    session_suffix = ""
    suffix_match = _SESSION_SUFFIX_RE.search(raw)
    if suffix_match:
        session_suffix = f" (s{suffix_match.group(1)})"
        raw = raw[: suffix_match.start()]

    kept: list[str] = []
    for token in (t.strip() for t in raw.split("_")):
        if not token or _AY_TOKEN_RE.match(token) or _SUBJECT_TOKEN_RE.match(token):
            continue
        if len(token) == 1 and token.isalpha():
            continue
        kept.append(token)

    date = ""
    if kept and CLASS_DATE_RE.search(kept[-1]):
        date = kept.pop()
    topic = re.sub(r"\s+", " ", " ".join(kept)).strip()

    if topic and date:
        return f"{topic} — {date}{session_suffix}"
    if date:
        return f"{date}{session_suffix}"
    return f"{topic or class_name}{session_suffix}"


class StudentSummary(BaseModel):
    student_id: str
    student_name: str
    class_names: list[str] = Field(default_factory=list)
    chunk_count: int = 0


def student_summary(store: Any, student_id: str, student_name: str = "") -> StudentSummary:
    chunks = store.get_student_chunks(student_id)
    class_names = sorted({c.class_name for c in chunks if c.class_name})
    resolved_name = student_name or (chunks[0].student_name if chunks else student_id)
    return StudentSummary(
        student_id=student_id,
        student_name=resolved_name,
        class_names=class_names,
        chunk_count=len(chunks),
    )


def top_score(result: RetrievalResult) -> float | None:
    scores = [c.score for c in result.retrieved_chunks if c.score is not None]
    return max(scores) if scores else None


def log_turn(
    store: Any,
    *,
    student_id: str,
    grade: str,
    answer_source: str,
    scoped_class: str | None,
    question_len: int,
) -> None:
    logger_fn = getattr(store, "log_query", None)
    if not callable(logger_fn):
        return
    try:
        logger_fn(student_id, grade, answer_source, scoped_class, question_len)
    except Exception as exc:
        logger.warning("Query telemetry hook failed for %r: %s", student_id, exc)


def answer_for_student(
    *,
    student_id: str,
    student_name: str,
    question: str,
    store: Any,
    embedder: QueryEmbedder,
    chat_backend: SupportsGenerate,
    db_url: str,
    class_name: str | None = None,
    history_turns: Sequence[ChatTurnRecord] = (),
    top_k: int = 5,
    groq_model: str = DEFAULT_GROQ_MODEL,
    max_history_turns: int = 3,
) -> ChatTurnRecord:
    if is_class_overview_question(question):
        top_k = max(top_k, OVERVIEW_TOP_K)
    retrieval_result = retrieve_from_pgvector(
        student_id=student_id,
        query=question,
        top_k=top_k,
        chunk_types=select_retrieval_chunk_types(question, ()),
        class_name=class_name,
        db_url=db_url,
        store=store,
        embedder=embedder,
    )
    turn = answer_turn(
        student_id=student_id,
        student_name=student_name,
        question=question,
        retrieval_result=retrieval_result,
        llm_backend=chat_backend,
        groq_model=groq_model,
        history_turns=history_turns,
        max_history_turns=max_history_turns,
        student_classes=resolve_student_classes(store, student_id),
        now=utc_now(),
        turn_index=len(history_turns) + 1,
    )
    log_turn(
        store,
        student_id=student_id,
        grade=turn.grade,
        answer_source=turn.answer_source,
        scoped_class=class_name,
        question_len=len(question),
    )
    return turn
