"""Non-UI logic for the Streamlit teacher-evaluation demo (``app.py``).

Everything here is plain, tested, and Streamlit-free so the UI layer stays a
thin glue shell. It reuses the committed pipeline (retrieval + chat) rather than
reimplementing any of it.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from scripts.chat import (
    DEFAULT_GROQ_MODEL,
    ChatTurnRecord,
    PromptMessage,
    SupportsGenerate,
    build_empty_context_answer,
    build_prompt_messages,
    collect_trust_flags,
    effective_top_k,
    iso_timestamp,
    select_retrieval_chunk_types,
    utc_now,
)
from scripts.retrieval import QueryEmbedder, RetrievalResult, retrieve_from_pgvector

# Per-question retrieval breadth (general questions widen to GENERAL_QUESTION_TOP_K) is
# owned by scripts.chat.effective_top_k, so the demo, CLI, and eval harness widen identically.

_SESSION_SUFFIX_RE = re.compile(r"_s(\d+)$", re.IGNORECASE)
_SUBJECT_TOKEN_RE = re.compile(r"^[A-Za-z]+\.\d+$")
_AY_TOKEN_RE = re.compile(r"^AY\d{4}-\d{2}$", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*",
    re.IGNORECASE,
)


def session_display_label(class_name: str) -> str:
    """Best-effort human label for a ``class_name`` (topic + date) for the session picker.

    Presentation only: the picker still FILTERS on the raw ``class_name``. Strips the
    "Subject.NN", section, and "AY2025-26" tokens, pulls the trailing date out, and keeps
    a trailing ``_s1``/``_s2`` meeting marker. Falls back to the raw value if unparseable.
    """
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
        if len(token) == 1 and token.isalpha():  # stray section letter, e.g. Math "A"
            continue
        kept.append(token)

    date = ""
    if kept and _DATE_RE.search(kept[-1]):
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
    """Summarize a student's stored corpus: class name(s) and chunk count.

    Drives the per-student header in the demo so the teacher sees how much
    grounding data exists before asking anything.
    """
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
    """Highest similarity score among retrieved chunks, or None if unscored/empty."""
    scores = [c.score for c in result.retrieved_chunks if c.score is not None]
    return max(scores) if scores else None


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
    """Retrieve student-scoped context and answer one question.

    Mirrors ``ChatService.ask_question`` without the session-file side effects:
    on zero retrieval it returns the grounded "no evidence" fallback (no Groq
    call); otherwise it builds the same prompt and calls the LLM backend. The
    returned ``ChatTurnRecord`` carries the retrieval result so the UI can show
    the grounding chunks.
    """
    retrieval_result = retrieve_from_pgvector(
        student_id=student_id,
        query=question,
        top_k=effective_top_k(question, top_k),
        chunk_types=select_retrieval_chunk_types(question, ()),
        class_name=class_name,
        db_url=db_url,
        store=store,
        embedder=embedder,
    )
    if retrieval_result.result_count == 0:
        answer = build_empty_context_answer(student_name, retrieval_result)
        answer_source: Literal["fallback", "groq"] = "fallback"
        model_name: str | None = None
        prompt_messages: list[PromptMessage] = []
    else:
        prompt_messages = build_prompt_messages(
            student_id=student_id,
            student_name=student_name,
            question=question,
            retrieval_result=retrieval_result,
            history_turns=history_turns,
            max_history_turns=max_history_turns,
        )
        answer = chat_backend.generate(messages=prompt_messages, model=groq_model)
        answer_source = "groq"
        model_name = groq_model
    return ChatTurnRecord(
        answer=answer,
        answer_source=answer_source,
        asked_at=iso_timestamp(utc_now()),
        model=model_name,
        prompt_messages=prompt_messages,
        question=question,
        retrieval_result=retrieval_result,
        turn_index=len(history_turns) + 1,
        trust_flags=collect_trust_flags(retrieval_result),
    )
