"""Non-UI logic for the Streamlit teacher-evaluation demo (``app.py``).

Everything here is plain, tested, and Streamlit-free so the UI layer stays a
thin glue shell. It reuses the committed pipeline (retrieval + chat) rather than
reimplementing any of it.
"""

from __future__ import annotations

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
    iso_timestamp,
    select_retrieval_chunk_types,
    utc_now,
)
from scripts.retrieval import QueryEmbedder, RetrievalResult, retrieve_from_pgvector


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
        top_k=top_k,
        chunk_types=select_retrieval_chunk_types(question, ()),
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
