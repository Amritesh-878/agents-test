from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path
from typing import Any, Sequence

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from scripts.auth import (
    Principal,
    allowed_student_ids,
    can_access_student,
    load_teacher_sections,
    principal_from_identity,
)
from scripts.chat import ChatError, GroqChatBackend, SupportsGenerate, load_groq_api_key
from scripts.demo_backend import answer_for_student, section_of_class, session_display_label
from scripts.retrieval import QueryEmbedder, RetrievalResult
from scripts.utils.db_url import resolve_db_url

logger = logging.getLogger(__name__)

SERVICE_TOKEN_HEADER = "X-Service-Token"
RATE_LIMIT_RETRY_SECONDS = 20

_TEACHER_SECTIONS_ENV = "TEACHER_SECTIONS_CSV"
_SERVICE_TOKEN_ENV = "CHATBOT_SERVICE_TOKEN"


class AskRequest(BaseModel):
    email: str
    lms_role: str
    question: str
    class_name: str | None = None
    student_id: str | None = None


class SourceItem(BaseModel):
    rank: int
    chunk_type: str
    score: float | None = None
    speaker: str = ""
    start: float = 0.0
    end: float = 0.0
    text: str = ""


class AskResponse(BaseModel):
    student_id: str
    student_name: str
    question: str
    answer: str
    grade: str
    answer_source: str
    sources: list[SourceItem] = Field(default_factory=list)


class StudentItem(BaseModel):
    student_id: str
    student_name: str
    sections: list[str] = Field(default_factory=list)


class SessionItem(BaseModel):
    class_name: str
    label: str


def rate_limit_error_types() -> tuple[type[BaseException], ...]:
    try:
        from groq import RateLimitError
    except ImportError:
        return ()
    return (RateLimitError,)


def sources_from_result(result: RetrievalResult) -> list[SourceItem]:
    return [
        SourceItem(
            rank=chunk.rank,
            chunk_type=chunk.chunk_type,
            score=chunk.score,
            speaker=chunk.source_speaker,
            start=chunk.start,
            end=chunk.end,
            text=chunk.text,
        )
        for chunk in result.retrieved_chunks
    ]


def resolve_service_token(service_token: str | None) -> str:
    token = service_token if service_token is not None else os.getenv(_SERVICE_TOKEN_ENV, "")
    if not token.strip():
        raise ChatError(
            f"{_SERVICE_TOKEN_ENV} is empty. Set a shared service secret before starting the API."
        )
    return token


def resolve_teacher_sections(
    teacher_sections: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    if teacher_sections is not None:
        return teacher_sections
    path = os.getenv(_TEACHER_SECTIONS_ENV, "").strip()
    if not path:
        return {}
    return load_teacher_sections(Path(path))


def student_name_for(pairs: Sequence[tuple[str, str, str]], student_id: str) -> str:
    for pid, name, _ in pairs:
        if pid == student_id and name:
            return name
    return student_id


def create_app(
    *,
    store: Any | None = None,
    embedder: QueryEmbedder | None = None,
    chat_backend: SupportsGenerate | None = None,
    teacher_sections: dict[str, list[str]] | None = None,
    service_token: str | None = None,
) -> FastAPI:
    expected_token = resolve_service_token(service_token)
    sections = resolve_teacher_sections(teacher_sections)
    db_url = resolve_db_url(None) if store is None else ""

    app = FastAPI(title="Adira Learning Assistant Service")
    resolved_store: Any | None = store
    resolved_embedder: QueryEmbedder | None = embedder
    resolved_backend: SupportsGenerate | None = chat_backend

    def require_service_token(x_service_token: str = Header(default="")) -> None:
        if not hmac.compare_digest(x_service_token.encode("utf-8"), expected_token.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Invalid service token.")

    def get_store() -> Any:
        nonlocal resolved_store
        if resolved_store is None:
            from scripts.utils.pg_store import connect_pg_store

            resolved_store = connect_pg_store(db_url)
        return resolved_store

    def get_embedder() -> QueryEmbedder:
        nonlocal resolved_embedder
        if resolved_embedder is None:
            from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL

            resolved_embedder = QueryEmbedder(DEFAULT_EMBEDDING_MODEL)
        return resolved_embedder

    def get_chat_backend() -> SupportsGenerate:
        nonlocal resolved_backend
        if resolved_backend is None:
            resolved_backend = GroqChatBackend(load_groq_api_key())
        return resolved_backend

    def require_principal(email: str, lms_role: str) -> Principal:
        principal = principal_from_identity(email, lms_role, teacher_sections=sections)
        if principal is None:
            raise HTTPException(status_code=403, detail="Not authorized for this service.")
        return principal

    def resolve_target(
        principal: Principal, requested_student_id: str | None, pairs: Sequence[tuple[str, str, str]]
    ) -> str:
        if principal.role == "student":
            if not principal.student_id:
                raise HTTPException(status_code=403, detail="Not authorized for this student.")
            return principal.student_id
        if not requested_student_id:
            raise HTTPException(status_code=400, detail="student_id is required for teachers.")
        if not can_access_student(principal, requested_student_id, pairs):
            raise HTTPException(status_code=403, detail="Not authorized for this student.")
        return requested_student_id

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_service_token)])
    def ask(body: AskRequest) -> AskResponse:
        principal = require_principal(body.email, body.lms_role)
        store_obj = get_store()
        pairs = store_obj.list_student_class_pairs()
        target_id = resolve_target(principal, body.student_id, pairs)
        try:
            turn = answer_for_student(
                student_id=target_id,
                student_name=student_name_for(pairs, target_id),
                question=body.question,
                store=store_obj,
                embedder=get_embedder(),
                chat_backend=get_chat_backend(),
                db_url=db_url,
                class_name=body.class_name,
            )
        except rate_limit_error_types() as exc:
            logger.warning("Groq rate limit hit: %s", exc)
            raise HTTPException(
                status_code=503,
                detail={"retry_after_seconds": RATE_LIMIT_RETRY_SECONDS},
            ) from exc
        except Exception as exc:
            logger.exception("Answer generation failed for %r", target_id)
            raise HTTPException(status_code=502, detail="Upstream answer generation failed.") from exc
        return AskResponse(
            student_id=target_id,
            student_name=student_name_for(pairs, target_id),
            question=body.question,
            answer=turn.answer,
            grade=turn.grade,
            answer_source=turn.answer_source,
            sources=sources_from_result(turn.retrieval_result),
        )

    @app.get(
        "/students",
        response_model=list[StudentItem],
        dependencies=[Depends(require_service_token)],
    )
    def students(email: str, lms_role: str) -> list[StudentItem]:
        principal = require_principal(email, lms_role)
        if principal.role != "teacher":
            raise HTTPException(status_code=403, detail="Teacher role required.")
        pairs = get_store().list_student_class_pairs()
        allowed = allowed_student_ids(principal, pairs)
        own_sections = set(principal.sections)
        by_id: dict[str, StudentItem] = {}
        for student_id, student_name, class_name in pairs:
            if student_id not in allowed:
                continue
            section = section_of_class(class_name)
            if section not in own_sections:
                continue
            item = by_id.get(student_id)
            if item is None:
                by_id[student_id] = StudentItem(
                    student_id=student_id,
                    student_name=student_name or student_id,
                    sections=[section],
                )
            elif section not in item.sections:
                item.sections.append(section)
        for item in by_id.values():
            item.sections.sort()
        return sorted(by_id.values(), key=lambda i: (i.student_name.lower(), i.student_id))

    @app.get(
        "/sessions",
        response_model=list[SessionItem],
        dependencies=[Depends(require_service_token)],
    )
    def sessions(email: str, lms_role: str, student_id: str | None = None) -> list[SessionItem]:
        principal = require_principal(email, lms_role)
        store_obj = get_store()
        pairs = store_obj.list_student_class_pairs()
        target_id = resolve_target(principal, student_id, pairs)
        class_names = sorted({cname for pid, _, cname in pairs if pid == target_id and cname})
        return [
            SessionItem(class_name=cname, label=session_display_label(cname))
            for cname in class_names
        ]

    return app
