from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from scripts.auth import AuthError, AuthService
from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL
from scripts.retrieval import (
    QueryEmbedder,
    RetrievalError,
    RetrievalResult,
    retrieve_from_pgvector,
)
from scripts.utils.chunker import ChunkType
from scripts.utils.db_url import resolve_db_url

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_EGRESS_NOTICE = (
    "Notice: your questions and the retrieved transcript excerpts are sent to Groq "
    "(an external US LLM API) to generate answers."
)
EXIT_COMMANDS = {"exit", "quit"}
HELP_COMMANDS = {"?", "help"}
CONTEXT_COMMANDS = {"context"}
SOURCES_COMMANDS = {"sources"}


class ChatArgs(BaseModel):
    db_url: str
    # student_id / student_name are NOT taken from the CLI; they are populated only
    # after a successful login against the credentials CSV (see run_login / main).
    student_id: str = ""
    student_name: str = ""
    credentials_path: Path = Path("data/credentials.csv")
    chunk_types: list[ChunkType] = Field(default_factory=list)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    groq_model: str = DEFAULT_GROQ_MODEL
    max_history_turns: int = 3
    question: str | None = None
    save_session_dir: Path = Path("output/chat_sessions")
    top_k: int = 5


class PromptMessage(BaseModel):
    content: str
    role: Literal["system", "user", "assistant"]


class ChatTurnRecord(BaseModel):
    answer: str
    answer_source: Literal["fallback", "groq"]
    asked_at: str
    model: str | None = None
    prompt_messages: list[PromptMessage] = Field(default_factory=list)
    question: str
    retrieval_result: RetrievalResult
    turn_index: int
    trust_flags: list[str] = Field(default_factory=list)


class ChatSessionRecord(BaseModel):
    embedding_model: str
    groq_model: str
    last_updated_at: str
    max_history_turns: int
    save_session_dir: str
    session_id: str
    started_at: str
    student_id: str
    student_name: str
    top_k: int
    turns: list[ChatTurnRecord] = Field(default_factory=list)


class ChatError(RuntimeError):
    pass


class SupportsRetrieve(Protocol):
    def retrieve(self, args: ChatArgs, question: str) -> RetrievalResult:
        ...


class SupportsGenerate(Protocol):
    def generate(self, *, messages: Sequence[PromptMessage], model: str) -> str:
        ...


def parse_args(argv: Sequence[str] | None = None) -> ChatArgs:
    parser = argparse.ArgumentParser(
        description="Student-scoped CLI chatbot backed by pgvector retrieval and Groq generation."
    )
    parser.add_argument(
        "--db-url",
        default=None,
        dest="db_url",
        help="PostgreSQL connection URL. Falls back to DATABASE_URL env var.",
    )
    parser.add_argument(
        "--credentials",
        default="data/credentials.csv",
        dest="credentials_path",
        help="Path to the login credentials CSV (student_id,password).",
    )
    parser.add_argument("--question", default=None)
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument(
        "--chunk-type",
        action="append",
        choices=("spoken", "missed", "class_context"),
        dest="chunk_types",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, dest="embedding_model")
    parser.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL, dest="groq_model")
    parser.add_argument(
        "--save-session-dir", default="output/chat_sessions", dest="save_session_dir"
    )
    parser.add_argument("--max-history-turns", type=int, default=3, dest="max_history_turns")
    namespace = parser.parse_args(argv)
    return ChatArgs(
        db_url=resolve_db_url(namespace.db_url),
        credentials_path=Path(namespace.credentials_path),
        chunk_types=list(namespace.chunk_types or []),
        embedding_model=namespace.embedding_model,
        groq_model=namespace.groq_model,
        max_history_turns=namespace.max_history_turns,
        question=namespace.question,
        save_session_dir=Path(namespace.save_session_dir),
        top_k=namespace.top_k,
    )


def validate_inputs(args: ChatArgs) -> None:
    # student_id / student_name are not validated here: they are not CLI inputs,
    # they are established by a successful login (run_login) before use.
    if not args.db_url.strip():
        raise ValueError(
            "Database URL is required. Pass --db-url or set DATABASE_URL in .env."
        )
    if args.question is not None and not args.question.strip():
        raise ValueError("Single-turn question must not be empty when provided.")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if args.max_history_turns < 0:
        raise ValueError("max_history_turns must be zero or greater.")


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "session"


def build_session_id(student_id: str, timestamp: datetime) -> str:
    return f"{timestamp.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{slugify(student_id)}"


def build_session_path(save_session_dir: Path, session_id: str) -> Path:
    return save_session_dir / f"{session_id}.json"


def load_groq_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key
    raise ChatError("GROQ_API_KEY is missing. Add it to .env before starting the chatbot.")


def prompt_password(prompt: str) -> str:
    return getpass.getpass(prompt)


def run_login(
    auth: AuthService,
    *,
    id_provider: Callable[[str], str],
    password_provider: Callable[[str], str],
    output_fn: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Prompt for student id + password, looping until authentication succeeds.

    Returns the authenticated ``student_id`` — the only way the queried id is
    established (it is never taken from the CLI). A clean EOF/Ctrl-C raises
    ``ChatError`` so the caller exits without a traceback. Each failed attempt
    backs off ~1s so the loop isn't an unthrottled password-guessing oracle.
    """
    while True:
        try:
            student_id = id_provider("Student id: ").strip()
            password = password_provider("Password: ")
        except (EOFError, KeyboardInterrupt) as exc:
            raise ChatError("Login aborted.") from exc
        if student_id and auth.authenticate(student_id, password):
            return student_id
        output_fn("Login failed. Check your student id and password.")
        sleep_fn(1.0)


def resolve_display_name(store: Any, student_id: str) -> str:
    name = store.get_student_name(student_id)
    return name if name else student_id


def collect_trust_flags(result: RetrievalResult) -> list[str]:
    flags: list[str] = list(result.warnings)
    seen = {f.casefold() for f in flags}
    for chunk in result.retrieved_chunks:
        for flag in chunk.trust_flags:
            if flag.casefold() not in seen:
                flags.append(flag)
                seen.add(flag.casefold())
    return flags


def summarize_trust_flags(result: RetrievalResult) -> str:
    flags = collect_trust_flags(result)
    return "No explicit trust warnings." if not flags else "; ".join(flags)


def build_history_messages(
    turns: Sequence[ChatTurnRecord], max_history_turns: int
) -> list[PromptMessage]:
    if max_history_turns == 0:
        return []
    history: list[PromptMessage] = []
    for turn in turns[-max_history_turns:]:
        history.append(PromptMessage(role="user", content=turn.question))
        history.append(PromptMessage(role="assistant", content=turn.answer))
    return history


def build_prompt_messages(
    *,
    student_id: str,
    student_name: str,
    question: str,
    retrieval_result: RetrievalResult,
    history_turns: Sequence[ChatTurnRecord],
    max_history_turns: int,
) -> list[PromptMessage]:
    history = build_history_messages(history_turns, max_history_turns)
    trust_summary = summarize_trust_flags(retrieval_result)
    system = PromptMessage(
        role="system",
        content="\n".join(
            [
                "You are a student-support chatbot for a recorded class session.",
                "Answer only from the retrieved transcript context you are given.",
                "Never invent details that do not appear in the retrieved chunks.",
                "If the retrieved context is estimated, low-confidence, sparse, or incomplete, say that clearly.",
                "If the retrieved context does not support the question, say you do not have enough evidence.",
                "Keep the answer concise and personalized to the student when the evidence supports it.",
            ]
        ),
    )
    user = PromptMessage(
        role="user",
        content="\n\n".join(
            [
                f"Student name: {student_name}",
                f"Student id: {student_id}",
                f"Current question: {question}",
                f"Trust summary: {trust_summary}",
                "Retrieved context:",
                retrieval_result.context_string,
                "Answer requirements:",
                "- Use only the retrieved context above.",
                "- Acknowledge trust limitations when they matter.",
                "- If the context is not enough, say so instead of guessing.",
            ]
        ),
    )
    return [system, *history, user]


def build_empty_context_answer(student_name: str, retrieval_result: RetrievalResult) -> str:
    warning_text = summarize_trust_flags(retrieval_result)
    return (
        f"I do not have enough student-scoped class context to answer that reliably for "
        f"{student_name}. The retrieval step did not return supporting chunks for this question. "
        f"Current retrieval notes: {warning_text}"
    )


def build_sources_payload(result: RetrievalResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "result_count": result.result_count,
        "warnings": result.warnings,
        "retrieved_chunks": [
            {
                "rank": chunk.rank,
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type,
                "start": chunk.start,
                "end": chunk.end,
                "source_speaker": chunk.source_speaker,
                "trust_flags": chunk.trust_flags,
            }
            for chunk in result.retrieved_chunks
        ],
    }


def format_sources_output(result: RetrievalResult) -> str:
    return json.dumps(build_sources_payload(result), indent=2)


def write_session_record(record: ChatSessionRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


class GroqChatBackend:
    def __init__(self, api_key: str) -> None:
        try:
            from groq import Groq
        except ImportError as error:
            raise ChatError(
                "Groq SDK is not installed. Install from requirements.txt."
            ) from error
        self._client = Groq(api_key=api_key)

    def generate(self, *, messages: Sequence[PromptMessage], model: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[m.model_dump(mode="json") for m in messages],
            temperature=0.2,
        )
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ChatError("Groq returned no completion choices.")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ChatError("Groq returned an empty completion.")
        return content.strip()


class RetrievalBackend:
    def __init__(self, *, embedder: QueryEmbedder | None = None, store: Any | None = None) -> None:
        self._embedder = embedder
        self._store = store
        self._owns_store = store is None

    def retrieve(self, args: ChatArgs, question: str) -> RetrievalResult:
        if self._embedder is None:
            self._embedder = QueryEmbedder(args.embedding_model)
        if self._store is None:
            from scripts.utils.pg_store import connect_pg_store

            self._store = connect_pg_store(args.db_url)
        return retrieve_from_pgvector(
            student_id=args.student_id,
            query=question,
            top_k=args.top_k,
            chunk_types=args.chunk_types,
            db_url=args.db_url,
            embedding_model=args.embedding_model,
            store=self._store,
            embedder=self._embedder,
        )

    def close(self) -> None:
        if self._owns_store and self._store is not None:
            self._store.close()
        self._store = None


class ChatService:
    def __init__(
        self,
        args: ChatArgs,
        *,
        retrieval_backend: SupportsRetrieve | None = None,
        llm_backend: SupportsGenerate | None = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        now_provider: Callable[[], datetime] = utc_now,
        session_id: str | None = None,
    ) -> None:
        self.args = args
        self.retrieval_backend = retrieval_backend or RetrievalBackend()
        self.llm_backend = llm_backend
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.now_provider = now_provider
        started_at = iso_timestamp(now_provider())
        resolved_id = session_id or build_session_id(args.student_id, now_provider())
        self.session_path = build_session_path(args.save_session_dir, resolved_id)
        self.session_record = ChatSessionRecord(
            embedding_model=args.embedding_model,
            groq_model=args.groq_model,
            last_updated_at=started_at,
            max_history_turns=args.max_history_turns,
            save_session_dir=str(args.save_session_dir),
            session_id=resolved_id,
            started_at=started_at,
            student_id=args.student_id,
            student_name=args.student_name,
            top_k=args.top_k,
        )
        self.last_retrieval: RetrievalResult | None = None
        write_session_record(self.session_record, self.session_path)

    def print_banner(self) -> None:
        self.output_fn(
            f"Chat ready for {self.args.student_name} ({self.args.student_id}). "
            f"Session: {self.session_path}\n"
            f"{GROQ_EGRESS_NOTICE}\n"
            "Commands: context, sources, help, quit"
        )

    def build_turn_record(
        self, question: str, answer: str, retrieval_result: RetrievalResult
    ) -> ChatTurnRecord:
        if retrieval_result.result_count == 0:
            prompt_messages: list[PromptMessage] = []
            answer_source: Literal["fallback", "groq"] = "fallback"
            model_name: str | None = None
        else:
            if self.llm_backend is None:
                raise ChatError("No language model backend was configured.")
            prompt_messages = build_prompt_messages(
                student_id=self.args.student_id,
                student_name=self.args.student_name,
                question=question,
                retrieval_result=retrieval_result,
                history_turns=self.session_record.turns,
                max_history_turns=self.args.max_history_turns,
            )
            answer_source = "groq"
            model_name = self.args.groq_model
        return ChatTurnRecord(
            answer=answer,
            answer_source=answer_source,
            asked_at=iso_timestamp(self.now_provider()),
            model=model_name,
            prompt_messages=prompt_messages,
            question=question,
            retrieval_result=retrieval_result,
            turn_index=len(self.session_record.turns) + 1,
            trust_flags=collect_trust_flags(retrieval_result),
        )

    def ask_question(self, question: str) -> ChatTurnRecord:
        normalized = question.strip()
        if not normalized:
            raise ChatError("Question must not be empty.")
        self.output_fn("Retrieving student-scoped context...")
        retrieval_result = self.retrieval_backend.retrieve(self.args, normalized)
        self.last_retrieval = retrieval_result
        if retrieval_result.result_count == 0:
            answer = build_empty_context_answer(self.args.student_name, retrieval_result)
        else:
            if self.llm_backend is None:
                raise ChatError("No language model backend was configured.")
            prompt_messages = build_prompt_messages(
                student_id=self.args.student_id,
                student_name=self.args.student_name,
                question=normalized,
                retrieval_result=retrieval_result,
                history_turns=self.session_record.turns,
                max_history_turns=self.args.max_history_turns,
            )
            self.output_fn("Generating grounded answer with Groq...")
            answer = self.llm_backend.generate(messages=prompt_messages, model=self.args.groq_model)
        turn = self.build_turn_record(normalized, answer, retrieval_result)
        self.session_record.turns.append(turn)
        self.session_record.last_updated_at = iso_timestamp(self.now_provider())
        write_session_record(self.session_record, self.session_path)
        return turn

    def print_context(self) -> None:
        if self.last_retrieval is None:
            self.output_fn("No retrieval trace yet. Ask a question first.")
            return
        self.output_fn(self.last_retrieval.context_string)

    def print_sources(self) -> None:
        if self.last_retrieval is None:
            self.output_fn("No retrieval trace yet. Ask a question first.")
            return
        self.output_fn(format_sources_output(self.last_retrieval))

    def print_help(self) -> None:
        self.output_fn("Commands: ask a question, or use context, sources, help, quit")

    def handle_user_input(self, raw_input: str) -> bool:
        normalized = raw_input.strip()
        if not normalized:
            self.output_fn("Enter a question, or use context, sources, help, or quit.")
            return True
        cmd = normalized.casefold()
        if cmd in EXIT_COMMANDS:
            self.output_fn("Ending chat session.")
            return False
        if cmd in HELP_COMMANDS:
            self.print_help()
            return True
        if cmd in CONTEXT_COMMANDS:
            self.print_context()
            return True
        if cmd in SOURCES_COMMANDS:
            self.print_sources()
            return True
        turn = self.ask_question(normalized)
        self.output_fn(f"Assistant: {turn.answer}")
        return True

    def run(self) -> None:
        self.print_banner()
        if self.args.question is not None:
            self.handle_user_input(self.args.question)
            return
        keep_running = True
        while keep_running:
            keep_running = self.handle_user_input(self.input_fn("You: "))

    def close(self) -> None:
        closer = getattr(self.retrieval_backend, "close", None)
        if callable(closer):
            closer()


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from scripts.utils.pg_store import connect_pg_store

        args = parse_args(argv)
        validate_inputs(args)
        auth = AuthService.from_csv(args.credentials_path)
        store = connect_pg_store(args.db_url)
        try:
            student_id = run_login(
                auth, id_provider=input, password_provider=prompt_password
            )
            student_name = resolve_display_name(store, student_id)
            args = args.model_copy(
                update={"student_id": student_id, "student_name": student_name}
            )
            chat_service = ChatService(
                args,
                retrieval_backend=RetrievalBackend(store=store),
                llm_backend=GroqChatBackend(load_groq_api_key()),
            )
            chat_service.run()
        finally:
            store.close()
    except (AuthError, ChatError, RetrievalError, ValueError, OSError) as error:
        print(f"Chat failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
