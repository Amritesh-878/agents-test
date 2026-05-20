from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.chunk_and_embed import CHROMA_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
from scripts.retrieval import RetrievalError, RetrievalResult, retrieve_from_chroma
from scripts.utils.chunker import ChunkType

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
EXIT_COMMANDS = {"exit", "quit"}
HELP_COMMANDS = {"?", "help"}
CONTEXT_COMMANDS = {"context"}
SOURCES_COMMANDS = {"sources"}


class ChatArgs(BaseModel):
    chroma_dir: Path = Path("data/chroma")
    chunk_types: list[ChunkType] = Field(default_factory=list)
    collection_name: str = CHROMA_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    groq_model: str = DEFAULT_GROQ_MODEL
    max_history_turns: int = 3
    question: str | None = None
    save_session_dir: Path = Path("output/chat_sessions")
    student_id: str
    student_name: str
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
    chroma_dir: str
    collection_name: str
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
        description=(
            "Run a student-scoped CLI chatbot backed by TASK-008 retrieval and Groq generation."
        )
    )
    parser.add_argument("--student-id", required=True, help="Stable student scope key from TASK-007.")
    parser.add_argument("--student-name", required=True, help="Display name used in the chat prompt.")
    parser.add_argument(
        "--question",
        help="Optional single-turn question. If omitted, an interactive chat loop starts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of ranked chunks to retrieve per turn.",
    )
    parser.add_argument(
        "--chunk-type",
        action="append",
        choices=("spoken", "missed", "class_context"),
        dest="chunk_types",
        help="Optional chunk-type filter. Repeat to allow multiple types.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/chroma",
        help="Directory containing the persistent TASK-007 ChromaDB store.",
    )
    parser.add_argument(
        "--collection-name",
        default=CHROMA_COLLECTION_NAME,
        help="ChromaDB collection name to query.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence Transformer model name used for retrieval query embedding.",
    )
    parser.add_argument(
        "--groq-model",
        default=DEFAULT_GROQ_MODEL,
        help="Groq model id used for answer generation.",
    )
    parser.add_argument(
        "--save-session-dir",
        default="output/chat_sessions",
        help="Directory where structured chat session JSON traces are written.",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=3,
        help="Maximum number of prior turns to retain in the prompt history.",
    )
    namespace = parser.parse_args(argv)
    return ChatArgs(
        chroma_dir=Path(namespace.chroma_dir),
        chunk_types=list(namespace.chunk_types or []),
        collection_name=namespace.collection_name,
        embedding_model=namespace.embedding_model,
        groq_model=namespace.groq_model,
        max_history_turns=namespace.max_history_turns,
        question=namespace.question,
        save_session_dir=Path(namespace.save_session_dir),
        student_id=namespace.student_id,
        student_name=namespace.student_name,
        top_k=namespace.top_k,
    )


def validate_inputs(args: ChatArgs) -> None:
    if not args.student_id.strip():
        raise ValueError("Student id must not be empty.")
    if not args.student_name.strip():
        raise ValueError("Student name must not be empty.")
    if args.question is not None and not args.question.strip():
        raise ValueError("Single-turn question must not be empty when provided.")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if args.max_history_turns < 0:
        raise ValueError("max_history_turns must be zero or greater.")
    if args.chroma_dir.exists() and args.chroma_dir.is_file():
        raise ValueError(f"Chroma directory path must be a directory: {args.chroma_dir}")
    if args.save_session_dir.exists() and args.save_session_dir.is_file():
        raise ValueError(f"Session output path must be a directory: {args.save_session_dir}")


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


def collect_trust_flags(result: RetrievalResult) -> list[str]:
    flags: list[str] = list(result.warnings)
    seen_flags = {flag.casefold() for flag in flags}
    for chunk in result.retrieved_chunks:
        for trust_flag in chunk.trust_flags:
            normalized_flag = trust_flag.casefold()
            if normalized_flag in seen_flags:
                continue
            flags.append(trust_flag)
            seen_flags.add(normalized_flag)
    return flags


def summarize_trust_flags(result: RetrievalResult) -> str:
    trust_flags = collect_trust_flags(result)
    if not trust_flags:
        return "No explicit trust warnings."
    return "; ".join(trust_flags)


def build_history_messages(
    turns: Sequence[ChatTurnRecord],
    max_history_turns: int,
) -> list[PromptMessage]:
    if max_history_turns == 0:
        return []

    history_messages: list[PromptMessage] = []
    for turn in turns[-max_history_turns:]:
        history_messages.append(PromptMessage(role="user", content=turn.question))
        history_messages.append(PromptMessage(role="assistant", content=turn.answer))
    return history_messages


def build_prompt_messages(
    *,
    student_id: str,
    student_name: str,
    question: str,
    retrieval_result: RetrievalResult,
    history_turns: Sequence[ChatTurnRecord],
    max_history_turns: int,
) -> list[PromptMessage]:
    history_messages = build_history_messages(history_turns, max_history_turns)
    trust_summary = summarize_trust_flags(retrieval_result)
    system_prompt = PromptMessage(
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
    user_prompt = PromptMessage(
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
    return [system_prompt, *history_messages, user_prompt]


def build_empty_context_answer(student_name: str, retrieval_result: RetrievalResult) -> str:
    warning_text = summarize_trust_flags(retrieval_result)
    return (
        f"I do not have enough student-scoped class context to answer that reliably for {student_name}. "
        f"The retrieval step did not return supporting chunks for this question. "
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
                "source_segment_ids": chunk.source_segment_ids,
                "source_segment_refs": [reference.model_dump(mode="json") for reference in chunk.source_segment_refs],
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
                "Groq SDK is not installed. Install the pinned dependency from requirements.txt."
            ) from error
        self._client = Groq(api_key=api_key)

    def generate(self, *, messages: Sequence[PromptMessage], model: str) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[message.model_dump(mode="json") for message in messages],
            temperature=0.2,
        )
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise ChatError("Groq returned no completion choices.")
        first_message = getattr(choices[0], "message", None)
        content = getattr(first_message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ChatError("Groq returned an empty completion.")
        return content.strip()


class RetrievalBackend:
    def retrieve(self, args: ChatArgs, question: str) -> RetrievalResult:
        return retrieve_from_chroma(
            student_id=args.student_id,
            query=question,
            top_k=args.top_k,
            chunk_types=args.chunk_types,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
        )


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
        resolved_session_id = session_id or build_session_id(args.student_id, now_provider())
        self.session_path = build_session_path(args.save_session_dir, resolved_session_id)
        self.session_record = ChatSessionRecord(
            chroma_dir=str(args.chroma_dir),
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            groq_model=args.groq_model,
            last_updated_at=started_at,
            max_history_turns=args.max_history_turns,
            save_session_dir=str(args.save_session_dir),
            session_id=resolved_session_id,
            started_at=started_at,
            student_id=args.student_id,
            student_name=args.student_name,
            top_k=args.top_k,
        )
        self.last_retrieval: RetrievalResult | None = None
        write_session_record(self.session_record, self.session_path)

    def print_banner(self) -> None:
        self.output_fn(
            "\n".join(
                [
                    (
                        f"Chat ready for {self.args.student_name} ({self.args.student_id}). "
                        f"Session trace: {self.session_path}"
                    ),
                    "Commands: context, sources, help, quit",
                ]
            )
        )

    def build_turn_record(self, question: str, answer: str, retrieval_result: RetrievalResult) -> ChatTurnRecord:
        if retrieval_result.result_count == 0:
            prompt_messages: list[PromptMessage] = []
            answer_source: Literal["fallback", "groq"] = "fallback"
            model_name: str | None = None
        else:
            if self.llm_backend is None:
                raise ChatError("No language model backend was configured for chat generation.")
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
        normalized_question = question.strip()
        if not normalized_question:
            raise ChatError("Question must not be empty.")

        self.output_fn("Retrieving student-scoped context...")
        retrieval_result = self.retrieval_backend.retrieve(self.args, normalized_question)
        self.last_retrieval = retrieval_result

        if retrieval_result.result_count == 0:
            answer = build_empty_context_answer(self.args.student_name, retrieval_result)
        else:
            if self.llm_backend is None:
                raise ChatError("No language model backend was configured for chat generation.")
            prompt_messages = build_prompt_messages(
                student_id=self.args.student_id,
                student_name=self.args.student_name,
                question=normalized_question,
                retrieval_result=retrieval_result,
                history_turns=self.session_record.turns,
                max_history_turns=self.args.max_history_turns,
            )
            self.output_fn("Generating grounded answer with Groq...")
            answer = self.llm_backend.generate(messages=prompt_messages, model=self.args.groq_model)

        turn_record = self.build_turn_record(normalized_question, answer, retrieval_result)
        self.session_record.turns.append(turn_record)
        self.session_record.last_updated_at = iso_timestamp(self.now_provider())
        write_session_record(self.session_record, self.session_path)
        return turn_record

    def print_context(self) -> None:
        if self.last_retrieval is None:
            self.output_fn("No retrieval trace is available yet. Ask a question first.")
            return
        self.output_fn(self.last_retrieval.context_string)

    def print_sources(self) -> None:
        if self.last_retrieval is None:
            self.output_fn("No retrieval trace is available yet. Ask a question first.")
            return
        self.output_fn(format_sources_output(self.last_retrieval))

    def print_help(self) -> None:
        self.output_fn("Commands: ask a question, or use context, sources, help, quit")

    def handle_user_input(self, raw_input: str) -> bool:
        normalized_input = raw_input.strip()
        if not normalized_input:
            self.output_fn("Enter a question, or use context, sources, help, or quit.")
            return True

        normalized_command = normalized_input.casefold()
        if normalized_command in EXIT_COMMANDS:
            self.output_fn("Ending chat session.")
            return False
        if normalized_command in HELP_COMMANDS:
            self.print_help()
            return True
        if normalized_command in CONTEXT_COMMANDS:
            self.print_context()
            return True
        if normalized_command in SOURCES_COMMANDS:
            self.print_sources()
            return True

        turn_record = self.ask_question(normalized_input)
        self.output_fn(f"Assistant: {turn_record.answer}")
        return True

    def run(self) -> None:
        self.print_banner()
        if self.args.question is not None:
            self.handle_user_input(self.args.question)
            return

        keep_running = True
        while keep_running:
            keep_running = self.handle_user_input(self.input_fn("You: "))


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        validate_inputs(args)
        chat_service = ChatService(args, llm_backend=GroqChatBackend(load_groq_api_key()))
        chat_service.run()
    except (ChatError, RetrievalError, ValueError, OSError) as error:
        print(f"Chat failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()