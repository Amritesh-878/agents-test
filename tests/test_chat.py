from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from scripts.auth import AuthService
from scripts.chat import (
    ChatArgs,
    ChatError,
    ChatService,
    RetrievalBackend,
    parse_args,
    resolve_display_name,
    run_login,
)
from scripts.retrieval import RetrievalResult


def make_args(tmp_path: Path) -> ChatArgs:
    return ChatArgs(
        db_url="postgresql://postgres:sup3rsecret@localhost:5432/adira",
        student_id="2302",
        student_name="Bhagyashree",
        save_session_dir=tmp_path / "chat_sessions",
    )


# --- #1: session traces hold no secrets ---


def test_session_trace_excludes_db_secret(tmp_path: Path) -> None:
    service = ChatService(make_args(tmp_path))
    contents = service.session_path.read_text(encoding="utf-8")
    assert "sup3rsecret" not in contents
    assert "postgresql://" not in contents
    assert "db_url" not in contents


def test_session_record_has_no_db_url_field() -> None:
    from scripts.chat import ChatSessionRecord

    assert "db_url" not in ChatSessionRecord.model_fields


# --- #3/#4 access half: no CLI path to set the queried student_id ---


def test_parse_args_rejects_student_id_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--student-id", "9999"])


def test_parse_args_rejects_student_name_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--student-name", "Someone"])


class _RecordingBackend:
    def __init__(self) -> None:
        self.seen_student_id: str | None = None

    def retrieve(self, args: ChatArgs, question: str) -> RetrievalResult:
        self.seen_student_id = args.student_id
        return RetrievalResult(
            context_string="none",
            embedding_model="m",
            query=question,
            result_count=0,
            student_id=args.student_id,
            top_k=args.top_k,
        )


def test_logged_in_retrieval_is_scoped_to_authed_id(tmp_path: Path) -> None:
    auth = AuthService({"2302": "alpha"})
    student_id = run_login(
        auth,
        id_provider=lambda _: "2302",
        password_provider=lambda _: "alpha",
        sleep_fn=lambda _: None,
    )
    assert student_id == "2302"

    args = make_args(tmp_path).model_copy(update={"student_id": student_id})
    backend = _RecordingBackend()
    service = ChatService(args, retrieval_backend=backend)
    service.ask_question("what did I miss?")

    # The only id ever queried is the authenticated one — there is no input that
    # could redirect retrieval to another student.
    assert backend.seen_student_id == "2302"


# --- run_login behavior ---


def test_run_login_loops_until_correct_with_backoff() -> None:
    auth = AuthService({"2302": "alpha"})
    passwords = iter(["wrong", "alpha"])
    sleeps: list[float] = []

    student_id = run_login(
        auth,
        id_provider=lambda _: "2302",
        password_provider=lambda _: next(passwords),
        output_fn=lambda _: None,
        sleep_fn=sleeps.append,
    )

    assert student_id == "2302"
    assert sleeps == [1.0]  # one backoff after the single failed attempt


def test_run_login_aborts_cleanly_on_eof() -> None:
    auth = AuthService({"2302": "alpha"})

    def raise_eof(_: str) -> str:
        raise EOFError

    with pytest.raises(ChatError, match="aborted"):
        run_login(
            auth,
            id_provider=lambda _: "2302",
            password_provider=raise_eof,
        )


# --- resolve_display_name ---


class _NameStore:
    def __init__(self, name: str | None) -> None:
        self._name = name

    def get_student_name(self, student_id: str) -> str | None:
        return self._name


def test_resolve_display_name_uses_store_name() -> None:
    assert resolve_display_name(_NameStore("Bhagyashree"), "2302") == "Bhagyashree"


def test_resolve_display_name_falls_back_to_id() -> None:
    assert resolve_display_name(_NameStore(None), "2302") == "2302"


# --- RetrievalBackend still does not leak an injected store on close ---


def test_retrieval_backend_close_is_safe_without_store() -> None:
    backend = RetrievalBackend()
    backend.close()  # should not raise even though no store was ever opened


# --- #5: Groq egress disclosure surfaced in the banner ---


def test_banner_includes_groq_egress_notice(tmp_path: Path) -> None:
    from scripts.chat import GROQ_EGRESS_NOTICE

    captured: list[str] = []
    service = ChatService(make_args(tmp_path), output_fn=captured.append)
    service.print_banner()

    assert GROQ_EGRESS_NOTICE in "\n".join(captured)


# --- Finding A: "what did I say" must not surface the teacher's class context ---


def test_is_self_referential_question_matches_own_speech() -> None:
    from scripts.chat import is_self_referential_question

    assert is_self_referential_question("What did I personally say during class today?")
    assert is_self_referential_question("What did I say about the determinants of supply?")
    assert is_self_referential_question("What did I ask the teacher?")
    assert is_self_referential_question("Did I contribute anything today?")
    assert is_self_referential_question("What was my answer to the supply question?")


def test_is_self_referential_question_ignores_class_and_passive_questions() -> None:
    from scripts.chat import is_self_referential_question

    # "we"/topic questions are about class content, not the student's own words.
    assert not is_self_referential_question("What did we cover in class today?")
    assert not is_self_referential_question(
        "What's the difference between a supply schedule and a supply curve?"
    )
    assert not is_self_referential_question("What were Jagruti and Kalyani being asked about?")
    # First-person but NOT about the student's own speech (passive / "I missed").
    assert not is_self_referential_question(
        "I missed the part about why quantity supplied can be negative — what was said?"
    )
    assert not is_self_referential_question("I joined late — what was the plan for today's class?")


def test_select_retrieval_chunk_types_scopes_self_referential_to_spoken() -> None:
    from scripts.chat import select_retrieval_chunk_types

    assert select_retrieval_chunk_types("What did I say today?", ()) == ["spoken"]
    # Non-self-referential questions stay unfiltered (full class context available).
    assert select_retrieval_chunk_types("What did we cover today?", ()) == []
    # An explicit caller filter always wins, even for a self-referential question.
    assert select_retrieval_chunk_types("What did I say today?", ["missed"]) == ["missed"]


def test_build_prompt_messages_teaches_speaker_attribution() -> None:
    from scripts.chat import build_prompt_messages

    result = RetrievalResult(
        context_string="ctx",
        embedding_model="m",
        query="q",
        result_count=1,
        student_id="2302",
        top_k=5,
    )
    messages = build_prompt_messages(
        student_id="2302",
        student_name="Bhagyashree",
        question="What did I say today?",
        retrieval_result=result,
        history_turns=[],
        max_history_turns=0,
    )
    system = messages[0].content.casefold()
    assert "speaker=teacher" in system
    assert "not the student" in system
    assert "type=spoken" in system


class _CapturingStore:
    def __init__(self) -> None:
        self.search_chunk_types: list[str] | None = None

    def search(
        self,
        query_embedding: list[float],
        student_id: str,
        top_k: int = 5,
        chunk_types: Sequence[str] | None = None,
    ) -> list[object]:
        self.search_chunk_types = list(chunk_types or [])
        return []


class _FixedEmbedder:
    def encode(self, query: str) -> list[float]:
        return [0.1, 0.2]


def test_retrieval_backend_scopes_self_referential_to_spoken(tmp_path: Path) -> None:
    store = _CapturingStore()
    backend = RetrievalBackend(store=store, embedder=_FixedEmbedder())  # type: ignore[arg-type]
    args = make_args(tmp_path)

    backend.retrieve(args, "What did I say during class today?")
    assert store.search_chunk_types == ["spoken"]

    backend.retrieve(args, "What did we cover in class today?")
    assert store.search_chunk_types == []
