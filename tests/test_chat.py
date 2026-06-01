from __future__ import annotations

from pathlib import Path

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
