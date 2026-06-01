from __future__ import annotations

from pathlib import Path

from scripts.chat import ChatArgs, ChatService


def make_args(tmp_path: Path) -> ChatArgs:
    return ChatArgs(
        db_url="postgresql://postgres:sup3rsecret@localhost:5432/adira",
        student_id="2302",
        student_name="Bhagyashree",
        save_session_dir=tmp_path / "chat_sessions",
    )


def test_session_trace_excludes_db_secret(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    service = ChatService(args)

    contents = service.session_path.read_text(encoding="utf-8")

    assert "sup3rsecret" not in contents
    assert "postgresql://" not in contents
    assert "db_url" not in contents


def test_session_record_has_no_db_url_field() -> None:
    assert "db_url" not in ChatService.__init__.__annotations__
    from scripts.chat import ChatSessionRecord

    assert "db_url" not in ChatSessionRecord.model_fields
