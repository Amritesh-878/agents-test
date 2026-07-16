from __future__ import annotations

from pathlib import Path

from scripts.parse_chat import parse_chat_file


def write_chat(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_public_messages_are_parsed(tmp_path: Path) -> None:
    chat = write_chat(
        tmp_path / "chat.txt",
        "14:14:41\t From A_Kalyani_2511 : can I get the code please\n"
        "14:14:51\t From A_Siddhi_2524 : the answer is 5 days I think\n",
    )
    messages = parse_chat_file(chat)
    assert [m.sender for m in messages] == ["A_Kalyani_2511", "A_Siddhi_2524"]
    assert messages[0].text == "can I get the code please"
    assert messages[0].timestamp_seconds == 0.0
    assert messages[1].timestamp_seconds == 10.0


def test_direct_messages_are_dropped(tmp_path: Path) -> None:
    chat = write_chat(
        tmp_path / "chat.txt",
        "14:44:34\t From A_Swarnima_2527  to  Nisha(direct message) : my private answer 25 litres\n"
        "14:45:00\t From A_Kalyani_2511 : a public message to everyone here\n",
    )
    messages = parse_chat_file(chat)
    assert len(messages) == 1
    assert messages[0].sender == "A_Kalyani_2511"
    assert all("private answer" not in m.text for m in messages)
    assert all("Swarnima" not in m.sender for m in messages)


def test_direct_message_continuation_lines_are_also_dropped(tmp_path: Path) -> None:
    chat = write_chat(
        tmp_path / "chat.txt",
        "10:00:00\t From A_Disha_2504  to  A_Sanaya(direct message) : secret line one\n"
        "secret line two continued\n"
        "10:00:05\t From A_Disha_2504 : a normal public message for everyone\n",
    )
    messages = parse_chat_file(chat)
    assert len(messages) == 1
    assert "secret" not in messages[0].text
    assert messages[0].text == "a normal public message for everyone"


def test_to_everyone_is_public(tmp_path: Path) -> None:
    chat = write_chat(
        tmp_path / "chat.txt",
        "09:00:00\t From A_Gunjan_2507 to Everyone : this went to everyone in the room\n",
    )
    messages = parse_chat_file(chat)
    assert len(messages) == 1
    assert messages[0].sender == "A_Gunjan_2507"


def test_multiline_public_message_is_merged(tmp_path: Path) -> None:
    chat = write_chat(
        tmp_path / "chat.txt",
        "14:44:53\t From A_Sanaya_2522 : V=300\nA=300\nS=150\n",
    )
    messages = parse_chat_file(chat)
    assert len(messages) == 1
    assert messages[0].text == "V=300 A=300 S=150"


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert parse_chat_file(tmp_path / "nope.txt") == []
