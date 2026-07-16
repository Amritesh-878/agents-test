
from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^(\d{1,2}:\d{2}:\d{2})\s+From\s+(.*)$")
_PRIVATE_MARKERS = ("(direct message)", "(privately)")


class ChatMessage(BaseModel):
    time_str: str
    timestamp_seconds: float
    sender: str
    text: str


def _clock_to_seconds(time_str: str) -> int:
    h, m, s = (int(part) for part in time_str.split(":"))
    return h * 3600 + m * 60 + s


def _is_public(routing: str) -> bool:
    lowered = routing.lower()
    if any(marker in lowered for marker in _PRIVATE_MARKERS):
        return False
    if " to " in routing:
        recipient = routing.split(" to ", 1)[1].split("(")[0].strip()
        return recipient.casefold() == "everyone"
    return True


def _sender_of(routing: str) -> str:
    return routing.split(" to ", 1)[0].strip()


class _RawMessage(BaseModel):
    time_str: str
    routing: str
    text_parts: list[str]


def _iter_raw_messages(text: str) -> list[_RawMessage]:
    messages: list[_RawMessage] = []
    current: _RawMessage | None = None
    for raw_line in text.splitlines():
        header = _HEADER_RE.match(raw_line)
        if header:
            routing, _, body = header.group(2).partition(" : ")
            current = _RawMessage(
                time_str=header.group(1), routing=routing.strip(), text_parts=[body.strip()]
            )
            messages.append(current)
        elif current is not None and raw_line.strip():
            current.text_parts.append(raw_line.strip())
    return messages


def parse_chat_file(path: Path) -> list[ChatMessage]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read chat file %s: %s", path, exc)
        return []

    raw_messages = _iter_raw_messages(text)
    if not raw_messages:
        return []

    base_seconds = _clock_to_seconds(raw_messages[0].time_str)
    public: list[ChatMessage] = []
    dropped_private = 0
    for raw in raw_messages:
        if not _is_public(raw.routing):
            dropped_private += 1
            continue
        body = " ".join(part for part in raw.text_parts if part).strip()
        if not body:
            continue
        public.append(
            ChatMessage(
                time_str=raw.time_str,
                timestamp_seconds=float(max(0, _clock_to_seconds(raw.time_str) - base_seconds)),
                sender=_sender_of(raw.routing),
                text=body,
            )
        )
    if dropped_private:
        logger.info("Dropped %d private direct message(s) from %s", dropped_private, path.name)
    return public
