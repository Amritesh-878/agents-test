"""Parse a Zoom saved-chat file into PUBLIC messages only.

Privacy is the whole point of this module: Zoom chat mixes public messages (sent to
everyone) with private direct messages between two participants. A private DM must NEVER
be ingested — it would leak one student's words into a bot. ``parse_chat_file`` therefore
returns ONLY public messages; direct messages (and their continuation lines) are dropped
at parse time and never leave this module.

Line format (one message, optionally followed by un-prefixed continuation lines)::

    HH:MM:SS<tab> From <sender> : <text>                      # public
    HH:MM:SS<tab> From <sender>  to  <recipient> : <text>      # public iff recipient=Everyone
    HH:MM:SS<tab> From <sender>  to  <recipient>(direct message) : <text>   # PRIVATE -> dropped
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# A message header: timestamp, then "From <routing> : <text>". The routing carries the
# sender and (optionally) the recipient; the first " : " separates routing from text.
_HEADER_RE = re.compile(r"^(\d{1,2}:\d{2}:\d{2})\s+From\s+(.*)$")
_PRIVATE_MARKERS = ("(direct message)", "(privately)")


class ChatMessage(BaseModel):
    time_str: str
    # Seconds relative to the first message in the file (the chat is on its own clock;
    # absolute alignment to the audio timeline is not available, so timestamps give
    # ordering and a usable present/missed window anchor).
    timestamp_seconds: float
    sender: str
    text: str


def _clock_to_seconds(time_str: str) -> int:
    h, m, s = (int(part) for part in time_str.split(":"))
    return h * 3600 + m * 60 + s


def _is_public(routing: str) -> bool:
    """True only for messages sent to everyone; False for any direct/private message."""
    lowered = routing.lower()
    if any(marker in lowered for marker in _PRIVATE_MARKERS):
        return False
    if " to " in routing:
        recipient = routing.split(" to ", 1)[1].split("(")[0].strip()
        return recipient.casefold() == "everyone"
    return True  # "From <sender> : ..." with no recipient is a public message


def _sender_of(routing: str) -> str:
    return routing.split(" to ", 1)[0].strip()


class _RawMessage(BaseModel):
    time_str: str
    routing: str
    text_parts: list[str]


def _iter_raw_messages(text: str) -> list[_RawMessage]:
    """Group lines into messages, attaching un-prefixed continuation lines to the prior one."""
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
            # Continuation of the current message (multi-line paste). It inherits the
            # current message's public/private status, so a DM's continuation is dropped too.
            current.text_parts.append(raw_line.strip())
    return messages


def parse_chat_file(path: Path) -> list[ChatMessage]:
    """Return the PUBLIC chat messages from a Zoom chat file (direct messages dropped)."""
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
