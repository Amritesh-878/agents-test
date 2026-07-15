from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from scripts.match_identity import (
    _NAME_MATCH_THRESHOLD,
    _name_score,
    _normalize_plain_tokens,
    load_attendance,
    load_roster,
    resolve_chat_sender,
)
from scripts.models.context import (
    AbsentStudentSummary,
    BuildContextMetadata,
    ContextSegment,
    StudentContext,
    StudentContextDocument,
)
from scripts.parse_chat import ChatMessage, parse_chat_file
from scripts.models.identity import (
    AttendanceRecord,
    IdentityMap,
    IdentityMapEntry,
    RosterEntry,
)
from scripts.models.transcript import (
    MergedSegment,
    MergedTranscriptDocument,
    PerStudentTranscript,
)
from scripts.utils.topics import extract_topics

logger = logging.getLogger(__name__)

# Tag attached when no usable attendance window is known for a present student.
# Without a per-class attendance duration, window_end == class_duration, so
# missed_segments is ALWAYS empty — which means "missed is unknown", not "nothing
# was missed". This flag makes that ambiguity explicit instead of silent.
MISSED_UNKNOWN_TAG = "missed_unknown_no_attendance"

ATTENDANCE_ONLY_PRESENCE_TAG = "attendance_only_presence"
ATTENDANCE_PRESENCE_THRESHOLD_MINUTES = 5.0


class ContextArgs(BaseModel):
    transcript_path: Path
    identity_map_path: Path
    teacher_transcript_path: Path | None = None
    roster_path: Path | None = None
    attendance_path: Path | None = None
    chat_path: Path | None = None
    output_path: Path
    review_md_path: Path | None = None
    review_csv_path: Path | None = None
    top_topics: int = 10
    skip_absent_summaries: bool = False
    attendance_presence_threshold: float = ATTENDANCE_PRESENCE_THRESHOLD_MINUTES


def parse_args(argv: Sequence[str] | None = None) -> ContextArgs:
    parser = argparse.ArgumentParser(
        description="Build per-student context objects for every enrolled student."
    )
    parser.add_argument("--transcript", required=True, type=Path, dest="transcript_path")
    parser.add_argument("--identity-map", required=True, type=Path, dest="identity_map_path")
    parser.add_argument(
        "--teacher-transcript",
        type=Path,
        dest="teacher_transcript_path",
        default=None,
        help="Teacher's isolated-M4A transcript JSON. When given, it is the primary "
        "source of class_context/missed; without it, the merged session timeline is used.",
    )
    parser.add_argument("--roster", type=Path, dest="roster_path", default=None)
    parser.add_argument("--attendance", type=Path, dest="attendance_path", default=None)
    parser.add_argument(
        "--chat-file",
        type=Path,
        dest="chat_path",
        default=None,
        help="Zoom saved-chat file. PUBLIC messages are attributed to senders via the "
        "roster; private direct messages are dropped. Requires --roster.",
    )
    parser.add_argument("--output", required=True, type=Path, dest="output_path")
    parser.add_argument("--review-md", type=Path, dest="review_md_path", default=None)
    parser.add_argument("--review-csv", type=Path, dest="review_csv_path", default=None)
    parser.add_argument("--top-topics", type=int, default=10, dest="top_topics")
    parser.add_argument(
        "--no-absent-summaries",
        action="store_true",
        dest="skip_absent_summaries",
        help="Do not generate topic-only bots for pure-absent students (no audio AND no "
        "chat). Present and chat-only students are unaffected.",
    )
    parser.add_argument(
        "--attendance-presence-threshold",
        type=float,
        default=ATTENDANCE_PRESENCE_THRESHOLD_MINUTES,
        dest="attendance_presence_threshold",
        help="Attendance minutes at/above which a roster student with no audio and no "
        "chat is treated as present. Default: 5.",
    )
    namespace = parser.parse_args(argv)
    return ContextArgs.model_validate(vars(namespace))


def validate_inputs(args: ContextArgs) -> None:
    if not args.transcript_path.exists():
        raise ValueError(f"Transcript not found: {args.transcript_path}")
    if not args.identity_map_path.exists():
        raise ValueError(f"Identity map not found: {args.identity_map_path}")
    if args.teacher_transcript_path is not None and not args.teacher_transcript_path.exists():
        raise ValueError(f"Teacher transcript not found: {args.teacher_transcript_path}")
    if args.roster_path is not None and not args.roster_path.exists():
        raise ValueError(f"Roster not found: {args.roster_path}")
    if args.attendance_path is not None and not args.attendance_path.exists():
        raise ValueError(f"Attendance not found: {args.attendance_path}")
    if args.chat_path is not None and not args.chat_path.exists():
        raise ValueError(f"Chat file not found: {args.chat_path}")


def merged_seg_to_context(seg: MergedSegment) -> ContextSegment:
    return ContextSegment(
        start=seg.start,
        end=seg.end,
        text=seg.text,
        speakers=seg.speakers,
        source=seg.source,
    )


def full_transcript_text(transcript: MergedTranscriptDocument) -> str:
    return " ".join(s.text for s in transcript.segments if s.text.strip())


def teacher_segments_from_transcript(
    teacher_doc: PerStudentTranscript, teacher_name: str
) -> list[ContextSegment]:
    """Convert the teacher's isolated-M4A transcript into class-context segments.

    The teacher's mic captures only her voice, so this is far cleaner than the mixed
    session MP4. Segments are attributed to the teacher and tagged ``source="teacher"``
    so downstream chunking treats them as clean class content, not noisy fallback.
    """
    segments: list[ContextSegment] = []
    for seg in teacher_doc.transcript.segments:
        if not seg.text.strip():
            continue
        segments.append(
            ContextSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speakers=[teacher_name],
                source="teacher",
            )
        )
    return segments


def class_context_text(
    transcript: MergedTranscriptDocument,
    teacher_doc: PerStudentTranscript | None,
) -> str:
    """Return the text used for TF-IDF topic extraction.

    Prefer the teacher's clean isolated-mic transcript when available; fall back to the
    full merged timeline (driven by the noisy session MP4) when no teacher M4A exists.
    """
    if teacher_doc is not None:
        return " ".join(
            s.text for s in teacher_doc.transcript.segments if s.text.strip()
        )
    return full_transcript_text(transcript)


def chat_message_to_segment(message: ChatMessage) -> ContextSegment:
    """A student's own public chat message as a context segment (source ``chat``)."""
    return ContextSegment(
        start=message.timestamp_seconds,
        end=message.timestamp_seconds,
        text=message.text,
        speakers=[message.sender],
        source="chat",
    )


def attribute_chat_to_students(
    messages: list[ChatMessage],
    roster: list[RosterEntry],
    roster_by_roll: dict[str, RosterEntry],
) -> dict[str, list[ContextSegment]]:
    """Group PUBLIC chat messages by the roster roll of their sender.

    Each message is attributed to exactly one student (or dropped if the sender can't be
    confidently/uniquely matched), so a student's chat only ever lands in their own
    bucket — never another student's. Requires a roster; without one, returns empty.
    """
    by_roll: dict[str, list[ContextSegment]] = {}
    for message in messages:
        roll = resolve_chat_sender(message.sender, roster, roster_by_roll)
        if roll is None:
            continue
        by_roll.setdefault(roll, []).append(chat_message_to_segment(message))
    return by_roll


def build_chat_only_context(
    student: RosterEntry,
    chat_segments: list[ContextSegment],
    transcript: MergedTranscriptDocument,
    topics: list[str],
    teacher_segments: list[ContextSegment] | None,
) -> StudentContext:
    """Context for a student who participated only by typing (no audio = never unmuted).

    They still get the teacher's class context (what was taught) plus their OWN chat. This
    is the point of chat ingestion: quiet students get a personal bot from their typed
    contributions instead of being treated as fully absent.
    """
    present_segments = list(teacher_segments) if teacher_segments is not None else []
    return StudentContext(
        name=student.name,
        roll_no=student.roll_no,
        email=student.email,
        status="present",
        spoken_segments=[],
        chat_segments=chat_segments,
        present_segments=present_segments,
        missed_segments=[],
        topics_discussed=topics,
        class_duration_seconds=transcript.duration_seconds,
        teacher_name=transcript.teacher_name,
        tags=["chat_only_no_audio"],
    )


def _attendance_name_agrees(attendance: AttendanceRecord, student: RosterEntry) -> bool:
    tokens = _normalize_plain_tokens(attendance.name)
    if not tokens:
        return True
    if _name_score(tokens, student.name) >= _NAME_MATCH_THRESHOLD:
        return True
    logger.warning(
        "Attendance roll %s carries name %r but roster student is %r; presence not granted",
        student.roll_no,
        attendance.name,
        student.name,
    )
    return False


def build_attendance_only_context(
    student: RosterEntry,
    transcript: MergedTranscriptDocument,
    topics: list[str],
    teacher_segments: list[ContextSegment] | None,
    attendance: AttendanceRecord,
) -> StudentContext:
    if teacher_segments is not None:
        present_segments = list(teacher_segments)
    else:
        present_segments = [merged_seg_to_context(s) for s in transcript.segments]
    return StudentContext(
        name=student.name,
        roll_no=student.roll_no,
        email=student.email,
        status="present",
        attendance_duration_minutes=attendance.duration_minutes,
        spoken_segments=[],
        chat_segments=[],
        present_segments=present_segments,
        missed_segments=[],
        topics_discussed=topics,
        class_duration_seconds=transcript.duration_seconds,
        teacher_name=transcript.teacher_name,
        tags=[ATTENDANCE_ONLY_PRESENCE_TAG, MISSED_UNKNOWN_TAG],
    )


def build_present_context(
    student: RosterEntry,
    entry: IdentityMapEntry,
    transcript: MergedTranscriptDocument,
    attendance_by_roll: dict[str, AttendanceRecord],
    topics: list[str],
    teacher_segments: list[ContextSegment] | None = None,
    chat_segments: list[ContextSegment] | None = None,
) -> StudentContext:
    """Build a present student's context.

    Class-context source: when ``teacher_segments`` is provided (a teacher M4A was
    identified), present/missed are built from the teacher's clean isolated-mic speech
    PLUS this student's own spoken segments — peer students and the noisy session-MP4
    fallback are excluded. When it is ``None``, the full merged timeline is used
    (pre-teacher-M4A behavior). ``spoken_segments`` always come from the merged
    timeline and are unaffected by this choice.

    ALIGNMENT ASSUMPTION: combining ``teacher_segments`` with the student's spoken
    segments on one shared clock and splitting on ``window_end`` assumes the teacher
    M4A starts at session start (offset 0.0) — the same "Zoom per-student M4As are
    always session-aligned" assumption documented in
    ``merge_transcripts.detect_alignment`` (finding #10). No teacher-track offset is
    derived; a non-session-aligned source would break this the same way #10's does.
    """
    att = attendance_by_roll.get(student.roll_no or "")
    class_duration = transcript.duration_seconds
    attendance_known = bool(att and att.duration_minutes)
    window_end = (
        min(att.duration_minutes * 60, class_duration)
        if attendance_known and att
        else class_duration
    )

    all_segs = [merged_seg_to_context(s) for s in transcript.segments]
    spoken_name = entry.matched_name or student.name

    # Attribute a spoken segment to a student ONLY when they are the PRIMARY speaker
    # (speakers[0]). Overlapping clusters carry every overlapping speaker in `speakers`,
    # but the segment's TEXT is only the primary (longest-overlap) speaker's words — see
    # merge_transcripts._cluster_to_merged. Matching on mere membership credited a student
    # with a peer's words on any overlap and duplicated the segment under every speaker.
    # Solo/non-overlapping speech is unaffected (the sole speaker is primary).
    spoken_segments = [s for s in all_segs if s.speakers and s.speakers[0] == spoken_name]
    if teacher_segments is not None:
        class_timeline = sorted(teacher_segments + spoken_segments, key=lambda s: s.start)
    else:
        class_timeline = all_segs
    present_segments = [s for s in class_timeline if s.start < window_end]
    missed_segments = [s for s in class_timeline if s.start >= window_end]

    tags = list(entry.tags) if entry.tags else []
    if att and att.tags:
        for t in att.tags:
            if t not in tags:
                tags.append(t)
    if not attendance_known and MISSED_UNKNOWN_TAG not in tags:
        tags.append(MISSED_UNKNOWN_TAG)

    return StudentContext(
        name=student.name,
        roll_no=student.roll_no,
        email=student.email,
        status="present",
        attendance_duration_minutes=att.duration_minutes if att else None,
        spoken_segments=spoken_segments,
        chat_segments=chat_segments or [],
        present_segments=present_segments,
        missed_segments=missed_segments,
        topics_discussed=topics,
        class_duration_seconds=class_duration,
        teacher_name=transcript.teacher_name,
        tags=tags,
    )


def build_absent_summary(
    student: RosterEntry,
    transcript: MergedTranscriptDocument,
    topics: list[str],
) -> AbsentStudentSummary:
    return AbsentStudentSummary(
        name=student.name,
        roll_no=student.roll_no,
        email=student.email,
        class_duration_seconds=transcript.duration_seconds,
        teacher_name=transcript.teacher_name,
        topics_discussed=topics,
    )


def build_context_document(
    transcript: MergedTranscriptDocument,
    identity_map: IdentityMap,
    roster: list[RosterEntry],
    attendance: list[AttendanceRecord],
    topics: list[str],
    teacher_doc: PerStudentTranscript | None = None,
    chat_messages: list[ChatMessage] | None = None,
    skip_absent_summaries: bool = False,
    attendance_presence_threshold: float = ATTENDANCE_PRESENCE_THRESHOLD_MINUTES,
) -> StudentContextDocument:
    class_name = transcript.class_name
    # Clean class-context source: the teacher's isolated mic when available, else None
    # (each student then falls back to the full merged timeline — pre-teacher behavior).
    teacher_segments = (
        teacher_segments_from_transcript(teacher_doc, transcript.teacher_name)
        if teacher_doc is not None
        else None
    )
    identity_by_roll: dict[str, IdentityMapEntry] = {
        e.matched_roll_no: e
        for e in identity_map.entries
        if e.matched_roll_no
    }
    attendance_by_roll: dict[str, AttendanceRecord] = {
        a.roll_no: a for a in attendance if a.roll_no
    }
    # PUBLIC chat attributed to each student's own roll (direct messages were dropped at
    # parse time; attribution is roster-gated, so this is empty without a roster).
    roster_by_roll: dict[str, RosterEntry] = {r.roll_no: r for r in roster}
    chat_by_roll = attribute_chat_to_students(chat_messages or [], roster, roster_by_roll)

    if roster:
        for att_roll, record in attendance_by_roll.items():
            if att_roll not in roster_by_roll:
                logger.warning(
                    "Attendance roll %s (%s) is absent from the roster; no bot generated",
                    att_roll,
                    record.name,
                )

    present_students: dict[str, StudentContext] = {}
    absent_students: dict[str, AbsentStudentSummary] = {}
    covered_roll_nos: set[str] = set()

    for student in roster:
        key = student.roll_no or student.name
        roll = student.roll_no or ""
        entry = identity_by_roll.get(roll)
        student_chat = chat_by_roll.get(roll, [])
        if entry is not None:
            covered_roll_nos.add(roll)
            present_students[key] = build_present_context(
                student, entry, transcript, attendance_by_roll, topics,
                teacher_segments, student_chat,
            )
        elif student_chat:
            # Typed but never unmuted: a chat-only present student, not absent.
            covered_roll_nos.add(roll)
            present_students[key] = build_chat_only_context(
                student, student_chat, transcript, topics, teacher_segments
            )
        else:
            att = attendance_by_roll.get(roll)
            if (
                att is not None
                and att.duration_minutes >= attendance_presence_threshold
                and _attendance_name_agrees(att, student)
            ):
                covered_roll_nos.add(roll)
                present_students[key] = build_attendance_only_context(
                    student, transcript, topics, teacher_segments, att
                )
            elif not skip_absent_summaries:
                absent_students[key] = build_absent_summary(student, transcript, topics)

    # Students matched via attendance (or other means) but absent from the roster CSV.
    # This handles the no-roster case: every matched entry still gets a context object.
    for entry in identity_map.entries:
        if entry.is_unmatched or entry.is_teacher:
            continue
        roll = entry.matched_roll_no or ""
        if roll in covered_roll_nos:
            continue
        key = roll or entry.matched_name or entry.audio_file
        synthetic = RosterEntry(
            name=entry.matched_name or entry.audio_file,
            roll_no=roll,
            email=entry.matched_email or "",
        )
        covered_roll_nos.add(roll)
        present_students[key] = build_present_context(
            synthetic, entry, transcript, attendance_by_roll, topics,
            teacher_segments, chat_by_roll.get(roll, []),
        )

    # Unmatched M4A entries (no roster, and no roll/attendance match — e.g. a filename
    # with no parseable roll, or a roll-collision casualty) have no usable student_id, so
    # they are NOT embedded (a filename-keyed chatbot can't be logged into). They are
    # counted and logged so the omission is visible for manual review, not silent.
    unmatched_count = len(identity_map.unmatched_entries)
    for entry in identity_map.unmatched_entries:
        logger.warning(
            "Unmatched audio not embedded (no usable roll): %s tags=%s",
            entry.audio_file,
            entry.tags,
        )

    all_enrolled = [s.roll_no or s.name for s in roster]
    metadata = BuildContextMetadata(
        total_enrolled=len(roster),
        present_count=len(present_students),
        absent_count=len(absent_students),
        unmatched_count=unmatched_count,
    )

    return StudentContextDocument(
        class_name=class_name,
        present_students=present_students,
        absent_students=absent_students,
        all_enrolled=all_enrolled,
        topics=topics,
        metadata=metadata,
    )


def format_review_md(doc: StudentContextDocument) -> str:
    lines = [
        f"# Student Context Review: {doc.class_name}",
        "",
        f"**Enrolled:** {doc.metadata.total_enrolled}  "
        f"**Present:** {doc.metadata.present_count}  "
        f"**Absent:** {doc.metadata.absent_count}  "
        f"**Unmatched:** {doc.metadata.unmatched_count}",
        "",
        f"**Topics:** {', '.join(doc.topics[:10]) or 'none'}",
        "",
        "| Student | Status | Spoken | Present | Missed | Tags |",
        "|---------|--------|--------|---------|--------|------|",
    ]
    for key, ctx in doc.present_students.items():
        tags = ", ".join(ctx.tags) or "—"
        lines.append(
            f"| {ctx.name} | present | {len(ctx.spoken_segments)} "
            f"| {len(ctx.present_segments)} | {len(ctx.missed_segments)} | {tags} |"
        )
    for key, summ in doc.absent_students.items():
        lines.append(f"| {summ.name} | absent | 0 | 0 | all | — |")
    return "\n".join(lines)


def write_review_csv(doc: StudentContextDocument, path: Path) -> None:
    rows = []
    for key, ctx in doc.present_students.items():
        for seg in ctx.spoken_segments:
            rows.append(
                {
                    "student": ctx.name,
                    "roll_no": ctx.roll_no or "",
                    "status": "present",
                    "segment_type": "spoken",
                    "start": seg.start,
                    "end": seg.end,
                    "speakers": "|".join(seg.speakers),
                    "text": seg.text[:120],
                }
            )
    for key, summ in doc.absent_students.items():
        rows.append(
            {
                "student": summ.name,
                "roll_no": summ.roll_no or "",
                "status": "absent",
                "segment_type": "summary",
                "start": 0.0,
                "end": summ.class_duration_seconds,
                "speakers": "",
                "text": ", ".join(summ.topics_discussed[:5]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("student,roll_no,status,segment_type,start,end,speakers,text\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as exc:
        logger.error("Input validation failed: %s", exc)
        raise SystemExit(2) from exc

    transcript = MergedTranscriptDocument.model_validate_json(
        args.transcript_path.read_text(encoding="utf-8")
    )
    identity_map = IdentityMap.model_validate_json(
        args.identity_map_path.read_text(encoding="utf-8")
    )
    roster = load_roster(args.roster_path) if args.roster_path else []
    attendance = load_attendance(args.attendance_path) if args.attendance_path else []

    teacher_doc: PerStudentTranscript | None = None
    if args.teacher_transcript_path is not None:
        teacher_doc = PerStudentTranscript.model_validate_json(
            args.teacher_transcript_path.read_text(encoding="utf-8")
        )

    chat_messages = parse_chat_file(args.chat_path) if args.chat_path else []

    text = class_context_text(transcript, teacher_doc)
    topics = extract_topics(text, top_n=args.top_topics)

    doc = build_context_document(
        transcript, identity_map, roster, attendance, topics, teacher_doc, chat_messages,
        skip_absent_summaries=args.skip_absent_summaries,
        attendance_presence_threshold=args.attendance_presence_threshold,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

    md_path = args.review_md_path or args.output_path.parent / "student_context_review.md"
    md_path.write_text(format_review_md(doc), encoding="utf-8")

    csv_path = args.review_csv_path or args.output_path.parent / "student_context_segments.csv"
    write_review_csv(doc, csv_path)

    print(f"Student contexts -> {args.output_path}")
    print(f"  Present: {doc.metadata.present_count}  Absent: {doc.metadata.absent_count}  "
          f"Unmatched: {doc.metadata.unmatched_count}")
    print(f"  Topics: {', '.join(topics[:5]) or 'none'}")


if __name__ == "__main__":
    main()
