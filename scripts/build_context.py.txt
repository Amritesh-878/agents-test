from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from dateutil import parser as dateutil_parser  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.merge import DiarizedTranscriptDocument, DiarizedTranscriptSegment

AUTOMATION_NAME_TOKENS = (
    "notetaker",
    "otter.ai",
    "read.ai",
    "meeting notes",
    "ai companion",
)
EXACT_ATTENDANCE_MODE = "exact_join_leave"
HEURISTIC_ATTENDANCE_MODE = "duration_only_estimated"
EXACT_MAPPING_METHOD = "join_time_first_speaker"
HEURISTIC_MAPPING_METHOD = "duration_rank_estimate"
FULL_CLASS_RATIO = 0.95


class BuildContextArgs(BaseModel):
    attendance_path: Path = Path("data/sample_attendance.csv")
    output_path: Path = Path("output/student_contexts.json")
    review_markdown_path: Path = Path("output/student_context_review.md")
    review_segments_path: Path = Path("output/student_context_segments.csv")
    transcript_path: Path = Path("output/transcript_diarized.json")


class AttendanceRecord(BaseModel):
    duration_minutes: float
    email: str | None = None
    guest: bool | None = None
    join_time: datetime | None = None
    leave_time: datetime | None = None
    name: str
    participant_key: str
    source_rows: int = 1


class AttendanceWindow(BaseModel):
    duration_minutes: float
    duration_seconds: float
    estimated: bool
    exact: bool
    joined_at: float
    left_at: float
    method: str
    note: str


class ContextSegment(BaseModel):
    approximate: bool = False
    end: float
    source_speaker: str | None = None
    start: float
    text: str


class SpeakerStats(BaseModel):
    first_segment_start: float
    last_segment_end: float
    sample_utterances: list[str] = Field(default_factory=list)
    segment_count: int
    speaker: str
    total_speaking_seconds: float


class SpeakerReview(BaseModel):
    confidence: str
    evidence: str
    first_segment_start: float
    last_segment_end: float
    manual_review_required: bool = True
    mapped_student: str | None = None
    sample_utterances: list[str] = Field(default_factory=list)
    segment_count: int
    speaker: str
    total_speaking_seconds: float


class StudentContext(BaseModel):
    attendance: AttendanceWindow
    email: str | None = None
    guest: bool | None = None
    manual_review_required: bool = True
    mapped_speaker: str | None = None
    mapping_confidence: str | None = None
    mapping_notes: str | None = None
    missed_segments: list[ContextSegment] = Field(default_factory=list)
    participant_kind: str = "student"
    spoken_segments: list[ContextSegment] = Field(default_factory=list)
    was_present_full_class: bool


class BuildContextMetadata(BaseModel):
    approximate_missed_segments: bool
    attendance_source_mode: str
    attendance_window_accuracy: str
    manual_review_required: bool
    meeting_duration_seconds: float
    notes: list[str] = Field(default_factory=list)
    speaker_mapping_method: str
    transcript_segment_count: int


class StudentContextDocument(BaseModel):
    metadata: BuildContextMetadata
    speaker_mapping: dict[str, str | None] = Field(default_factory=dict)
    speaker_reviews: list[SpeakerReview] = Field(default_factory=list)
    students: dict[str, StudentContext] = Field(default_factory=dict)


class BuildContextError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> BuildContextArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-student context objects from the diarized transcript plus Zoom attendance CSV."
        )
    )
    parser.add_argument(
        "--transcript",
        default="output/transcript_diarized.json",
        help="Path to the TASK-005 diarized transcript JSON.",
    )
    parser.add_argument(
        "--attendance",
        default="data/sample_attendance.csv",
        help="Path to the Zoom attendance CSV.",
    )
    parser.add_argument(
        "--output",
        default="output/student_contexts.json",
        help="Path to the machine-readable student context JSON.",
    )
    parser.add_argument(
        "--review-markdown",
        default="output/student_context_review.md",
        help="Path to the human-readable Markdown review artifact.",
    )
    parser.add_argument(
        "--review-segments",
        default="output/student_context_segments.csv",
        help="Path to the flattened CSV review artifact.",
    )
    namespace = parser.parse_args(argv)
    return BuildContextArgs(
        attendance_path=Path(namespace.attendance),
        output_path=Path(namespace.output),
        review_markdown_path=Path(namespace.review_markdown),
        review_segments_path=Path(namespace.review_segments),
        transcript_path=Path(namespace.transcript),
    )


def validate_inputs(args: BuildContextArgs) -> None:
    for path_name, path_value, expected_suffix in (
        ("Transcript", args.transcript_path, ".json"),
        ("Attendance", args.attendance_path, ".csv"),
    ):
        if not path_value.exists():
            raise ValueError(f"{path_name} file does not exist: {path_value}")
        if not path_value.is_file():
            raise ValueError(f"{path_name} path is not a file: {path_value}")
        if path_value.suffix.lower() != expected_suffix:
            raise ValueError(f"{path_name} file must use the {expected_suffix} extension.")

    if args.output_path.suffix.lower() != ".json":
        raise ValueError("Output file must use the .json extension.")
    if args.review_markdown_path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("Review markdown file must use the .md or .txt extension.")
    if args.review_segments_path.suffix.lower() != ".csv":
        raise ValueError("Review segments file must use the .csv extension.")


def normalize_header(header: str) -> str:
    return "".join(character for character in header.casefold() if character.isalnum())


def normalize_name(name: str) -> str:
    collapsed = " ".join(name.split())
    return collapsed.strip()


def participant_key(name: str, email: str | None) -> str:
    if email:
        return email.casefold()
    return normalize_name(name).casefold()


def maybe_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def parse_duration_minutes(value: str | None) -> float:
    if value is None or not value.strip():
        raise BuildContextError("Attendance CSV is missing a duration value.")
    try:
        duration = float(value)
    except ValueError as error:
        raise BuildContextError(f"Attendance duration is not numeric: {value!r}") from error
    if duration < 0:
        raise BuildContextError(f"Attendance duration cannot be negative: {value!r}")
    return duration


def parse_guest(value: str | None) -> bool | None:
    text = maybe_text(value)
    if text is None:
        return None
    lowered = text.casefold()
    if lowered in {"yes", "true", "1"}:
        return True
    if lowered in {"no", "false", "0"}:
        return False
    return None


def parse_datetime_value(value: str | None) -> datetime | None:
    text = maybe_text(value)
    if text is None:
        return None
    try:
        return dateutil_parser.parse(text)
    except (OverflowError, TypeError, ValueError) as error:
        raise BuildContextError(f"Unable to parse attendance timestamp: {text!r}") from error


def is_automation_participant(name: str, email: str | None) -> bool:
    haystack = " ".join(part for part in (name, email or "") if part).casefold()
    return any(token in haystack for token in AUTOMATION_NAME_TOKENS)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def round_seconds(value: float) -> float:
    return round(value, 3)


def load_transcript(path: Path) -> DiarizedTranscriptDocument:
    try:
        return DiarizedTranscriptDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BuildContextError(f"Failed to read diarized transcript JSON: {error}") from error
    except ValidationError as error:
        raise BuildContextError(
            f"Diarized transcript JSON does not match the expected schema: {error}"
        ) from error


def get_row_value(row: dict[str, str | None], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        if alias in row:
            return row[alias]
    return None


def aggregate_attendance_rows(rows: Sequence[AttendanceRecord]) -> list[AttendanceRecord]:
    grouped: dict[str, AttendanceRecord] = {}
    for row in rows:
        existing = grouped.get(row.participant_key)
        if existing is None:
            grouped[row.participant_key] = row
            continue

        existing.duration_minutes += row.duration_minutes
        existing.source_rows += row.source_rows
        if existing.guest is None:
            existing.guest = row.guest
        elif row.guest is False:
            existing.guest = False
        if existing.join_time is None or (row.join_time is not None and row.join_time < existing.join_time):
            existing.join_time = row.join_time
        if existing.leave_time is None or (
            row.leave_time is not None and row.leave_time > existing.leave_time
        ):
            existing.leave_time = row.leave_time
        if existing.email is None and row.email is not None:
            existing.email = row.email
    return sorted(grouped.values(), key=lambda record: (record.name.casefold(), record.email or ""))


def load_attendance_records(path: Path) -> tuple[list[AttendanceRecord], str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise BuildContextError("Attendance CSV has no header row.")
            rows: list[AttendanceRecord] = []
            saw_join_leave = False
            for raw_row in reader:
                row = {
                    normalize_header(key): value
                    for key, value in raw_row.items()
                    if key is not None
                }
                name = maybe_text(
                    get_row_value(row, ("nameoriginalname", "name", "participantname"))
                )
                if name is None:
                    raise BuildContextError("Attendance CSV is missing a participant name.")
                email = maybe_text(get_row_value(row, ("useremail", "email")))
                join_time = parse_datetime_value(get_row_value(row, ("jointime",)))
                leave_time = parse_datetime_value(get_row_value(row, ("leavetime",)))
                if join_time is not None or leave_time is not None:
                    saw_join_leave = True
                duration_value = get_row_value(
                    row,
                    (
                        "durationminutes",
                        "totaldurationminutes",
                    ),
                )
                if duration_value is None and join_time is not None and leave_time is not None:
                    duration_minutes = max(0.0, (leave_time - join_time).total_seconds() / 60.0)
                else:
                    duration_minutes = parse_duration_minutes(duration_value)
                guest = parse_guest(get_row_value(row, ("guest",)))
                normalized_name = normalize_name(name)
                rows.append(
                    AttendanceRecord(
                        duration_minutes=round(duration_minutes, 3),
                        email=email,
                        guest=guest,
                        join_time=join_time,
                        leave_time=leave_time,
                        name=normalized_name,
                        participant_key=participant_key(normalized_name, email),
                    )
                )
    except OSError as error:
        raise BuildContextError(f"Failed to read attendance CSV: {error}") from error

    if not rows:
        raise BuildContextError("Attendance CSV did not contain any participant rows.")

    attendance_mode = EXACT_ATTENDANCE_MODE if saw_join_leave else HEURISTIC_ATTENDANCE_MODE
    return aggregate_attendance_rows(rows), attendance_mode


def build_speaker_stats(
    segments: Sequence[DiarizedTranscriptSegment],
) -> list[SpeakerStats]:
    grouped: dict[str, list[DiarizedTranscriptSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.speaker, []).append(segment)

    stats: list[SpeakerStats] = []
    for speaker, speaker_segments in grouped.items():
        ordered_segments = sorted(speaker_segments, key=lambda item: (item.start, item.end))
        sample_utterances: list[str] = []
        for segment in ordered_segments:
            text = segment.text.strip()
            if not text:
                continue
            shortened = text if len(text) <= 120 else f"{text[:117]}..."
            if shortened not in sample_utterances:
                sample_utterances.append(shortened)
            if len(sample_utterances) == 3:
                break
        stats.append(
            SpeakerStats(
                first_segment_start=round_seconds(ordered_segments[0].start),
                last_segment_end=round_seconds(ordered_segments[-1].end),
                sample_utterances=sample_utterances,
                segment_count=len(ordered_segments),
                speaker=speaker,
                total_speaking_seconds=round_seconds(
                    sum(segment.end - segment.start for segment in ordered_segments)
                ),
            )
        )
    return sorted(
        stats,
        key=lambda item: (-item.total_speaking_seconds, item.first_segment_start, item.speaker),
    )


def build_exact_speaker_mapping(
    attendance_records: Sequence[AttendanceRecord],
    transcript_segments: Sequence[DiarizedTranscriptSegment],
) -> dict[str, str]:
    records_with_join = [
        record
        for record in attendance_records
        if record.join_time is not None and not is_automation_participant(record.name, record.email)
    ]
    if not records_with_join:
        return {}

    ordered_segments = [segment for segment in transcript_segments if segment.speaker != "UNKNOWN"]
    meeting_start = min(record.join_time for record in records_with_join if record.join_time is not None)
    claimed_speakers: set[str] = set()
    mapping: dict[str, str] = {}

    for record in sorted(records_with_join, key=lambda item: (item.join_time or meeting_start, item.name)):
        assert record.join_time is not None
        join_offset = max(0.0, (record.join_time - meeting_start).total_seconds())
        for segment in ordered_segments:
            if segment.speaker in claimed_speakers:
                continue
            if segment.start >= join_offset:
                mapping[segment.speaker] = record.name
                claimed_speakers.add(segment.speaker)
                break
    return mapping


def build_duration_rank_mapping(
    attendance_records: Sequence[AttendanceRecord],
    speaker_stats: Sequence[SpeakerStats],
) -> dict[str, str]:
    ranked_participants = sorted(
        [
            record
            for record in attendance_records
            if not is_automation_participant(record.name, record.email)
        ],
        key=lambda item: (-item.duration_minutes, item.name.casefold()),
    )
    ranked_speakers = [stat for stat in speaker_stats if stat.speaker != "UNKNOWN"]
    mapping: dict[str, str] = {}
    for speaker_stat, participant in zip(ranked_speakers, ranked_participants):
        mapping[speaker_stat.speaker] = participant.name
    return mapping


def build_speaker_mapping(
    attendance_records: Sequence[AttendanceRecord],
    transcript_segments: Sequence[DiarizedTranscriptSegment],
    speaker_stats: Sequence[SpeakerStats],
    attendance_mode: str,
) -> tuple[dict[str, str], str]:
    if attendance_mode == EXACT_ATTENDANCE_MODE:
        mapping = build_exact_speaker_mapping(attendance_records, transcript_segments)
        return mapping, EXACT_MAPPING_METHOD
    mapping = build_duration_rank_mapping(attendance_records, speaker_stats)
    return mapping, HEURISTIC_MAPPING_METHOD


def speaker_segments_for(
    transcript_segments: Sequence[DiarizedTranscriptSegment],
    speaker: str | None,
) -> list[DiarizedTranscriptSegment]:
    if speaker is None:
        return []
    return [segment for segment in transcript_segments if segment.speaker == speaker]


def stretch_window_to_duration(
    anchor_start: float,
    anchor_end: float,
    target_duration: float,
    meeting_duration: float,
) -> tuple[float, float]:
    span = max(0.0, anchor_end - anchor_start)
    bounded_duration = clamp(max(target_duration, span), 0.0, meeting_duration)
    extra = max(0.0, bounded_duration - span)
    join_estimate = anchor_start - (extra / 2.0)
    leave_estimate = anchor_end + (extra / 2.0)

    if join_estimate < 0.0:
        leave_estimate = min(meeting_duration, leave_estimate - join_estimate)
        join_estimate = 0.0
    if leave_estimate > meeting_duration:
        join_estimate = max(0.0, join_estimate - (leave_estimate - meeting_duration))
        leave_estimate = meeting_duration

    actual_duration = leave_estimate - join_estimate
    if actual_duration < bounded_duration:
        remaining = bounded_duration - actual_duration
        if join_estimate == 0.0:
            leave_estimate = min(meeting_duration, leave_estimate + remaining)
        elif leave_estimate == meeting_duration:
            join_estimate = max(0.0, join_estimate - remaining)

    return round_seconds(join_estimate), round_seconds(leave_estimate)


def center_window(duration_seconds: float, meeting_duration: float) -> tuple[float, float]:
    bounded_duration = clamp(duration_seconds, 0.0, meeting_duration)
    join_estimate = max(0.0, (meeting_duration - bounded_duration) / 2.0)
    leave_estimate = min(meeting_duration, join_estimate + bounded_duration)
    return round_seconds(join_estimate), round_seconds(leave_estimate)


def build_attendance_window(
    record: AttendanceRecord,
    mapped_segments: Sequence[DiarizedTranscriptSegment],
    meeting_duration_seconds: float,
    attendance_mode: str,
    meeting_start: datetime | None,
) -> AttendanceWindow:
    duration_seconds = clamp(record.duration_minutes * 60.0, 0.0, meeting_duration_seconds)

    if (
        attendance_mode == EXACT_ATTENDANCE_MODE
        and record.join_time is not None
        and record.leave_time is not None
        and meeting_start is not None
    ):
        joined_at = clamp((record.join_time - meeting_start).total_seconds(), 0.0, meeting_duration_seconds)
        left_at = clamp((record.leave_time - meeting_start).total_seconds(), joined_at, meeting_duration_seconds)
        return AttendanceWindow(
            duration_minutes=round(record.duration_minutes, 3),
            duration_seconds=round_seconds(left_at - joined_at),
            estimated=False,
            exact=True,
            joined_at=round_seconds(joined_at),
            left_at=round_seconds(left_at),
            method="csv_join_leave",
            note="Derived from explicit Zoom join and leave timestamps.",
        )

    if meeting_duration_seconds == 0.0:
        return AttendanceWindow(
            duration_minutes=round(record.duration_minutes, 3),
            duration_seconds=0.0,
            estimated=True,
            exact=False,
            joined_at=0.0,
            left_at=0.0,
            method="empty_transcript",
            note="Transcript had no timed segments, so no attendance window could be estimated.",
        )

    if duration_seconds >= meeting_duration_seconds * FULL_CLASS_RATIO:
        return AttendanceWindow(
            duration_minutes=round(record.duration_minutes, 3),
            duration_seconds=round_seconds(meeting_duration_seconds),
            estimated=True,
            exact=False,
            joined_at=0.0,
            left_at=round_seconds(meeting_duration_seconds),
            method="duration_full_class",
            note=(
                "Estimated as full-class attendance because the reported duration covers nearly the "
                "entire diarized meeting."
            ),
        )

    if mapped_segments:
        ordered_segments = sorted(mapped_segments, key=lambda item: (item.start, item.end))
        joined_at, left_at = stretch_window_to_duration(
            ordered_segments[0].start,
            ordered_segments[-1].end,
            duration_seconds,
            meeting_duration_seconds,
        )
        return AttendanceWindow(
            duration_minutes=round(record.duration_minutes, 3),
            duration_seconds=round_seconds(left_at - joined_at),
            estimated=True,
            exact=False,
            joined_at=joined_at,
            left_at=left_at,
            method="duration_anchored_to_speech",
            note=(
                "Estimated from total duration and the mapped speaker's first and last diarized segments; "
                "window placement remains approximate because the CSV does not include join or leave timestamps."
            ),
        )

    joined_at, left_at = center_window(duration_seconds, meeting_duration_seconds)
    return AttendanceWindow(
        duration_minutes=round(record.duration_minutes, 3),
        duration_seconds=round_seconds(left_at - joined_at),
        estimated=True,
        exact=False,
        joined_at=joined_at,
        left_at=left_at,
        method="duration_centered_without_speech_anchor",
        note=(
            "Estimated from total duration only and centered in the meeting because no join or leave timestamps "
            "or mapped speech anchor were available."
        ),
    )


def build_context_segment(
    segment: DiarizedTranscriptSegment,
    approximate: bool,
) -> ContextSegment:
    return ContextSegment(
        approximate=approximate,
        end=round_seconds(segment.end),
        source_speaker=segment.speaker,
        start=round_seconds(segment.start),
        text=segment.text.strip(),
    )


def compute_missed_segments(
    transcript_segments: Sequence[DiarizedTranscriptSegment],
    joined_at: float,
    left_at: float,
    approximate: bool,
) -> list[ContextSegment]:
    return [
        build_context_segment(segment, approximate=approximate)
        for segment in transcript_segments
        if segment.end <= joined_at or segment.start >= left_at
    ]


def build_speaker_reviews(
    speaker_stats: Sequence[SpeakerStats],
    speaker_mapping: dict[str, str],
    attendance_records: Sequence[AttendanceRecord],
    mapping_method: str,
) -> list[SpeakerReview]:
    attendee_index = {record.name: record for record in attendance_records}
    reviews: list[SpeakerReview] = []
    for stat in speaker_stats:
        mapped_student = speaker_mapping.get(stat.speaker)
        if mapped_student is None:
            confidence = "none"
            evidence = "No participant was auto-mapped to this speaker; manual review is required."
        else:
            participant = attendee_index[mapped_student]
            if mapping_method == EXACT_MAPPING_METHOD:
                confidence = "medium"
                evidence = (
                    "Mapped with the first-speaker-after-join heuristic using Zoom join timestamps. "
                    f"Participant reported {participant.duration_minutes:.1f} attendance minutes."
                )
            else:
                confidence = "low"
                evidence = (
                    "Mapped with the approved duration-rank fallback because the attendance CSV lacks join and "
                    f"leave timestamps. Participant reported {participant.duration_minutes:.1f} attendance minutes."
                )
        reviews.append(
            SpeakerReview(
                confidence=confidence,
                evidence=evidence,
                first_segment_start=stat.first_segment_start,
                last_segment_end=stat.last_segment_end,
                mapped_student=mapped_student,
                sample_utterances=list(stat.sample_utterances),
                segment_count=stat.segment_count,
                speaker=stat.speaker,
                total_speaking_seconds=stat.total_speaking_seconds,
            )
        )
    return reviews


def build_student_context_document(
    transcript: DiarizedTranscriptDocument,
    attendance_records: Sequence[AttendanceRecord],
    attendance_mode: str,
) -> StudentContextDocument:
    meeting_duration_seconds = round_seconds(
        max((segment.end for segment in transcript.segments), default=0.0)
    )
    speaker_stats = build_speaker_stats(transcript.segments)
    speaker_mapping, mapping_method = build_speaker_mapping(
        attendance_records,
        transcript.segments,
        speaker_stats,
        attendance_mode,
    )
    speaker_to_segments = {
        speaker: speaker_segments_for(transcript.segments, speaker)
        for speaker in {segment.speaker for segment in transcript.segments}
    }
    meeting_start = min(
        (record.join_time for record in attendance_records if record.join_time is not None),
        default=None,
    )
    reverse_mapping = {student_name: speaker for speaker, student_name in speaker_mapping.items()}

    students: dict[str, StudentContext] = {}
    for record in attendance_records:
        mapped_speaker = reverse_mapping.get(record.name)
        mapped_segments = [] if mapped_speaker is None else speaker_to_segments.get(mapped_speaker, [])
        attendance_window = build_attendance_window(
            record,
            mapped_segments,
            meeting_duration_seconds,
            attendance_mode,
            meeting_start,
        )
        spoken_segments = [
            build_context_segment(segment, approximate=False) for segment in mapped_segments
        ]
        missed_segments = compute_missed_segments(
            transcript.segments,
            attendance_window.joined_at,
            attendance_window.left_at,
            approximate=attendance_window.estimated,
        )
        students[record.name] = StudentContext(
            attendance=attendance_window,
            email=record.email,
            guest=record.guest,
            mapped_speaker=mapped_speaker,
            mapping_confidence=("medium" if mapping_method == EXACT_MAPPING_METHOD else "low")
            if mapped_speaker is not None
            else None,
            mapping_notes=(
                "Speaker mapping uses explicit Zoom join/leave times."
                if mapping_method == EXACT_MAPPING_METHOD
                else "Speaker mapping uses the approved duration-rank fallback and must be manually reviewed."
            )
            if mapped_speaker is not None
            else "No speaker could be auto-mapped for this participant.",
            missed_segments=missed_segments,
            participant_kind=(
                "system" if is_automation_participant(record.name, record.email) else "student"
            ),
            spoken_segments=spoken_segments,
            was_present_full_class=(
                attendance_window.joined_at <= 0.001
                and attendance_window.left_at >= meeting_duration_seconds - 0.001
            ),
        )

    notes = [
        "Speaker mapping is surfaced for manual evaluation before any personalized-learning use.",
    ]
    if attendance_mode == HEURISTIC_ATTENDANCE_MODE:
        notes.append(
            "Attendance windows and missed segments are estimated because the real CSV lacks Join Time and Leave Time columns."
        )
    else:
        notes.append("Attendance windows come from explicit Zoom join and leave timestamps.")

    return StudentContextDocument(
        metadata=BuildContextMetadata(
            approximate_missed_segments=attendance_mode == HEURISTIC_ATTENDANCE_MODE,
            attendance_source_mode=attendance_mode,
            attendance_window_accuracy=(
                "estimated" if attendance_mode == HEURISTIC_ATTENDANCE_MODE else "exact"
            ),
            manual_review_required=True,
            meeting_duration_seconds=meeting_duration_seconds,
            notes=notes,
            speaker_mapping_method=mapping_method,
            transcript_segment_count=len(transcript.segments),
        ),
        speaker_mapping={speaker: speaker_mapping.get(speaker) for speaker in transcript.speakers},
        speaker_reviews=build_speaker_reviews(
            speaker_stats,
            speaker_mapping,
            attendance_records,
            mapping_method,
        ),
        students=students,
    )


def build_review_markdown(document: StudentContextDocument) -> str:
    lines = [
        "# Student Context Review",
        "",
        f"- Meeting duration (seconds): {document.metadata.meeting_duration_seconds:.3f}",
        f"- Attendance source mode: {document.metadata.attendance_source_mode}",
        f"- Attendance window accuracy: {document.metadata.attendance_window_accuracy}",
        f"- Speaker mapping method: {document.metadata.speaker_mapping_method}",
        f"- Manual review required: {'yes' if document.metadata.manual_review_required else 'no'}",
        "",
        "## Notes",
        "",
    ]
    for note in document.metadata.notes:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Speaker Mapping Review",
            "",
            "| Speaker | Suggested participant | Confidence | Speaking seconds | Segment count | Window | Sample utterance |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for review in document.speaker_reviews:
        sample = review.sample_utterances[0] if review.sample_utterances else ""
        lines.append(
            "| "
            f"{review.speaker} | {review.mapped_student or 'UNMAPPED'} | {review.confidence} | "
            f"{review.total_speaking_seconds:.3f} | {review.segment_count} | "
            f"{review.first_segment_start:.3f}-{review.last_segment_end:.3f} | {sample} |"
        )
        lines.append(f"Evidence: {review.evidence}")

    lines.extend(["", "## Student Contexts", ""])
    for student_name, context in document.students.items():
        lines.append(f"### {student_name}")
        lines.append("")
        lines.append(f"- Participant kind: {context.participant_kind}")
        lines.append(f"- Email: {context.email or 'N/A'}")
        lines.append(f"- Mapped speaker: {context.mapped_speaker or 'UNMAPPED'}")
        lines.append(f"- Mapping notes: {context.mapping_notes or 'N/A'}")
        lines.append(
            "- Attendance window: "
            f"{context.attendance.joined_at:.3f}-{context.attendance.left_at:.3f} seconds "
            f"({context.attendance.method})"
        )
        lines.append(f"- Attendance note: {context.attendance.note}")
        lines.append(f"- Spoken segments: {len(context.spoken_segments)}")
        lines.append(f"- Missed segments: {len(context.missed_segments)}")
        if context.missed_segments:
            lines.append("- Approximate missed segments: yes")
        else:
            lines.append("- Approximate missed segments: no")
        if context.spoken_segments:
            lines.append("- Spoken sample:")
            for segment in context.spoken_segments[:3]:
                lines.append(
                    f"  - [{segment.start:.3f}-{segment.end:.3f}] {segment.text}"
                )
        if context.missed_segments:
            lines.append("- Missed sample:")
            for segment in context.missed_segments[:3]:
                lines.append(
                    f"  - [{segment.start:.3f}-{segment.end:.3f}] {segment.text}"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_output(document: StudentContextDocument, args: BuildContextArgs) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    args.review_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.review_markdown_path.write_text(build_review_markdown(document), encoding="utf-8")
    args.review_segments_path.parent.mkdir(parents=True, exist_ok=True)
    with args.review_segments_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "student_name",
                "participant_kind",
                "mapped_speaker",
                "segment_type",
                "approximate",
                "start",
                "end",
                "source_speaker",
                "text",
            ],
        )
        writer.writeheader()
        for student_name, context in document.students.items():
            for segment_type, segments in (
                ("spoken", context.spoken_segments),
                ("missed", context.missed_segments),
            ):
                for segment in segments:
                    writer.writerow(
                        {
                            "student_name": student_name,
                            "participant_kind": context.participant_kind,
                            "mapped_speaker": context.mapped_speaker or "",
                            "segment_type": segment_type,
                            "approximate": "yes" if segment.approximate else "no",
                            "start": f"{segment.start:.3f}",
                            "end": f"{segment.end:.3f}",
                            "source_speaker": segment.source_speaker or "",
                            "text": segment.text,
                        }
                    )


class BuildContextService:
    def __init__(self, args: BuildContextArgs) -> None:
        self.args = args

    def run(self) -> StudentContextDocument:
        print(f"Loading diarized transcript from {self.args.transcript_path}...")
        transcript = load_transcript(self.args.transcript_path)
        print(f"Loading attendance CSV from {self.args.attendance_path}...")
        attendance_records, attendance_mode = load_attendance_records(self.args.attendance_path)
        print("Building per-student context with manual-review speaker mapping...")
        document = build_student_context_document(transcript, attendance_records, attendance_mode)
        save_output(document, self.args)
        mapped_speakers = sum(1 for student in document.students.values() if student.mapped_speaker)
        print(
            "Student context saved: "
            f"{len(document.students)} participants, {mapped_speakers} mapped speakers, "
            f"attendance mode={document.metadata.attendance_source_mode}"
        )
        print(f"JSON output: {self.args.output_path}")
        print(f"Review markdown: {self.args.review_markdown_path}")
        print(f"Review CSV: {self.args.review_segments_path}")
        return document


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        validate_inputs(args)
        BuildContextService(args).run()
    except (BuildContextError, ValueError) as error:
        print(f"Student context build failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()