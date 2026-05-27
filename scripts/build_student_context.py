from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from scripts.match_identity import load_attendance, load_roster
from scripts.models.context import (
    AbsentStudentSummary,
    BuildContextMetadata,
    ContextSegment,
    StudentContext,
    StudentContextDocument,
)
from scripts.models.identity import (
    AttendanceRecord,
    IdentityMap,
    IdentityMapEntry,
    RosterEntry,
)
from scripts.models.transcript import MergedSegment, MergedTranscriptDocument
from scripts.utils.topics import extract_topics

logger = logging.getLogger(__name__)


class ContextArgs(BaseModel):
    transcript_path: Path
    identity_map_path: Path
    roster_path: Path | None = None
    attendance_path: Path | None = None
    output_path: Path
    review_md_path: Path | None = None
    review_csv_path: Path | None = None
    top_topics: int = 10


def parse_args(argv: Sequence[str] | None = None) -> ContextArgs:
    parser = argparse.ArgumentParser(
        description="Build per-student context objects for every enrolled student."
    )
    parser.add_argument("--transcript", required=True, type=Path, dest="transcript_path")
    parser.add_argument("--identity-map", required=True, type=Path, dest="identity_map_path")
    parser.add_argument("--roster", type=Path, dest="roster_path", default=None)
    parser.add_argument("--attendance", type=Path, dest="attendance_path", default=None)
    parser.add_argument("--output", required=True, type=Path, dest="output_path")
    parser.add_argument("--review-md", type=Path, dest="review_md_path", default=None)
    parser.add_argument("--review-csv", type=Path, dest="review_csv_path", default=None)
    parser.add_argument("--top-topics", type=int, default=10, dest="top_topics")
    namespace = parser.parse_args(argv)
    return ContextArgs.model_validate(vars(namespace))


def validate_inputs(args: ContextArgs) -> None:
    if not args.transcript_path.exists():
        raise ValueError(f"Transcript not found: {args.transcript_path}")
    if not args.identity_map_path.exists():
        raise ValueError(f"Identity map not found: {args.identity_map_path}")
    if args.roster_path is not None and not args.roster_path.exists():
        raise ValueError(f"Roster not found: {args.roster_path}")
    if args.attendance_path is not None and not args.attendance_path.exists():
        raise ValueError(f"Attendance not found: {args.attendance_path}")


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


def build_present_context(
    student: RosterEntry,
    entry: IdentityMapEntry,
    transcript: MergedTranscriptDocument,
    attendance_by_roll: dict[str, AttendanceRecord],
    topics: list[str],
) -> StudentContext:
    att = attendance_by_roll.get(student.roll_no or "")
    class_duration = transcript.duration_seconds
    window_end = (
        min(att.duration_minutes * 60, class_duration)
        if att and att.duration_minutes
        else class_duration
    )

    all_segs = [merged_seg_to_context(s) for s in transcript.segments]
    spoken_name = entry.matched_name or student.name

    spoken_segments = [s for s in all_segs if spoken_name in s.speakers]
    present_segments = [s for s in all_segs if s.start < window_end]
    missed_segments = [s for s in all_segs if s.start >= window_end]

    tags = list(entry.tags) if entry.tags else []
    if att and att.tags:
        for t in att.tags:
            if t not in tags:
                tags.append(t)

    return StudentContext(
        name=student.name,
        roll_no=student.roll_no,
        email=student.email,
        status="present",
        attendance_duration_minutes=att.duration_minutes if att else None,
        spoken_segments=spoken_segments,
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


def build_unmatched_context(
    entry: IdentityMapEntry,
    transcript: MergedTranscriptDocument,
    topics: list[str],
) -> StudentContext:
    all_segs = [merged_seg_to_context(s) for s in transcript.segments]
    name = entry.audio_file
    spoken = [s for s in all_segs if name in s.speakers]
    return StudentContext(
        name=name,
        roll_no=None,
        email=None,
        status="present",
        spoken_segments=spoken,
        present_segments=all_segs,
        missed_segments=[],
        topics_discussed=topics,
        class_duration_seconds=transcript.duration_seconds,
        teacher_name=transcript.teacher_name,
        tags=["unmatched"],
    )


def build_context_document(
    transcript: MergedTranscriptDocument,
    identity_map: IdentityMap,
    roster: list[RosterEntry],
    attendance: list[AttendanceRecord],
    topics: list[str],
) -> StudentContextDocument:
    class_name = transcript.class_name
    identity_by_roll: dict[str, IdentityMapEntry] = {
        e.matched_roll_no: e
        for e in identity_map.entries
        if e.matched_roll_no
    }
    attendance_by_roll: dict[str, AttendanceRecord] = {
        a.roll_no: a for a in attendance if a.roll_no
    }

    present_students: dict[str, StudentContext] = {}
    absent_students: dict[str, AbsentStudentSummary] = {}

    for student in roster:
        key = student.roll_no or student.name
        entry = identity_by_roll.get(student.roll_no or "")
        if entry is not None:
            present_students[key] = build_present_context(
                student, entry, transcript, attendance_by_roll, topics
            )
        else:
            absent_students[key] = build_absent_summary(student, transcript, topics)

    # Unmatched M4A entries (not in roster)
    unmatched_count = 0
    for entry in identity_map.unmatched_entries:
        key = entry.audio_file
        present_students[key] = build_unmatched_context(entry, transcript, topics)
        unmatched_count += 1

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

    text = full_transcript_text(transcript)
    topics = extract_topics(text, top_n=args.top_topics)

    doc = build_context_document(transcript, identity_map, roster, attendance, topics)

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
