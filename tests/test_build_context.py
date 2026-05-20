from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_context import (
    BuildContextArgs,
    DiarizedTranscriptDocument,
    DiarizedTranscriptSegment,
    EXACT_ATTENDANCE_MODE,
    HEURISTIC_ATTENDANCE_MODE,
    AttendanceRecord,
    BuildContextError,
    build_attendance_window,
    build_review_markdown,
    build_speaker_mapping,
    build_speaker_stats,
    build_student_context_document,
    compute_missed_segments,
    load_attendance_records,
    parse_args,
    validate_inputs,
)


def build_transcript_document() -> DiarizedTranscriptDocument:
    return DiarizedTranscriptDocument(
        language="en",
        model="small",
        metadata={
            "merge_method": "majority_overlap",
            "total_segments": 4,
            "unknown_ratio": 0.0,
            "unknown_segments": 0,
        },
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[
            DiarizedTranscriptSegment(
                start=0.0,
                end=15.0,
                speaker="SPEAKER_00",
                text="Welcome back everyone.",
                words=[],
            ),
            DiarizedTranscriptSegment(
                start=40.0,
                end=50.0,
                speaker="SPEAKER_01",
                text="Can you explain that again?",
                words=[],
            ),
            DiarizedTranscriptSegment(
                start=70.0,
                end=90.0,
                speaker="SPEAKER_00",
                text="Sure, let us review the method.",
                words=[],
            ),
            DiarizedTranscriptSegment(
                start=110.0,
                end=120.0,
                speaker="SPEAKER_01",
                text="I think I understand now.",
                words=[],
            ),
        ],
    )


def test_parse_args_uses_task_defaults() -> None:
    args = parse_args([])

    assert args == BuildContextArgs(
        attendance_path=Path("data/sample_attendance.csv"),
        output_path=Path("output/student_contexts.json"),
        review_markdown_path=Path("output/student_context_review.md"),
        review_segments_path=Path("output/student_context_segments.csv"),
        transcript_path=Path("output/transcript_diarized.json"),
    )


def test_validate_inputs_rejects_missing_attendance(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Attendance file does not exist"):
        validate_inputs(
            BuildContextArgs(
                transcript_path=transcript_path,
                attendance_path=tmp_path / "attendance.csv",
            )
        )


def test_load_attendance_records_groups_duplicate_duration_only_rows(tmp_path: Path) -> None:
    attendance_path = tmp_path / "attendance.csv"
    attendance_path.write_text(
        "Name (original name),Email,Total duration (minutes),Guest\n"
        "Ada Lovelace,ada@example.com,30,No\n"
        "Ada Lovelace,ada@example.com,15,No\n"
        "read.ai meeting notes,,5,Yes\n",
        encoding="utf-8",
    )

    records, mode = load_attendance_records(attendance_path)

    assert mode == HEURISTIC_ATTENDANCE_MODE
    assert records[0].name == "Ada Lovelace"
    assert records[0].duration_minutes == 45.0
    assert records[0].source_rows == 2


def test_load_attendance_records_supports_join_leave_rows(tmp_path: Path) -> None:
    attendance_path = tmp_path / "attendance.csv"
    attendance_path.write_text(
        "Name (Original Name),User Email,Join Time,Leave Time,Duration (Minutes)\n"
        "Priya Sharma,priya@example.com,2026-04-05 10:00:00,2026-04-05 10:30:00,30\n",
        encoding="utf-8",
    )

    records, mode = load_attendance_records(attendance_path)

    assert mode == EXACT_ATTENDANCE_MODE
    assert records[0].join_time is not None
    assert records[0].leave_time is not None


def test_build_speaker_mapping_uses_duration_rank_fallback() -> None:
    transcript = build_transcript_document()
    speaker_stats = build_speaker_stats(transcript.segments)
    records = [
        AttendanceRecord(
            name="Teacher",
            email="teacher@example.com",
            duration_minutes=60.0,
            participant_key="teacher@example.com",
        ),
        AttendanceRecord(
            name="Student",
            email="student@example.com",
            duration_minutes=20.0,
            participant_key="student@example.com",
        ),
        AttendanceRecord(
            name="read.ai meeting notes",
            email=None,
            duration_minutes=50.0,
            participant_key="read.ai meeting notes",
        ),
    ]

    mapping, method = build_speaker_mapping(
        records,
        transcript.segments,
        speaker_stats,
        HEURISTIC_ATTENDANCE_MODE,
    )

    assert method == "duration_rank_estimate"
    assert mapping == {"SPEAKER_00": "Teacher", "SPEAKER_01": "Student"}


def test_build_attendance_window_anchors_duration_to_speech() -> None:
    record = AttendanceRecord(
        name="Student",
        email="student@example.com",
        duration_minutes=4.0,
        participant_key="student@example.com",
    )
    mapped_segments = [
        DiarizedTranscriptSegment(
            start=100.0,
            end=120.0,
            speaker="SPEAKER_01",
            text="question",
            words=[],
        ),
        DiarizedTranscriptSegment(
            start=180.0,
            end=210.0,
            speaker="SPEAKER_01",
            text="follow up",
            words=[],
        ),
    ]

    window = build_attendance_window(
        record,
        mapped_segments,
        meeting_duration_seconds=300.0,
        attendance_mode=HEURISTIC_ATTENDANCE_MODE,
        meeting_start=None,
    )

    assert window.estimated is True
    assert window.method == "duration_anchored_to_speech"
    assert window.joined_at < 100.0
    assert window.left_at > 210.0
    assert window.duration_seconds == pytest.approx(240.0)


def test_compute_missed_segments_marks_approximate_windows() -> None:
    transcript = build_transcript_document()

    missed_segments = compute_missed_segments(
        transcript.segments,
        joined_at=20.0,
        left_at=100.0,
        approximate=True,
    )

    assert [segment.text for segment in missed_segments] == [
        "Welcome back everyone.",
        "I think I understand now.",
    ]
    assert all(segment.approximate for segment in missed_segments)


def test_build_student_context_document_marks_fallback_outputs_for_review() -> None:
    transcript = build_transcript_document()
    attendance_records = [
        AttendanceRecord(
            name="Teacher",
            email="teacher@example.com",
            duration_minutes=5.0,
            participant_key="teacher@example.com",
        ),
        AttendanceRecord(
            name="Student",
            email="student@example.com",
            duration_minutes=1.0,
            participant_key="student@example.com",
        ),
    ]

    document = build_student_context_document(
        transcript,
        attendance_records,
        HEURISTIC_ATTENDANCE_MODE,
    )

    assert document.metadata.attendance_window_accuracy == "estimated"
    assert document.metadata.approximate_missed_segments is True
    assert document.students["Teacher"].mapped_speaker == "SPEAKER_00"
    assert document.students["Teacher"].mapping_confidence == "low"
    assert document.students["Student"].missed_segments


def test_build_review_markdown_includes_mapping_and_approximation_notes() -> None:
    transcript = build_transcript_document()
    attendance_records = [
        AttendanceRecord(
            name="Teacher",
            email="teacher@example.com",
            duration_minutes=5.0,
            participant_key="teacher@example.com",
        ),
    ]
    document = build_student_context_document(
        transcript,
        attendance_records,
        HEURISTIC_ATTENDANCE_MODE,
    )

    review = build_review_markdown(document)

    assert "## Speaker Mapping Review" in review
    assert "Attendance window accuracy: estimated" in review
    assert "Approximate missed segments" in review


def test_load_attendance_records_rejects_missing_duration(tmp_path: Path) -> None:
    attendance_path = tmp_path / "attendance.csv"
    attendance_path.write_text(
        "Name (original name),Email,Total duration (minutes),Guest\n"
        "Ada Lovelace,ada@example.com,,No\n",
        encoding="utf-8",
    )

    with pytest.raises(BuildContextError, match="missing a duration value"):
        load_attendance_records(attendance_path)