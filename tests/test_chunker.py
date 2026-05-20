from __future__ import annotations

from scripts.utils.chunker import (
    ChunkProjectionSegment,
    ChunkerConfig,
    build_chunk_record,
    chunk_projection_segments,
)


def build_segment(
    *,
    segment_id: str,
    segment_index: int,
    start: float,
    end: float,
    text: str,
    chunk_type: str = "class_context",
    source_speaker: str = "SPEAKER_00",
    source_mapped_student: str | None = "Teacher",
    approximate: bool = False,
) -> ChunkProjectionSegment:
    return ChunkProjectionSegment(
        approximate=approximate,
        attendance_accuracy="estimated",
        attendance_estimated=True,
        attendance_source_mode="duration_only_estimated",
        chunk_type=chunk_type,
        end=end,
        source_manual_review_required=True,
        source_mapped_student=source_mapped_student,
        source_mapping_confidence="low",
        source_segment_id=segment_id,
        source_segment_index=segment_index,
        source_speaker=source_speaker,
        start=start,
        student_email="student@example.com",
        student_id="student-example-com",
        student_manual_review_required=True,
        student_mapped_speaker="SPEAKER_01",
        student_mapping_confidence="low",
        student_name="Student Example",
        text=text,
    )


def test_chunk_projection_segments_merges_adjacent_segments_with_same_provenance() -> None:
    segments = [
        build_segment(
            segment_id="seg-0000",
            segment_index=0,
            start=0.0,
            end=4.0,
            text="Today we will revise time and work.",
        ),
        build_segment(
            segment_id="seg-0001",
            segment_index=1,
            start=4.2,
            end=8.4,
            text="Please keep your notebooks open for the next example.",
        ),
    ]

    chunks = chunk_projection_segments(segments, ChunkerConfig(target_chars=140, max_chars=220))

    assert len(chunks) == 1
    assert chunks[0].source_segment_ids == ["seg-0000", "seg-0001"]
    assert "time and work" in chunks[0].text
    assert chunks[0].source_speaker == "SPEAKER_00"


def test_chunk_projection_segments_keeps_chunk_type_boundaries_visible() -> None:
    segments = [
        build_segment(
            segment_id="seg-0000",
            segment_index=0,
            start=0.0,
            end=4.0,
            text="You missed this explanation while you were absent.",
            chunk_type="missed",
        ),
        build_segment(
            segment_id="seg-0001",
            segment_index=1,
            start=4.1,
            end=8.0,
            text="Now you are back in the shared class discussion.",
            chunk_type="class_context",
        ),
    ]

    chunks = chunk_projection_segments(segments, ChunkerConfig(target_chars=180, max_chars=260))

    assert [chunk.chunk_type for chunk in chunks] == ["missed", "class_context"]
    assert chunks[0].source_segment_ids == ["seg-0000"]
    assert chunks[1].source_segment_ids == ["seg-0001"]


def test_chunk_projection_segments_splits_long_regions_when_limits_are_reached() -> None:
    segments = [
        build_segment(
            segment_id=f"seg-000{index}",
            segment_index=index,
            start=float(index * 5),
            end=float(index * 5 + 4),
            text="This is a compact but traceable transcript sentence for retrieval review.",
        )
        for index in range(4)
    ]

    chunks = chunk_projection_segments(
        segments,
        ChunkerConfig(target_chars=80, max_chars=150, max_segments=4),
    )

    assert len(chunks) == 2
    assert chunks[0].source_segment_ids == ["seg-0000", "seg-0001"]
    assert chunks[1].source_segment_ids == ["seg-0002", "seg-0003"]
    assert all(chunk.source_segment_count == 2 for chunk in chunks)


def test_build_chunk_record_preserves_low_confidence_provenance_flags() -> None:
    chunk = build_chunk_record(
        [
            build_segment(
                segment_id="seg-0007",
                segment_index=7,
                start=35.0,
                end=42.0,
                text="I think the answer should be ten days.",
                chunk_type="spoken",
                source_speaker="SPEAKER_01",
                source_mapped_student="Student Example",
            )
        ]
    )

    assert chunk.chunk_type == "spoken"
    assert chunk.source_mapping_confidence == "low"
    assert chunk.student_mapping_confidence == "low"
    assert chunk.source_segment_refs[0].segment_index == 7


def test_chunk_projection_segments_keeps_students_isolated_during_sorting() -> None:
    first_student = build_segment(
        segment_id="seg-0000",
        segment_index=0,
        start=0.0,
        end=4.0,
        text="Teacher explanation for the first student view.",
    )
    second_student = build_segment(
        segment_id="seg-0001",
        segment_index=1,
        start=0.0,
        end=4.0,
        text="Teacher explanation for the second student view.",
    )
    second_student.student_id = "student-two"
    second_student.student_name = "Student Two"

    chunks = chunk_projection_segments(
        [second_student, first_student],
        ChunkerConfig(target_chars=120, max_chars=200),
    )

    assert [chunk.student_id for chunk in chunks] == ["student-example-com", "student-two"]