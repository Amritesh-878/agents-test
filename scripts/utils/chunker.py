from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, Field, model_validator

ChunkType = Literal["class_context", "missed", "spoken", "chat", "material"]


class ChunkerConfig(BaseModel):
    max_chars: int = 700
    max_gap_seconds: float = 15.0
    max_segments: int = 6
    target_chars: int = 420

    @model_validator(mode="after")
    def validate_limits(self) -> "ChunkerConfig":
        if self.target_chars <= 0:
            raise ValueError("Chunk target_chars must be positive.")
        if self.max_chars < self.target_chars:
            raise ValueError("Chunk max_chars must be greater than or equal to target_chars.")
        if self.max_gap_seconds < 0:
            raise ValueError("Chunk max_gap_seconds cannot be negative.")
        if self.max_segments <= 0:
            raise ValueError("Chunk max_segments must be positive.")
        return self


class SourceSegmentReference(BaseModel):
    end: float
    segment_id: str
    segment_index: int
    source_speaker: str
    start: float
    text: str


class ChunkProjectionSegment(BaseModel):
    approximate: bool = False
    attendance_accuracy: str
    attendance_estimated: bool
    attendance_source_mode: str
    chunk_type: ChunkType
    end: float
    participant_kind: str = "student"
    source_manual_review_required: bool = True
    source_mapped_student: str | None = None
    source_mapping_confidence: str | None = None
    source_segment_id: str
    source_segment_index: int
    source_speaker: str
    start: float
    student_email: str | None = None
    student_id: str
    student_manual_review_required: bool = True
    student_mapped_speaker: str | None = None
    student_mapping_confidence: str | None = None
    student_name: str
    text: str


class ChunkRecord(BaseModel):
    approximate: bool = False
    attendance_accuracy: str
    attendance_estimated: bool
    attendance_source_mode: str
    chunk_type: ChunkType
    duration_seconds: float
    end: float
    participant_kind: str = "student"
    source_manual_review_required: bool = True
    source_mapped_student: str | None = None
    source_mapping_confidence: str | None = None
    source_segment_count: int
    source_segment_ids: list[str] = Field(default_factory=list)
    source_segment_indices: list[int] = Field(default_factory=list)
    source_segment_refs: list[SourceSegmentReference] = Field(default_factory=list)
    source_speaker: str
    start: float
    student_email: str | None = None
    student_id: str
    student_manual_review_required: bool = True
    student_mapped_speaker: str | None = None
    student_mapping_confidence: str | None = None
    student_name: str
    text: str


def round_seconds(value: float) -> float:
    return round(value, 3)


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def source_reference_for(segment: ChunkProjectionSegment) -> SourceSegmentReference:
    return SourceSegmentReference(
        end=round_seconds(segment.end),
        segment_id=segment.source_segment_id,
        segment_index=segment.source_segment_index,
        source_speaker=segment.source_speaker,
        start=round_seconds(segment.start),
        text=normalize_text(segment.text),
    )


def boundary_key(segment: ChunkProjectionSegment) -> tuple[str | bool | None, ...]:
    return (
        segment.student_id,
        segment.chunk_type,
        segment.source_speaker,
        segment.source_mapped_student,
        segment.source_mapping_confidence,
        segment.source_manual_review_required,
        segment.attendance_accuracy,
        segment.attendance_estimated,
        segment.attendance_source_mode,
        segment.approximate,
        segment.student_mapped_speaker,
        segment.student_mapping_confidence,
        segment.student_manual_review_required,
        segment.participant_kind,
    )


def sort_projection_segments(
    segments: Sequence[ChunkProjectionSegment],
) -> list[ChunkProjectionSegment]:
    return sorted(
        segments,
        key=lambda item: (
            item.student_id,
            item.start,
            item.end,
            item.source_segment_index,
            item.source_segment_id,
        ),
    )


def joined_text_length(segments: Sequence[ChunkProjectionSegment]) -> int:
    return len(" ".join(normalize_text(segment.text) for segment in segments if segment.text.strip()))


def can_merge_segment(
    current_segments: Sequence[ChunkProjectionSegment],
    next_segment: ChunkProjectionSegment,
    config: ChunkerConfig,
) -> bool:
    if not current_segments:
        return True

    last_segment = current_segments[-1]
    if boundary_key(last_segment) != boundary_key(next_segment):
        return False

    gap_seconds = max(0.0, next_segment.start - last_segment.end)
    if gap_seconds > config.max_gap_seconds:
        return False

    if len(current_segments) + 1 > config.max_segments:
        return False

    projected_length = joined_text_length(current_segments) + 1 + len(normalize_text(next_segment.text))
    if projected_length > config.max_chars:
        return False

    return joined_text_length(current_segments) < config.target_chars


def build_chunk_record(segments: Sequence[ChunkProjectionSegment]) -> ChunkRecord:
    if not segments:
        raise ValueError("Chunk record requires at least one projection segment.")

    ordered_segments = sort_projection_segments(segments)
    first_segment = ordered_segments[0]
    last_segment = ordered_segments[-1]
    text = " ".join(
        normalize_text(segment.text)
        for segment in ordered_segments
        if normalize_text(segment.text)
    )
    source_refs = [source_reference_for(segment) for segment in ordered_segments]
    return ChunkRecord(
        approximate=first_segment.approximate,
        attendance_accuracy=first_segment.attendance_accuracy,
        attendance_estimated=first_segment.attendance_estimated,
        attendance_source_mode=first_segment.attendance_source_mode,
        chunk_type=first_segment.chunk_type,
        duration_seconds=round_seconds(last_segment.end - first_segment.start),
        end=round_seconds(last_segment.end),
        participant_kind=first_segment.participant_kind,
        source_manual_review_required=first_segment.source_manual_review_required,
        source_mapped_student=first_segment.source_mapped_student,
        source_mapping_confidence=first_segment.source_mapping_confidence,
        source_segment_count=len(source_refs),
        source_segment_ids=[reference.segment_id for reference in source_refs],
        source_segment_indices=[reference.segment_index for reference in source_refs],
        source_segment_refs=source_refs,
        source_speaker=first_segment.source_speaker,
        start=round_seconds(first_segment.start),
        student_email=first_segment.student_email,
        student_id=first_segment.student_id,
        student_manual_review_required=first_segment.student_manual_review_required,
        student_mapped_speaker=first_segment.student_mapped_speaker,
        student_mapping_confidence=first_segment.student_mapping_confidence,
        student_name=first_segment.student_name,
        text=text,
    )


def chunk_projection_segments(
    segments: Sequence[ChunkProjectionSegment],
    config: ChunkerConfig | None = None,
) -> list[ChunkRecord]:
    chunker_config = ChunkerConfig() if config is None else config
    ordered_segments = sort_projection_segments(segments)
    if not ordered_segments:
        return []

    chunked_records: list[ChunkRecord] = []
    current_segments: list[ChunkProjectionSegment] = []

    for segment in ordered_segments:
        if not normalize_text(segment.text):
            continue
        if not current_segments:
            current_segments = [segment]
            continue
        if can_merge_segment(current_segments, segment, chunker_config):
            current_segments.append(segment)
            continue
        chunked_records.append(build_chunk_record(current_segments))
        current_segments = [segment]

    if current_segments:
        chunked_records.append(build_chunk_record(current_segments))
    return chunked_records