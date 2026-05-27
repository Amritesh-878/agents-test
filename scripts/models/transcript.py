from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TranscriptWord(BaseModel):
    start: float
    end: float
    word: str
    score: float | None = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)
    language: str | None = None


class TranscriptDocument(BaseModel):
    language: str | None = None
    model: str
    segments: list[TranscriptSegment] = Field(default_factory=list)


class DualLanguageWord(BaseModel):
    start: float
    end: float
    word: str
    score: float
    source_language: str


class PerStudentTranscript(BaseModel):
    audio_file: str
    student_name: str | None = None
    roll_no: str | None = None
    is_teacher: bool = False
    transcript: TranscriptDocument
    merged_words: list[DualLanguageWord] = Field(default_factory=list)
    hi_avg_score: float = 0.0
    en_avg_score: float = 0.0
    dominant_language: str = "en"


class AlignmentResult(BaseModel):
    mode: Literal["session_aligned", "join_offset"]
    offset: float = 0.0
    uncertain: bool = False


class MergedSegment(BaseModel):
    start: float
    end: float
    text: str
    speakers: list[str] = Field(default_factory=list)
    source: Literal["per_student", "session_fallback"]
    words: list[DualLanguageWord] = Field(default_factory=list)
    confidence: float = 1.0


class MergeMetadata(BaseModel):
    total_segments: int
    per_student_segments: int
    session_fallback_segments: int
    multi_speaker_segments: int
    alignment_mode: str
    merge_method: str


class MergedTranscriptDocument(BaseModel):
    class_name: str
    duration_seconds: float
    segments: list[MergedSegment] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    teacher_name: str
    alignment_results: dict[str, AlignmentResult] = Field(default_factory=dict)
    metadata: MergeMetadata
