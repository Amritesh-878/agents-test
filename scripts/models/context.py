from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ContextSegment(BaseModel):
    start: float
    end: float
    text: str
    speakers: list[str] = Field(default_factory=list)
    source: str = "per_student"


class BuildContextMetadata(BaseModel):
    total_enrolled: int
    present_count: int
    absent_count: int
    unmatched_count: int


class StudentContext(BaseModel):
    name: str
    roll_no: str | None = None
    email: str | None = None
    status: Literal["present", "absent"]
    attendance_duration_minutes: float | None = None
    spoken_segments: list[ContextSegment] = Field(default_factory=list)
    present_segments: list[ContextSegment] = Field(default_factory=list)
    missed_segments: list[ContextSegment] = Field(default_factory=list)
    topics_discussed: list[str] = Field(default_factory=list)
    class_date: str | None = None
    class_duration_seconds: float = 0.0
    teacher_name: str = ""
    is_teacher: bool = False
    tags: list[str] = Field(default_factory=list)


class AbsentStudentSummary(BaseModel):
    name: str
    roll_no: str | None = None
    email: str | None = None
    class_date: str | None = None
    class_duration_seconds: float = 0.0
    teacher_name: str = ""
    topics_discussed: list[str] = Field(default_factory=list)


class StudentContextDocument(BaseModel):
    class_name: str
    present_students: dict[str, StudentContext] = Field(default_factory=dict)
    absent_students: dict[str, AbsentStudentSummary] = Field(default_factory=dict)
    all_enrolled: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    metadata: BuildContextMetadata
