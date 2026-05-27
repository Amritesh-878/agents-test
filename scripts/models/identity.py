from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RosterEntry(BaseModel):
    name: str
    roll_no: str
    email: str


class AttendanceRecord(BaseModel):
    name: str
    roll_no: str | None = None
    duration_minutes: float
    guest: bool | None = None
    tags: list[str] = Field(default_factory=list)


class AudioFileIdentity(BaseModel):
    filename: str
    extracted_number: str | None = None
    roll_no_4digit: str | None = None
    display_name: str | None = None


class StudentIdentity(BaseModel):
    name: str
    roll_no: str | None = None
    email: str | None = None
    source: Literal["roster", "attendance", "audio_file"]


class MatchResult(BaseModel):
    audio_file: str
    matched_student: StudentIdentity | None = None
    method: Literal["roll_no", "fuzzy_name", "none"]
    confidence: float


class PerStudentAudioFile(BaseModel):
    path: Path
    filename: str
    display_name: str | None = None
    extracted_number: str | None = None
    roll_no_4digit: str | None = None


class ZoomFileManifest(BaseModel):
    class_name: str
    raw_dir: Path
    session_mp4: Path | None = None
    mixed_m4a: Path | None = None
    per_student_m4as: list[PerStudentAudioFile] = Field(default_factory=list)
    recording_conf: dict[str, str] | None = None
    zoomver_tag: str | None = None


class IdentityMapEntry(BaseModel):
    audio_file: str
    roll_no_4digit: str | None = None
    matched_name: str | None = None
    matched_roll_no: str | None = None
    matched_email: str | None = None
    match_method: Literal["roll_no", "fuzzy_name", "none"]
    match_confidence: float
    is_teacher: bool = False
    is_unmatched: bool = False
    attendance_duration_minutes: float | None = None
    tags: list[str] = Field(default_factory=list)


class IdentityMap(BaseModel):
    teacher_name: str
    teacher_audio_file: str | None = None
    entries: list[IdentityMapEntry] = Field(default_factory=list)
    unmatched_entries: list[IdentityMapEntry] = Field(default_factory=list)
    roster_students_without_audio: list[str] = Field(default_factory=list)
