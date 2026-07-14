from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    input_path: Path
    output_dir: Path
    teacher_names: list[str] = Field(default_factory=list)
    roster_path: Path | None = None
    attendance_path: Path | None = None
    db_url: str | None = None
    single_language: str | None = None


class ClassSession(BaseModel):
    class_name: str
    zip_path: Path | None = None
    raw_dir: Path
    output_dir: Path
    teacher_name: str


class EmbeddingRecord(BaseModel):
    id: str
    student_id: str
    student_name: str
    class_name: str
    chunk_type: str
    text: str
    embedding: list[float] = Field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None
    speaker: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk_id: str
    student_id: str
    student_name: str
    class_name: str
    chunk_type: str
    text: str
    distance: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    speaker: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MigrationResult(BaseModel):
    tables_created: list[str] = Field(default_factory=list)
    indexes_created: list[str] = Field(default_factory=list)
    extensions_created: list[str] = Field(default_factory=list)
    success: bool


class StepResult(BaseModel):
    step_name: str
    success: bool
    duration_seconds: float = 0.0
    output_files: list[str] = Field(default_factory=list)
    error: str | None = None


class ClassSessionReport(BaseModel):
    class_name: str
    zip_file: str
    output_dir: str
    step_results: dict[str, StepResult] = Field(default_factory=dict)
    success: bool
    error: str | None = None


class PipelineReport(BaseModel):
    input_path: str
    sessions: list[ClassSessionReport] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    total_classes: int = 0
    successful_classes: int = 0
    failed_classes: int = 0


class DriveFile(BaseModel):
    id: str
    name: str


class DriveFileResult(BaseModel):
    drive_file_id: str
    name: str
    class_name: str
    status: str  # "processed" | "skipped_duplicate" | "failed"
    error: str | None = None


class DriveSyncReport(BaseModel):
    folder_id: str
    total_listed: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[DriveFileResult] = Field(default_factory=list)
