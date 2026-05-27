from __future__ import annotations

from pathlib import Path

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
