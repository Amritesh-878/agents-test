from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.match_identity import load_attendance
from scripts.utils.attendance_resolver import resolve_attendance_file
from scripts.utils.class_date import parse_class_date, parse_filename_date

_HEADER = "Name (original name),Email,Total duration (minutes),Guest"


def _write_attendance(directory: Path, meeting_id: str, year: int, month: int, day: int) -> Path:
    path = directory / f"participants_{meeting_id}_{year:04d}_{month:02d}_{day:02d}.csv"
    path.write_text(f"{_HEADER}\nAsha_2401,asha@example.com,42,No\n", encoding="utf-8")
    return path


def test_parse_class_date_derives_year_from_academic_year() -> None:
    assert parse_class_date("English.03_AY26-27_Nouns_7 Jul") == date(2026, 7, 7)
    assert parse_class_date("Economics.02_AY2025-26_ Supply Function_16 April") == date(2026, 4, 16)
    assert parse_class_date("Math.01_A _AY2025-26_Scaffolding_31 Mar") == date(2026, 3, 31)
    assert parse_class_date("English.04_AY26-27_Cornell Notetaking_29 Jun") == date(2026, 6, 29)


def test_parse_class_date_requires_both_day_and_academic_year() -> None:
    assert parse_class_date("English.03_Nouns_7 Jul") is None
    assert parse_class_date("English.03_AY26-27_Nouns") is None
    assert parse_class_date("freeform-name") is None


def test_parse_class_date_rejects_impossible_dates() -> None:
    assert parse_class_date("English.03_AY26-27_Nouns_31 Jun") is None


def test_parse_filename_date_reads_full_date() -> None:
    assert parse_filename_date("participants_84538437552_2025_06_20") == date(2025, 6, 20)
    assert parse_filename_date("participants_84538437552_2026_06_28") == date(2026, 6, 28)
    assert parse_filename_date("not-a-participant-file") is None
    assert parse_filename_date("participants_800_2026_13_40") is None


def test_resolves_to_the_previous_day_file(tmp_path: Path) -> None:
    wanted = _write_attendance(tmp_path, "800", 2026, 7, 6)
    _write_attendance(tmp_path, "801", 2026, 7, 7)
    resolved = resolve_attendance_file("English.03_AY26-27_Nouns_7 Jul", tmp_path)
    assert resolved == wanted


def test_previous_day_crosses_month_boundary(tmp_path: Path) -> None:
    wanted = _write_attendance(tmp_path, "800", 2026, 6, 30)
    resolved = resolve_attendance_file("English.04_AY26-27_Verbs_1 Jul", tmp_path)
    assert resolved == wanted


def test_previous_day_crosses_year_boundary(tmp_path: Path) -> None:
    wanted = _write_attendance(tmp_path, "800", 2026, 12, 31)
    resolved = resolve_attendance_file("English.04_AY26-27_Poetry_1 Jan", tmp_path)
    assert resolved == wanted


def test_multi_year_archive_is_not_ambiguous(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2025, 6, 28)
    wanted = _write_attendance(tmp_path, "800", 2026, 6, 28)
    resolved = resolve_attendance_file("English.04_AY26-27_Cornell Notetaking_29 Jun", tmp_path)
    assert resolved == wanted


def test_no_matching_file_returns_none(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2026, 7, 7)
    assert resolve_attendance_file("English.03_AY26-27_Nouns_7 Jul", tmp_path) is None


def test_unparseable_class_name_returns_none(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2026, 7, 6)
    assert resolve_attendance_file("no-date-here", tmp_path) is None


def test_duplicate_previous_day_files_raise(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2026, 7, 6)
    _write_attendance(tmp_path, "801", 2026, 7, 6)
    with pytest.raises(ValueError, match="Ambiguous attendance"):
        resolve_attendance_file("English.03_AY26-27_Nouns_7 Jul", tmp_path)


def test_resolved_file_is_loadable_by_load_attendance(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2026, 7, 6)
    resolved = resolve_attendance_file("English.03_AY26-27_Nouns_7 Jul", tmp_path)
    assert resolved is not None
    records = load_attendance(resolved)
    assert records[0].roll_no == "2401"
    assert records[0].duration_minutes == 42.0
