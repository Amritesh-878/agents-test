from __future__ import annotations

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


def test_parse_class_date_extracts_month_and_day() -> None:
    assert parse_class_date("English.03_AY2025-26_Nouns_7 Jul") == (7, 7)
    assert parse_class_date("Economics.02_AY2025-26_ Supply Function_16 April") == (4, 16)
    assert parse_class_date("freeform-name") is None


def test_parse_filename_date_ignores_meeting_id() -> None:
    assert parse_filename_date("participants_84538437552_2025_06_20") == (6, 20)
    assert parse_filename_date("not-a-participant-file") is None


def test_resolves_same_day_ignoring_filename_year(tmp_path: Path) -> None:
    wanted = _write_attendance(tmp_path, "800", 2025, 7, 7)
    resolved = resolve_attendance_file("English.03_AY2025-26_Nouns_7 Jul", tmp_path)
    assert resolved == wanted


def test_prefers_same_day_over_day_before(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2025, 7, 6)
    same_day = _write_attendance(tmp_path, "801", 2025, 7, 7)
    resolved = resolve_attendance_file("English.03_AY2025-26_Nouns_7 Jul", tmp_path)
    assert resolved == same_day


def test_day_before_fallback_when_no_same_day(tmp_path: Path) -> None:
    prior = _write_attendance(tmp_path, "800", 2025, 7, 6)
    resolved = resolve_attendance_file("English.03_AY2025-26_Nouns_7 Jul", tmp_path)
    assert resolved == prior


def test_day_before_fallback_crosses_month_boundary(tmp_path: Path) -> None:
    prior = _write_attendance(tmp_path, "800", 2025, 6, 30)
    resolved = resolve_attendance_file("English.04_AY2025-26_Verbs_1 Jul", tmp_path)
    assert resolved == prior


def test_no_matching_file_returns_none(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2025, 1, 1)
    assert resolve_attendance_file("English.03_AY2025-26_Nouns_7 Jul", tmp_path) is None


def test_unparseable_class_name_returns_none(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2025, 7, 7)
    assert resolve_attendance_file("no-date-here", tmp_path) is None


def test_ambiguous_same_day_raises(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2025, 7, 7)
    _write_attendance(tmp_path, "801", 2025, 7, 7)
    with pytest.raises(ValueError, match="Ambiguous attendance"):
        resolve_attendance_file("English.03_AY2025-26_Nouns_7 Jul", tmp_path)


def test_resolved_file_is_loadable_by_load_attendance(tmp_path: Path) -> None:
    _write_attendance(tmp_path, "800", 2025, 7, 7)
    resolved = resolve_attendance_file("English.03_AY2025-26_Nouns_7 Jul", tmp_path)
    assert resolved is not None
    records = load_attendance(resolved)
    assert records[0].roll_no == "2401"
    assert records[0].duration_minutes == 42.0
