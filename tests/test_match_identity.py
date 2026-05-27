from __future__ import annotations

from pathlib import Path

import pytest

from scripts.match_identity import (
    MatchArgs,
    fuzzy_match_score,
    load_attendance,
    load_roster,
    match_files,
    parse_attendance_name,
    validate_inputs,
)
from scripts.models.identity import (
    IdentityMap,
    PerStudentAudioFile,
    RosterEntry,
    ZoomFileManifest,
)


# --- Helpers ---


def make_manifest(tmp_path: Path, audio_files: list[PerStudentAudioFile]) -> ZoomFileManifest:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return ZoomFileManifest(class_name="TestClass", raw_dir=raw_dir, per_student_m4as=audio_files)


def make_audio(
    tmp_path: Path,
    filename: str,
    display_name: str | None = None,
    roll_no_4digit: str | None = None,
    extracted_number: str | None = None,
) -> PerStudentAudioFile:
    return PerStudentAudioFile(
        path=tmp_path / "raw" / filename,
        filename=filename,
        display_name=display_name,
        roll_no_4digit=roll_no_4digit,
        extracted_number=extracted_number,
    )


def write_roster_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "RollNo", "Email"])
        writer.writeheader()
        writer.writerows(rows)


def write_attendance_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Name (original name)", "Email", "Total duration (minutes)", "Guest"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


# --- parse_attendance_name ---


def test_parse_attendance_name_standard() -> None:
    name, roll = parse_attendance_name("Anshi_2301")
    assert name == "Anshi"
    assert roll == "2301"


def test_parse_attendance_name_underscore_in_name() -> None:
    name, roll = parse_attendance_name("A_Disha_2504")
    assert name == "A_Disha"
    assert roll == "2504"


def test_parse_attendance_name_no_roll() -> None:
    name, roll = parse_attendance_name("Amritesh Praveen")
    assert name == "Amritesh Praveen"
    assert roll is None


def test_parse_attendance_name_space_before_digits() -> None:
    name, roll = parse_attendance_name("Bhagyashree_ 2302")
    assert name == "Bhagyashree"
    assert roll == "2302"


def test_parse_attendance_name_strip_parenthetical() -> None:
    name, roll = parse_attendance_name("A_Siddhi_2524 (Siddhi Ujgare)")
    assert name == "A_Siddhi"
    assert roll == "2524"


def test_parse_attendance_name_notetaker() -> None:
    name, roll = parse_attendance_name("Lakshmi's Notetaker (Otter.ai)")
    assert roll is None
    assert "Notetaker" in name


# --- load_roster ---


def test_load_roster_valid(tmp_path: Path) -> None:
    path = tmp_path / "roster.csv"
    write_roster_csv(
        path,
        [
            {"Name": "Anshi", "RollNo": "2301", "Email": "anshi@example.com"},
            {"Name": "Disha", "RollNo": "2504", "Email": "disha@example.com"},
        ],
    )
    roster = load_roster(path)
    assert len(roster) == 2
    assert roster[0].roll_no == "2301"
    assert roster[1].name == "Disha"


def test_load_roster_empty(tmp_path: Path) -> None:
    path = tmp_path / "roster.csv"
    write_roster_csv(path, [])
    assert load_roster(path) == []


# --- load_attendance ---


def test_load_attendance_valid(tmp_path: Path) -> None:
    path = tmp_path / "attendance.csv"
    write_attendance_csv(
        path,
        [
            {"Name (original name)": "Anshi_2301", "Email": "", "Total duration (minutes)": "60", "Guest": "Yes"},
            {"Name (original name)": "Amritesh Praveen", "Email": "a@b.com", "Total duration (minutes)": "90", "Guest": "No"},
        ],
    )
    records = load_attendance(path)
    assert len(records) == 2
    assert records[0].roll_no == "2301"
    assert records[0].duration_minutes == 60.0
    assert records[1].roll_no is None
    assert records[1].duration_minutes == 90.0


def test_load_attendance_empty(tmp_path: Path) -> None:
    path = tmp_path / "attendance.csv"
    write_attendance_csv(path, [])
    assert load_attendance(path) == []


# --- fuzzy_match_score ---


def test_fuzzy_match_score_high() -> None:
    # "Amritesh Praveen" vs "Amritesh Praveen": exact match
    score = fuzzy_match_score("Amritesh Praveen", "Amritesh Praveen")
    assert score == 1.0


def test_fuzzy_match_score_low() -> None:
    score = fuzzy_match_score("Alice", "Bob")
    assert score < 0.5


def test_fuzzy_match_score_exact() -> None:
    score = fuzzy_match_score("Amritesh Praveen", "Amritesh Praveen")
    assert score == 1.0


# --- Roll number matching ---


def test_match_roll_no_success(tmp_path: Path) -> None:
    audio = make_audio(tmp_path, "audioAnshi_23013186578705.m4a", "Anshi", "2301", "23013186578705")
    manifest = make_manifest(tmp_path, [audio])
    roster = [RosterEntry(name="Anshi Kumar", roll_no="2301", email="anshi@example.com")]

    identity_map = match_files(manifest, roster, [], ["Amritesh Praveen"])

    assert len(identity_map.entries) == 1
    assert identity_map.entries[0].matched_roll_no == "2301"
    assert identity_map.entries[0].match_method == "roll_no"
    assert identity_map.entries[0].match_confidence == 1.0


def test_match_roll_no_no_match(tmp_path: Path) -> None:
    audio = make_audio(tmp_path, "audioAnshi_99991234567890.m4a", "Anshi", "9999")
    manifest = make_manifest(tmp_path, [audio])
    roster = [RosterEntry(name="Anshi Kumar", roll_no="2301", email="anshi@example.com")]

    identity_map = match_files(manifest, roster, [], ["Amritesh Praveen"])

    assert len(identity_map.entries) == 0
    assert len(identity_map.unmatched_entries) == 1


# --- Teacher matching ---


def test_match_teacher_fuzzy(tmp_path: Path) -> None:
    # Use a name that scores above 0.75 against the teacher — exact display name match.
    audio = make_audio(tmp_path, "audioAmriteshPraveen_23456789.m4a", "Amritesh Praveen", None)
    manifest = make_manifest(tmp_path, [audio])

    identity_map = match_files(manifest, [], [], ["Amritesh Praveen"])

    assert identity_map.teacher_audio_file == "audioAmriteshPraveen_23456789.m4a"
    assert len(identity_map.entries) == 0
    assert len(identity_map.unmatched_entries) == 0


def test_match_teacher_no_audio(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, [])
    identity_map = match_files(manifest, [], [], ["Amritesh Praveen"])
    assert identity_map.teacher_audio_file is None


def test_match_teacher_exact(tmp_path: Path) -> None:
    audio = make_audio(tmp_path, "audioAmritesh Praveen_1234.m4a", "Amritesh Praveen", None)
    manifest = make_manifest(tmp_path, [audio])

    identity_map = match_files(manifest, [], [], ["Amritesh Praveen"])

    assert identity_map.teacher_audio_file is not None


# --- Tagging ---


def test_tagging_short_duration(tmp_path: Path) -> None:
    from scripts.models.identity import AttendanceRecord

    audio = make_audio(tmp_path, "audioAnshi_23013186578705.m4a", "Anshi", "2301")
    manifest = make_manifest(tmp_path, [audio])
    roster = [RosterEntry(name="Anshi Kumar", roll_no="2301", email="anshi@example.com")]
    attendance = [AttendanceRecord(name="Anshi", roll_no="2301", duration_minutes=3.0)]

    identity_map = match_files(manifest, roster, attendance, ["Teacher"], short_duration_threshold=5.0)

    assert "short_duration" in identity_map.entries[0].tags


def test_tagging_normal_duration(tmp_path: Path) -> None:
    from scripts.models.identity import AttendanceRecord

    audio = make_audio(tmp_path, "audioAnshi_23013186578705.m4a", "Anshi", "2301")
    manifest = make_manifest(tmp_path, [audio])
    roster = [RosterEntry(name="Anshi Kumar", roll_no="2301", email="anshi@example.com")]
    attendance = [AttendanceRecord(name="Anshi", roll_no="2301", duration_minutes=60.0)]

    identity_map = match_files(manifest, roster, attendance, ["Teacher"])

    assert "short_duration" not in identity_map.entries[0].tags


def test_tagging_unmatched(tmp_path: Path) -> None:
    audio = make_audio(tmp_path, "audioUnknown_99991234567890.m4a", "Unknown", "9999")
    manifest = make_manifest(tmp_path, [audio])

    identity_map = match_files(manifest, [], [], ["Teacher"])

    entry = identity_map.unmatched_entries[0]
    assert entry.is_unmatched is True
    assert "unmatched" in entry.tags


# --- Roster students without audio ---


def test_roster_without_audio(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, [])
    roster = [RosterEntry(name="Anshi Kumar", roll_no="2301", email="anshi@example.com")]

    identity_map = match_files(manifest, roster, [], ["Teacher"])

    assert "2301" in identity_map.roster_students_without_audio


# --- Duplicate roll number guard ---


def test_duplicate_roll_no_raises(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, [])
    roster = [
        RosterEntry(name="Anshi", roll_no="2301", email="a@b.com"),
        RosterEntry(name="Other", roll_no="2301", email="c@d.com"),
    ]
    with pytest.raises(ValueError, match="Duplicate roll numbers"):
        match_files(manifest, roster, [], ["Teacher"])


# --- validate_inputs ---


def test_validate_inputs_missing_manifest(tmp_path: Path) -> None:
    args = MatchArgs(
        manifest_path=tmp_path / "missing.json",
        teacher=["Teacher"],
    )
    with pytest.raises(ValueError, match="Manifest file not found"):
        validate_inputs(args)


def test_validate_inputs_no_teacher(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = MatchArgs(manifest_path=manifest, teacher=[])
    with pytest.raises(ValueError, match="--teacher"):
        validate_inputs(args)


# --- identity_map.json written ---


def test_identity_map_json_written(tmp_path: Path) -> None:
    import json

    audio = make_audio(tmp_path, "audioAnshi_23013186578705.m4a", "Anshi", "2301")
    manifest = make_manifest(tmp_path, [audio])
    roster = [RosterEntry(name="Anshi Kumar", roll_no="2301", email="anshi@example.com")]

    identity_map = match_files(manifest, roster, [], ["Teacher"])

    output_path = tmp_path / "identity_map.json"
    output_path.write_text(identity_map.model_dump_json(indent=2), encoding="utf-8")

    data = json.loads(output_path.read_text())
    assert data["entries"][0]["matched_roll_no"] == "2301"
    restored = IdentityMap.model_validate(data)
    assert len(restored.entries) == 1
