from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pytest

from scripts.ingest_zip import (
    IngestArgs,
    classify_files,
    parse_m4a_filename,
    parse_recording_conf,
    process_batch,
    process_zip,
    validate_inputs,
)
from scripts.models.identity import ZoomFileManifest


# --- Helpers ---


def make_zoom_zip(zip_path: Path, nested: bool = False) -> None:
    prefix = "GMT20240101_Recording/" if nested else ""
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{prefix}session.mp4", "fake mp4 data")
        zf.writestr(f"{prefix}mixed.m4a", "fake mixed audio")
        zf.writestr(f"{prefix}Audio Record/audioAnshi_23013186578705.m4a", "fake student audio")
        zf.writestr(f"{prefix}recording.conf", "meetingId=99999\nhost=Dr Smith\n")


# --- Filename Parsing ---


def test_parse_m4a_standard() -> None:
    name, number, roll = parse_m4a_filename("audioAnshi_23013186578705.m4a")
    assert name == "Anshi"
    assert number == "23013186578705"
    assert roll == "2301"


def test_parse_m4a_underscore_in_name() -> None:
    name, number, roll = parse_m4a_filename("audioA_Disha_25043186578705.m4a")
    assert name == "A_Disha"
    assert number == "25043186578705"
    assert roll == "2504"


def test_parse_m4a_no_number() -> None:
    name, number, roll = parse_m4a_filename("audioTeacher.m4a")
    assert name == "Teacher"
    assert number is None
    assert roll is None


def test_parse_m4a_short_number() -> None:
    name, number, roll = parse_m4a_filename("audio_Test_12.m4a")
    assert number == "12"
    assert roll == "12"


def test_parse_m4a_not_audio_prefix() -> None:
    name, number, roll = parse_m4a_filename("mixed_session.m4a")
    assert name is None
    assert number is None
    assert roll is None


def test_parse_m4a_no_underscore_teacher_style() -> None:
    # e.g. audioNisha11031110282.m4a — teacher file with no roll, no underscore
    name, number, roll = parse_m4a_filename("audioNisha11031110282.m4a")
    assert name == "Nisha"
    assert number is None
    assert roll is None


# --- File Classification ---


def test_classify_files_standard_layout(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "session.mp4").write_bytes(b"")
    (raw_dir / "mixed.m4a").write_bytes(b"")
    audio_dir = raw_dir / "Audio Record"
    audio_dir.mkdir()
    (audio_dir / "audioAnshi_23013186578705.m4a").write_bytes(b"")
    (audio_dir / "audioA_Disha_25043186578705.m4a").write_bytes(b"")

    manifest = classify_files(raw_dir, "CS101")

    assert manifest.class_name == "CS101"
    assert manifest.session_mp4 is not None
    assert manifest.mixed_m4a is not None
    assert len(manifest.per_student_m4as) == 2


def test_classify_files_nested_top_level(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    nested = raw_dir / "GMT20240101_Recording"
    nested.mkdir(parents=True)
    (nested / "session.mp4").write_bytes(b"")
    audio_dir = nested / "Audio Record"
    audio_dir.mkdir()
    (audio_dir / "audioBob_12341234567890.m4a").write_bytes(b"")

    manifest = classify_files(raw_dir, "class1")

    assert manifest.session_mp4 is not None
    assert len(manifest.per_student_m4as) == 1


def test_classify_files_no_mp4_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "audioAnshi_23013186578705.m4a").write_bytes(b"")

    with caplog.at_level(logging.WARNING, logger="scripts.ingest_zip"):
        manifest = classify_files(raw_dir, "class1")

    assert manifest.session_mp4 is None
    assert "No session MP4" in caplog.text


def test_classify_files_roll_numbers_parsed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    audio_dir = raw_dir / "Audio Record"
    audio_dir.mkdir(parents=True)
    (audio_dir / "audioAnshi_23013186578705.m4a").write_bytes(b"")

    manifest = classify_files(raw_dir, "class1")

    student = manifest.per_student_m4as[0]
    assert student.roll_no_4digit == "2301"
    assert student.display_name == "Anshi"


# --- Manifest Generation ---


def test_manifest_round_trip(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    audio_dir = raw_dir / "Audio Record"
    audio_dir.mkdir(parents=True)
    (raw_dir / "session.mp4").write_bytes(b"")
    (audio_dir / "audioAnshi_23013186578705.m4a").write_bytes(b"")

    manifest = classify_files(raw_dir, "CS101")
    data = manifest.model_dump()
    restored = ZoomFileManifest.model_validate(data)

    assert restored.class_name == "CS101"
    assert len(restored.per_student_m4as) == 1


def test_manifest_recording_conf_parsed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    conf = raw_dir / "recording.conf"
    conf.write_text("meetingId=12345\nhost=Dr Smith\n", encoding="utf-8")

    manifest = classify_files(raw_dir, "class1")

    assert manifest.recording_conf is not None
    assert manifest.recording_conf["meetingId"] == "12345"
    assert manifest.recording_conf["host"] == "Dr Smith"


def test_manifest_recording_conf_absent(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    manifest = classify_files(raw_dir, "class1")

    assert manifest.recording_conf is None


# --- parse_recording_conf ---


def test_parse_recording_conf_key_value(tmp_path: Path) -> None:
    conf = tmp_path / "recording.conf"
    conf.write_text("meetingId=12345\nhost=Dr Smith\nduration=3600\n", encoding="utf-8")

    result = parse_recording_conf(conf)

    assert result["meetingId"] == "12345"
    assert result["host"] == "Dr Smith"
    assert result["duration"] == "3600"


def test_parse_recording_conf_skips_comments(tmp_path: Path) -> None:
    conf = tmp_path / "recording.conf"
    conf.write_text("# This is a comment\nkey=value\n", encoding="utf-8")

    result = parse_recording_conf(conf)

    assert "key" in result
    assert len(result) == 1


# --- Zip Extraction + process_zip ---


def test_process_zip_creates_manifest(tmp_path: Path) -> None:
    zip_path = tmp_path / "CS101.zip"
    make_zoom_zip(zip_path)
    output_dir = tmp_path / "output"

    manifest = process_zip(zip_path, output_dir)

    assert manifest.class_name == "CS101"
    assert manifest.session_mp4 is not None
    assert len(manifest.per_student_m4as) == 1
    assert (output_dir / "CS101" / "manifest.json").exists()


def test_process_zip_manifest_json_valid(tmp_path: Path) -> None:
    zip_path = tmp_path / "CS101.zip"
    make_zoom_zip(zip_path)
    output_dir = tmp_path / "output"

    process_zip(zip_path, output_dir)

    import json

    raw = json.loads((output_dir / "CS101" / "manifest.json").read_text())
    assert raw["class_name"] == "CS101"
    assert len(raw["per_student_m4as"]) == 1


def test_process_zip_nested_dir(tmp_path: Path) -> None:
    zip_path = tmp_path / "MyClass.zip"
    make_zoom_zip(zip_path, nested=True)
    output_dir = tmp_path / "output"

    manifest = process_zip(zip_path, output_dir)

    assert manifest.session_mp4 is not None
    assert len(manifest.per_student_m4as) == 1


# --- Batch Mode ---


def test_process_batch_two_zips(tmp_path: Path) -> None:
    zips_dir = tmp_path / "zips"
    zips_dir.mkdir()
    for name in ["ClassA", "ClassB"]:
        make_zoom_zip(zips_dir / f"{name}.zip")

    output_dir = tmp_path / "output"
    manifests = process_batch(zips_dir, output_dir)

    assert len(manifests) == 2
    assert {m.class_name for m in manifests} == {"ClassA", "ClassB"}


def test_process_batch_no_zips_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No .zip files found"):
        process_batch(empty_dir, tmp_path / "output")


# --- validate_inputs ---


def test_validate_inputs_missing_path(tmp_path: Path) -> None:
    args = IngestArgs(input_path=tmp_path / "nonexistent.zip", output_dir=tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        validate_inputs(args)


def test_validate_inputs_wrong_extension(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    args = IngestArgs(input_path=f, output_dir=tmp_path)
    with pytest.raises(ValueError, match=".zip"):
        validate_inputs(args)


def test_validate_inputs_directory_accepted(tmp_path: Path) -> None:
    args = IngestArgs(input_path=tmp_path, output_dir=tmp_path)
    validate_inputs(args)  # should not raise
