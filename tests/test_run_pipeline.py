from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.models.pipeline import ClassSessionReport, PipelineReport, StepResult
from scripts.models.transcript import (
    PerStudentTranscript,
    TranscriptDocument,
    TranscriptSegment,
)
from scripts.run_pipeline import RunArgs, validate_inputs


def _write_transcript(
    path: Path,
    segments: list[tuple[float, float, str]],
    audio_file: str,
    is_teacher: bool = False,
) -> None:
    doc = PerStudentTranscript(
        audio_file=audio_file,
        is_teacher=is_teacher,
        transcript=TranscriptDocument(
            model="small",
            segments=[TranscriptSegment(start=s, end=e, text=t) for s, e, t in segments],
        ),
        merged_words=[],
    )
    path.write_text(doc.model_dump_json(), encoding="utf-8")


def _stage_class(tmp_path: Path, *, stage_teacher_transcript: bool) -> tuple[Path, RunArgs]:
    """Stage a class zip + pre-built transcripts for a --skip-transcribe run.

    The student (Anshi, roll 2301) speaks once; the session MP4 transcript is noisy
    UNKNOWN filler; the teacher (Nisha) M4A has clean class content. Returns the zip
    path and a RunArgs configured to skip transcription and embedding.
    """
    roster = tmp_path / "roster.csv"
    roster.write_text("Name,RollNo,Email\nAnshi Kumar,2301,anshi@example.com\n", encoding="utf-8")

    zip_path = tmp_path / "CS101.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("GMT_Session.mp4", "x")
        zf.writestr("Audio Record/audioNisha.m4a", "x")
        zf.writestr("Audio Record/audioAnshi_2301.m4a", "x")

    out = tmp_path / "out"
    transcripts = out / "CS101" / "transcripts"
    transcripts.mkdir(parents=True, exist_ok=True)
    _write_transcript(
        transcripts / "session.json",
        [(0, 5, "noisy session unknown filler"), (5, 10, "more mixed noise")],
        "GMT_Session.mp4",
    )
    _write_transcript(
        transcripts / "audioAnshi_2301.m4a.json",
        [(2, 4, "anshi own answer")],
        "audioAnshi_2301.m4a",
    )
    if stage_teacher_transcript:
        _write_transcript(
            transcripts / "audioNisha.m4a.json",
            [(0, 20, "teacher explains supply and demand clearly")],
            "audioNisha.m4a",
            is_teacher=True,
        )

    args = RunArgs(
        input_path=zip_path,
        output_dir=out,
        teacher=["Nisha"],
        roster_path=roster,
        db_url="",
        skip_transcribe=True,
        skip_embed=True,
    )
    return zip_path, args


# --- validate_inputs ---


def test_validate_inputs_missing_input(tmp_path: Path) -> None:
    args = RunArgs(
        input_path=tmp_path / "nonexistent.zip",
        output_dir=tmp_path / "out",
        teacher=["Dr Smith"],
    )
    with pytest.raises(ValueError, match="not found"):
        validate_inputs(args)


def test_validate_inputs_no_teacher(tmp_path: Path) -> None:
    args = RunArgs(input_path=tmp_path, output_dir=tmp_path / "out", teacher=[])
    with pytest.raises(ValueError, match="teacher"):
        validate_inputs(args)


def test_validate_inputs_directory_ok(tmp_path: Path) -> None:
    args = RunArgs(input_path=tmp_path, output_dir=tmp_path / "out", teacher=["Dr Smith"])
    validate_inputs(args)  # should not raise


# --- PipelineReport model ---


def test_pipeline_report_model() -> None:
    report = PipelineReport(
        input_path="/data/zips",
        sessions=[],
        total_duration_seconds=10.5,
        total_classes=0,
        successful_classes=0,
        failed_classes=0,
    )
    assert report.total_classes == 0
    data = report.model_dump()
    assert "sessions" in data


def test_step_result_model() -> None:
    sr = StepResult(step_name="ingest_zip", success=True, duration_seconds=1.2)
    assert sr.success
    assert sr.error is None


def test_class_session_report_model() -> None:
    sr = StepResult(step_name="ingest_zip", success=True)
    report = ClassSessionReport(
        class_name="CS101",
        zip_file="CS101.zip",
        output_dir="/out/CS101",
        step_results={"ingest_zip": sr},
        success=True,
    )
    assert report.success
    assert "ingest_zip" in report.step_results


# --- run_pipeline batch ---


def test_run_pipeline_no_zips_raises(tmp_path: Path) -> None:
    from scripts.run_pipeline import run_pipeline

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    args = RunArgs(input_path=empty_dir, output_dir=tmp_path / "out", teacher=["T"])
    with pytest.raises(ValueError, match="No .zip files"):
        run_pipeline(args)


def test_run_pipeline_single_zip_calls_process(tmp_path: Path) -> None:
    from scripts.run_pipeline import run_pipeline

    zip_path = tmp_path / "CS101.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dummy.txt", "data")

    mock_report = ClassSessionReport(
        class_name="CS101",
        zip_file=str(zip_path),
        output_dir=str(tmp_path / "CS101"),
        step_results={},
        success=True,
    )

    with patch("scripts.run_pipeline.process_single_class", return_value=mock_report) as mock_proc:
        args = RunArgs(
            input_path=zip_path, output_dir=tmp_path / "out", teacher=["Dr Smith"]
        )
        report = run_pipeline(args)

    mock_proc.assert_called_once()
    assert report.total_classes == 1
    assert report.successful_classes == 1


def test_run_pipeline_batch_two_zips(tmp_path: Path) -> None:
    from scripts.run_pipeline import run_pipeline

    zips_dir = tmp_path / "zips"
    zips_dir.mkdir()
    for name in ["ClassA", "ClassB"]:
        with zipfile.ZipFile(zips_dir / f"{name}.zip", "w") as zf:
            zf.writestr("dummy.txt", "data")

    def mock_process(zip_path: Path, config: RunArgs) -> ClassSessionReport:
        return ClassSessionReport(
            class_name=zip_path.stem,
            zip_file=str(zip_path),
            output_dir=str(tmp_path / zip_path.stem),
            step_results={},
            success=True,
        )

    with patch("scripts.run_pipeline.process_single_class", side_effect=mock_process):
        args = RunArgs(input_path=zips_dir, output_dir=tmp_path / "out", teacher=["T"])
        report = run_pipeline(args)

    assert report.total_classes == 2
    assert report.successful_classes == 2


def test_run_pipeline_failed_class_continues(tmp_path: Path) -> None:
    from scripts.run_pipeline import run_pipeline

    zips_dir = tmp_path / "zips"
    zips_dir.mkdir()
    for name in ["ClassA", "ClassB"]:
        with zipfile.ZipFile(zips_dir / f"{name}.zip", "w") as zf:
            zf.writestr("dummy.txt", "data")

    call_count = 0

    def mock_process(zip_path: Path, config: RunArgs) -> ClassSessionReport:
        nonlocal call_count
        call_count += 1
        success = zip_path.stem != "ClassA"
        return ClassSessionReport(
            class_name=zip_path.stem,
            zip_file=str(zip_path),
            output_dir=str(tmp_path / zip_path.stem),
            step_results={},
            success=success,
            error=None if success else "simulated failure",
        )

    with patch("scripts.run_pipeline.process_single_class", side_effect=mock_process):
        args = RunArgs(input_path=zips_dir, output_dir=tmp_path / "out", teacher=["T"])
        report = run_pipeline(args)

    assert call_count == 2  # both processed even though ClassA failed
    assert report.failed_classes == 1
    assert report.successful_classes == 1


def test_parse_args_speed_flags(tmp_path: Path) -> None:
    from scripts.run_pipeline import parse_args

    args = parse_args(
        [
            "--input", str(tmp_path),
            "--output-dir", str(tmp_path / "o"),
            "--teacher", "Nisha",
            "--vad-filter",
            "--gate-monolingual",
            "--beam-size", "1",
        ]
    )
    assert args.vad_filter is True
    assert args.gate_monolingual is True
    assert args.beam_size == 1


def _capture_transcribe_argv(tmp_path: Path, **overrides: object) -> list[str]:
    from scripts.run_pipeline import process_single_class

    zip_path, base = _stage_class(tmp_path, stage_teacher_transcript=True)
    args = base.model_copy(update={"skip_transcribe": False, **overrides})
    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str]) -> None:
        captured["argv"] = list(argv)

    with patch("scripts.transcribe_dual.main", side_effect=fake_main):
        report = process_single_class(zip_path, args)
    assert report.success, report.error
    return captured["argv"]


def test_speed_flags_forwarded_to_transcribe(tmp_path: Path) -> None:
    argv = _capture_transcribe_argv(
        tmp_path, vad_filter=True, gate_monolingual=True, beam_size=1
    )
    assert "--vad-filter" in argv
    assert "--gate-monolingual" in argv
    assert argv[argv.index("--beam-size") + 1] == "1"


def test_default_speed_flags_not_forwarded(tmp_path: Path) -> None:
    argv = _capture_transcribe_argv(tmp_path)
    assert "--vad-filter" not in argv
    assert "--gate-monolingual" not in argv
    assert "--beam-size" not in argv


# --- teacher M4A as primary class-context source (process_single_class wiring) ---


def test_process_single_class_uses_teacher_transcript(tmp_path: Path) -> None:
    import json

    from scripts.run_pipeline import process_single_class

    zip_path, args = _stage_class(tmp_path, stage_teacher_transcript=True)
    report = process_single_class(zip_path, args)
    assert report.success, report.error

    contexts = json.loads(
        (args.output_dir / "CS101" / "student_contexts.json").read_text(encoding="utf-8")
    )
    present = contexts["present_students"]["2301"]["present_segments"]
    texts = [s["text"] for s in present]
    assert "teacher explains supply and demand clearly" in texts
    assert "anshi own answer" in texts  # student's own contribution kept
    assert not any("noisy session" in t for t in texts)  # session fallback excluded


def test_process_single_class_falls_back_without_teacher_transcript(tmp_path: Path) -> None:
    import json

    from scripts.run_pipeline import process_single_class

    # teacher_audio_file is still detected, but its transcript JSON is absent ->
    # warn + fall back to the full merged (session-MP4-driven) timeline.
    zip_path, args = _stage_class(tmp_path, stage_teacher_transcript=False)
    report = process_single_class(zip_path, args)
    assert report.success, report.error

    contexts = json.loads(
        (args.output_dir / "CS101" / "student_contexts.json").read_text(encoding="utf-8")
    )
    present = contexts["present_students"]["2301"]["present_segments"]
    texts = [s["text"] for s in present]
    assert any("noisy session" in t for t in texts)  # fallback uses session timeline


# --- retrieval layer ---


def test_distance_to_score() -> None:
    from scripts.retrieval import distance_to_score

    assert distance_to_score(0.0) == 1.0
    assert distance_to_score(None) is None
    score = distance_to_score(1.0)
    assert score is not None
    assert 0 < score < 1


def test_retrieve_validate_inputs_empty_student_id() -> None:
    from scripts.retrieval import RetrievalArgs, validate_inputs

    args = RetrievalArgs(student_id="", query="hello", db_url="postgresql://localhost/db")
    with pytest.raises(ValueError, match="Student id"):
        validate_inputs(args)


def test_retrieve_validate_inputs_no_db_url() -> None:
    from scripts.retrieval import RetrievalArgs, validate_inputs

    args = RetrievalArgs(student_id="2301", query="hello", db_url="")
    with pytest.raises(ValueError, match="db-url"):
        validate_inputs(args)


# --- deprecation check ---


def test_old_scripts_renamed_to_txt() -> None:
    base = Path(__file__).parent.parent / "scripts"
    assert (base / "merge.py.txt").exists()
    assert (base / "build_context.py.txt").exists()
    assert (base / "chunk_and_embed.py.txt").exists()
    assert not (base / "merge.py").exists()
    assert not (base / "chunk_and_embed.py").exists()
