from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from scripts.models.pipeline import (
    ClassSessionReport,
    PipelineReport,
    StepResult,
)
from scripts.utils.db_url import resolve_db_url

logger = logging.getLogger(__name__)


class RunArgs(BaseModel):
    input_path: Path
    output_dir: Path
    teacher: list[str]
    roster_path: Path | None = None
    attendance_path: Path | None = None
    db_url: str = ""
    model: str = "small"
    single_language: str | None = None
    allow_cpu: bool = False
    skip_transcribe: bool = False
    skip_embed: bool = False


def parse_args(argv: Sequence[str] | None = None) -> RunArgs:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: zip -> transcript -> context -> pgvector."
    )
    parser.add_argument("--input", required=True, type=Path, dest="input_path")
    parser.add_argument("--output-dir", required=True, type=Path, dest="output_dir")
    parser.add_argument("--teacher", action="append", default=[], dest="teacher")
    parser.add_argument("--roster", type=Path, dest="roster_path", default=None)
    parser.add_argument("--attendance", type=Path, dest="attendance_path", default=None)
    parser.add_argument(
        "--db-url",
        default=None,
        dest="db_url",
        help="PostgreSQL connection URL. Falls back to DATABASE_URL env var.",
    )
    parser.add_argument("--model", default="small")
    parser.add_argument("--single-language", default=None, dest="single_language")
    parser.add_argument("--allow-cpu", action="store_true", dest="allow_cpu")
    parser.add_argument("--skip-transcribe", action="store_true", dest="skip_transcribe")
    parser.add_argument("--skip-embed", action="store_true", dest="skip_embed")
    namespace = parser.parse_args(argv)
    data = vars(namespace)
    data["db_url"] = resolve_db_url(data.get("db_url"))
    return RunArgs.model_validate(data)


def validate_inputs(args: RunArgs) -> None:
    if not args.input_path.exists():
        raise ValueError(f"Input path not found: {args.input_path}")
    if not args.teacher:
        raise ValueError("At least one --teacher name is required.")


def _timed_step(name: str, fn: object, *a: object, **kw: object) -> StepResult:
    t0 = time.monotonic()
    try:

        if callable(fn):
            result = fn(*a, **kw)  # type: ignore[operator]
        elapsed = time.monotonic() - t0
        files: list[str] = []
        if hasattr(result, "__fspath__"):
            files = [str(result)]
        elif isinstance(result, list):
            files = [str(r) for r in result if hasattr(r, "__fspath__")]
        return StepResult(
            step_name=name, success=True, duration_seconds=elapsed, output_files=files
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        # Per-step isolation is intentional (one class failing must not abort the
        # batch); log the full traceback at debug so real defects stay diagnosable.
        logger.error("Step %s failed: %s", name, exc)
        logger.debug("Step %s traceback", name, exc_info=True)
        return StepResult(
            step_name=name, success=False, duration_seconds=elapsed, error=str(exc)
        )


def process_single_class(zip_path: Path, config: RunArgs) -> ClassSessionReport:
    from scripts.build_student_context import build_context_document, class_context_text
    from scripts.ingest_zip import process_zip
    from scripts.match_identity import load_attendance, load_roster, match_files
    from scripts.merge_transcripts import format_review_md, merge_all
    from scripts.models.identity import IdentityMap
    from scripts.models.transcript import MergedTranscriptDocument, PerStudentTranscript
    from scripts.utils.topics import extract_topics

    class_name = zip_path.stem
    output_dir = config.output_dir / class_name
    transcripts_dir = output_dir / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    step_results: dict[str, StepResult] = {}

    # Step 1: ingest zip
    sr = _timed_step("ingest_zip", process_zip, zip_path, config.output_dir)
    step_results["ingest_zip"] = sr
    if not sr.success:
        return ClassSessionReport(
            class_name=class_name, zip_file=str(zip_path),
            output_dir=str(output_dir), step_results=step_results,
            success=False, error=sr.error,
        )
    manifest_path = output_dir / "manifest.json"

    # Step 2: identity matching
    from scripts.models.identity import ZoomFileManifest

    manifest = ZoomFileManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    roster = load_roster(config.roster_path) if config.roster_path else []
    attendance = load_attendance(config.attendance_path) if config.attendance_path else []

    def _match() -> Path:
        identity_map = match_files(manifest, roster, attendance, config.teacher)
        p = output_dir / "identity_map.json"
        p.write_text(identity_map.model_dump_json(indent=2), encoding="utf-8")
        return p

    sr = _timed_step("match_identity", _match)
    step_results["match_identity"] = sr
    if not sr.success:
        return ClassSessionReport(
            class_name=class_name, zip_file=str(zip_path),
            output_dir=str(output_dir), step_results=step_results,
            success=False, error=sr.error,
        )
    identity_map = IdentityMap.model_validate_json(
        (output_dir / "identity_map.json").read_text(encoding="utf-8")
    )

    # Step 3: transcription (optional skip for faster testing)
    if not config.skip_transcribe:
        def _transcribe() -> Path:
            from scripts.transcribe_dual import main as tdmain
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            tdmain([
                "--manifest", str(manifest_path),
                "--output-dir", str(transcripts_dir),
                "--model", config.model,
                *(["--single-language", config.single_language] if config.single_language else []),
                *(["--allow-cpu"] if config.allow_cpu else []),
            ])
            return transcripts_dir

        sr = _timed_step("transcribe_dual", _transcribe)
        step_results["transcribe_dual"] = sr
        if not sr.success:
            return ClassSessionReport(
                class_name=class_name, zip_file=str(zip_path),
                output_dir=str(output_dir), step_results=step_results,
                success=False, error=sr.error,
            )

    # Step 4: transcript merge
    session_transcript_path = transcripts_dir / "session.json"

    def _merge() -> Path:
        if not session_transcript_path.exists():
            raise FileNotFoundError(f"Session transcript not found: {session_transcript_path}")
        session_doc = PerStudentTranscript.model_validate_json(
            session_transcript_path.read_text(encoding="utf-8")
        )
        student_docs: dict[str, PerStudentTranscript] = {}
        for entry in identity_map.entries:
            tf = transcripts_dir / f"{entry.audio_file}.json"
            if tf.exists():
                doc = PerStudentTranscript.model_validate_json(tf.read_text(encoding="utf-8"))
                student_docs[entry.matched_name or entry.audio_file] = doc
        merged = merge_all(session_doc, student_docs, identity_map, class_name)
        p = output_dir / "transcript_merged.json"
        p.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
        (output_dir / "transcript_review.md").write_text(
            format_review_md(merged),
            encoding="utf-8",
        )
        return p

    sr = _timed_step("merge_transcripts", _merge)
    step_results["merge_transcripts"] = sr
    if not sr.success:
        return ClassSessionReport(
            class_name=class_name, zip_file=str(zip_path),
            output_dir=str(output_dir), step_results=step_results,
            success=False, error=sr.error,
        )
    merged_transcript = MergedTranscriptDocument.model_validate_json(
        (output_dir / "transcript_merged.json").read_text(encoding="utf-8")
    )

    # Step 5: student context building
    def _build_context() -> Path:
        # The teacher's isolated M4A is the primary, clean source of class_context/missed.
        # It was transcribed alongside the students to transcripts_dir/{file}.json. When it
        # is absent or its transcript is missing, fall back to the merged session timeline.
        teacher_doc: PerStudentTranscript | None = None
        if identity_map.teacher_audio_file:
            teacher_tf = transcripts_dir / f"{identity_map.teacher_audio_file}.json"
            if teacher_tf.exists():
                teacher_doc = PerStudentTranscript.model_validate_json(
                    teacher_tf.read_text(encoding="utf-8")
                )
            else:
                logger.warning(
                    "Teacher transcript not found: %s - class context falls back to session MP4",
                    teacher_tf.name,
                )
        text = class_context_text(merged_transcript, teacher_doc)
        topics = extract_topics(text)
        doc = build_context_document(
            merged_transcript, identity_map, roster, attendance, topics, teacher_doc
        )
        p = output_dir / "student_contexts.json"
        p.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        return p

    sr = _timed_step("build_context", _build_context)
    step_results["build_context"] = sr
    if not sr.success:
        return ClassSessionReport(
            class_name=class_name, zip_file=str(zip_path),
            output_dir=str(output_dir), step_results=step_results,
            success=False, error=sr.error,
        )

    # Step 6: embed and store (optional skip)
    if not config.skip_embed and config.db_url:
        def _embed() -> None:
            from scripts.embed_and_store import main as emain
            emain([
                "--contexts", str(output_dir / "student_contexts.json"),
                "--db-url", config.db_url,
            ])

        sr = _timed_step("embed_and_store", _embed)
        step_results["embed_and_store"] = sr

    overall_success = all(s.success for s in step_results.values())
    return ClassSessionReport(
        class_name=class_name,
        zip_file=str(zip_path),
        output_dir=str(output_dir),
        step_results=step_results,
        success=overall_success,
    )


def run_pipeline(args: RunArgs) -> PipelineReport:
    if args.input_path.is_dir():
        zips = sorted(args.input_path.glob("*.zip"))
        if not zips:
            raise ValueError(f"No .zip files found in {args.input_path}")
    else:
        zips = [args.input_path]

    t0 = time.monotonic()
    sessions: list[ClassSessionReport] = []
    for zip_path in zips:
        logger.info("Processing %s", zip_path.name)
        report = process_single_class(zip_path, args)
        sessions.append(report)
        if report.success:
            logger.info("  OK %s complete", zip_path.name)
        else:
            logger.warning("  FAIL %s failed: %s", zip_path.name, report.error)

    elapsed = time.monotonic() - t0
    successful = sum(1 for s in sessions if s.success)
    return PipelineReport(
        input_path=str(args.input_path),
        sessions=sessions,
        total_duration_seconds=elapsed,
        total_classes=len(sessions),
        successful_classes=successful,
        failed_classes=len(sessions) - successful,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    report = run_pipeline(args)
    report_path = args.output_dir / "pipeline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nPipeline complete: {report.successful_classes}/{report.total_classes} classes succeeded")
    print(f"Duration: {report.total_duration_seconds:.1f}s")
    print(f"Report: {report_path}")

    if report.failed_classes > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
