from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from scripts.models.identity import PerStudentAudioFile, ZoomFileManifest

logger = logging.getLogger(__name__)

MAX_ZIP_ENTRY_COUNT = 1000
MAX_ZIP_UNCOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024


class ZipSafetyError(RuntimeError):
    pass


class IngestArgs(BaseModel):
    input_path: Path
    output_dir: Path


def parse_args(argv: Sequence[str] | None = None) -> IngestArgs:
    parser = argparse.ArgumentParser(
        description="Extract a Zoom .zip and build a file discovery manifest."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        dest="input_path",
        help="Path to a .zip file or a directory of .zip files (batch mode).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        dest="output_dir",
        help="Base output directory. Each class gets a subdirectory here.",
    )
    namespace = parser.parse_args(argv)
    return IngestArgs.model_validate(vars(namespace))


def validate_inputs(args: IngestArgs) -> None:
    if not args.input_path.exists():
        raise ValueError(f"Input path does not exist: {args.input_path}")
    if args.input_path.is_file() and args.input_path.suffix.lower() != ".zip":
        raise ValueError(f"Input file must be a .zip: {args.input_path}")


def parse_m4a_filename(filename: str) -> tuple[str | None, str | None, str | None]:
    stem = filename[:-4] if filename.lower().endswith(".m4a") else filename
    if not stem.startswith("audio"):
        return None, None, None
    rest = stem[len("audio"):]
    if not rest:
        return None, None, None

    last_underscore = rest.rfind("_")
    if last_underscore == -1:
        m = re.match(r"^(.*?)(\d{9,})$", rest)
        if m:
            name_part = m.group(1).rstrip("0123456789") or None
            return name_part, None, None
        return rest, None, None

    candidate_name = rest[:last_underscore]
    candidate_number = rest[last_underscore + 1:]

    if candidate_number.isdigit():
        roll_no = candidate_number[:4] if candidate_number else None
        return candidate_name or None, candidate_number, roll_no

    return rest, None, None


def parse_recording_conf(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read recording.conf: %s", exc)
        return result
    for line in text.splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def classify_files(raw_dir: Path, class_name: str) -> ZoomFileManifest:
    session_mp4: Path | None = None
    mixed_m4a: Path | None = None
    per_student_m4as: list[PerStudentAudioFile] = []
    chat_file: Path | None = None
    recording_conf: dict[str, str] | None = None
    zoomver_tag: str | None = None

    for file_path in sorted(raw_dir.rglob("*")):
        if not file_path.is_file():
            continue
        name = file_path.name
        suffix = file_path.suffix.lower()

        if suffix == ".mp4":
            if session_mp4 is None:
                session_mp4 = file_path
            else:
                logger.warning("Multiple MP4 files found; using first: %s", session_mp4)
        elif suffix == ".m4a":
            in_audio_record = "audio record" in str(file_path.parent).lower()
            if name.startswith("audio") and in_audio_record:
                display_name, extracted_number, roll_no_4digit = parse_m4a_filename(name)
                per_student_m4as.append(
                    PerStudentAudioFile(
                        path=file_path,
                        filename=name,
                        display_name=display_name,
                        extracted_number=extracted_number,
                        roll_no_4digit=roll_no_4digit,
                    )
                )
            else:
                if mixed_m4a is None:
                    mixed_m4a = file_path
                else:
                    logger.warning("Multiple non-audio M4A files found; using first: %s", mixed_m4a)
        elif suffix == ".txt" and "chat" in name.lower():
            if chat_file is None:
                chat_file = file_path
            else:
                logger.warning("Multiple chat files found; using first: %s", chat_file)
        elif name == "recording.conf":
            parsed = parse_recording_conf(file_path)
            recording_conf = parsed if parsed else None
        elif name == "zoomver.tag":
            try:
                zoomver_tag = file_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError as exc:
                logger.warning("Could not read zoomver.tag: %s", exc)
        else:
            logger.debug("Unclassified file: %s", file_path)

    if session_mp4 is None:
        logger.warning("No session MP4 found in %s", raw_dir)
    if not per_student_m4as:
        logger.warning("No per-student M4A files found in %s", raw_dir)

    return ZoomFileManifest(
        class_name=class_name,
        raw_dir=raw_dir,
        session_mp4=session_mp4,
        mixed_m4a=mixed_m4a,
        per_student_m4as=per_student_m4as,
        chat_file=chat_file,
        recording_conf=recording_conf,
        zoomver_tag=zoomver_tag,
    )


def check_zip_safety(
    zf: zipfile.ZipFile,
    *,
    max_entries: int = MAX_ZIP_ENTRY_COUNT,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
) -> None:
    infos = zf.infolist()
    if len(infos) > max_entries:
        raise ZipSafetyError(
            f"Zip has {len(infos)} entries, exceeding the limit of {max_entries}."
        )
    total = sum(info.file_size for info in infos)
    if total > max_uncompressed_bytes:
        raise ZipSafetyError(
            f"Zip uncompressed size {total} bytes exceeds the limit of "
            f"{max_uncompressed_bytes} bytes."
        )


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    class_name = zip_path.stem
    raw_dir = output_dir / class_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        check_zip_safety(zf)
        zf.extractall(raw_dir)
    logger.info("Extracted %s -> %s", zip_path.name, raw_dir)
    return raw_dir


def write_manifest(manifest: ZoomFileManifest, output_dir: Path) -> Path:
    manifest_path = output_dir / manifest.class_name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest_path


def process_zip(zip_path: Path, output_dir: Path) -> ZoomFileManifest:
    class_name = zip_path.stem
    raw_dir = extract_zip(zip_path, output_dir)
    manifest = classify_files(raw_dir, class_name)
    write_manifest(manifest, output_dir)
    return manifest


def process_batch(input_dir: Path, output_dir: Path) -> list[ZoomFileManifest]:
    zips = sorted(input_dir.glob("*.zip"))
    if not zips:
        raise ValueError(f"No .zip files found in {input_dir}")
    manifests: list[ZoomFileManifest] = []
    for zip_path in zips:
        logger.info("Processing %s", zip_path.name)
        manifests.append(process_zip(zip_path, output_dir))
    return manifests


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as exc:
        logger.error("Input validation failed: %s", exc)
        raise SystemExit(2) from exc

    if args.input_path.is_dir():
        manifests = process_batch(args.input_path, args.output_dir)
        print(f"Processed {len(manifests)} zip(s).")
        batch_path = args.output_dir / "batch_manifest.json"
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(
            json.dumps([m.model_dump(mode="json") for m in manifests], indent=2),
            encoding="utf-8",
        )
        print(f"Batch manifest written to {batch_path}")
    else:
        manifest = process_zip(args.input_path, args.output_dir)
        print(f"Manifest written for class '{manifest.class_name}'.")
        print(f"  Session MP4:      {manifest.session_mp4}")
        print(f"  Mixed M4A:        {manifest.mixed_m4a}")
        print(f"  Per-student M4As: {len(manifest.per_student_m4as)}")


if __name__ == "__main__":
    main()
