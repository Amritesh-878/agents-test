from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel


class ExtractionArgs(BaseModel):
    input_path: Path
    output_path: Path = Path("output/audio.wav")


class AudioProperties(BaseModel):
    channels: int
    duration_seconds: float
    sample_rate: int


class AudioExtractionError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> ExtractionArgs:
    parser = argparse.ArgumentParser(
        description="Extract 16kHz mono WAV audio from a Zoom MP4 recording."
    )
    parser.add_argument("--input", required=True, help="Path to the input MP4 recording.")
    parser.add_argument(
        "--output",
        default="output/audio.wav",
        help="Path to the output WAV file.",
    )
    namespace = parser.parse_args(argv)
    return ExtractionArgs(
        input_path=Path(namespace.input),
        output_path=Path(namespace.output),
    )


def validate_inputs(args: ExtractionArgs) -> None:
    if not args.input_path.exists():
        raise ValueError(f"Input file does not exist: {args.input_path}")
    if not args.input_path.is_file():
        raise ValueError(f"Input path is not a file: {args.input_path}")
    if args.input_path.suffix.lower() != ".mp4":
        raise ValueError("Input file must use the .mp4 extension.")
    if args.output_path.suffix.lower() != ".wav":
        raise ValueError("Output file must use the .wav extension.")


def build_ffmpeg_command(input_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-vn",
        str(output_path),
    ]


def build_ffprobe_command(output_path: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(output_path),
    ]


def run_command(command: Sequence[str], tool_name: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as error:
        raise AudioExtractionError(
            f"{tool_name} is not available in PATH. Install ffmpeg and ensure both ffmpeg and ffprobe are available."
        ) from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip() or str(error)
        raise AudioExtractionError(f"{tool_name} failed: {details}") from error


def extract_audio(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting audio from {input_path} to {output_path}...")
    run_command(build_ffmpeg_command(input_path, output_path), "ffmpeg")
    if not output_path.exists():
        raise AudioExtractionError(f"ffmpeg did not create the expected output file: {output_path}")
    if output_path.stat().st_size <= 0:
        raise AudioExtractionError(f"ffmpeg created an empty output file: {output_path}")


def parse_ffprobe_output(raw_output: str) -> AudioProperties:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise AudioExtractionError(f"ffprobe returned invalid JSON: {error}") from error

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise AudioExtractionError("ffprobe returned no stream metadata.")

    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        streams[0],
    )
    format_payload = payload.get("format")
    if not isinstance(format_payload, dict):
        raise AudioExtractionError("ffprobe returned no format metadata.")

    sample_rate_text = str(audio_stream.get("sample_rate", "")).strip()
    channels_value = audio_stream.get("channels")
    duration_text = str(format_payload.get("duration", audio_stream.get("duration", ""))).strip()

    try:
        sample_rate = int(sample_rate_text)
    except ValueError as error:
        raise AudioExtractionError(f"ffprobe returned an invalid sample rate: {sample_rate_text!r}") from error

    if not isinstance(channels_value, int):
        raise AudioExtractionError(f"ffprobe returned invalid channel metadata: {channels_value!r}")

    try:
        duration_seconds = float(duration_text)
    except ValueError as error:
        raise AudioExtractionError(f"ffprobe returned an invalid duration: {duration_text!r}") from error

    return AudioProperties(
        channels=channels_value,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
    )


def probe_audio(output_path: Path) -> AudioProperties:
    print(f"Validating extracted audio with ffprobe: {output_path}")
    completed = run_command(build_ffprobe_command(output_path), "ffprobe")
    return parse_ffprobe_output(completed.stdout)


def validate_audio_properties(properties: AudioProperties) -> None:
    if properties.sample_rate != 16000:
        raise AudioExtractionError(
            f"Expected 16kHz audio, got {properties.sample_rate}Hz instead."
        )
    if properties.channels != 1:
        raise AudioExtractionError(
            f"Expected mono audio, got {properties.channels} channels instead."
        )
    if properties.duration_seconds <= 0:
        raise AudioExtractionError("Expected a positive audio duration from ffprobe.")


def format_duration(duration_seconds: float) -> str:
    total_seconds = int(round(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        validate_inputs(args)
        extract_audio(args.input_path, args.output_path)
        properties = probe_audio(args.output_path)
        validate_audio_properties(properties)
    except (AudioExtractionError, ValueError) as error:
        print(f"Audio extraction failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(
        "Extraction complete: "
        f"{format_duration(properties.duration_seconds)} at {properties.sample_rate}Hz mono -> {args.output_path}"
    )


if __name__ == "__main__":
    main()