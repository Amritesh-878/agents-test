from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_audio import (
    AudioExtractionError,
    AudioProperties,
    ExtractionArgs,
    build_ffmpeg_command,
    format_duration,
    parse_args,
    parse_ffprobe_output,
    validate_audio_properties,
    validate_inputs,
)


def test_parse_args_uses_default_output() -> None:
    args = parse_args(["--input", "recording.mp4"])

    assert args == ExtractionArgs(
        input_path=Path("recording.mp4"),
        output_path=Path("output/audio.wav"),
    )


def test_validate_inputs_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(ValueError, match="does not exist"):
        validate_inputs(ExtractionArgs(input_path=missing_path))


def test_validate_inputs_rejects_non_mp4_input(tmp_path: Path) -> None:
    input_path = tmp_path / "recording.mov"
    input_path.write_text("not a real video", encoding="utf-8")

    with pytest.raises(ValueError, match=".mp4"):
        validate_inputs(ExtractionArgs(input_path=input_path))


def test_validate_inputs_rejects_non_wav_output(tmp_path: Path) -> None:
    input_path = tmp_path / "recording.mp4"
    input_path.write_text("not a real video", encoding="utf-8")

    with pytest.raises(ValueError, match=".wav"):
        validate_inputs(ExtractionArgs(input_path=input_path, output_path=tmp_path / "audio.mp3"))


def test_build_ffmpeg_command_uses_expected_flags() -> None:
    input_path = Path("input.mp4")
    output_path = Path("output/audio.wav")
    command = build_ffmpeg_command(
        input_path=input_path,
        output_path=output_path,
    )

    assert command == [
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


def test_parse_ffprobe_output_reads_audio_properties() -> None:
    payload = json.dumps(
        {
            "streams": [{"codec_type": "audio", "sample_rate": "16000", "channels": 1}],
            "format": {"duration": "3723.4"},
        }
    )

    assert parse_ffprobe_output(payload) == AudioProperties(
        channels=1,
        duration_seconds=3723.4,
        sample_rate=16000,
    )


def test_validate_audio_properties_rejects_wrong_sample_rate() -> None:
    with pytest.raises(AudioExtractionError, match="16kHz"):
        validate_audio_properties(
            AudioProperties(channels=1, duration_seconds=12.0, sample_rate=44100)
        )


def test_validate_audio_properties_rejects_stereo_audio() -> None:
    with pytest.raises(AudioExtractionError, match="mono"):
        validate_audio_properties(
            AudioProperties(channels=2, duration_seconds=12.0, sample_rate=16000)
        )


def test_format_duration_formats_hour_long_audio() -> None:
    assert format_duration(3723.4) == "01:02:03"