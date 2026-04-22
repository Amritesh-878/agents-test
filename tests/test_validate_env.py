from __future__ import annotations

import pytest

from scripts.validate_env import ValidationArgs, ValidationCheck, ValidationSummary, format_vram_gib, parse_args, validate_inputs


def test_parse_args_reads_skip_flags() -> None:
    args = parse_args(["--allow-cpu", "--skip-pyannote", "--skip-whisperx", "--whisperx-model", "small.en"])

    assert args == ValidationArgs(
        allow_cpu=True,
        skip_pyannote=True,
        skip_whisperx=True,
        whisperx_model="small.en",
    )


def test_validate_inputs_rejects_empty_model_names() -> None:
    with pytest.raises(ValueError):
        validate_inputs(ValidationArgs(pyannote_model="   "))

    with pytest.raises(ValueError):
        validate_inputs(ValidationArgs(whisperx_model="   "))


def test_format_vram_gib_formats_expected_units() -> None:
    assert format_vram_gib(4 * 1024 * 1024 * 1024) == "4.0 GiB"


def test_validation_summary_exit_code_tracks_failures() -> None:
    summary = ValidationSummary(
        checks=[
            ValidationCheck(name="python", success=True, details="ok"),
            ValidationCheck(name="cuda", success=False, details="missing"),
        ]
    )

    assert summary.has_failures() is True
    assert summary.exit_code() == 1