from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.diarize import (
    DiarizationArgs,
    DiarizationDocument,
    DiarizationError,
    DiarizationSegment,
    RuntimeOptions,
    annotation_to_raw_segments,
    build_diarization_document,
    build_pipeline_kwargs,
    load_hf_token,
    parse_args,
    resolve_runtime_options,
    validate_inputs,
)


def test_parse_args_uses_default_paths() -> None:
    args = parse_args([])

    assert args == DiarizationArgs(
        input_path=Path("output/audio.wav"),
        output_path=Path("output/diarization.json"),
        pipeline_name="pyannote/speaker-diarization-3.1",
        max_speakers=6,
    )


def test_validate_inputs_rejects_missing_wav(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        validate_inputs(DiarizationArgs(input_path=tmp_path / "audio.wav"))


def test_validate_inputs_rejects_non_json_output(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.wav"
    input_path.write_text("fake audio", encoding="utf-8")

    with pytest.raises(ValueError, match=".json"):
        validate_inputs(
            DiarizationArgs(input_path=input_path, output_path=tmp_path / "diarization.txt")
        )


def test_validate_inputs_rejects_min_speakers_above_max(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.wav"
    input_path.write_text("fake audio", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be greater"):
        validate_inputs(
            DiarizationArgs(input_path=input_path, min_speakers=4, max_speakers=3)
        )


def test_resolve_runtime_options_prefers_cuda() -> None:
    assert resolve_runtime_options(cuda_available=True, allow_cpu=False) == RuntimeOptions(
        device="cuda"
    )


def test_resolve_runtime_options_allows_cpu_fallback() -> None:
    assert resolve_runtime_options(cuda_available=False, allow_cpu=True) == RuntimeOptions(
        device="cpu"
    )


def test_resolve_runtime_options_requires_cuda_by_default() -> None:
    with pytest.raises(DiarizationError, match="CUDA is not available"):
        resolve_runtime_options(cuda_available=False, allow_cpu=False)


def test_load_hf_token_prefers_primary_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", " primary-token ")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "fallback-token")

    assert load_hf_token() == "primary-token"


def test_build_pipeline_kwargs_omits_none_values() -> None:
    assert build_pipeline_kwargs(None, 6) == {"max_speakers": 6}
    assert build_pipeline_kwargs(2, None) == {"min_speakers": 2}


def test_annotation_to_raw_segments_reads_pyannote_like_tracks() -> None:
    class FakeAnnotation:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield (SimpleNamespace(start=0.1, end=1.2), None, "SPEAKER_00")
            yield (SimpleNamespace(start=1.3, end=2.5), None, "SPEAKER_01")

    assert annotation_to_raw_segments(FakeAnnotation()) == [
        (0.1, 1.2, "SPEAKER_00"),
        (1.3, 2.5, "SPEAKER_01"),
    ]


def test_build_diarization_document_sorts_segments_and_speakers() -> None:
    raw_segments = [
        (2.0049, 3.1239, "SPEAKER_01"),
        (0.0, 1.9999, "SPEAKER_00"),
    ]

    assert build_diarization_document(raw_segments) == DiarizationDocument(
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
            DiarizationSegment(speaker="SPEAKER_01", start=2.005, end=3.124),
        ],
    )


def test_build_diarization_document_requires_segments() -> None:
    with pytest.raises(DiarizationError, match="no diarization segments"):
        build_diarization_document([])


def test_build_diarization_document_rejects_invalid_segment_spans() -> None:
    with pytest.raises(DiarizationError, match="invalid segment span"):
        build_diarization_document([(1.0, 1.0, "SPEAKER_00")])