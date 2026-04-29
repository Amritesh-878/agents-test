from __future__ import annotations

from email.message import Message
import inspect
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from faster_whisper.transcribe import TranscriptionOptions

from scripts.transcribe import (
    RuntimeOptions,
    TranscriptDocument,
    TranscriptionArgs,
    TranscriptionError,
    TranscriptSegment,
    TranscriptWord,
    build_asr_options,
    build_backend_transcription_result,
    build_segment_words,
    build_transcript_document,
    get_batch_size_attempts,
    is_vad_bootstrap_redirect,
    normalize_language_code,
    parse_args,
    resolve_alignment_language,
    resolve_runtime_options,
    validate_inputs,
)


def test_parse_args_uses_default_paths() -> None:
    args = parse_args([])

    assert args == TranscriptionArgs(
        input_path=Path("output/audio.wav"),
        output_path=Path("output/transcript_raw.json"),
        whisper_model="small",
    )


def test_validate_inputs_rejects_missing_wav(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        validate_inputs(TranscriptionArgs(input_path=tmp_path / "audio.wav"))


def test_validate_inputs_rejects_non_wav_input(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.mp3"
    input_path.write_text("fake audio", encoding="utf-8")

    with pytest.raises(ValueError, match=".wav"):
        validate_inputs(TranscriptionArgs(input_path=input_path))


def test_validate_inputs_rejects_non_json_output(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.wav"
    input_path.write_text("fake audio", encoding="utf-8")

    with pytest.raises(ValueError, match=".json"):
        validate_inputs(
            TranscriptionArgs(input_path=input_path, output_path=tmp_path / "transcript.txt")
        )


def test_resolve_runtime_options_prefers_cuda() -> None:
    assert resolve_runtime_options(cuda_available=True, allow_cpu=False) == RuntimeOptions(
        device="cuda",
        compute_type="float16",
    )


def test_resolve_runtime_options_allows_cpu_fallback() -> None:
    assert resolve_runtime_options(cuda_available=False, allow_cpu=True) == RuntimeOptions(
        device="cpu",
        compute_type="int8",
    )


def test_resolve_runtime_options_requires_cuda_by_default() -> None:
    with pytest.raises(TranscriptionError, match="CUDA is not available"):
        resolve_runtime_options(cuda_available=False, allow_cpu=False)


def test_get_batch_size_attempts_adds_retry_at_four() -> None:
    assert get_batch_size_attempts(8) == [8, 4]
    assert get_batch_size_attempts(4) == [4]


def test_build_asr_options_matches_current_faster_whisper_requirements() -> None:
    supported_options = set(inspect.signature(TranscriptionOptions).parameters)
    options = build_asr_options("small")

    assert set(options).issubset(supported_options)
    if "multilingual" in supported_options:
        assert options["multilingual"] is True
    english_only_options = build_asr_options("small.en")
    if "multilingual" in supported_options:
        assert english_only_options["multilingual"] is False


def test_is_vad_bootstrap_redirect_matches_http_301() -> None:
    redirected = HTTPError("https://example.test", 301, "Moved", hdrs=Message(), fp=None)
    other_error = HTTPError("https://example.test", 500, "Broken", hdrs=Message(), fp=None)

    assert is_vad_bootstrap_redirect(redirected) is True
    assert is_vad_bootstrap_redirect(other_error) is False


def test_normalize_language_code_strips_regional_suffix() -> None:
    assert normalize_language_code(" en-IN ") == "en"


def test_resolve_alignment_language_prefers_override() -> None:
    assert resolve_alignment_language("hi", "en") == "en"


def test_build_transcript_document_keeps_word_timestamps() -> None:
    result = {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 1.2,
                "text": "hello class",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.4},
                    {"word": "class", "start": 0.5, "end": 1.2},
                ],
            }
        ],
    }

    assert build_transcript_document(result, "small") == TranscriptDocument(
        language="en",
        model="small",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=1.2,
                text="hello class",
                words=[
                    TranscriptWord(word="hello", start=0.0, end=0.4),
                    TranscriptWord(word="class", start=0.5, end=1.2),
                ],
            )
        ],
    )


def test_build_backend_transcription_result_normalizes_segments() -> None:
    segments = [
        SimpleNamespace(start=0.01, end=0.49, text=" hello "),
        SimpleNamespace(start=0.5, end=1.0, text="world"),
    ]

    assert build_backend_transcription_result(segments, "en-IN") == {
        "language": "en",
        "segments": [
            {"start": 0.01, "end": 0.49, "text": "hello"},
            {"start": 0.5, "end": 1.0, "text": "world"},
        ],
    }


def test_build_segment_words_falls_back_to_segment_span_when_alignment_is_missing() -> None:
    words = build_segment_words(
        raw_words=[{"word": "27.", "start": None, "end": None}],
        segment_start=12.0,
        segment_end=12.4,
        segment_text="27.",
    )

    assert words == [TranscriptWord(word="27.", start=12.0, end=12.4)]


def test_build_transcript_document_falls_back_when_segment_has_no_words_key() -> None:
    result = {
        "language": "en",
        "segments": [{"start": 2.0, "end": 2.4, "text": "27."}],
    }

    assert build_transcript_document(result, "small") == TranscriptDocument(
        language="en",
        model="small",
        segments=[
            TranscriptSegment(
                start=2.0,
                end=2.4,
                text="27.",
                words=[TranscriptWord(word="27.", start=2.0, end=2.4)],
            )
        ],
    )


def test_build_transcript_document_requires_word_timestamps() -> None:
    result = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 0.8, "text": "hello", "words": "bad"}],
    }

    with pytest.raises(TranscriptionError, match="invalid aligned word metadata"):
        build_transcript_document(result, "small")