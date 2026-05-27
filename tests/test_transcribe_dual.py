from __future__ import annotations

from pathlib import Path

import pytest

from scripts.extract_audio import validate_inputs as ea_validate_inputs
from scripts.extract_audio import ExtractionArgs
from scripts.models.transcript import DualLanguageWord, TranscriptWord
from scripts.transcribe_dual import (
    TranscribeArgs,
    build_transcript_document,
    compute_language_stats,
    merge_by_word_probability,
    resegment,
    validate_inputs,
)


# --- Helpers ---


def w(start: float, end: float, word: str, score: float | None = None) -> TranscriptWord:
    return TranscriptWord(start=start, end=end, word=word, score=score)


def dw(start: float, end: float, word: str, score: float, lang: str) -> DualLanguageWord:
    return DualLanguageWord(start=start, end=end, word=word, score=score, source_language=lang)


# ---------------------------------------------------------------------------
# merge_by_word_probability
# ---------------------------------------------------------------------------


def test_merge_overlap_hi_wins() -> None:
    hi = [w(0.0, 0.5, "yeh", 0.9)]
    en = [w(0.0, 0.5, "yeh", 0.4)]
    result = merge_by_word_probability(hi, en)
    assert len(result) == 1
    assert result[0].source_language == "hi"
    assert result[0].score == 0.9


def test_merge_overlap_en_wins() -> None:
    hi = [w(0.5, 1.0, "string", 0.3)]
    en = [w(0.5, 1.0, "string", 0.8)]
    result = merge_by_word_probability(hi, en)
    assert len(result) == 1
    assert result[0].source_language == "en"
    assert result[0].score == 0.8


def test_merge_equal_scores_hindi_preferred() -> None:
    hi = [w(0.0, 0.5, "kya", 0.7)]
    en = [w(0.0, 0.5, "kya", 0.7)]
    result = merge_by_word_probability(hi, en)
    assert result[0].source_language == "hi"


def test_merge_only_hi_word() -> None:
    hi = [w(0.0, 0.5, "hello", 0.8), w(0.5, 1.0, "world", 0.7)]
    en = [w(0.0, 0.5, "hello", 0.6)]
    result = merge_by_word_probability(hi, en)
    assert len(result) == 2
    assert result[0].source_language == "hi"
    assert result[1].word == "world"
    assert result[1].source_language == "hi"


def test_merge_only_en_word() -> None:
    hi = [w(0.0, 0.5, "yeh", 0.6)]
    en = [w(0.0, 0.5, "yeh", 0.5), w(0.5, 1.0, "function", 0.9)]
    result = merge_by_word_probability(hi, en)
    assert len(result) == 2
    assert result[1].word == "function"
    assert result[1].source_language == "en"


def test_merge_no_overlap_time_order() -> None:
    hi = [w(0.0, 0.5, "one", 0.9)]
    en = [w(1.0, 1.5, "two", 0.8)]
    result = merge_by_word_probability(hi, en)
    assert len(result) == 2
    assert result[0].word == "one"
    assert result[0].source_language == "hi"
    assert result[1].word == "two"
    assert result[1].source_language == "en"


def test_merge_none_score_treated_as_zero() -> None:
    hi = [w(0.0, 0.5, "kya", None)]
    en = [w(0.0, 0.5, "kya", 0.1)]
    result = merge_by_word_probability(hi, en)
    # 0.0 < 0.1 → en wins
    assert result[0].source_language == "en"


def test_merge_negative_score_treated_as_zero() -> None:
    hi = [w(0.0, 0.5, "kya", -0.5)]
    en = [w(0.0, 0.5, "kya", 0.0)]
    # Both treated as 0.0 → tie → Hindi preferred
    result = merge_by_word_probability(hi, en)
    assert result[0].source_language == "hi"


def test_merge_empty_both() -> None:
    assert merge_by_word_probability([], []) == []


def test_merge_empty_hi() -> None:
    en = [w(0.0, 0.5, "hello", 0.8)]
    result = merge_by_word_probability([], en)
    assert len(result) == 1
    assert result[0].source_language == "en"


def test_merge_empty_en() -> None:
    hi = [w(0.0, 0.5, "namaste", 0.9)]
    result = merge_by_word_probability(hi, [])
    assert len(result) == 1
    assert result[0].source_language == "hi"


# ---------------------------------------------------------------------------
# resegment
# ---------------------------------------------------------------------------


def test_resegment_within_gap() -> None:
    words = [
        dw(0.0, 0.5, "hello", 0.9, "en"),
        dw(0.6, 1.0, "world", 0.8, "en"),
    ]
    segs = resegment(words, gap_threshold=1.5)
    assert len(segs) == 1
    assert segs[0].text == "hello world"


def test_resegment_across_gap() -> None:
    words = [
        dw(0.0, 0.5, "hello", 0.9, "en"),
        dw(2.5, 3.0, "world", 0.8, "en"),
    ]
    segs = resegment(words, gap_threshold=1.5)
    assert len(segs) == 2
    assert segs[0].text == "hello"
    assert segs[1].text == "world"


def test_resegment_single_word() -> None:
    words = [dw(0.0, 0.5, "kya", 0.9, "hi")]
    segs = resegment(words)
    assert len(segs) == 1
    assert segs[0].start == 0.0
    assert segs[0].end == 0.5


def test_resegment_empty() -> None:
    assert resegment([]) == []


def test_resegment_segment_timestamps() -> None:
    words = [
        dw(1.0, 1.5, "first", 0.8, "en"),
        dw(1.6, 2.0, "second", 0.7, "hi"),
    ]
    segs = resegment(words)
    assert segs[0].start == 1.0
    assert segs[0].end == 2.0


# ---------------------------------------------------------------------------
# compute_language_stats
# ---------------------------------------------------------------------------


def test_compute_stats_hi_dominant() -> None:
    words = [
        dw(0.0, 0.5, "yeh", 0.9, "hi"),
        dw(0.5, 1.0, "kya", 0.8, "hi"),
        dw(1.0, 1.5, "is", 0.7, "en"),
    ]
    hi_avg, en_avg, dominant = compute_language_stats(words)
    assert dominant == "hi"
    assert abs(hi_avg - 0.85) < 0.01
    assert en_avg == 0.7


def test_compute_stats_en_dominant() -> None:
    words = [
        dw(0.0, 0.5, "function", 0.9, "en"),
        dw(0.5, 1.0, "returns", 0.8, "en"),
    ]
    _, _, dominant = compute_language_stats(words)
    assert dominant == "en"


def test_compute_stats_empty() -> None:
    hi_avg, en_avg, dominant = compute_language_stats([])
    assert hi_avg == 0.0
    assert en_avg == 0.0


# ---------------------------------------------------------------------------
# build_transcript_document
# ---------------------------------------------------------------------------


def test_build_transcript_document_structure() -> None:
    words = [
        dw(0.0, 0.5, "yeh", 0.9, "hi"),
        dw(0.6, 1.0, "function", 0.8, "en"),
        dw(3.0, 3.5, "hai", 0.85, "hi"),
    ]
    doc = build_transcript_document(words, "small")
    assert doc.model == "small"
    assert len(doc.segments) == 2  # gap > 1.5s splits at 3.0
    assert doc.segments[0].text == "yeh function"
    assert doc.segments[1].text == "hai"


# ---------------------------------------------------------------------------
# validate_inputs for transcribe_dual
# ---------------------------------------------------------------------------


def test_validate_inputs_missing_manifest(tmp_path: Path) -> None:
    args = TranscribeArgs(
        manifest_path=tmp_path / "missing.json",
        output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="Manifest not found"):
        validate_inputs(args)


def test_validate_inputs_invalid_single_language(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = TranscribeArgs(
        manifest_path=manifest,
        output_dir=tmp_path / "out",
        single_language="fr",
    )
    with pytest.raises(ValueError, match="--single-language"):
        validate_inputs(args)


def test_validate_inputs_single_language_hi_ok(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = TranscribeArgs(
        manifest_path=manifest,
        output_dir=tmp_path / "out",
        single_language="hi",
    )
    validate_inputs(args)  # should not raise


# ---------------------------------------------------------------------------
# M4A extraction — extract_audio.validate_inputs
# ---------------------------------------------------------------------------


def test_extract_audio_accepts_mp4(tmp_path: Path) -> None:
    f = tmp_path / "session.mp4"
    f.write_bytes(b"")
    args = ExtractionArgs(input_path=f, output_path=tmp_path / "out.wav")
    ea_validate_inputs(args)  # should not raise


def test_extract_audio_accepts_m4a(tmp_path: Path) -> None:
    f = tmp_path / "student.m4a"
    f.write_bytes(b"")
    args = ExtractionArgs(input_path=f, output_path=tmp_path / "out.wav")
    ea_validate_inputs(args)  # should not raise


def test_extract_audio_rejects_mp3(tmp_path: Path) -> None:
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"")
    args = ExtractionArgs(input_path=f, output_path=tmp_path / "out.wav")
    with pytest.raises(ValueError, match=r"\.(mp4|m4a)"):
        ea_validate_inputs(args)
