from __future__ import annotations

from pathlib import Path

import pytest

from scripts.extract_audio import validate_inputs as ea_validate_inputs
from scripts.extract_audio import ExtractionArgs
from scripts.models.transcript import DualLanguageWord, TranscriptSegment, TranscriptWord
from scripts.transcribe_dual import (
    TranscribeArgs,
    _decide_gate_language,
    _segments_from_raw,
    build_transcript_document,
    compute_language_stats,
    resegment,
    select_language_per_segment,
    validate_inputs,
)


# --- Helpers ---


def w(start: float, end: float, word: str, score: float | None = None) -> TranscriptWord:
    return TranscriptWord(start=start, end=end, word=word, score=score)


def dw(start: float, end: float, word: str, score: float, lang: str) -> DualLanguageWord:
    return DualLanguageWord(start=start, end=end, word=word, score=score, source_language=lang)


def seg(*words: TranscriptWord) -> TranscriptSegment:
    return TranscriptSegment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(x.word for x in words),
        words=list(words),
    )


class _RawWord:
    """Minimal stand-in for a faster-whisper word object (start/end/word/probability)."""

    def __init__(self, start: float | None, end: float | None, word: str, probability: float) -> None:
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


class _RawSeg:
    def __init__(self, start: float, end: float, words: list[_RawWord]) -> None:
        self.start = start
        self.end = end
        self.words = words


def _has_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


# ---------------------------------------------------------------------------
# select_language_per_segment — per-segment (not per-word) language choice
# ---------------------------------------------------------------------------


def test_select_picks_higher_mean_language_no_interleave() -> None:
    # One clause: en is confident, hi is the garbled low-confidence pass over the same words.
    en = [seg(w(0.0, 0.4, "supply", 0.9), w(0.5, 0.9, "function", 0.9), w(1.0, 1.4, "is", 0.85))]
    hi = [seg(w(0.0, 0.4, "सप्लाई", 0.4), w(0.5, 0.9, "ज़ोग", 0.3), w(1.0, 1.4, "है", 0.35))]
    result = select_language_per_segment(hi, en)
    assert [r.word for r in result] == ["supply", "function", "is"]
    assert {r.source_language for r in result} == {"en"}  # single language, no interleaving


def test_select_bilingual_keeps_both_languages_per_window() -> None:
    # Window A (English clause) then a >1.5s gap then window B (Hindi clause).
    en = [
        seg(w(0.0, 0.4, "the", 0.9), w(0.5, 0.9, "price", 0.9)),
        seg(w(5.0, 5.4, "garbleden", 0.3), w(5.5, 5.9, "noise", 0.25)),
    ]
    hi = [
        seg(w(0.0, 0.4, "गलत", 0.3), w(0.5, 0.9, "शोर", 0.2)),
        seg(w(5.0, 5.4, "क्यों", 0.9), w(5.5, 5.9, "बढ़ा", 0.88)),
    ]
    result = select_language_per_segment(hi, en)
    langs = {r.source_language for r in result}
    assert langs == {"en", "hi"}  # both languages survive across the track
    window_a = [r for r in result if r.start < 1.0]
    window_b = [r for r in result if r.start >= 5.0]
    assert {r.source_language for r in window_a} == {"en"}
    assert {r.source_language for r in window_b} == {"hi"}


def test_select_monolingual_track_produces_zero_devanagari() -> None:
    # Teacher analog: clean confident English vs garbled low-confidence Devanagari, same windows.
    en = [
        seg(w(0.0, 0.4, "okay", 0.88), w(0.5, 0.9, "great", 0.9)),
        seg(w(3.0, 3.4, "now", 0.87), w(3.5, 3.9, "tell", 0.86), w(4.0, 4.4, "me", 0.85)),
    ]
    hi = [
        seg(w(0.0, 0.4, "ज़द़ोंग्या", 0.4), w(0.5, 0.9, "ख़ोगना", 0.45)),
        seg(w(3.0, 3.4, "थाब", 0.5), w(3.5, 3.9, "लग", 0.42), w(4.0, 4.4, "ड़ोंगे", 0.3)),
    ]
    result = select_language_per_segment(hi, en)
    assert all(not _has_devanagari(r.word) for r in result)


def test_select_near_tie_breaks_on_mass_then_hindi() -> None:
    # Means within eps (0.05): hi mean 0.70 over 3 words (mass 2.10), en mean 0.71 over 2 (mass 1.42).
    hi = [seg(w(0.0, 0.3, "एक", 0.70), w(0.4, 0.7, "दो", 0.70), w(0.8, 1.1, "तीन", 0.70))]
    en = [seg(w(0.0, 0.3, "one", 0.71), w(0.4, 0.7, "two", 0.71))]
    result = select_language_per_segment(hi, en)
    assert {r.source_language for r in result} == {"hi"}  # higher mass wins the near-tie


def test_select_exact_tie_prefers_hindi() -> None:
    hi = [seg(w(0.0, 0.4, "हाँ", 0.6))]
    en = [seg(w(0.0, 0.4, "yes", 0.6))]
    result = select_language_per_segment(hi, en)
    assert result[0].source_language == "hi"


def test_select_low_confidence_both_still_emits_higher_mean() -> None:
    # Both passes junk on a noisy window; nothing is dropped here (downstream filter handles it).
    hi = [seg(w(0.0, 0.4, "घ", 0.10), w(0.5, 0.9, "घ", 0.08))]
    en = [seg(w(0.0, 0.4, "uh", 0.15), w(0.5, 0.9, "um", 0.12))]
    result = select_language_per_segment(hi, en)
    assert len(result) == 2
    assert {r.source_language for r in result} == {"en"}  # higher (still low) mean wins


def test_select_one_side_empty_in_window_uses_other() -> None:
    en = [seg(w(0.0, 0.4, "only", 0.7), w(0.5, 0.9, "english", 0.7))]
    result = select_language_per_segment([], en)
    assert [r.word for r in result] == ["only", "english"]
    assert {r.source_language for r in result} == {"en"}


def test_select_empty_inputs() -> None:
    assert select_language_per_segment([], []) == []


# ---------------------------------------------------------------------------
# _segments_from_raw — shaping + hallucination filter (no GPU)
# ---------------------------------------------------------------------------


def test_segments_from_raw_shapes_words_and_scores() -> None:
    raw = [_RawSeg(0.0, 1.0, [_RawWord(0.0, 0.5, "hello", 0.9), _RawWord(0.5, 1.0, "world", 0.8)])]
    segs = _segments_from_raw(raw)
    assert len(segs) == 1
    assert segs[0].text == "hello world"
    assert segs[0].words[0].score == 0.9
    assert segs[0].start == 0.0 and segs[0].end == 1.0


def test_segments_from_raw_drops_hallucinated_segment() -> None:
    # 8+ words dominated by one token -> hallucination -> dropped.
    raw = [_RawSeg(0.0, 4.0, [_RawWord(i * 0.5, i * 0.5 + 0.4, "the", 0.9) for i in range(10)])]
    assert _segments_from_raw(raw) == []


def test_segments_from_raw_skips_words_without_timestamps() -> None:
    raw = [_RawSeg(0.0, 1.0, [_RawWord(None, None, "x", 0.5), _RawWord(0.5, 1.0, "kept", 0.7)])]
    segs = _segments_from_raw(raw)
    assert len(segs) == 1
    assert [x.word for x in segs[0].words] == ["kept"]


# ---------------------------------------------------------------------------
# _decide_gate_language — conservative multi-probe gate (pure logic)
# ---------------------------------------------------------------------------


def test_gate_all_probes_agree_high_prob_returns_language() -> None:
    assert _decide_gate_language([("en", 0.95), ("en", 0.9), ("en", 0.88)], 0.85) == "en"


def test_gate_disagreeing_probes_returns_none() -> None:
    # Opens English, switches Hindi mid-track -> not gated, run both passes.
    assert _decide_gate_language([("en", 0.95), ("hi", 0.92), ("en", 0.9)], 0.85) is None


def test_gate_low_probability_probe_returns_none() -> None:
    assert _decide_gate_language([("en", 0.95), ("en", 0.6), ("en", 0.9)], 0.85) is None


def test_gate_invalid_language_returns_none() -> None:
    assert _decide_gate_language([("fr", 0.99)], 0.85) is None


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
