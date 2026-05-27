from __future__ import annotations

from pathlib import Path

import pytest

from scripts.merge_transcripts import (
    MergeArgs,
    SpeechEvent,
    _cluster_events,
    _word_overlap_ratio,
    build_merged_segments,
    build_speech_events,
    compute_merge_metadata,
    detect_alignment,
    format_review_md,
    merge_all,
    validate_inputs,
)
from scripts.transcribe_dual import is_hallucinated_segment
from scripts.models.identity import IdentityMap
from scripts.models.transcript import (
    AlignmentResult,
    MergedTranscriptDocument,
    MergeMetadata,
    PerStudentTranscript,
    TranscriptDocument,
    TranscriptSegment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def doc(segments: list[TranscriptSegment]) -> PerStudentTranscript:
    return PerStudentTranscript(
        audio_file="test.m4a",
        transcript=TranscriptDocument(model="small", segments=segments),
        merged_words=[],
    )


def event(start: float, end: float, speaker: str, text: str = "hello") -> SpeechEvent:
    return SpeechEvent(start=start, end=end, speaker=speaker, text=text)


def dummy_identity_map(teacher: str = "Teacher") -> IdentityMap:
    return IdentityMap(teacher_name=teacher)


# ---------------------------------------------------------------------------
# _word_overlap_ratio
# ---------------------------------------------------------------------------


def test_word_overlap_ratio_identical() -> None:
    assert _word_overlap_ratio("hello world", "hello world") == 1.0


def test_word_overlap_ratio_no_overlap() -> None:
    assert _word_overlap_ratio("hello world", "foo bar baz") == 0.0


def test_word_overlap_ratio_partial() -> None:
    score = _word_overlap_ratio("hello world", "hello there")
    assert 0.0 < score < 1.0


def test_word_overlap_ratio_empty() -> None:
    assert _word_overlap_ratio("", "hello") == 0.0
    assert _word_overlap_ratio("hello", "") == 0.0


# ---------------------------------------------------------------------------
# detect_alignment
# ---------------------------------------------------------------------------


def test_alignment_session_aligned() -> None:
    student_segs = [seg(10.0, 15.0, "hello world")]
    session_segs = [seg(10.5, 15.0, "hello world")]
    result = detect_alignment(student_segs, session_segs)
    assert result.mode == "session_aligned"
    assert result.offset == 0.0
    assert result.uncertain is False


def test_alignment_join_offset() -> None:
    student_segs = [seg(0.0, 5.0, "hello world")]
    session_segs = [seg(300.0, 305.0, "hello world")]
    result = detect_alignment(student_segs, session_segs)
    assert result.mode == "join_offset"
    assert abs(result.offset - 300.0) < 0.1


def test_alignment_no_match_uncertain() -> None:
    student_segs = [seg(10.0, 15.0, "xyz abc")]
    session_segs = [seg(100.0, 105.0, "completely different content here")]
    result = detect_alignment(student_segs, session_segs)
    assert result.uncertain is True


def test_alignment_empty_student_uncertain() -> None:
    result = detect_alignment([], [seg(0.0, 5.0, "anything")])
    assert result.uncertain is True
    assert result.mode == "session_aligned"


def test_alignment_blank_text_session_aligned_by_duration() -> None:
    # Same duration -> session_aligned via duration check (doesn't need text)
    student_segs = [seg(0.0, 5.0, "   ")]
    session_segs = [seg(0.0, 5.0, "something")]
    result = detect_alignment(student_segs, session_segs)
    assert result.mode == "session_aligned"
    assert result.offset == 0.0


# ---------------------------------------------------------------------------
# build_speech_events
# ---------------------------------------------------------------------------


def test_build_speech_events_no_offset() -> None:
    student_doc = doc([seg(10.0, 20.0, "hello")])
    alignment = AlignmentResult(mode="session_aligned", offset=0.0)
    events = build_speech_events("Alice", student_doc, alignment)
    assert len(events) == 1
    assert events[0].start == 10.0
    assert events[0].speaker == "Alice"


def test_build_speech_events_with_offset() -> None:
    student_doc = doc([seg(5.0, 10.0, "hello")])
    alignment = AlignmentResult(mode="join_offset", offset=300.0)
    events = build_speech_events("Bob", student_doc, alignment)
    assert abs(events[0].start - 305.0) < 0.01
    assert abs(events[0].end - 310.0) < 0.01


def test_build_speech_events_skips_blank() -> None:
    student_doc = doc([seg(0.0, 5.0, "  "), seg(5.0, 10.0, "hello")])
    alignment = AlignmentResult(mode="session_aligned", offset=0.0)
    events = build_speech_events("Alice", student_doc, alignment)
    assert len(events) == 1
    assert events[0].text == "hello"


def test_build_speech_events_muted_student() -> None:
    student_doc = doc([])
    alignment = AlignmentResult(mode="session_aligned", offset=0.0)
    events = build_speech_events("Silent", student_doc, alignment)
    assert events == []


# ---------------------------------------------------------------------------
# _cluster_events
# ---------------------------------------------------------------------------


def test_cluster_no_overlap() -> None:
    events = [event(0.0, 5.0, "A"), event(6.0, 10.0, "B")]
    clusters = _cluster_events(events)
    assert len(clusters) == 2


def test_cluster_overlap() -> None:
    events = [event(0.0, 6.0, "A"), event(4.0, 10.0, "B")]
    clusters = _cluster_events(events)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_empty() -> None:
    assert _cluster_events([]) == []


def test_cluster_adjacent_not_merged() -> None:
    # End of A == start of B → no overlap → separate clusters
    events = [event(0.0, 5.0, "A"), event(5.0, 10.0, "B")]
    clusters = _cluster_events(events)
    assert len(clusters) == 2


# ---------------------------------------------------------------------------
# build_merged_segments — speaker attribution + text replacement
# ---------------------------------------------------------------------------


def test_merged_single_student() -> None:
    session_segs = [seg(0.0, 10.0, "session text")]
    events = [event(0.0, 10.0, "Alice", "student text")]
    result = build_merged_segments(session_segs, events)
    assert len(result) == 1
    assert result[0].speakers == ["Alice"]
    assert result[0].text == "student text"
    assert result[0].source == "per_student"


def test_merged_two_students_overlapping() -> None:
    session_segs = [seg(0.0, 10.0, "session")]
    events = [event(0.0, 8.0, "Alice"), event(2.0, 10.0, "Bob")]
    result = build_merged_segments(session_segs, events)
    multi = [s for s in result if s.source == "per_student"]
    assert len(multi) == 1
    assert len(multi[0].speakers) == 2


def test_merged_no_students_all_fallback() -> None:
    session_segs = [seg(0.0, 5.0, "session A"), seg(5.0, 10.0, "session B")]
    result = build_merged_segments(session_segs, [])
    assert all(s.source == "session_fallback" for s in result)
    assert all("UNKNOWN" in s.speakers for s in result)


def test_merged_gap_filled_with_session() -> None:
    session_segs = [seg(0.0, 10.0, "session")]
    events = [event(0.0, 3.0, "Alice"), event(7.0, 10.0, "Bob")]
    result = build_merged_segments(session_segs, events)
    per_student = [s for s in result if s.source == "per_student"]
    fallbacks = [s for s in result if s.source == "session_fallback"]
    assert len(per_student) == 2
    assert len(fallbacks) >= 1  # gap [3,7] filled with session


def test_merged_session_spans_two_sequential_events() -> None:
    session_segs = [seg(0.0, 10.0, "whole session")]
    events = [event(0.0, 4.0, "Alice"), event(6.0, 10.0, "Bob")]
    result = build_merged_segments(session_segs, events)
    per_student = [s for s in result if s.source == "per_student"]
    assert len(per_student) == 2
    speakers = {s.speakers[0] for s in per_student}
    assert "Alice" in speakers
    assert "Bob" in speakers


def test_merged_muted_student_no_crash() -> None:
    session_segs = [seg(0.0, 5.0, "session")]
    result = build_merged_segments(session_segs, [])
    assert len(result) >= 1


def test_merged_multi_speaker_primary_has_most_overlap() -> None:
    session_segs = [seg(0.0, 10.0, "session")]
    # Alice covers 0-8 (8s), Bob covers 7-10 (3s) — Alice is primary
    events = [event(0.0, 8.0, "Alice", "alice text"), event(7.0, 10.0, "Bob", "bob text")]
    result = build_merged_segments(session_segs, events)
    multi = [s for s in result if s.source == "per_student"]
    assert multi[0].speakers[0] == "Alice"
    assert multi[0].text == "alice text"


def test_merged_session_fallback_text_used_for_gaps() -> None:
    session_segs = [seg(0.0, 10.0, "session text here")]
    events = [event(3.0, 7.0, "Alice", "student text")]
    result = build_merged_segments(session_segs, events)
    fallbacks = [s for s in result if s.source == "session_fallback"]
    assert any("session text" in s.text for s in fallbacks)


# ---------------------------------------------------------------------------
# compute_merge_metadata
# ---------------------------------------------------------------------------


def test_compute_metadata_counts() -> None:
    from scripts.models.transcript import MergedSegment

    segments = [
        MergedSegment(start=0, end=5, text="a", speakers=["Alice"], source="per_student"),
        MergedSegment(start=5, end=10, text="b", speakers=["UNKNOWN"], source="session_fallback"),
        MergedSegment(start=10, end=15, text="c", speakers=["Alice", "Bob"], source="per_student"),
    ]
    meta = compute_merge_metadata(segments)
    assert meta.total_segments == 3
    assert meta.per_student_segments == 2
    assert meta.session_fallback_segments == 1
    assert meta.multi_speaker_segments == 1


# ---------------------------------------------------------------------------
# format_review_md
# ---------------------------------------------------------------------------


def _make_merged_doc() -> MergedTranscriptDocument:
    from scripts.models.transcript import MergedSegment

    meta = MergeMetadata(
        total_segments=2,
        per_student_segments=1,
        session_fallback_segments=1,
        multi_speaker_segments=0,
        alignment_mode="per_student_canonical",
        merge_method="cluster_then_fill",
    )
    return MergedTranscriptDocument(
        class_name="CS101",
        duration_seconds=600.0,
        segments=[
            MergedSegment(start=0, end=5, text="hello", speakers=["Alice"], source="per_student"),
            MergedSegment(start=5, end=10, text="gap", speakers=["UNKNOWN"], source="session_fallback"),
        ],
        speakers=["Alice"],
        teacher_name="Dr Smith",
        alignment_results={"Alice": AlignmentResult(mode="session_aligned", offset=0.0)},
        metadata=meta,
    )


def test_review_md_contains_alignment_info() -> None:
    merged_doc = _make_merged_doc()
    md = format_review_md(merged_doc)
    assert "session_aligned" in md
    assert "Alice" in md


def test_review_md_contains_stats() -> None:
    merged_doc = _make_merged_doc()
    md = format_review_md(merged_doc)
    assert "Total segments" in md
    assert "Per-student" in md


# ---------------------------------------------------------------------------
# merge_all integration
# ---------------------------------------------------------------------------


def test_merge_all_single_student() -> None:
    session = doc([seg(0.0, 10.0, "hello world"), seg(10.0, 20.0, "goodbye")])
    student = doc([seg(0.0, 10.0, "hello world")])
    imap = dummy_identity_map()
    result = merge_all(session, {"Alice": student}, imap, "TestClass")
    assert result.class_name == "TestClass"
    assert "Alice" in result.speakers
    per_student = [s for s in result.segments if s.source == "per_student"]
    assert len(per_student) >= 1


def test_merge_all_no_students() -> None:
    session = doc([seg(0.0, 10.0, "session text")])
    imap = dummy_identity_map()
    result = merge_all(session, {}, imap, "TestClass")
    assert all(s.source == "session_fallback" for s in result.segments)
    assert result.speakers == []


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------


def test_validate_inputs_missing_session(tmp_path: Path) -> None:
    args = MergeArgs(
        session_transcript=tmp_path / "missing.json",
        student_transcripts_dir=tmp_path,
        identity_map_path=tmp_path / "identity_map.json",
        output_path=tmp_path / "out.json",
    )
    with pytest.raises(ValueError, match="Session transcript not found"):
        validate_inputs(args)


def test_validate_inputs_missing_identity_map(tmp_path: Path) -> None:
    session = tmp_path / "session.json"
    session.write_text("{}")
    args = MergeArgs(
        session_transcript=session,
        student_transcripts_dir=tmp_path,
        identity_map_path=tmp_path / "missing.json",
        output_path=tmp_path / "out.json",
    )
    with pytest.raises(ValueError, match="Identity map not found"):
        validate_inputs(args)


# ---------------------------------------------------------------------------
# Hallucination filter (is_hallucinated_segment)
# ---------------------------------------------------------------------------


def test_hallucination_detected_repeated_word() -> None:
    # "अपने" repeated 10 times — classic Whisper silence artifact
    words = ["अपने"] * 10
    assert is_hallucinated_segment(words) is True


def test_hallucination_not_triggered_short_segment() -> None:
    # Fewer than 8 words — too short to call hallucination
    assert is_hallucinated_segment(["yes"] * 5) is False


def test_hallucination_not_triggered_real_speech() -> None:
    words = "how many days did Mohit take to finish the work".split()
    assert is_hallucinated_segment(words) is False


def test_hallucination_real_repetition_edge() -> None:
    # 8 of 10 identical = 80% > threshold(0.7) → hallucination
    words = ["yes"] * 8 + ["okay", "right"]
    assert is_hallucinated_segment(words) is True


def test_alignment_real_world_same_duration() -> None:
    # Simulate Zoom cloud recording: both student and session end at ~1027s
    student_segs = [seg(0.0, 1027.6, "kya karein aur kaise")]
    session_segs = [seg(0.0, 1027.6, "completely different text here")]
    result = detect_alignment(student_segs, session_segs)
    assert result.mode == "session_aligned"
    assert result.offset == 0.0
    assert result.uncertain is False
