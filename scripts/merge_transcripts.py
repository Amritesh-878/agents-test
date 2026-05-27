from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from scripts.models.identity import IdentityMap
from scripts.models.transcript import (
    AlignmentResult,
    DualLanguageWord,
    MergeMetadata,
    MergedSegment,
    MergedTranscriptDocument,
    PerStudentTranscript,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

_ALIGNED_TOLERANCE = 2.0     # seconds: delta < this → session-aligned
_SIMILARITY_THRESHOLD = 0.5  # word-overlap ratio to consider two segments matching
_GAP_EPSILON = 0.05          # seconds: ignore gaps smaller than this


# ---------------------------------------------------------------------------
# Internal data structure
# ---------------------------------------------------------------------------


@dataclass
class SpeechEvent:
    start: float
    end: float
    speaker: str
    text: str
    words: list[DualLanguageWord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure helper functions — fully testable without I/O
# ---------------------------------------------------------------------------


def _word_overlap_ratio(text_a: str, text_b: str) -> float:
    """Dice-coefficient word overlap between two strings."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return 2 * len(words_a & words_b) / (len(words_a) + len(words_b))


def detect_alignment(
    student_segments: list[TranscriptSegment],
    session_segments: list[TranscriptSegment],
    aligned_tolerance: float = _ALIGNED_TOLERANCE,
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
) -> AlignmentResult:
    """Compare first student speech against the session transcript to detect offset.

    Returns session_aligned (offset≈0) or join_offset (offset=delta) or uncertain.
    """
    first_student = next(
        (s for s in student_segments if s.text.strip()),
        None,
    )
    if first_student is None:
        return AlignmentResult(mode="session_aligned", offset=0.0, uncertain=True)

    best_offset: float | None = None
    best_similarity = 0.0

    for session_seg in session_segments:
        sim = _word_overlap_ratio(first_student.text, session_seg.text)
        if sim >= similarity_threshold and sim > best_similarity:
            best_similarity = sim
            best_offset = session_seg.start - first_student.start

    if best_offset is None:
        return AlignmentResult(mode="session_aligned", offset=0.0, uncertain=True)

    if abs(best_offset) <= aligned_tolerance:
        return AlignmentResult(mode="session_aligned", offset=0.0)
    return AlignmentResult(mode="join_offset", offset=best_offset)


def build_speech_events(
    speaker: str,
    doc: PerStudentTranscript,
    alignment: AlignmentResult,
) -> list[SpeechEvent]:
    """Convert a student transcript into time-corrected SpeechEvents."""
    offset = alignment.offset
    events: list[SpeechEvent] = []
    for seg in doc.transcript.segments:
        if not seg.text.strip():
            continue
        adj_start = seg.start + offset
        adj_end = seg.end + offset
        # Carry words that fall within this segment's original time range
        seg_words = [
            DualLanguageWord(
                start=w.start + offset,
                end=w.end + offset,
                word=w.word,
                score=w.score,
                source_language=w.source_language,
            )
            for w in doc.merged_words
            if w.start >= seg.start and w.end <= seg.end
        ]
        events.append(
            SpeechEvent(
                start=adj_start,
                end=adj_end,
                speaker=speaker,
                text=seg.text,
                words=seg_words,
            )
        )
    return events


def _cluster_events(events: list[SpeechEvent]) -> list[list[SpeechEvent]]:
    """Group overlapping SpeechEvents into clusters (maximal overlapping sets)."""
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: e.start)
    clusters: list[list[SpeechEvent]] = [[sorted_events[0]]]
    for event in sorted_events[1:]:
        cluster_end = max(e.end for e in clusters[-1])
        if event.start < cluster_end:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return clusters


def _cluster_to_merged(cluster: list[SpeechEvent]) -> MergedSegment:
    """Convert a cluster of overlapping events into a single MergedSegment."""
    c_start = min(e.start for e in cluster)
    c_end = max(e.end for e in cluster)

    # Sort by overlap duration with the full cluster window (primary speaker first)
    ranked = sorted(
        cluster,
        key=lambda e: min(e.end, c_end) - max(e.start, c_start),
        reverse=True,
    )
    speakers = [e.speaker for e in ranked]
    primary = ranked[0]
    return MergedSegment(
        start=c_start,
        end=c_end,
        text=primary.text,
        speakers=speakers,
        source="per_student",
        words=primary.words,
        confidence=1.0,
    )


def _fill_gap_with_session(
    session_segments: list[TranscriptSegment],
    gap_start: float,
    gap_end: float,
) -> list[MergedSegment]:
    """Return session-fallback MergedSegments for the gap [gap_start, gap_end]."""
    result: list[MergedSegment] = []
    for seg in session_segments:
        overlap_start = max(seg.start, gap_start)
        overlap_end = min(seg.end, gap_end)
        if overlap_end - overlap_start > _GAP_EPSILON and seg.text.strip():
            result.append(
                MergedSegment(
                    start=overlap_start,
                    end=overlap_end,
                    text=seg.text,
                    speakers=["UNKNOWN"],
                    source="session_fallback",
                    confidence=0.5,
                )
            )
    return result


def build_merged_segments(
    session_segments: list[TranscriptSegment],
    speech_events: list[SpeechEvent],
) -> list[MergedSegment]:
    """Merge per-student speech events with session transcript into a unified timeline.

    - Per-student speech is canonical wherever available.
    - Gaps between student events are filled with session-transcript fallback.
    - Overlapping student events produce multi-speaker segments.
    - Non-overlapping sequential student events within one session segment are split.
    """
    if not speech_events:
        return [
            MergedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speakers=["UNKNOWN"],
                source="session_fallback",
                confidence=0.5,
            )
            for seg in session_segments
            if seg.text.strip()
        ]

    clusters = _cluster_events(speech_events)
    result: list[MergedSegment] = []

    timeline_start = session_segments[0].start if session_segments else 0.0
    timeline_end = session_segments[-1].end if session_segments else 0.0
    covered_until = timeline_start

    for cluster in clusters:
        cluster_start = min(e.start for e in cluster)
        cluster_end = max(e.end for e in cluster)

        # Clamp to session timeline
        cluster_start = max(cluster_start, timeline_start)
        cluster_end = min(cluster_end, timeline_end)
        if cluster_end <= cluster_start:
            continue

        # Fill gap before this cluster
        if cluster_start - covered_until > _GAP_EPSILON:
            result.extend(
                _fill_gap_with_session(session_segments, covered_until, cluster_start)
            )

        result.append(_cluster_to_merged(cluster))
        covered_until = cluster_end

    # Fill gap after last cluster
    if timeline_end - covered_until > _GAP_EPSILON:
        result.extend(_fill_gap_with_session(session_segments, covered_until, timeline_end))

    return result


def compute_merge_metadata(segments: list[MergedSegment]) -> MergeMetadata:
    total = len(segments)
    per_student = sum(1 for s in segments if s.source == "per_student")
    session_fallback = sum(1 for s in segments if s.source == "session_fallback")
    multi_speaker = sum(1 for s in segments if len(s.speakers) > 1)
    mode = "per_student_canonical" if per_student > 0 else "session_only"
    return MergeMetadata(
        total_segments=total,
        per_student_segments=per_student,
        session_fallback_segments=session_fallback,
        multi_speaker_segments=multi_speaker,
        alignment_mode=mode,
        merge_method="cluster_then_fill",
    )


def format_review_md(doc: MergedTranscriptDocument) -> str:
    lines = [
        f"# Transcript Merge Review: {doc.class_name}",
        "",
        "## Alignment Results",
        "",
    ]
    for student, alignment in doc.alignment_results.items():
        flag = "  ⚠️ uncertain" if alignment.uncertain else ""
        lines.append(
            f"- **{student}**: `{alignment.mode}` "
            f"(offset={alignment.offset:.1f}s){flag}"
        )

    lines += [
        "",
        "## Merge Statistics",
        "",
        f"- Total segments:            {doc.metadata.total_segments}",
        f"- Per-student segments:      {doc.metadata.per_student_segments}",
        f"- Session fallback segments: {doc.metadata.session_fallback_segments}",
        f"- Multi-speaker segments:    {doc.metadata.multi_speaker_segments}",
        f"- Duration:                  {doc.duration_seconds:.1f}s",
        "",
        "## Speakers",
        "",
    ]
    for speaker in doc.speakers:
        lines.append(f"- {speaker}")

    lines += ["", "## Segment Preview (first 20)", ""]
    for seg in doc.segments[:20]:
        speaker_str = ", ".join(seg.speakers)
        icon = "[session]" if seg.source == "session_fallback" else "[student]"
        preview = seg.text[:100].replace("\n", " ")
        lines.append(
            f"{icon} `[{seg.start:.1f}–{seg.end:.1f}s]` **{speaker_str}**: {preview}"
        )
    if len(doc.segments) > 20:
        lines.append(f"\n... ({len(doc.segments) - 20} more segments not shown)")

    return "\n".join(lines)


def merge_all(
    session_doc: PerStudentTranscript,
    student_docs: dict[str, PerStudentTranscript],
    identity_map: IdentityMap,
    class_name: str,
) -> MergedTranscriptDocument:
    session_segments = session_doc.transcript.segments

    # Detect alignment for every student
    alignment_results: dict[str, AlignmentResult] = {}
    for name, doc in student_docs.items():
        result = detect_alignment(doc.transcript.segments, session_segments)
        alignment_results[name] = result
        logger.info(
            "Alignment — %s: mode=%s offset=%.1fs uncertain=%s",
            name,
            result.mode,
            result.offset,
            result.uncertain,
        )

    # Build speech events with offset correction applied
    all_events: list[SpeechEvent] = []
    for name, doc in student_docs.items():
        events = build_speech_events(name, doc, alignment_results[name])
        if not events:
            logger.warning("No speech events for student: %s (muted?)", name)
        all_events.extend(events)

    merged_segments = build_merged_segments(session_segments, all_events)

    duration = (
        session_segments[-1].end - session_segments[0].start
        if session_segments
        else 0.0
    )
    metadata = compute_merge_metadata(merged_segments)
    all_speakers = sorted(
        {s for seg in merged_segments for s in seg.speakers if s != "UNKNOWN"}
    )

    return MergedTranscriptDocument(
        class_name=class_name,
        duration_seconds=duration,
        segments=merged_segments,
        speakers=all_speakers,
        teacher_name=identity_map.teacher_name,
        alignment_results=alignment_results,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class MergeArgs(BaseModel):
    session_transcript: Path
    student_transcripts_dir: Path
    identity_map_path: Path
    output_path: Path
    review_path: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> MergeArgs:
    parser = argparse.ArgumentParser(
        description="Merge per-student and session transcripts into a speaker-attributed document."
    )
    parser.add_argument(
        "--session-transcript",
        required=True,
        type=Path,
        dest="session_transcript",
        help="Path to session.json produced by transcribe_dual.py.",
    )
    parser.add_argument(
        "--student-transcripts",
        required=True,
        type=Path,
        dest="student_transcripts_dir",
        help="Directory containing per-student transcript JSON files.",
    )
    parser.add_argument(
        "--identity-map",
        required=True,
        type=Path,
        dest="identity_map_path",
        help="Path to identity_map.json produced by match_identity.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        dest="output_path",
        help="Output path for transcript_merged.json.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        dest="review_path",
        default=None,
        help="Output path for transcript_review.md (default: alongside --output).",
    )
    namespace = parser.parse_args(argv)
    return MergeArgs.model_validate(vars(namespace))


def validate_inputs(args: MergeArgs) -> None:
    if not args.session_transcript.exists():
        raise ValueError(f"Session transcript not found: {args.session_transcript}")
    if not args.student_transcripts_dir.exists():
        raise ValueError(f"Student transcripts directory not found: {args.student_transcripts_dir}")
    if not args.identity_map_path.exists():
        raise ValueError(f"Identity map not found: {args.identity_map_path}")


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as exc:
        logger.error("Input validation failed: %s", exc)
        raise SystemExit(2) from exc

    session_doc = PerStudentTranscript.model_validate_json(
        args.session_transcript.read_text(encoding="utf-8")
    )
    identity_map = IdentityMap.model_validate_json(
        args.identity_map_path.read_text(encoding="utf-8")
    )

    student_docs: dict[str, PerStudentTranscript] = {}
    for entry in identity_map.entries:
        transcript_file = args.student_transcripts_dir / f"{entry.audio_file}.json"
        if not transcript_file.exists():
            logger.warning("Student transcript not found: %s — skipping", transcript_file.name)
            continue
        doc = PerStudentTranscript.model_validate_json(
            transcript_file.read_text(encoding="utf-8")
        )
        speaker_name = entry.matched_name or entry.audio_file
        student_docs[speaker_name] = doc

    class_name = args.output_path.parent.name
    merged_doc = merge_all(session_doc, student_docs, identity_map, class_name)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(merged_doc.model_dump_json(indent=2), encoding="utf-8")

    review_path = args.review_path or args.output_path.parent / "transcript_review.md"
    review_path.write_text(format_review_md(merged_doc), encoding="utf-8")

    print(f"Merged transcript -> {args.output_path}")
    print(f"Review artifact   -> {review_path}")
    print(f"  Total segments:   {merged_doc.metadata.total_segments}")
    print(f"  Per-student:      {merged_doc.metadata.per_student_segments}")
    print(f"  Session fallback: {merged_doc.metadata.session_fallback_segments}")
    print(f"  Multi-speaker:    {merged_doc.metadata.multi_speaker_segments}")


if __name__ == "__main__":
    main()
