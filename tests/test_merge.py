from __future__ import annotations

from pathlib import Path

import pytest

from scripts.diarize import DiarizationDocument, DiarizationSegment
from scripts.merge import (
    MERGE_METHOD,
    UNKNOWN_SPEAKER,
    DiarizedTranscriptDocument,
    DiarizedTranscriptSegment,
    MergeArgs,
    MergeError,
    build_diarized_transcript,
    compute_overlap,
    format_segment_line,
    get_majority_speaker,
    load_diarization,
    parse_args,
    validate_inputs,
)
from scripts.transcribe import TranscriptDocument, TranscriptSegment, TranscriptWord


def test_parse_args_uses_default_paths() -> None:
    args = parse_args([])

    assert args == MergeArgs(
        diarization_path=Path("output/diarization.json"),
        output_path=Path("output/transcript_diarized.json"),
        transcript_path=Path("output/transcript_raw.json"),
    )


def test_validate_inputs_rejects_missing_transcript(tmp_path: Path) -> None:
    diarization_path = tmp_path / "diarization.json"
    diarization_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Transcript file does not exist"):
        validate_inputs(
            MergeArgs(
                transcript_path=tmp_path / "transcript_raw.json",
                diarization_path=diarization_path,
            )
        )


def test_validate_inputs_rejects_non_json_output(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript_raw.json"
    diarization_path = tmp_path / "diarization.json"
    transcript_path.write_text("{}", encoding="utf-8")
    diarization_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Output file must use the .json extension"):
        validate_inputs(
            MergeArgs(
                transcript_path=transcript_path,
                diarization_path=diarization_path,
                output_path=tmp_path / "transcript.txt",
            )
        )


def test_compute_overlap_returns_shared_duration() -> None:
    assert compute_overlap(0.5, 3.5, 0.0, 3.0) == pytest.approx(2.5)
    assert compute_overlap(0.5, 3.5, 3.0, 6.0) == pytest.approx(0.5)
    assert compute_overlap(10.0, 12.0, 0.0, 3.0) == 0.0


def test_get_majority_speaker_prefers_largest_overlap() -> None:
    diarization_segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=3.0),
        DiarizationSegment(speaker="SPEAKER_01", start=3.0, end=6.0),
    ]

    assert get_majority_speaker(0.5, 3.5, diarization_segments) == "SPEAKER_00"
    assert get_majority_speaker(3.5, 5.5, diarization_segments) == "SPEAKER_01"


def test_get_majority_speaker_returns_unknown_when_no_overlap() -> None:
    diarization_segments = [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=3.0)]

    assert get_majority_speaker(10.0, 12.0, diarization_segments) == UNKNOWN_SPEAKER


def test_build_diarized_transcript_preserves_words_and_counts_unknowns() -> None:
    transcript = TranscriptDocument(
        language="en",
        model="small",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="hello class",
                words=[
                    TranscriptWord(word="hello", start=0.0, end=0.8),
                    TranscriptWord(word="class", start=0.9, end=2.0),
                ],
            ),
            TranscriptSegment(
                start=6.0,
                end=7.0,
                text="late question",
                words=[TranscriptWord(word="late", start=6.0, end=6.4)],
            ),
        ],
    )
    diarization = DiarizationDocument(
        speakers=["SPEAKER_00"],
        segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0)],
    )

    with pytest.warns(UserWarning, match="could not be matched"):
        document = build_diarized_transcript(transcript, diarization)

    assert document == DiarizedTranscriptDocument(
        language="en",
        model="small",
        speakers=["SPEAKER_00", UNKNOWN_SPEAKER],
        metadata={
            "merge_method": MERGE_METHOD,
            "total_segments": 2,
            "unknown_segments": 1,
            "unknown_ratio": 0.5,
        },
        segments=[
            DiarizedTranscriptSegment(
                start=0.0,
                end=2.0,
                speaker="SPEAKER_00",
                text="hello class",
                words=[
                    TranscriptWord(word="hello", start=0.0, end=0.8),
                    TranscriptWord(word="class", start=0.9, end=2.0),
                ],
            ),
            DiarizedTranscriptSegment(
                start=6.0,
                end=7.0,
                speaker=UNKNOWN_SPEAKER,
                text="late question",
                words=[TranscriptWord(word="late", start=6.0, end=6.4)],
            ),
        ],
    )


def test_load_diarization_rejects_invalid_schema(tmp_path: Path) -> None:
    diarization_path = tmp_path / "diarization.json"
    diarization_path.write_text('{"speakers": [], "segments": "bad"}', encoding="utf-8")

    with pytest.raises(MergeError, match="expected schema"):
        load_diarization(diarization_path)


def test_format_segment_line_matches_manual_review_shape() -> None:
    segment = DiarizedTranscriptSegment(
        start=0.5,
        end=4.2,
        speaker="SPEAKER_00",
        text="So today we are going to cover gradient descent.",
        words=[],
    )

    assert (
        format_segment_line(segment)
        == "[0.500-4.200] SPEAKER_00: So today we are going to cover gradient descent."
    )