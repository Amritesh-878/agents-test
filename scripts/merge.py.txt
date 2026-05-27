from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field, ValidationError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diarize import DiarizationDocument, DiarizationSegment
from scripts.transcribe import TranscriptDocument, TranscriptSegment, TranscriptWord

MERGE_METHOD = "majority_overlap"
UNKNOWN_SPEAKER = "UNKNOWN"


class MergeArgs(BaseModel):
    diarization_path: Path = Path("output/diarization.json")
    output_path: Path = Path("output/transcript_diarized.json")
    transcript_path: Path = Path("output/transcript_raw.json")


class DiarizedTranscriptSegment(BaseModel):
    end: float
    speaker: str
    start: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)


class MergeMetadata(BaseModel):
    merge_method: str = MERGE_METHOD
    total_segments: int
    unknown_ratio: float
    unknown_segments: int


class DiarizedTranscriptDocument(BaseModel):
    language: str
    metadata: MergeMetadata
    model: str
    segments: list[DiarizedTranscriptSegment] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)


class MergeError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> MergeArgs:
    parser = argparse.ArgumentParser(
        description="Merge a WhisperX transcript with pyannote diarization using timestamp overlap."
    )
    parser.add_argument(
        "--transcript",
        default="output/transcript_raw.json",
        help="Path to the TASK-003 transcript JSON.",
    )
    parser.add_argument(
        "--diarization",
        default="output/diarization.json",
        help="Path to the TASK-004 diarization JSON.",
    )
    parser.add_argument(
        "--output",
        default="output/transcript_diarized.json",
        help="Path to the merged transcript JSON.",
    )
    namespace = parser.parse_args(argv)
    return MergeArgs(
        diarization_path=Path(namespace.diarization),
        output_path=Path(namespace.output),
        transcript_path=Path(namespace.transcript),
    )


def validate_inputs(args: MergeArgs) -> None:
    for path_name, path_value in (
        ("Transcript", args.transcript_path),
        ("Diarization", args.diarization_path),
    ):
        if not path_value.exists():
            raise ValueError(f"{path_name} file does not exist: {path_value}")
        if not path_value.is_file():
            raise ValueError(f"{path_name} path is not a file: {path_value}")
        if path_value.suffix.lower() != ".json":
            raise ValueError(f"{path_name} file must use the .json extension.")

    if args.output_path.suffix.lower() != ".json":
        raise ValueError("Output file must use the .json extension.")


def load_transcript(path: Path) -> TranscriptDocument:
    try:
        return TranscriptDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise MergeError(f"Failed to read transcript JSON: {error}") from error
    except ValidationError as error:
        raise MergeError(f"Transcript JSON does not match the expected schema: {error}") from error


def load_diarization(path: Path) -> DiarizationDocument:
    try:
        return DiarizationDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise MergeError(f"Failed to read diarization JSON: {error}") from error
    except ValidationError as error:
        raise MergeError(f"Diarization JSON does not match the expected schema: {error}") from error


def compute_overlap(
    transcript_start: float,
    transcript_end: float,
    diarization_start: float,
    diarization_end: float,
) -> float:
    return max(0.0, min(transcript_end, diarization_end) - max(transcript_start, diarization_start))


def get_majority_speaker(
    segment_start: float,
    segment_end: float,
    diarization_segments: Sequence[DiarizationSegment],
) -> str:
    overlaps: dict[str, float] = {}
    for diarization_segment in diarization_segments:
        overlap = compute_overlap(
            segment_start,
            segment_end,
            diarization_segment.start,
            diarization_segment.end,
        )
        if overlap <= 0:
            continue
        overlaps[diarization_segment.speaker] = overlaps.get(diarization_segment.speaker, 0.0) + overlap

    if not overlaps:
        return UNKNOWN_SPEAKER

    return max(overlaps.items(), key=lambda item: item[1])[0]


def build_diarized_segment(
    transcript_segment: TranscriptSegment,
    diarization_segments: Sequence[DiarizationSegment],
) -> DiarizedTranscriptSegment:
    speaker = get_majority_speaker(
        transcript_segment.start,
        transcript_segment.end,
        diarization_segments,
    )
    return DiarizedTranscriptSegment(
        end=transcript_segment.end,
        speaker=speaker,
        start=transcript_segment.start,
        text=transcript_segment.text.strip(),
        words=list(transcript_segment.words),
    )


def build_diarized_transcript(
    transcript: TranscriptDocument,
    diarization: DiarizationDocument,
) -> DiarizedTranscriptDocument:
    ordered_diarization_segments = sorted(
        diarization.segments,
        key=lambda segment: (segment.start, segment.end, segment.speaker),
    )
    merged_segments = [
        build_diarized_segment(transcript_segment, ordered_diarization_segments)
        for transcript_segment in transcript.segments
    ]
    unknown_segments = sum(1 for segment in merged_segments if segment.speaker == UNKNOWN_SPEAKER)
    if unknown_segments > 0:
        warnings.warn(
            (
                f"{unknown_segments} transcript segments could not be matched to a diarization speaker "
                f"and were marked {UNKNOWN_SPEAKER}."
            ),
            stacklevel=2,
        )

    total_segments = len(merged_segments)
    unknown_ratio = 0.0 if total_segments == 0 else round(unknown_segments / total_segments, 4)
    speakers = sorted({segment.speaker for segment in merged_segments})
    return DiarizedTranscriptDocument(
        language=transcript.language,
        metadata=MergeMetadata(
            total_segments=total_segments,
            unknown_ratio=unknown_ratio,
            unknown_segments=unknown_segments,
        ),
        model=transcript.model,
        segments=merged_segments,
        speakers=speakers,
    )


def save_output(document: DiarizedTranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def format_segment_line(segment: DiarizedTranscriptSegment) -> str:
    return f"[{segment.start:.3f}-{segment.end:.3f}] {segment.speaker}: {segment.text}"


class MergeService:
    def __init__(self, args: MergeArgs) -> None:
        self.args = args

    def run(self) -> DiarizedTranscriptDocument:
        print(f"Loading transcript from {self.args.transcript_path}...")
        transcript = load_transcript(self.args.transcript_path)
        print(f"Loading diarization from {self.args.diarization_path}...")
        diarization = load_diarization(self.args.diarization_path)
        print("Merging transcript segments with diarization speakers...")
        document = build_diarized_transcript(transcript, diarization)
        save_output(document, self.args.output_path)
        print(
            "Merged transcript saved: "
            f"{document.metadata.total_segments} segments, "
            f"{document.metadata.unknown_segments} UNKNOWN -> {self.args.output_path}"
        )
        preview_count = min(3, len(document.segments))
        if preview_count > 0:
            print("Preview:")
            for segment in document.segments[:preview_count]:
                print(format_segment_line(segment))
        return document


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        validate_inputs(args)
        MergeService(args).run()
    except (MergeError, ValueError) as error:
        print(f"Transcript merge failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()