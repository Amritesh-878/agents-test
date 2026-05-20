from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pytest

from scripts.build_context import (
    AttendanceWindow,
    BuildContextMetadata,
    ContextSegment,
    SpeakerReview,
    StudentContext,
    StudentContextDocument,
)
from scripts.chunk_and_embed import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    ChunkAndEmbedArgs,
    ChunkAndEmbedError,
    ChunkAndEmbedService,
    build_chunk_catalog_records,
    build_review_markdown,
    chunk_record_to_chroma_metadata,
    load_contexts,
    parse_args,
    sync_collection,
    validate_inputs,
    write_jsonl,
    write_review_csv,
)
from scripts.merge import DiarizedTranscriptDocument, DiarizedTranscriptSegment

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*legacy embedding function config.*:DeprecationWarning"
)


class DeterministicEmbeddingFunction:
    def name(self) -> str:
        return "deterministic-test-embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), float(index + 1), float(text.count(" ") + 1)]
            for index, text in enumerate(input)
        ]


def build_transcript_document() -> DiarizedTranscriptDocument:
    return DiarizedTranscriptDocument(
        language="en",
        model="small",
        metadata={
            "merge_method": "majority_overlap",
            "total_segments": 3,
            "unknown_ratio": 0.0,
            "unknown_segments": 0,
        },
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[
            DiarizedTranscriptSegment(
                start=0.0,
                end=8.0,
                speaker="SPEAKER_00",
                text="The teacher introduces the new problem.",
                words=[],
            ),
            DiarizedTranscriptSegment(
                start=10.0,
                end=14.0,
                speaker="SPEAKER_01",
                text="I think the answer is ten days.",
                words=[],
            ),
            DiarizedTranscriptSegment(
                start=16.0,
                end=24.0,
                speaker="SPEAKER_00",
                text="Let us check that method step by step.",
                words=[],
            ),
        ],
    )


def build_student_context_document() -> StudentContextDocument:
    return StudentContextDocument(
        metadata=BuildContextMetadata(
            approximate_missed_segments=True,
            attendance_source_mode="duration_only_estimated",
            attendance_window_accuracy="estimated",
            manual_review_required=True,
            meeting_duration_seconds=24.0,
            notes=["Estimated for review."],
            speaker_mapping_method="duration_rank_estimate",
            transcript_segment_count=3,
        ),
        speaker_mapping={
            "SPEAKER_00": "Teacher Example",
            "SPEAKER_01": "Student Example",
        },
        speaker_reviews=[
            SpeakerReview(
                confidence="low",
                evidence="Fallback mapping.",
                first_segment_start=0.0,
                last_segment_end=24.0,
                mapped_student="Teacher Example",
                sample_utterances=["The teacher introduces the new problem."],
                segment_count=2,
                speaker="SPEAKER_00",
                total_speaking_seconds=16.0,
            ),
            SpeakerReview(
                confidence="low",
                evidence="Fallback mapping.",
                first_segment_start=10.0,
                last_segment_end=14.0,
                mapped_student="Student Example",
                sample_utterances=["I think the answer is ten days."],
                segment_count=1,
                speaker="SPEAKER_01",
                total_speaking_seconds=4.0,
            ),
        ],
        students={
            "Student Example": StudentContext(
                attendance=AttendanceWindow(
                    duration_minutes=0.2,
                    duration_seconds=12.0,
                    estimated=True,
                    exact=False,
                    joined_at=9.0,
                    left_at=21.0,
                    method="duration_anchored_to_speech",
                    note="Estimated from speech.",
                ),
                email="student@example.com",
                guest=False,
                manual_review_required=True,
                mapped_speaker="SPEAKER_01",
                mapping_confidence="low",
                mapping_notes="Fallback mapping.",
                missed_segments=[
                    ContextSegment(
                        approximate=True,
                        start=0.0,
                        end=8.0,
                        source_speaker="SPEAKER_00",
                        text="The teacher introduces the new problem.",
                    )
                ],
                participant_kind="student",
                spoken_segments=[
                    ContextSegment(
                        approximate=False,
                        start=10.0,
                        end=14.0,
                        source_speaker="SPEAKER_01",
                        text="I think the answer is ten days.",
                    )
                ],
                was_present_full_class=False,
            )
        },
    )


def test_parse_args_uses_task_defaults() -> None:
    args = parse_args([])

    assert args == ChunkAndEmbedArgs(
        chroma_dir=Path("data/chroma"),
        chunk_debug_path=Path("output/rag_chunks.jsonl"),
        collection_name=CHROMA_COLLECTION_NAME,
        contexts_path=Path("output/student_contexts.json"),
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        max_chars=700,
        max_gap_seconds=15.0,
        max_segments=6,
        review_csv_path=Path("output/rag_chunk_review.csv"),
        review_markdown_path=Path("output/rag_chunk_review.md"),
        target_chars=420,
        transcript_path=Path("output/transcript_diarized.json"),
    )


def test_validate_inputs_rejects_non_json_context_path(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.json"
    contexts_path = tmp_path / "contexts.txt"
    transcript_path.write_text("{}", encoding="utf-8")
    contexts_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Contexts file must use the .json extension"):
        validate_inputs(
            ChunkAndEmbedArgs(
                transcript_path=transcript_path,
                contexts_path=contexts_path,
            )
        )


def test_build_chunk_catalog_records_emits_deterministic_provenance_rich_chunks() -> None:
    transcript = build_transcript_document()
    contexts = build_student_context_document()

    first_run = build_chunk_catalog_records(transcript, contexts)
    second_run = build_chunk_catalog_records(transcript, contexts)

    assert [record.chunk_type for record in first_run] == ["missed", "spoken", "class_context"]
    assert [record.chunk_id for record in first_run] == [record.chunk_id for record in second_run]
    assert first_run[0].attendance_estimated is True
    assert first_run[0].approximate is True
    assert first_run[1].source_mapped_student == "Student Example"
    assert first_run[1].source_mapping_confidence == "low"
    assert first_run[2].source_segment_ids == ["seg-0002"]


def test_write_artifacts_and_sync_collection_are_idempotent(tmp_path: Path) -> None:
    records = build_chunk_catalog_records(
        build_transcript_document(),
        build_student_context_document(),
    )
    jsonl_path = tmp_path / "rag_chunks.jsonl"
    csv_path = tmp_path / "rag_chunk_review.csv"

    write_jsonl(records, jsonl_path)
    write_review_csv(records, csv_path)

    jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == len(records)
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == len(records) + 1

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        name="test-chunks",
        embedding_function=DeterministicEmbeddingFunction(),
    )
    first_count = sync_collection(collection, records)
    second_count = sync_collection(collection, records)
    stored = collection.get(ids=[records[0].chunk_id])

    assert first_count == len(records)
    assert second_count == len(records)
    assert collection.count() == len(records)
    assert stored["metadatas"][0]["chunk_id"] == records[0].chunk_id


def test_chunk_service_writes_matching_review_outputs(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.json"
    contexts_path = tmp_path / "contexts.json"
    transcript_path.write_text(build_transcript_document().model_dump_json(indent=2), encoding="utf-8")
    contexts_path.write_text(build_student_context_document().model_dump_json(indent=2), encoding="utf-8")

    args = ChunkAndEmbedArgs(
        transcript_path=transcript_path,
        contexts_path=contexts_path,
        chroma_dir=tmp_path / "chroma",
        chunk_debug_path=tmp_path / "rag_chunks.jsonl",
        review_csv_path=tmp_path / "rag_chunk_review.csv",
        review_markdown_path=tmp_path / "rag_chunk_review.md",
    )
    client = chromadb.PersistentClient(path=str(args.chroma_dir))

    records = ChunkAndEmbedService(
        args,
        chroma_client=client,
        embedding_function=DeterministicEmbeddingFunction(),
    ).run()
    jsonl_lines = args.chunk_debug_path.read_text(encoding="utf-8").strip().splitlines()
    markdown = args.review_markdown_path.read_text(encoding="utf-8")

    assert len(records) == 3
    assert len(jsonl_lines) == len(records)
    assert f"Total chunks: {len(records)}" in markdown
    assert "Student Example" in markdown


def test_load_contexts_rejects_invalid_schema(tmp_path: Path) -> None:
    contexts_path = tmp_path / "contexts.json"
    contexts_path.write_text('{"metadata": [], "students": {}}', encoding="utf-8")

    with pytest.raises(ChunkAndEmbedError, match="expected schema"):
        load_contexts(contexts_path)


def test_chunk_record_to_chroma_metadata_serializes_source_refs() -> None:
    record = build_chunk_catalog_records(
        build_transcript_document(),
        build_student_context_document(),
    )[0]

    metadata = chunk_record_to_chroma_metadata(record)

    assert metadata["student_id"] == "student-example-com"
    source_segment_ids_json = metadata["source_segment_ids_json"]
    source_segment_refs_json = metadata["source_segment_refs_json"]

    assert isinstance(source_segment_ids_json, str)
    assert isinstance(source_segment_refs_json, str)
    assert json.loads(source_segment_ids_json) == ["seg-0000"]
    assert json.loads(source_segment_refs_json)[0]["segment_id"] == "seg-0000"


def test_build_review_markdown_mentions_machine_readable_exports() -> None:
    markdown = build_review_markdown(
        build_chunk_catalog_records(
            build_transcript_document(),
            build_student_context_document(),
        ),
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )

    assert "Chunk Type Counts" in markdown
    assert "JSONL" in markdown