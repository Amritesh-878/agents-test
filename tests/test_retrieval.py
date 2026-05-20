from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import pytest

from scripts.chunk_and_embed import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    ChunkCatalogRecord,
    chunk_record_to_chroma_metadata,
)
from scripts.retrieval import (
    RetrievalArgs,
    RetrievalService,
    build_where_filter,
    parse_args,
    retrieve_from_chroma,
    validate_inputs,
)
from scripts.utils.chunker import SourceSegmentReference

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*legacy embedding function config.*:DeprecationWarning"
)


class KeywordEmbeddingFunction:
    def __init__(self) -> None:
        self.vocabulary = [
            "equivalent",
            "fractions",
            "third",
            "sixths",
            "homework",
            "chapter",
            "teacher",
            "example",
        ]

    def name(self) -> str:
        return "keyword-test-embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(input, list):
            return [self.embed_text(text) for text in input]
        return self.embed_text(input)

    def embed_text(self, text: str) -> list[float]:
        normalized = text.casefold()
        return [float(normalized.count(token)) for token in self.vocabulary]


def build_record(
    *,
    chunk_id: str,
    student_id: str,
    student_name: str,
    chunk_type: str,
    text: str,
    start: float,
    end: float,
    source_speaker: str,
    source_segment_id: str,
    source_mapping_confidence: str | None = None,
    student_mapping_confidence: str | None = None,
    approximate: bool = False,
    attendance_estimated: bool = False,
    source_manual_review_required: bool = False,
    student_manual_review_required: bool = False,
) -> ChunkCatalogRecord:
    return ChunkCatalogRecord(
        approximate=approximate,
        attendance_accuracy=("estimated" if attendance_estimated else "exact"),
        attendance_estimated=attendance_estimated,
        attendance_source_mode=("duration_only_estimated" if attendance_estimated else "exact_join_leave"),
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        collection_name=CHROMA_COLLECTION_NAME,
        duration_seconds=end - start,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        end=end,
        participant_kind="student",
        source_manual_review_required=source_manual_review_required,
        source_mapped_student=student_name,
        source_mapping_confidence=source_mapping_confidence,
        source_segment_count=1,
        source_segment_ids=[source_segment_id],
        source_segment_indices=[0],
        source_segment_refs=[
            SourceSegmentReference(
                end=end,
                segment_id=source_segment_id,
                segment_index=0,
                source_speaker=source_speaker,
                start=start,
                text=text,
            )
        ],
        source_speaker=source_speaker,
        start=start,
        student_email=None,
        student_id=student_id,
        student_manual_review_required=student_manual_review_required,
        student_mapped_speaker=source_speaker,
        student_mapping_confidence=student_mapping_confidence,
        student_name=student_name,
        text=text,
    )


def seed_collection(tmp_path: Path) -> tuple[Any, KeywordEmbeddingFunction]:
    embedding_function = KeywordEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    records = [
        build_record(
            chunk_id="student-a:missed:0001",
            student_id="student-a",
            student_name="Student A",
            chunk_type="missed",
            text="You missed the explanation about equivalent fractions.",
            start=0.0,
            end=8.0,
            source_speaker="SPEAKER_00",
            source_segment_id="seg-0000",
            source_mapping_confidence="low",
            student_mapping_confidence="low",
            approximate=True,
            attendance_estimated=True,
            source_manual_review_required=True,
            student_manual_review_required=True,
        ),
        build_record(
            chunk_id="student-a:spoken:0002",
            student_id="student-a",
            student_name="Student A",
            chunk_type="spoken",
            text="I answered that one third equals two sixths.",
            start=9.0,
            end=13.0,
            source_speaker="SPEAKER_01",
            source_segment_id="seg-0001",
            source_mapping_confidence="high",
            student_mapping_confidence="high",
        ),
        build_record(
            chunk_id="student-a:class_context:0003",
            student_id="student-a",
            student_name="Student A",
            chunk_type="class_context",
            text="The teacher compared the denominators step by step.",
            start=14.0,
            end=20.0,
            source_speaker="SPEAKER_00",
            source_segment_id="seg-0002",
        ),
        build_record(
            chunk_id="student-b:missed:0004",
            student_id="student-b",
            student_name="Student B",
            chunk_type="missed",
            text="You missed the homework reminder about chapter five.",
            start=21.0,
            end=27.0,
            source_speaker="SPEAKER_00",
            source_segment_id="seg-0003",
        ),
    ]
    collection.upsert(
        ids=[record.chunk_id for record in records],
        documents=[record.text for record in records],
        metadatas=[chunk_record_to_chroma_metadata(record) for record in records],
    )
    return client, embedding_function


def test_parse_args_uses_expected_defaults() -> None:
    args = parse_args(["--student-id", "student-a", "--query", "What did I miss?"])

    assert args == RetrievalArgs(
        chroma_dir=Path("data/chroma"),
        chunk_types=[],
        collection_name=CHROMA_COLLECTION_NAME,
        debug_output=None,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        query="What did I miss?",
        student_id="student-a",
        top_k=5,
    )


def test_validate_inputs_rejects_non_json_debug_output() -> None:
    with pytest.raises(ValueError, match=".json extension"):
        validate_inputs(
            RetrievalArgs(
                student_id="student-a",
                query="What did I miss?",
                debug_output=Path("output/retrieval_debug/sample.txt"),
            )
        )


def test_build_where_filter_combines_student_scope_and_chunk_type() -> None:
    where_filter = build_where_filter("student-a", ["missed"])

    assert where_filter == {"$and": [{"student_id": "student-a"}, {"chunk_type": "missed"}]}


def test_retrieve_from_chroma_enforces_student_scope(tmp_path: Path) -> None:
    client, embedding_function = seed_collection(tmp_path)

    result = retrieve_from_chroma(
        student_id="student-a",
        query="homework chapter five",
        top_k=3,
        chroma_dir=tmp_path / "chroma",
        chroma_client=client,
        embedding_function=embedding_function,
    )

    assert result.result_count == 3
    assert all(chunk.student_id == "student-a" for chunk in result.retrieved_chunks)
    assert all(not chunk.chunk_id.startswith("student-b") for chunk in result.retrieved_chunks)


def test_retrieve_from_chroma_preserves_provenance_and_formats_context(tmp_path: Path) -> None:
    client, embedding_function = seed_collection(tmp_path)

    result = retrieve_from_chroma(
        student_id="student-a",
        query="equivalent fractions",
        top_k=2,
        chroma_dir=tmp_path / "chroma",
        chroma_client=client,
        embedding_function=embedding_function,
    )

    first_chunk = result.retrieved_chunks[0]

    assert first_chunk.chunk_id == "student-a:missed:0001"
    assert first_chunk.source_segment_refs[0].segment_id == "seg-0000"
    assert first_chunk.source_mapping_confidence == "low"
    assert "attendance_estimated" in first_chunk.trust_flags
    assert "chunk_id=student-a:missed:0001" in result.context_string
    assert "type=missed" in result.context_string
    assert "span=0.000-8.000" in result.context_string


def test_retrieve_from_chroma_supports_chunk_type_filter(tmp_path: Path) -> None:
    client, embedding_function = seed_collection(tmp_path)

    result = retrieve_from_chroma(
        student_id="student-a",
        query="third sixths",
        top_k=2,
        chunk_types=["spoken"],
        chroma_dir=tmp_path / "chroma",
        chroma_client=client,
        embedding_function=embedding_function,
    )

    assert result.result_count == 1
    assert result.retrieved_chunks[0].chunk_type == "spoken"
    assert "Chunk types: spoken" in result.context_string


def test_retrieve_from_chroma_returns_safe_empty_result_for_unknown_student(tmp_path: Path) -> None:
    client, embedding_function = seed_collection(tmp_path)

    result = retrieve_from_chroma(
        student_id="missing-student",
        query="What did I miss?",
        top_k=2,
        chroma_dir=tmp_path / "chroma",
        chroma_client=client,
        embedding_function=embedding_function,
    )

    assert result.result_count == 0
    assert result.retrieved_chunks == []
    assert "No stored chunks found" in result.warnings[0]
    assert "No student-scoped transcript chunks matched this query." in result.context_string


def test_retrieval_service_writes_debug_output_from_structured_result(tmp_path: Path) -> None:
    client, embedding_function = seed_collection(tmp_path)
    debug_output = tmp_path / "output" / "retrieval_debug" / "sample_query.json"
    args = RetrievalArgs(
        student_id="student-a",
        query="equivalent fractions",
        top_k=2,
        chroma_dir=tmp_path / "chroma",
        debug_output=debug_output,
    )

    result = RetrievalService(
        args,
        chroma_client=client,
        embedding_function=embedding_function,
    ).run()
    payload = json.loads(debug_output.read_text(encoding="utf-8"))

    assert payload["student_id"] == result.student_id
    assert payload["context_string"] == result.context_string
    assert payload["retrieved_chunks"][0]["chunk_id"] == result.retrieved_chunks[0].chunk_id