from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ValidationError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_context import ContextSegment, StudentContext, StudentContextDocument
from scripts.merge import DiarizedTranscriptDocument
from scripts.utils.chunker import ChunkProjectionSegment, ChunkRecord, ChunkerConfig, chunk_projection_segments

CHROMA_COLLECTION_NAME = "student_transcript_chunks"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ChunkAndEmbedArgs(BaseModel):
    chroma_dir: Path = Path("data/chroma")
    chunk_debug_path: Path = Path("output/rag_chunks.jsonl")
    collection_name: str = CHROMA_COLLECTION_NAME
    contexts_path: Path = Path("output/student_contexts.json")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    max_chars: int = 700
    max_gap_seconds: float = 15.0
    max_segments: int = 6
    review_csv_path: Path = Path("output/rag_chunk_review.csv")
    review_markdown_path: Path = Path("output/rag_chunk_review.md")
    target_chars: int = 420
    transcript_path: Path = Path("output/transcript_diarized.json")


class SourceTranscriptSegment(BaseModel):
    end: float
    segment_id: str
    segment_index: int
    source_manual_review_required: bool = True
    source_mapped_student: str | None = None
    source_mapping_confidence: str | None = None
    source_speaker: str
    start: float
    text: str


class ChunkCatalogRecord(ChunkRecord):
    chunk_id: str
    collection_name: str = CHROMA_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


class ChunkAndEmbedError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> ChunkAndEmbedArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk the diarized transcript into per-student RAG records, write review artifacts, "
            "and upsert them into a local ChromaDB collection."
        )
    )
    parser.add_argument(
        "--transcript",
        default="output/transcript_diarized.json",
        help="Path to the TASK-005 diarized transcript JSON.",
    )
    parser.add_argument(
        "--contexts",
        default="output/student_contexts.json",
        help="Path to the TASK-006 student context JSON.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/chroma",
        help="Directory used for the persistent ChromaDB store.",
    )
    parser.add_argument(
        "--chunk-debug",
        default="output/rag_chunks.jsonl",
        help="Path to the machine-readable JSONL chunk export.",
    )
    parser.add_argument(
        "--review-csv",
        default="output/rag_chunk_review.csv",
        help="Path to the flat CSV review artifact.",
    )
    parser.add_argument(
        "--review-markdown",
        default="output/rag_chunk_review.md",
        help="Path to the human-readable Markdown review artifact.",
    )
    parser.add_argument(
        "--collection-name",
        default=CHROMA_COLLECTION_NAME,
        help="ChromaDB collection name for the student-scoped chunk catalog.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence Transformer model name used by ChromaDB for embedding.",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=420,
        help="Preferred chunk size target in characters before opening a new chunk.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=700,
        help="Hard limit for merged chunk text length in characters.",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=15.0,
        help="Maximum allowed gap between adjacent transcript segments when merging.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=6,
        help="Maximum number of source transcript segments merged into one chunk.",
    )
    namespace = parser.parse_args(argv)
    return ChunkAndEmbedArgs(
        chroma_dir=Path(namespace.chroma_dir),
        chunk_debug_path=Path(namespace.chunk_debug),
        collection_name=namespace.collection_name,
        contexts_path=Path(namespace.contexts),
        embedding_model=namespace.embedding_model,
        max_chars=namespace.max_chars,
        max_gap_seconds=namespace.max_gap_seconds,
        max_segments=namespace.max_segments,
        review_csv_path=Path(namespace.review_csv),
        review_markdown_path=Path(namespace.review_markdown),
        target_chars=namespace.target_chars,
        transcript_path=Path(namespace.transcript),
    )


def validate_inputs(args: ChunkAndEmbedArgs) -> None:
    for path_name, path_value in (
        ("Transcript", args.transcript_path),
        ("Contexts", args.contexts_path),
    ):
        if not path_value.exists():
            raise ValueError(f"{path_name} file does not exist: {path_value}")
        if not path_value.is_file():
            raise ValueError(f"{path_name} path is not a file: {path_value}")
        if path_value.suffix.lower() != ".json":
            raise ValueError(f"{path_name} file must use the .json extension.")

    if args.chunk_debug_path.suffix.lower() not in {".jsonl", ".json"}:
        raise ValueError("Chunk debug file must use the .jsonl or .json extension.")
    if args.review_csv_path.suffix.lower() != ".csv":
        raise ValueError("Review CSV file must use the .csv extension.")
    if args.review_markdown_path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("Review markdown file must use the .md or .txt extension.")
    if args.chroma_dir.exists() and args.chroma_dir.is_file():
        raise ValueError(f"Chroma directory path must be a directory: {args.chroma_dir}")

    ChunkerConfig(
        max_chars=args.max_chars,
        max_gap_seconds=args.max_gap_seconds,
        max_segments=args.max_segments,
        target_chars=args.target_chars,
    )


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def round_seconds(value: float) -> float:
    return round(value, 3)


def segment_lookup_key(
    start: float,
    end: float,
    source_speaker: str,
    text: str,
) -> tuple[float, float, str, str]:
    return (round_seconds(start), round_seconds(end), source_speaker, normalize_text(text))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "student"


def build_student_id(student_name: str, student_email: str | None) -> str:
    if student_email:
        return slugify(student_email)
    return slugify(student_name)


def load_transcript(path: Path) -> DiarizedTranscriptDocument:
    try:
        return DiarizedTranscriptDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ChunkAndEmbedError(f"Failed to read diarized transcript JSON: {error}") from error
    except ValidationError as error:
        raise ChunkAndEmbedError(
            f"Diarized transcript JSON does not match the expected schema: {error}"
        ) from error


def load_contexts(path: Path) -> StudentContextDocument:
    try:
        return StudentContextDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ChunkAndEmbedError(f"Failed to read student context JSON: {error}") from error
    except ValidationError as error:
        raise ChunkAndEmbedError(
            f"Student context JSON does not match the expected schema: {error}"
        ) from error


def build_source_segment_catalog(
    transcript: DiarizedTranscriptDocument,
    contexts: StudentContextDocument,
) -> tuple[list[SourceTranscriptSegment], dict[tuple[float, float, str, str], SourceTranscriptSegment]]:
    speaker_review_index = {review.speaker: review for review in contexts.speaker_reviews}
    catalog: list[SourceTranscriptSegment] = []
    lookup: dict[tuple[float, float, str, str], SourceTranscriptSegment] = {}

    for segment_index, segment in enumerate(transcript.segments):
        speaker_review = speaker_review_index.get(segment.speaker)
        catalog_segment = SourceTranscriptSegment(
            end=round_seconds(segment.end),
            segment_id=f"seg-{segment_index:04d}",
            segment_index=segment_index,
            source_manual_review_required=(
                True if speaker_review is None else speaker_review.manual_review_required
            ),
            source_mapped_student=contexts.speaker_mapping.get(segment.speaker),
            source_mapping_confidence=(None if speaker_review is None else speaker_review.confidence),
            source_speaker=segment.speaker,
            start=round_seconds(segment.start),
            text=normalize_text(segment.text),
        )
        lookup_key = segment_lookup_key(
            catalog_segment.start,
            catalog_segment.end,
            catalog_segment.source_speaker,
            catalog_segment.text,
        )
        if lookup_key in lookup:
            raise ChunkAndEmbedError(
                "Transcript segment lookup keys must be unique for deterministic provenance."
            )
        catalog.append(catalog_segment)
        lookup[lookup_key] = catalog_segment

    return catalog, lookup


def build_context_membership(
    segments: Sequence[ContextSegment],
) -> dict[tuple[float, float, str, str], bool]:
    membership: dict[tuple[float, float, str, str], bool] = {}
    for segment in segments:
        key = segment_lookup_key(
            segment.start,
            segment.end,
            segment.source_speaker or "UNKNOWN",
            segment.text,
        )
        membership[key] = segment.approximate
    return membership


def validate_context_segments_exist(
    student_name: str,
    segment_kind: str,
    membership: dict[tuple[float, float, str, str], bool],
    source_lookup: dict[tuple[float, float, str, str], SourceTranscriptSegment],
) -> None:
    missing_keys = sorted(key for key in membership if key not in source_lookup)
    if missing_keys:
        raise ChunkAndEmbedError(
            f"{student_name} has {segment_kind} segments that do not match the source transcript."
        )


def build_projection_segments_for_student(
    student_name: str,
    student_context: StudentContext,
    source_catalog: Sequence[SourceTranscriptSegment],
    source_lookup: dict[tuple[float, float, str, str], SourceTranscriptSegment],
    attendance_source_mode: str,
    attendance_accuracy: str,
) -> list[ChunkProjectionSegment]:
    spoken_membership = build_context_membership(student_context.spoken_segments)
    missed_membership = build_context_membership(student_context.missed_segments)
    validate_context_segments_exist(student_name, "spoken", spoken_membership, source_lookup)
    validate_context_segments_exist(student_name, "missed", missed_membership, source_lookup)

    student_id = build_student_id(student_name, student_context.email)
    projection_segments: list[ChunkProjectionSegment] = []

    for source_segment in source_catalog:
        lookup_key = segment_lookup_key(
            source_segment.start,
            source_segment.end,
            source_segment.source_speaker,
            source_segment.text,
        )
        if lookup_key in spoken_membership:
            chunk_type = "spoken"
            approximate = spoken_membership[lookup_key]
        elif lookup_key in missed_membership:
            chunk_type = "missed"
            approximate = missed_membership[lookup_key]
        else:
            chunk_type = "class_context"
            approximate = student_context.attendance.estimated

        projection_segments.append(
            ChunkProjectionSegment(
                approximate=approximate,
                attendance_accuracy=attendance_accuracy,
                attendance_estimated=student_context.attendance.estimated,
                attendance_source_mode=attendance_source_mode,
                chunk_type=chunk_type,
                end=source_segment.end,
                participant_kind=student_context.participant_kind,
                source_manual_review_required=source_segment.source_manual_review_required,
                source_mapped_student=source_segment.source_mapped_student,
                source_mapping_confidence=source_segment.source_mapping_confidence,
                source_segment_id=source_segment.segment_id,
                source_segment_index=source_segment.segment_index,
                source_speaker=source_segment.source_speaker,
                start=source_segment.start,
                student_email=student_context.email,
                student_id=student_id,
                student_manual_review_required=student_context.manual_review_required,
                student_mapped_speaker=student_context.mapped_speaker,
                student_mapping_confidence=student_context.mapping_confidence,
                student_name=student_name,
                text=source_segment.text,
            )
        )

    return projection_segments


def stable_chunk_id(chunk_record: ChunkRecord) -> str:
    digest_input = "|".join(
        [
            chunk_record.student_id,
            chunk_record.chunk_type,
            chunk_record.source_speaker,
            f"{chunk_record.start:.3f}",
            f"{chunk_record.end:.3f}",
            ",".join(chunk_record.source_segment_ids),
        ]
    )
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:16]
    return f"{chunk_record.student_id}:{chunk_record.chunk_type}:{digest}"


def build_chunk_catalog_records(
    transcript: DiarizedTranscriptDocument,
    contexts: StudentContextDocument,
    *,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunker_config: ChunkerConfig | None = None,
) -> list[ChunkCatalogRecord]:
    source_catalog, source_lookup = build_source_segment_catalog(transcript, contexts)
    records: list[ChunkCatalogRecord] = []
    resolved_chunker_config = ChunkerConfig() if chunker_config is None else chunker_config

    for student_name, student_context in sorted(contexts.students.items(), key=lambda item: item[0].casefold()):
        projection_segments = build_projection_segments_for_student(
            student_name,
            student_context,
            source_catalog,
            source_lookup,
            contexts.metadata.attendance_source_mode,
            contexts.metadata.attendance_window_accuracy,
        )
        chunk_records = chunk_projection_segments(projection_segments, resolved_chunker_config)
        for chunk_record in chunk_records:
            records.append(
                ChunkCatalogRecord(
                    **chunk_record.model_dump(),
                    chunk_id=stable_chunk_id(chunk_record),
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                )
            )

    return sorted(
        records,
        key=lambda item: (
            item.student_id,
            item.start,
            item.end,
            item.chunk_type,
            item.chunk_id,
        ),
    )


def serialize_source_segment_refs(record: ChunkCatalogRecord) -> str:
    return json.dumps(
        [reference.model_dump(mode="json") for reference in record.source_segment_refs],
        separators=(",", ":"),
    )


def chunk_record_to_chroma_metadata(record: ChunkCatalogRecord) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "approximate": record.approximate,
        "attendance_accuracy": record.attendance_accuracy,
        "attendance_estimated": record.attendance_estimated,
        "attendance_source_mode": record.attendance_source_mode,
        "chunk_id": record.chunk_id,
        "chunk_type": record.chunk_type,
        "collection_name": record.collection_name,
        "duration_seconds": record.duration_seconds,
        "embedding_model": record.embedding_model,
        "end": record.end,
        "participant_kind": record.participant_kind,
        "source_manual_review_required": record.source_manual_review_required,
        "source_segment_count": record.source_segment_count,
        "source_segment_ids_json": json.dumps(record.source_segment_ids),
        "source_segment_indices_json": json.dumps(record.source_segment_indices),
        "source_segment_refs_json": serialize_source_segment_refs(record),
        "source_speaker": record.source_speaker,
        "start": record.start,
        "student_id": record.student_id,
        "student_manual_review_required": record.student_manual_review_required,
        "student_name": record.student_name,
    }
    optional_fields = {
        "source_mapped_student": record.source_mapped_student,
        "source_mapping_confidence": record.source_mapping_confidence,
        "student_email": record.student_email,
        "student_mapped_speaker": record.student_mapped_speaker,
        "student_mapping_confidence": record.student_mapping_confidence,
    }
    for key, value in optional_fields.items():
        if value is not None:
            metadata[key] = value
    return metadata


def write_jsonl(records: Sequence[ChunkCatalogRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def review_csv_row(record: ChunkCatalogRecord) -> dict[str, str | int | float | bool]:
    return {
        "chunk_id": record.chunk_id,
        "student_id": record.student_id,
        "student_name": record.student_name,
        "student_email": record.student_email or "",
        "participant_kind": record.participant_kind,
        "chunk_type": record.chunk_type,
        "start": record.start,
        "end": record.end,
        "duration_seconds": record.duration_seconds,
        "source_speaker": record.source_speaker,
        "source_mapped_student": record.source_mapped_student or "",
        "source_mapping_confidence": record.source_mapping_confidence or "",
        "source_manual_review_required": record.source_manual_review_required,
        "student_mapped_speaker": record.student_mapped_speaker or "",
        "student_mapping_confidence": record.student_mapping_confidence or "",
        "student_manual_review_required": record.student_manual_review_required,
        "attendance_accuracy": record.attendance_accuracy,
        "attendance_source_mode": record.attendance_source_mode,
        "attendance_estimated": record.attendance_estimated,
        "approximate": record.approximate,
        "source_segment_count": record.source_segment_count,
        "source_segment_ids": "|".join(record.source_segment_ids),
        "source_segment_indices": "|".join(str(index) for index in record.source_segment_indices),
        "text": record.text,
    }


def write_review_csv(records: Sequence[ChunkCatalogRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chunk_id",
        "student_id",
        "student_name",
        "student_email",
        "participant_kind",
        "chunk_type",
        "start",
        "end",
        "duration_seconds",
        "source_speaker",
        "source_mapped_student",
        "source_mapping_confidence",
        "source_manual_review_required",
        "student_mapped_speaker",
        "student_mapping_confidence",
        "student_manual_review_required",
        "attendance_accuracy",
        "attendance_source_mode",
        "attendance_estimated",
        "approximate",
        "source_segment_count",
        "source_segment_ids",
        "source_segment_indices",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(review_csv_row(record))


def preview_text(value: str, limit: int = 90) -> str:
    normalized = normalize_text(value)
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def build_review_markdown(
    records: Sequence[ChunkCatalogRecord],
    *,
    collection_name: str,
    embedding_model: str,
) -> str:
    type_counts: Counter[str] = Counter(str(record.chunk_type) for record in records)
    student_counts: dict[str, Counter[str]] = {}
    for record in records:
        student_counts.setdefault(record.student_name, Counter())
        student_counts[record.student_name][record.chunk_type] += 1

    lines = [
        "# RAG Chunk Review",
        "",
        f"- Collection name: {collection_name}",
        f"- Embedding model: {embedding_model}",
        f"- Total chunks: {len(records)}",
        f"- Unique students: {len({record.student_id for record in records})}",
        f"- Estimated-attendance chunks: {sum(1 for record in records if record.attendance_estimated)}",
        (
            "- Manual-review chunks: "
            f"{sum(1 for record in records if record.source_manual_review_required or record.student_manual_review_required)}"
        ),
        "",
        "## Chunk Type Counts",
        "",
        "| Chunk Type | Count |",
        "| --- | ---: |",
    ]
    for chunk_type in ("spoken", "missed", "class_context"):
        lines.append(f"| {chunk_type} | {type_counts.get(chunk_type, 0)} |")

    lines.extend(
        [
            "",
            "## Student Summary",
            "",
            "| Student | Chunks | Spoken | Missed | Class Context |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for student_name in sorted(student_counts, key=str.casefold):
        counts = student_counts[student_name]
        lines.append(
            "| "
            f"{student_name} | {sum(counts.values())} | {counts.get('spoken', 0)} | "
            f"{counts.get('missed', 0)} | {counts.get('class_context', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Preview",
            "",
            "| Chunk ID | Student | Type | Speaker | Span | Source Segments | Text |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in records[:25]:
        lines.append(
            "| "
            f"{record.chunk_id} | {record.student_name} | {record.chunk_type} | {record.source_speaker} | "
            f"{record.start:.3f}-{record.end:.3f} | {';'.join(record.source_segment_ids)} | "
            f"{preview_text(record.text).replace('|', '/')} |"
        )

    lines.extend(
        [
            "",
            "The CSV contains the full row-level review catalog and the JSONL contains the full machine-readable chunk schema.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_review_markdown(
    records: Sequence[ChunkCatalogRecord],
    path: Path,
    *,
    collection_name: str,
    embedding_model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_review_markdown(
            records,
            collection_name=collection_name,
            embedding_model=embedding_model,
        ),
        encoding="utf-8",
    )


def build_embedding_function(model_name: str) -> Any:
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as error:
        raise ChunkAndEmbedError(
            "ChromaDB embedding support is unavailable. Install chromadb and sentence-transformers first."
        ) from error
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def create_persistent_client(chroma_dir: Path) -> Any:
    try:
        import chromadb
    except ImportError as error:
        raise ChunkAndEmbedError(
            "ChromaDB is not installed. Install chromadb before running chunk ingestion."
        ) from error
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def get_collection(client: Any, collection_name: str, embedding_function: Any) -> Any:
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )


def sync_collection(collection: Any, records: Sequence[ChunkCatalogRecord]) -> int:
    record_ids = [record.chunk_id for record in records]
    existing_ids: set[str] = set()
    existing_count = collection.count()
    if existing_count > 0:
        existing_result = collection.get(limit=existing_count)
        existing_raw_ids = existing_result.get("ids")
        if isinstance(existing_raw_ids, list):
            existing_ids = {str(item) for item in existing_raw_ids}

    stale_ids = sorted(existing_ids.difference(record_ids))
    if stale_ids:
        collection.delete(ids=stale_ids)

    collection.upsert(
        documents=[record.text for record in records],
        ids=record_ids,
        metadatas=[chunk_record_to_chroma_metadata(record) for record in records],
    )
    return collection.count()


class ChunkAndEmbedService:
    def __init__(
        self,
        args: ChunkAndEmbedArgs,
        *,
        chroma_client: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.args = args
        self.chroma_client = chroma_client
        self.embedding_function = embedding_function

    def run(self) -> list[ChunkCatalogRecord]:
        print(f"Loading diarized transcript from {self.args.transcript_path}...")
        transcript = load_transcript(self.args.transcript_path)
        print(f"Loading student contexts from {self.args.contexts_path}...")
        contexts = load_contexts(self.args.contexts_path)
        print("Building student-scoped chunk catalog...")
        chunk_records = build_chunk_catalog_records(
            transcript,
            contexts,
            collection_name=self.args.collection_name,
            embedding_model=self.args.embedding_model,
            chunker_config=ChunkerConfig(
                max_chars=self.args.max_chars,
                max_gap_seconds=self.args.max_gap_seconds,
                max_segments=self.args.max_segments,
                target_chars=self.args.target_chars,
            ),
        )
        if not chunk_records:
            raise ChunkAndEmbedError("No chunk records were generated from the supplied inputs.")

        print("Writing inspectable chunk review artifacts...")
        write_jsonl(chunk_records, self.args.chunk_debug_path)
        write_review_csv(chunk_records, self.args.review_csv_path)
        write_review_markdown(
            chunk_records,
            self.args.review_markdown_path,
            collection_name=self.args.collection_name,
            embedding_model=self.args.embedding_model,
        )

        print(f"Upserting {len(chunk_records)} chunks into ChromaDB at {self.args.chroma_dir}...")
        client = self.chroma_client or create_persistent_client(self.args.chroma_dir)
        embedding_function = self.embedding_function or build_embedding_function(self.args.embedding_model)
        collection = get_collection(client, self.args.collection_name, embedding_function)
        stored_count = sync_collection(collection, chunk_records)
        print(
            "Chunk ingestion complete: "
            f"{len(chunk_records)} chunks written, collection count now {stored_count}."
        )
        return chunk_records


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        validate_inputs(args)
        ChunkAndEmbedService(args).run()
    except (ChunkAndEmbedError, ValueError) as error:
        print(f"Chunk ingestion failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()