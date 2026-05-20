from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from pydantic import BaseModel, Field

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.chunk_and_embed import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    build_embedding_function,
    create_persistent_client,
)
from scripts.utils.chunker import ChunkType, SourceSegmentReference


class RetrievalArgs(BaseModel):
    chroma_dir: Path = Path("data/chroma")
    chunk_types: list[ChunkType] = Field(default_factory=list)
    collection_name: str = CHROMA_COLLECTION_NAME
    debug_output: Path | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    query: str
    student_id: str
    top_k: int = 5


class RetrievedChunk(BaseModel):
    approximate: bool = False
    attendance_accuracy: str
    attendance_estimated: bool
    attendance_source_mode: str
    chunk_id: str
    chunk_type: ChunkType
    distance: float | None = None
    duration_seconds: float
    end: float
    participant_kind: str = "student"
    rank: int
    score: float | None = None
    source_manual_review_required: bool = True
    source_mapped_student: str | None = None
    source_mapping_confidence: str | None = None
    source_segment_count: int
    source_segment_ids: list[str] = Field(default_factory=list)
    source_segment_indices: list[int] = Field(default_factory=list)
    source_segment_refs: list[SourceSegmentReference] = Field(default_factory=list)
    source_speaker: str
    start: float
    student_email: str | None = None
    student_id: str
    student_manual_review_required: bool = True
    student_mapped_speaker: str | None = None
    student_mapping_confidence: str | None = None
    student_name: str
    text: str
    trust_flags: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk_types: list[ChunkType] = Field(default_factory=list)
    collection_name: str
    context_string: str
    embedding_model: str
    query: str
    result_count: int
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    student_id: str
    top_k: int
    warnings: list[str] = Field(default_factory=list)


class RetrievalError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> RetrievalArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve student-scoped transcript chunks from ChromaDB and optionally write "
            "an inspectable JSON debug artifact."
        )
    )
    parser.add_argument("--student-id", required=True, help="Stable student scope key from TASK-007.")
    parser.add_argument("--query", required=True, help="Question to embed and use for retrieval.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of ranked chunks to return.",
    )
    parser.add_argument(
        "--chunk-type",
        action="append",
        choices=("spoken", "missed", "class_context"),
        dest="chunk_types",
        help="Optional chunk-type filter. Repeat the flag to allow multiple chunk types.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/chroma",
        help="Directory containing the persistent TASK-007 ChromaDB store.",
    )
    parser.add_argument(
        "--collection-name",
        default=CHROMA_COLLECTION_NAME,
        help="ChromaDB collection name to query.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence Transformer model name used for the retrieval query embedding.",
    )
    parser.add_argument(
        "--debug-output",
        help="Optional JSON output path for an inspectable retrieval debug artifact.",
    )
    namespace = parser.parse_args(argv)
    return RetrievalArgs(
        chroma_dir=Path(namespace.chroma_dir),
        chunk_types=list(namespace.chunk_types or []),
        collection_name=namespace.collection_name,
        debug_output=(None if namespace.debug_output is None else Path(namespace.debug_output)),
        embedding_model=namespace.embedding_model,
        query=namespace.query,
        student_id=namespace.student_id,
        top_k=namespace.top_k,
    )


def validate_inputs(args: RetrievalArgs) -> None:
    if not args.student_id.strip():
        raise ValueError("Student id must not be empty.")
    if not args.query.strip():
        raise ValueError("Query text must not be empty.")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if args.chroma_dir.exists() and args.chroma_dir.is_file():
        raise ValueError(f"Chroma directory path must be a directory: {args.chroma_dir}")
    if args.debug_output is not None and args.debug_output.suffix.lower() != ".json":
        raise ValueError("Debug output path must use the .json extension.")


def build_where_filter(student_id: str, chunk_types: Sequence[ChunkType]) -> dict[str, Any]:
    if not chunk_types:
        return {"student_id": student_id}
    if len(chunk_types) == 1:
        return {
            "$and": [
                {"student_id": student_id},
                {"chunk_type": chunk_types[0]},
            ]
        }
    return {
        "$and": [
            {"student_id": student_id},
            {"chunk_type": {"$in": list(chunk_types)}},
        ]
    }


def collection_names(client: Any) -> set[str]:
    names: set[str] = set()
    for item in client.list_collections():
        if isinstance(item, str):
            names.add(item)
            continue
        item_name = getattr(item, "name", None)
        if isinstance(item_name, str):
            names.add(item_name)
    return names


def get_existing_collection(client: Any, collection_name: str, embedding_function: Any) -> Any:
    if collection_name not in collection_names(client):
        raise RetrievalError(
            f"ChromaDB collection '{collection_name}' was not found. Run TASK-007 ingestion first."
        )
    return client.get_collection(name=collection_name, embedding_function=embedding_function)


def first_result_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not value:
        return []
    first_entry = value[0]
    if not isinstance(first_entry, list):
        return []
    return list(first_entry)


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def parse_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as error:
            raise RetrievalError(f"Collection metadata field '{field_name}' must be numeric.") from error
    raise RetrievalError(f"Collection metadata field '{field_name}' is missing or invalid.")


def parse_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise RetrievalError(f"Collection metadata field '{field_name}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise RetrievalError(f"Collection metadata field '{field_name}' must be an integer.")
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise RetrievalError(f"Collection metadata field '{field_name}' must be an integer.") from error
    raise RetrievalError(f"Collection metadata field '{field_name}' is missing or invalid.")


def parse_str(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    raise RetrievalError(f"Collection metadata field '{field_name}' is missing or invalid.")


def parse_optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def parse_json_list(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, str):
        raise RetrievalError(f"Collection metadata field '{field_name}' must be a JSON string.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise RetrievalError(f"Collection metadata field '{field_name}' is not valid JSON.") from error
    if not isinstance(parsed, list):
        raise RetrievalError(f"Collection metadata field '{field_name}' must decode to a list.")
    return parsed


def build_trust_flags(
    *,
    attendance_accuracy: str,
    attendance_estimated: bool,
    approximate: bool,
    source_manual_review_required: bool,
    source_mapping_confidence: str | None,
    student_manual_review_required: bool,
    student_mapping_confidence: str | None,
) -> list[str]:
    flags: list[str] = []
    if attendance_estimated:
        flags.append("attendance_estimated")
    if attendance_accuracy.casefold() != "exact":
        flags.append(f"attendance_accuracy={attendance_accuracy}")
    if approximate:
        flags.append("approximate_context")
    if source_manual_review_required:
        flags.append("source_manual_review_required")
    if student_manual_review_required:
        flags.append("student_manual_review_required")
    if source_mapping_confidence is not None and source_mapping_confidence.casefold() != "high":
        flags.append(f"source_mapping_confidence={source_mapping_confidence}")
    if student_mapping_confidence is not None and student_mapping_confidence.casefold() != "high":
        flags.append(f"student_mapping_confidence={student_mapping_confidence}")
    return flags


def distance_to_score(distance: float | None) -> float | None:
    if distance is None:
        return None
    return round(1.0 / (1.0 + max(distance, 0.0)), 6)


def parse_retrieved_chunk(
    *,
    rank: int,
    chunk_id: str,
    document_text: str,
    metadata: Mapping[str, Any],
    distance: float | None,
) -> RetrievedChunk:
    attendance_accuracy = parse_str(metadata.get("attendance_accuracy"), field_name="attendance_accuracy")
    attendance_estimated = parse_bool(metadata.get("attendance_estimated"))
    approximate = parse_bool(metadata.get("approximate"))
    source_manual_review_required = parse_bool(metadata.get("source_manual_review_required"), default=True)
    student_manual_review_required = parse_bool(metadata.get("student_manual_review_required"), default=True)
    source_mapping_confidence = parse_optional_str(metadata.get("source_mapping_confidence"))
    student_mapping_confidence = parse_optional_str(metadata.get("student_mapping_confidence"))
    source_segment_refs = [
        SourceSegmentReference.model_validate(item)
        for item in parse_json_list(metadata.get("source_segment_refs_json"), field_name="source_segment_refs_json")
    ]
    source_segment_ids = [
        parse_str(item, field_name="source_segment_ids_json[]")
        for item in parse_json_list(metadata.get("source_segment_ids_json"), field_name="source_segment_ids_json")
    ]
    source_segment_indices = [
        parse_int(item, field_name="source_segment_indices_json[]")
        for item in parse_json_list(
            metadata.get("source_segment_indices_json"),
            field_name="source_segment_indices_json",
        )
    ]
    return RetrievedChunk(
        approximate=approximate,
        attendance_accuracy=attendance_accuracy,
        attendance_estimated=attendance_estimated,
        attendance_source_mode=parse_str(
            metadata.get("attendance_source_mode"),
            field_name="attendance_source_mode",
        ),
        chunk_id=chunk_id,
        chunk_type=cast(ChunkType, parse_str(metadata.get("chunk_type"), field_name="chunk_type")),
        distance=distance,
        duration_seconds=parse_float(metadata.get("duration_seconds"), field_name="duration_seconds"),
        end=parse_float(metadata.get("end"), field_name="end"),
        participant_kind=parse_str(metadata.get("participant_kind"), field_name="participant_kind"),
        rank=rank,
        score=distance_to_score(distance),
        source_manual_review_required=source_manual_review_required,
        source_mapped_student=parse_optional_str(metadata.get("source_mapped_student")),
        source_mapping_confidence=source_mapping_confidence,
        source_segment_count=parse_int(metadata.get("source_segment_count"), field_name="source_segment_count"),
        source_segment_ids=source_segment_ids,
        source_segment_indices=source_segment_indices,
        source_segment_refs=source_segment_refs,
        source_speaker=parse_str(metadata.get("source_speaker"), field_name="source_speaker"),
        start=parse_float(metadata.get("start"), field_name="start"),
        student_email=parse_optional_str(metadata.get("student_email")),
        student_id=parse_str(metadata.get("student_id"), field_name="student_id"),
        student_manual_review_required=student_manual_review_required,
        student_mapped_speaker=parse_optional_str(metadata.get("student_mapped_speaker")),
        student_mapping_confidence=student_mapping_confidence,
        student_name=parse_str(metadata.get("student_name"), field_name="student_name"),
        text=document_text,
        trust_flags=build_trust_flags(
            attendance_accuracy=attendance_accuracy,
            attendance_estimated=attendance_estimated,
            approximate=approximate,
            source_manual_review_required=source_manual_review_required,
            source_mapping_confidence=source_mapping_confidence,
            student_manual_review_required=student_manual_review_required,
            student_mapping_confidence=student_mapping_confidence,
        ),
    )


def format_source_refs(source_segment_refs: Sequence[SourceSegmentReference]) -> str:
    if not source_segment_refs:
        return "none"
    return ", ".join(
        f"{reference.segment_id}@{reference.start:.3f}-{reference.end:.3f}"
        for reference in source_segment_refs
    )


def format_retrieved_chunk(chunk: RetrievedChunk) -> str:
    score_text = "n/a" if chunk.score is None else f"{chunk.score:.6f}"
    flags_text = "none" if not chunk.trust_flags else ", ".join(chunk.trust_flags)
    return "\n".join(
        [
            (
                f"[{chunk.rank}] chunk_id={chunk.chunk_id} type={chunk.chunk_type} "
                f"span={chunk.start:.3f}-{chunk.end:.3f} score={score_text}"
            ),
            (
                f"speaker={chunk.source_speaker} mapped_student={chunk.source_mapped_student or 'unknown'} "
                f"student={chunk.student_name}"
            ),
            f"trust_flags={flags_text}",
            f"source_refs={format_source_refs(chunk.source_segment_refs)}",
            f"text={chunk.text}",
        ]
    )


def build_context_string(
    *,
    student_id: str,
    query: str,
    chunk_types: Sequence[ChunkType],
    retrieved_chunks: Sequence[RetrievedChunk],
    warnings: Sequence[str],
) -> str:
    lines = [f"Student retrieval context for {student_id}", f"Query: {query}"]
    if chunk_types:
        lines.append(f"Chunk types: {', '.join(chunk_types)}")
    if warnings:
        lines.append(f"Warnings: {'; '.join(warnings)}")
    if not retrieved_chunks:
        lines.append("No student-scoped transcript chunks matched this query.")
        return "\n".join(lines)

    lines.append(f"Retrieved chunks: {len(retrieved_chunks)}")
    for chunk in retrieved_chunks:
        lines.extend(["", format_retrieved_chunk(chunk)])
    return "\n".join(lines)


def empty_result(
    *,
    student_id: str,
    query: str,
    top_k: int,
    chunk_types: Sequence[ChunkType],
    collection_name: str,
    embedding_model: str,
    warnings: Sequence[str],
) -> RetrievalResult:
    warning_list = list(warnings)
    return RetrievalResult(
        chunk_types=list(chunk_types),
        collection_name=collection_name,
        context_string=build_context_string(
            student_id=student_id,
            query=query,
            chunk_types=chunk_types,
            retrieved_chunks=[],
            warnings=warning_list,
        ),
        embedding_model=embedding_model,
        query=query,
        result_count=0,
        retrieved_chunks=[],
        student_id=student_id,
        top_k=top_k,
        warnings=warning_list,
    )


def retrieve_from_collection(
    collection: Any,
    *,
    student_id: str,
    query: str,
    top_k: int,
    chunk_types: Sequence[ChunkType],
    collection_name: str,
    embedding_model: str,
) -> RetrievalResult:
    where_filter = build_where_filter(student_id, chunk_types)
    scope_probe = collection.get(where=where_filter, limit=1)
    scope_ids = scope_probe.get("ids", [])
    if not isinstance(scope_ids, list) or not scope_ids:
        return empty_result(
            student_id=student_id,
            query=query,
            top_k=top_k,
            chunk_types=chunk_types,
            collection_name=collection_name,
            embedding_model=embedding_model,
            warnings=["No stored chunks found for the requested student scope."],
        )

    raw_result = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    raw_chunk_ids = [str(item) for item in first_result_list(raw_result, "ids")]
    raw_documents = first_result_list(raw_result, "documents")
    raw_metadatas = first_result_list(raw_result, "metadatas")
    raw_distances = first_result_list(raw_result, "distances")

    retrieved_chunks: list[RetrievedChunk] = []
    for index, chunk_id in enumerate(raw_chunk_ids):
        metadata = raw_metadatas[index] if index < len(raw_metadatas) else None
        if not isinstance(metadata, Mapping):
            raise RetrievalError(f"ChromaDB result for chunk '{chunk_id}' is missing metadata.")
        document_text = raw_documents[index] if index < len(raw_documents) else ""
        if not isinstance(document_text, str):
            raise RetrievalError(f"ChromaDB result for chunk '{chunk_id}' is missing document text.")
        raw_distance = raw_distances[index] if index < len(raw_distances) else None
        distance = None if raw_distance is None else parse_float(raw_distance, field_name="distance")
        retrieved_chunk = parse_retrieved_chunk(
            rank=index + 1,
            chunk_id=chunk_id,
            document_text=document_text,
            metadata=metadata,
            distance=distance,
        )
        if retrieved_chunk.student_id != student_id:
            raise RetrievalError(
                "Cross-student leakage detected in retrieval results. Check the Chroma filter path."
            )
        retrieved_chunks.append(retrieved_chunk)

    warnings: list[str] = []
    if not retrieved_chunks:
        warnings.append("The student scope exists, but no ranked chunks were returned for this query.")
    context_string = build_context_string(
        student_id=student_id,
        query=query,
        chunk_types=chunk_types,
        retrieved_chunks=retrieved_chunks,
        warnings=warnings,
    )
    return RetrievalResult(
        chunk_types=list(chunk_types),
        collection_name=collection_name,
        context_string=context_string,
        embedding_model=embedding_model,
        query=query,
        result_count=len(retrieved_chunks),
        retrieved_chunks=retrieved_chunks,
        student_id=student_id,
        top_k=top_k,
        warnings=warnings,
    )


def retrieve_from_chroma(
    *,
    student_id: str,
    query: str,
    top_k: int = 5,
    chunk_types: Sequence[ChunkType] | None = None,
    chroma_dir: Path = Path("data/chroma"),
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chroma_client: Any | None = None,
    embedding_function: Any | None = None,
) -> RetrievalResult:
    resolved_chunk_types = list(chunk_types or [])
    args = RetrievalArgs(
        chroma_dir=chroma_dir,
        chunk_types=resolved_chunk_types,
        collection_name=collection_name,
        embedding_model=embedding_model,
        query=query,
        student_id=student_id,
        top_k=top_k,
    )
    validate_inputs(args)

    client = chroma_client or create_persistent_client(chroma_dir)
    resolved_embedding_function = embedding_function or build_embedding_function(embedding_model)
    collection = get_existing_collection(client, collection_name, resolved_embedding_function)
    return retrieve_from_collection(
        collection,
        student_id=student_id,
        query=query,
        top_k=top_k,
        chunk_types=resolved_chunk_types,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )


def write_debug_output(result: RetrievalResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


class RetrievalService:
    def __init__(
        self,
        args: RetrievalArgs,
        *,
        chroma_client: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.args = args
        self.chroma_client = chroma_client
        self.embedding_function = embedding_function

    def run(self) -> RetrievalResult:
        print(
            f"Retrieving up to {self.args.top_k} chunks for student_id={self.args.student_id} "
            f"from {self.args.collection_name}..."
        )
        result = retrieve_from_chroma(
            student_id=self.args.student_id,
            query=self.args.query,
            top_k=self.args.top_k,
            chunk_types=self.args.chunk_types,
            chroma_dir=self.args.chroma_dir,
            collection_name=self.args.collection_name,
            embedding_model=self.args.embedding_model,
            chroma_client=self.chroma_client,
            embedding_function=self.embedding_function,
        )
        if self.args.debug_output is not None:
            print(f"Writing retrieval debug output to {self.args.debug_output}...")
            write_debug_output(result, self.args.debug_output)
        print(f"Retrieval complete: {result.result_count} chunks returned.")
        return result


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        validate_inputs(args)
        result = RetrievalService(args).run()
        print(result.context_string)
    except (RetrievalError, ValueError) as error:
        print(f"Retrieval failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()