"""Student-scoped pgvector retrieval.

Note: this module's CLI is an unauthenticated *local dev/debug tool* that trusts
``--student-id`` directly. The authenticated entry point for students is
``scripts.chat`` (which derives the id from login); do not expose this CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence, cast

from pydantic import BaseModel, Field

from scripts.embed_and_store import DEFAULT_EMBEDDING_MODEL
from scripts.models.pipeline import SearchResult
from scripts.utils.chunker import ChunkType, SourceSegmentReference
from scripts.utils.db_url import resolve_db_url
from scripts.utils.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

HYBRID_POOL_SIZE = 25


class RetrievalArgs(BaseModel):
    db_url: str
    student_id: str
    query: str
    top_k: int = 5
    chunk_types: list[ChunkType] = Field(default_factory=list)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    debug_output: Path | None = None


class RetrievedChunk(BaseModel):
    approximate: bool = False
    attendance_accuracy: str = "per_student_m4a"
    attendance_estimated: bool = False
    attendance_source_mode: str = "per_student_m4a"
    chunk_id: str
    chunk_type: ChunkType
    distance: float | None = None
    duration_seconds: float = 0.0
    end: float = 0.0
    participant_kind: str = "student"
    rank: int
    score: float | None = None
    source_manual_review_required: bool = False
    source_mapped_student: str | None = None
    source_mapping_confidence: str | None = None
    source_segment_count: int = 1
    source_segment_ids: list[str] = Field(default_factory=list)
    source_segment_indices: list[int] = Field(default_factory=list)
    source_segment_refs: list[SourceSegmentReference] = Field(default_factory=list)
    source_file: str | None = None
    source_speaker: str = ""
    start: float = 0.0
    student_email: str | None = None
    student_id: str
    student_manual_review_required: bool = False
    student_mapped_speaker: str | None = None
    student_mapping_confidence: str | None = None
    student_name: str
    text: str
    trust_flags: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk_types: list[ChunkType] = Field(default_factory=list)
    collection_name: str = "pgvector"
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
        description="Retrieve student-scoped transcript chunks from pgvector."
    )
    parser.add_argument("--student-id", required=True, dest="student_id")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--db-url",
        default=None,
        dest="db_url",
        help="PostgreSQL connection URL. Falls back to DATABASE_URL env var.",
    )
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument(
        "--chunk-type",
        action="append",
        choices=("spoken", "missed", "class_context", "chat", "material"),
        dest="chunk_types",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, dest="embedding_model")
    parser.add_argument("--debug-output", type=Path, default=None, dest="debug_output")
    namespace = parser.parse_args(argv)
    return RetrievalArgs(
        db_url=resolve_db_url(namespace.db_url),
        student_id=namespace.student_id,
        query=namespace.query,
        top_k=namespace.top_k,
        chunk_types=list(namespace.chunk_types or []),
        embedding_model=namespace.embedding_model,
        debug_output=namespace.debug_output,
    )


def validate_inputs(args: RetrievalArgs) -> None:
    if not args.student_id.strip():
        raise ValueError("Student id must not be empty.")
    if not args.query.strip():
        raise ValueError("Query text must not be empty.")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not args.db_url.strip():
        raise ValueError("--db-url is required.")


def distance_to_score(distance: float | None) -> float | None:
    if distance is None:
        return None
    return round(1.0 / (1.0 + max(distance, 0.0)), 6)


def search_result_to_chunk(result: SearchResult, rank: int) -> RetrievedChunk:
    meta = result.metadata
    return RetrievedChunk(
        chunk_id=result.chunk_id,
        chunk_type=cast(ChunkType, result.chunk_type),
        distance=result.distance,
        end=result.end_time or 0.0,
        rank=rank,
        score=distance_to_score(result.distance),
        source_file=meta.get("source_file"),
        source_speaker=result.speaker
        or (result.student_name if result.chunk_type == "spoken" else "teacher"),
        start=result.start_time or 0.0,
        student_email=meta.get("student_email"),
        student_id=result.student_id,
        student_name=result.student_name,
        text=result.text,
        duration_seconds=(result.end_time or 0.0) - (result.start_time or 0.0),
        attendance_accuracy=meta.get("attendance_accuracy", "per_student_m4a"),
        attendance_estimated=bool(meta.get("attendance_estimated", False)),
        attendance_source_mode=meta.get("attendance_source_mode", "per_student_m4a"),
    )


def format_retrieved_chunk(chunk: RetrievedChunk) -> str:
    score_text = "n/a" if chunk.score is None else f"{chunk.score:.6f}"
    source_text = f" source={chunk.source_file}" if chunk.source_file else ""
    return (
        f"[{chunk.rank}] id={chunk.chunk_id[:12]} type={chunk.chunk_type} "
        f"span={chunk.start:.1f}-{chunk.end:.1f}s score={score_text}\n"
        f"speaker={chunk.source_speaker} student={chunk.student_name}{source_text}\n"
        f"text={chunk.text}"
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


class QueryEmbedder:
    """Lazily load a sentence-transformer once and reuse it across queries.

    Constructing ``SentenceTransformer`` reloads ~80 MB from disk, so a long-lived
    embedder (e.g. one per chat session) avoids paying that cost on every turn.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def encode(self, query: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(query).tolist()


def fuse_search_results(
    dense: Sequence[SearchResult],
    lexical: Sequence[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    by_id: dict[str, SearchResult] = {}
    for result in lexical:
        by_id.setdefault(result.chunk_id, result)
    for result in dense:
        by_id[result.chunk_id] = result
    fused_ids = reciprocal_rank_fusion(
        [[result.chunk_id for result in dense], [result.chunk_id for result in lexical]]
    )
    return [by_id[chunk_id] for chunk_id in fused_ids][:top_k]


def retrieve_from_pgvector(
    *,
    student_id: str,
    query: str,
    top_k: int = 5,
    chunk_types: Sequence[ChunkType] | None = None,
    class_name: str | None = None,
    db_url: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    store: Any | None = None,
    embedder: QueryEmbedder | None = None,
) -> RetrievalResult:
    from scripts.utils.pg_store import PgVectorStore, connect_pg_store

    resolved_chunk_types = list(chunk_types or [])
    warnings: list[str] = []

    pg_store: PgVectorStore = store or connect_pg_store(db_url)
    close_after = store is None
    active_embedder = embedder if embedder is not None else QueryEmbedder(embedding_model)
    pool_size = max(top_k, HYBRID_POOL_SIZE)

    try:
        query_embedding = active_embedder.encode(query)
        dense_results = pg_store.search(
            query_embedding, student_id, pool_size, resolved_chunk_types, class_name
        )
        lexical_results = pg_store.search_lexical(
            query,
            student_id=student_id,
            chunk_types=resolved_chunk_types,
            limit=pool_size,
            class_name=class_name,
        )
    finally:
        if close_after:
            pg_store.close()

    if not dense_results and not lexical_results:
        warnings.append("No stored chunks found for this student scope.")

    fused_results = fuse_search_results(dense_results, lexical_results, top_k)
    chunks = [search_result_to_chunk(r, i + 1) for i, r in enumerate(fused_results)]
    for chunk in chunks:
        if chunk.student_id != student_id:
            raise RetrievalError(f"Cross-student leakage detected: {chunk.student_id}")

    context = build_context_string(
        student_id=student_id,
        query=query,
        chunk_types=resolved_chunk_types,
        retrieved_chunks=chunks,
        warnings=warnings,
    )
    return RetrievalResult(
        chunk_types=resolved_chunk_types,
        context_string=context,
        embedding_model=embedding_model,
        query=query,
        result_count=len(chunks),
        retrieved_chunks=chunks,
        student_id=student_id,
        top_k=top_k,
        warnings=warnings,
    )


def write_debug_output(result: RetrievalResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


class RetrievalService:
    def __init__(self, args: RetrievalArgs, *, store: Any | None = None) -> None:
        self.args = args
        self.store = store

    def run(self) -> RetrievalResult:
        result = retrieve_from_pgvector(
            student_id=self.args.student_id,
            query=self.args.query,
            top_k=self.args.top_k,
            chunk_types=self.args.chunk_types,
            db_url=self.args.db_url,
            embedding_model=self.args.embedding_model,
            store=self.store,
        )
        if self.args.debug_output is not None:
            write_debug_output(result, self.args.debug_output)
        return result


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
