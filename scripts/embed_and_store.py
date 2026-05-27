from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from pydantic import BaseModel

from scripts.models.context import (
    AbsentStudentSummary,
    StudentContext,
    StudentContextDocument,
)
from scripts.models.pipeline import EmbeddingRecord
from scripts.utils.pg_store import PgVectorStore, connect_pg_store

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MAX_CHUNK_CHARS = 700


class EmbedArgs(BaseModel):
    contexts_path: Path
    db_url: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_review_path: Path | None = None
    chunk_jsonl_path: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> EmbedArgs:
    parser = argparse.ArgumentParser(
        description="Chunk, embed, and upsert student contexts to pgvector."
    )
    parser.add_argument("--contexts", required=True, type=Path, dest="contexts_path")
    parser.add_argument("--db-url", dest="db_url", default="")
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL, dest="embedding_model"
    )
    parser.add_argument("--chunk-review", type=Path, dest="chunk_review_path", default=None)
    parser.add_argument("--chunk-jsonl", type=Path, dest="chunk_jsonl_path", default=None)
    namespace = parser.parse_args(argv)
    load_dotenv()
    db_url = namespace.db_url or os.getenv("DATABASE_URL", "")
    return EmbedArgs(
        contexts_path=namespace.contexts_path,
        db_url=db_url,
        embedding_model=namespace.embedding_model,
        chunk_review_path=namespace.chunk_review_path,
        chunk_jsonl_path=namespace.chunk_jsonl_path,
    )


def validate_inputs(args: EmbedArgs) -> None:
    if not args.contexts_path.exists():
        raise ValueError(f"Contexts file not found: {args.contexts_path}")
    if not args.db_url.strip():
        raise ValueError("Database URL is required. Pass --db-url or set DATABASE_URL.")


def is_quality_text(text: str, min_chars: int = 20) -> bool:
    """Return False for garbled, repetitive, or too-short text — don't embed junk.

    Catches:
    - Single-word or phrase repetitions (Whisper hallucination on silence)
    - Replacement-character-dense output (model failure on noisy audio)
    - Chunks too short to contain useful information
    """
    from collections import Counter

    stripped = text.strip()
    if len(stripped) < min_chars:
        return False

    words = stripped.split()
    if len(words) < 4:
        return False

    # Trigram repetition: any 3-word phrase appearing 4+ times → hallucinated
    if len(words) >= 12:
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        if Counter(trigrams).most_common(1)[0][1] >= 4:
            return False

    # Unicode replacement chars → garbled audio segment
    replacement_ratio = stripped.count("�") / max(len(stripped), 1)
    if replacement_ratio > 0.02:
        return False

    return True


def stable_chunk_id(student_id: str, chunk_type: str, text: str) -> str:
    payload = f"{student_id}|{chunk_type}|{text}"
    return hashlib.sha1(payload.encode()).hexdigest()


def _split_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current and current_len + 1 + len(word) > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += (1 if current else 0) + len(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_student_context(
    ctx: StudentContext, class_name: str
) -> list[EmbeddingRecord]:
    student_id = ctx.roll_no or ctx.name.lower().replace(" ", "_")
    records: list[EmbeddingRecord] = []

    for seg in ctx.spoken_segments:
        if seg.text.strip() and is_quality_text(seg.text):
            records.append(
                EmbeddingRecord(
                    id=stable_chunk_id(student_id, "spoken", seg.text),
                    student_id=student_id,
                    student_name=ctx.name,
                    class_name=class_name,
                    chunk_type="spoken",
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    speaker=", ".join(seg.speakers),
                    metadata={"status": ctx.status, "tags": ctx.tags},
                )
            )

    for seg in ctx.missed_segments:
        if seg.text.strip() and is_quality_text(seg.text):
            records.append(
                EmbeddingRecord(
                    id=stable_chunk_id(student_id, "missed", seg.text),
                    student_id=student_id,
                    student_name=ctx.name,
                    class_name=class_name,
                    chunk_type="missed",
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    speaker=None,
                    metadata={"status": ctx.status},
                )
            )

    # Filter individual segments first, then concatenate — keeps garbled/short
    # segments from polluting otherwise-good chunks when split.
    quality_segs = [
        s.text for s in ctx.present_segments
        if s.text.strip() and is_quality_text(s.text, min_chars=15)
    ]
    class_text = " ".join(quality_segs)
    for chunk_text in _split_text(class_text, _MAX_CHUNK_CHARS):
        if not is_quality_text(chunk_text):
            continue
        records.append(
            EmbeddingRecord(
                id=stable_chunk_id(student_id, "class_context", chunk_text),
                student_id=student_id,
                student_name=ctx.name,
                class_name=class_name,
                chunk_type="class_context",
                text=chunk_text,
                metadata={"status": ctx.status},
            )
        )

    return records


def chunk_absent_summary(
    summary: AbsentStudentSummary, class_name: str
) -> list[EmbeddingRecord]:
    student_id = summary.roll_no or summary.name.lower().replace(" ", "_")
    topics_text = "Topics covered: " + ", ".join(summary.topics_discussed)
    return [
        EmbeddingRecord(
            id=stable_chunk_id(student_id, "class_context", topics_text),
            student_id=student_id,
            student_name=summary.name,
            class_name=class_name,
            chunk_type="class_context",
            text=topics_text,
            metadata={"status": "absent", "topics": summary.topics_discussed},
        )
    ]


def collect_all_records(doc: StudentContextDocument) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for ctx in doc.present_students.values():
        records.extend(chunk_student_context(ctx, doc.class_name))
    for summary in doc.absent_students.values():
        records.extend(chunk_absent_summary(summary, doc.class_name))
    return records


def embed_records(
    records: list[EmbeddingRecord], model_name: str
) -> list[EmbeddingRecord]:
    if not records:
        return records
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [r.text for r in records]
    embeddings = model.encode(texts, show_progress_bar=True)
    for rec, emb in zip(records, embeddings):
        rec.embedding = emb.tolist()
    return records


def write_chunk_review(records: list[EmbeddingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "student_id", "student_name", "class_name", "chunk_type", "text_preview"]
        )
        for r in records:
            writer.writerow(
                [r.id[:12], r.student_id, r.student_name, r.class_name, r.chunk_type, r.text[:80]]
            )


def write_chunk_jsonl(records: list[EmbeddingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def run_embed_and_store(
    doc: StudentContextDocument,
    store: PgVectorStore,
    embedding_model: str,
) -> list[EmbeddingRecord]:
    records = collect_all_records(doc)
    if not records:
        logger.info("No records to embed.")
        return []
    records = embed_records(records, embedding_model)
    store.delete_class_chunks(doc.class_name)
    store.upsert_chunks(records)
    return records


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    doc = StudentContextDocument.model_validate_json(
        args.contexts_path.read_text(encoding="utf-8")
    )
    store = connect_pg_store(args.db_url)
    try:
        records = run_embed_and_store(doc, store, args.embedding_model)
    finally:
        store.close()

    review_path = args.chunk_review_path or args.contexts_path.parent / "rag_chunk_review.csv"
    write_chunk_review(records, review_path)

    jsonl_path = args.chunk_jsonl_path or args.contexts_path.parent / "rag_chunks.jsonl"
    write_chunk_jsonl(records, jsonl_path)

    print(f"Embedded and stored {len(records)} chunks -> pgvector")


if __name__ == "__main__":
    main()
