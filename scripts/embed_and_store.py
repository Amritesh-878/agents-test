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
    - Nukta-dense Devanagari (the Hindi pass hallucinating garbled transliterations)
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

    # Distinct-token ratio: short loops (a 3-5 word phrase repeated 2-3 times)
    # duck the trigram rule but collapse to very few unique tokens.
    # e.g. "A B C D A B C D A B C D" → 4 unique / 12 total = 0.33. Genuine answers
    # of 8+ words run well above 0.5, so this drops loops without touching real speech.
    if len(words) >= 8 and len(set(words)) / len(words) < 0.5:
        return False

    # No-space loop hallucination: the Hindi pass sometimes emits one unbroken run
    # with no spaces (e.g. "पाइपाइपाइ…"×100). That reads as a single very long token,
    # so the word-based checks above miss it. Real words are short; flag an over-long
    # token only when it is mostly non-ASCII (don't touch long ASCII like URLs).
    longest = max(words, key=len)
    if len(longest) > 40 and sum(c > "\x7f" for c in longest) > len(longest) / 2:
        return False

    # Unicode replacement chars → garbled audio segment
    replacement_ratio = stripped.count("�") / max(len(stripped), 1)
    if replacement_ratio > 0.02:
        return False

    # Nukta-dense Devanagari → the Hindi pass hallucinating on bilingual speech
    # (e.g. "तो आड़़ क्वान्टी अप गुड़"). Genuine Hindi runs ~15-40 nukta tokens per
    # 1k words; these loops run 200-400+, so a generous cap drops the garble without
    # touching real Hindi (a Devanagari-ratio cap would wrongly delete genuine Hindi).
    nukta_per_1k = (stripped.count("़") / len(words)) * 1000
    if nukta_per_1k > 120:
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

    for seg in ctx.chat_segments:
        # The student's own PUBLIC chat messages (typed contributions). Same student_id
        # scope and quality filter as spoken; a lower min_chars keeps short-but-real typed
        # answers ("answer is 20/3 days") while is_quality_text still drops junk/links.
        if seg.text.strip() and is_quality_text(seg.text, min_chars=8):
            records.append(
                EmbeddingRecord(
                    id=stable_chunk_id(student_id, "chat", seg.text),
                    student_id=student_id,
                    student_name=ctx.name,
                    class_name=class_name,
                    chunk_type="chat",
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


def chunk_material_blocks(
    blocks: Sequence[tuple[str, str]],
    *,
    student_id: str,
    student_name: str,
    class_name: str,
) -> list[EmbeddingRecord]:
    """Chunk extracted ``(source_filename, block_text)`` material blocks for one student.

    Materials are authoritative class content, not speech: no timestamps, and the
    speaker is set to "material" so retrieval never mis-labels them as the teacher's
    spoken words. The source filename is kept as provenance for TASK-020 labeling.
    """
    records: list[EmbeddingRecord] = []
    for source_filename, block_text in blocks:
        for chunk_text in _split_text(block_text, _MAX_CHUNK_CHARS):
            if not is_quality_text(chunk_text):
                continue
            records.append(
                EmbeddingRecord(
                    id=stable_chunk_id(student_id, "material", chunk_text),
                    student_id=student_id,
                    student_name=student_name,
                    class_name=class_name,
                    chunk_type="material",
                    text=chunk_text,
                    speaker="material",
                    metadata={"source_file": source_filename},
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


def embed_records_deduped(
    records: list[EmbeddingRecord], model_name: str
) -> list[EmbeddingRecord]:
    """Embed each distinct text once and share the vector across duplicates.

    Material chunks are identical for every enrolled student of a class, so
    encoding the per-student copies would redo the same work N times.
    """
    if not records:
        return records
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    unique_texts = list(dict.fromkeys(record.text for record in records))
    embeddings = model.encode(unique_texts, show_progress_bar=True)
    vector_by_text = {text: emb.tolist() for text, emb in zip(unique_texts, embeddings)}
    for record in records:
        record.embedding = vector_by_text[record.text]
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
