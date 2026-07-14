"""Ingest class materials (PPT/PDF/notes) into pgvector as ``material`` chunks.

Expected folder layout: ``materials/<class_name>/*.pptx|pdf|docx|txt|md`` — pass
that class folder as ``--materials-dir``. Chunks are embedded once and stored
per-student-in-class (mirroring ``class_context``), so retrieval and per-student
isolation are unchanged. Re-running replaces a class's ``material`` chunks;
spoken/chat/class_context chunks are never touched.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Protocol, Sequence

from pydantic import BaseModel, Field

from scripts.embed_and_store import (
    DEFAULT_EMBEDDING_MODEL,
    chunk_material_blocks,
    embed_records_deduped,
    write_chunk_review,
)
from scripts.models.identity import IdentityMap
from scripts.models.pipeline import EmbeddingRecord
from scripts.utils.db_url import resolve_db_url
from scripts.utils.material_extract import (
    MaterialBlock,
    MaterialExtractError,
    extract_materials_dir,
)

logger = logging.getLogger(__name__)


class MaterialStudent(BaseModel):
    student_id: str
    student_name: str


class IngestArgs(BaseModel):
    materials_dir: Path
    class_name: str
    db_url: str
    identity_map_path: Path | None = None
    student_ids: list[str] = Field(default_factory=list)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_review_path: Path | None = None


class SupportsMaterialStore(Protocol):
    def delete_student_material_chunks(self, class_name: str, student_id: str) -> int:
        ...

    def upsert_chunks(self, records: Sequence[EmbeddingRecord]) -> int:
        ...


def parse_args(argv: Sequence[str] | None = None) -> IngestArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Extract class materials (pptx/pdf/docx/txt/md) and embed them into "
            "pgvector as 'material' chunks for every enrolled student. Folder "
            "layout: materials/<class_name>/*.pptx|pdf|docx|txt|md — pass the "
            "class folder as --materials-dir."
        )
    )
    parser.add_argument("--materials-dir", required=True, type=Path, dest="materials_dir")
    parser.add_argument("--class-name", required=True, dest="class_name")
    parser.add_argument(
        "--identity-map",
        type=Path,
        dest="identity_map_path",
        default=None,
        help="identity_map.json from the class pipeline run; its matched entries "
        "define the enrolled students the materials are stored under.",
    )
    parser.add_argument(
        "--student-id",
        action="append",
        dest="student_ids",
        default=None,
        help="Explicit student id to ingest for (repeatable). Use when no identity "
        "map exists yet, or to add students the map does not cover.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        dest="db_url",
        help="PostgreSQL connection URL. Falls back to DATABASE_URL env var.",
    )
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL, dest="embedding_model"
    )
    parser.add_argument(
        "--chunk-review",
        type=Path,
        dest="chunk_review_path",
        default=None,
        help="Optional CSV path for spot-checking the stored material chunks.",
    )
    namespace = parser.parse_args(argv)
    return IngestArgs(
        materials_dir=namespace.materials_dir,
        class_name=namespace.class_name,
        db_url=resolve_db_url(namespace.db_url),
        identity_map_path=namespace.identity_map_path,
        student_ids=list(namespace.student_ids or []),
        embedding_model=namespace.embedding_model,
        chunk_review_path=namespace.chunk_review_path,
    )


def validate_inputs(args: IngestArgs) -> None:
    if not args.materials_dir.is_dir():
        raise ValueError(f"Materials folder not found: {args.materials_dir}")
    if not args.class_name.strip():
        raise ValueError("Class name must not be empty.")
    if not args.db_url.strip():
        raise ValueError("Database URL is required. Pass --db-url or set DATABASE_URL.")
    if args.identity_map_path is None and not args.student_ids:
        raise ValueError("Pass --identity-map or at least one --student-id.")
    if args.identity_map_path is not None and not args.identity_map_path.exists():
        raise ValueError(f"Identity map not found: {args.identity_map_path}")


def enrolled_students(identity_map: IdentityMap) -> list[MaterialStudent]:
    """Enrolled students from the identity map, with ids derived exactly as
    ``embed_and_store`` derives them (roll_no, else lowercased name slug) so the
    material chunks land under the same student_id as the transcript chunks."""
    students: list[MaterialStudent] = []
    seen: set[str] = set()
    for entry in identity_map.entries:
        if entry.is_teacher or entry.is_unmatched:
            continue
        name = entry.matched_name or entry.audio_file
        student_id = entry.matched_roll_no or name.lower().replace(" ", "_")
        if student_id in seen:
            continue
        seen.add(student_id)
        students.append(MaterialStudent(student_id=student_id, student_name=name))
    return students


def resolve_students(args: IngestArgs) -> list[MaterialStudent]:
    students: list[MaterialStudent] = []
    if args.identity_map_path is not None:
        identity_map = IdentityMap.model_validate_json(
            args.identity_map_path.read_text(encoding="utf-8")
        )
        students.extend(enrolled_students(identity_map))
        if identity_map.roster_students_without_audio:
            # Their roll numbers are not in the identity map, so a name-derived id
            # could mismatch the roster-derived id their other chunks use.
            logger.warning(
                "%d roster students without audio were skipped (%s); pass their ids "
                "via --student-id to include them.",
                len(identity_map.roster_students_without_audio),
                ", ".join(identity_map.roster_students_without_audio),
            )
    seen = {student.student_id for student in students}
    for student_id in args.student_ids:
        normalized = student_id.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            students.append(MaterialStudent(student_id=normalized, student_name=normalized))
    return students


def build_material_records(
    blocks: Sequence[MaterialBlock],
    students: Sequence[MaterialStudent],
    class_name: str,
) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for student in students:
        records.extend(
            chunk_material_blocks(
                blocks,
                student_id=student.student_id,
                student_name=student.student_name,
                class_name=class_name,
            )
        )
    return records


def run_ingest_materials(
    blocks: Sequence[MaterialBlock],
    students: Sequence[MaterialStudent],
    class_name: str,
    store: SupportsMaterialStore,
    embedding_model: str,
) -> list[EmbeddingRecord]:
    records = build_material_records(blocks, students, class_name)
    records = embed_records_deduped(records, embedding_model)
    for student in students:
        store.delete_student_material_chunks(class_name, student.student_id)
    store.upsert_chunks(records)
    return records


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from scripts.utils.pg_store import connect_pg_store

    try:
        args = parse_args(argv)
        validate_inputs(args)
        blocks = extract_materials_dir(args.materials_dir)
        if not blocks:
            raise ValueError(
                f"No usable material text extracted from {args.materials_dir}. "
                "Check the folder holds .pptx/.pdf/.docx/.txt/.md files with real content."
            )
        students = resolve_students(args)
        if not students:
            raise ValueError(
                "No enrolled students resolved from the identity map or --student-id."
            )
    except (MaterialExtractError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    store = connect_pg_store(args.db_url)
    try:
        records = run_ingest_materials(
            blocks, students, args.class_name, store, args.embedding_model
        )
    finally:
        store.close()

    if args.chunk_review_path is not None:
        write_chunk_review(records, args.chunk_review_path)
        print(f"Chunk review CSV: {args.chunk_review_path}")

    print(
        f"Ingested {len(records)} material chunks "
        f"({len(records) // max(len(students), 1)} per student x {len(students)} students) "
        f"for class {args.class_name} -> pgvector"
    )


if __name__ == "__main__":
    main()
