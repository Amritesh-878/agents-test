from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import BaseModel

from scripts.models.pipeline import MigrationResult
from scripts.utils.db_url import resolve_db_url

if TYPE_CHECKING:
    import psycopg

logger = logging.getLogger(__name__)

_DDL = [
    ("extension", "vector", "CREATE EXTENSION IF NOT EXISTS vector"),
    (
        "table",
        "embeddings",
        """CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    student_name TEXT NOT NULL,
    class_name TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384),
    start_time FLOAT,
    end_time FLOAT,
    speaker TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
)""",
    ),
    (
        "index",
        "idx_embeddings_student_class",
        "CREATE INDEX IF NOT EXISTS idx_embeddings_student_class ON embeddings (student_id, class_name)",
    ),
    (
        "index",
        "idx_embeddings_hnsw",
        """CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)""",
    ),
]


class MigrateArgs(BaseModel):
    db_url: str


def parse_args(argv: Sequence[str] | None = None) -> MigrateArgs:
    parser = argparse.ArgumentParser(
        description="Create pgvector schema for the Adira embedding store."
    )
    parser.add_argument(
        "--db-url",
        dest="db_url",
        default=None,
        help="PostgreSQL connection URL. Falls back to DATABASE_URL env var.",
    )
    namespace = parser.parse_args(argv)
    return MigrateArgs(db_url=resolve_db_url(namespace.db_url))


def validate_inputs(args: MigrateArgs) -> None:
    if not args.db_url.strip():
        raise ValueError(
            "Database URL is required. Pass --db-url or set DATABASE_URL in .env."
        )


def run_migration(conn: psycopg.Connection[Any]) -> MigrationResult:
    extensions_created: list[str] = []
    tables_created: list[str] = []
    indexes_created: list[str] = []

    with conn.cursor() as cur:
        for kind, name, sql in _DDL:
            logger.info("Running: %s %s", kind, name)
            cur.execute(sql)
            if kind == "extension":
                extensions_created.append(name)
            elif kind == "table":
                tables_created.append(name)
            elif kind == "index":
                indexes_created.append(name)
    conn.commit()

    return MigrationResult(
        extensions_created=extensions_created,
        tables_created=tables_created,
        indexes_created=indexes_created,
        success=True,
    )


def get_ddl_statements() -> list[tuple[str, str, str]]:
    return list(_DDL)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    try:
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(args.db_url)
        register_vector(conn)
    except Exception as exc:
        logger.error("Database connection failed: %s", exc)
        raise SystemExit(1) from exc

    result = run_migration(conn)
    conn.close()

    print(f"Migration complete: extensions={result.extensions_created} "
          f"tables={result.tables_created} indexes={result.indexes_created}")


if __name__ == "__main__":
    main()
