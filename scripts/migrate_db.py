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

DEFAULT_EMBEDDING_DIM = 384

_HNSW_INDEX_SQL = """CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)"""


def build_ddl(embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> list[tuple[str, str, str]]:
    return [
        ("extension", "vector", "CREATE EXTENSION IF NOT EXISTS vector"),
        (
            "table",
            "embeddings",
            f"""CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    student_name TEXT NOT NULL,
    class_name TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector({embedding_dim}),
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
            "table",
            "processed_files",
            """CREATE TABLE IF NOT EXISTS processed_files (
    drive_file_id TEXT PRIMARY KEY,
    class_name TEXT NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT NOW()
)""",
        ),
        (
            "table",
            "query_log",
            """CREATE TABLE IF NOT EXISTS query_log (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT NOW(),
    student_id TEXT,
    grade TEXT,
    answer_source TEXT,
    scoped_class TEXT,
    question_len INT
)""",
        ),
        (
            "index",
            "idx_query_log_student_ts",
            "CREATE INDEX IF NOT EXISTS idx_query_log_student_ts ON query_log (student_id, ts)",
        ),
        ("index", "idx_embeddings_hnsw", _HNSW_INDEX_SQL),
        (
            "index",
            "idx_embeddings_text_fts",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_text_fts "
            "ON embeddings USING gin (to_tsvector('simple', text))",
        ),
    ]


def build_trigram_cleanup_ddl() -> list[tuple[str, str, str]]:
    return [
        ("index", "idx_embeddings_text_trgm", "DROP INDEX IF EXISTS idx_embeddings_text_trgm"),
    ]


def build_dimension_migration_ddl(embedding_dim: int) -> list[tuple[str, str, str]]:
    return [
        ("index", "idx_embeddings_hnsw", "DROP INDEX IF EXISTS idx_embeddings_hnsw"),
        (
            "column",
            "embedding",
            f"ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector({embedding_dim})",
        ),
        ("index", "idx_embeddings_hnsw", _HNSW_INDEX_SQL),
    ]


class MigrateArgs(BaseModel):
    db_url: str
    embedding_dim: int = DEFAULT_EMBEDDING_DIM


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
    parser.add_argument(
        "--embedding-dim",
        dest="embedding_dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="pgvector column dimension. Set to the candidate model's dim (e.g. 768) to "
        "migrate the existing column; the store must be cleared and re-embedded at that dim.",
    )
    namespace = parser.parse_args(argv)
    return MigrateArgs(
        db_url=resolve_db_url(namespace.db_url),
        embedding_dim=namespace.embedding_dim,
    )


def validate_inputs(args: MigrateArgs) -> None:
    if not args.db_url.strip():
        raise ValueError(
            "Database URL is required. Pass --db-url or set DATABASE_URL in .env."
        )


def run_migration(
    conn: psycopg.Connection[Any], embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> MigrationResult:
    extensions_created: list[str] = []
    tables_created: list[str] = []
    indexes_created: list[str] = []

    statements = list(build_ddl(embedding_dim))
    statements.extend(build_trigram_cleanup_ddl())
    if embedding_dim != DEFAULT_EMBEDDING_DIM:
        statements.extend(build_dimension_migration_ddl(embedding_dim))

    with conn.cursor() as cur:
        for kind, name, sql in statements:
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


def get_ddl_statements(embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> list[tuple[str, str, str]]:
    return build_ddl(embedding_dim)


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

    result = run_migration(conn, args.embedding_dim)
    conn.close()

    print(f"Migration complete: extensions={result.extensions_created} "
          f"tables={result.tables_created} indexes={result.indexes_created}")


if __name__ == "__main__":
    main()
