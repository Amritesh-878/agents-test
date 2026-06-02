from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_EXISTS_SQL = "SELECT 1 FROM processed_files WHERE drive_file_id = %s LIMIT 1"

_INSERT_SQL = """
INSERT INTO processed_files (drive_file_id, class_name)
VALUES (%s, %s)
ON CONFLICT (drive_file_id) DO NOTHING
"""

_GET_ALL_SQL = "SELECT drive_file_id FROM processed_files"


class ProcessedFilesStore:
    """Idempotent dedup ledger for Drive ingestion, keyed by Drive file id.

    Mirrors :class:`scripts.utils.pg_store.PgVectorStore`'s raw-SQL pattern: every
    query is parameterized (``%s`` placeholders, no f-string SQL) and the caller
    owns the connection lifecycle.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def is_processed(self, drive_file_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(_EXISTS_SQL, (drive_file_id,))
            return cur.fetchone() is not None

    def mark_processed(self, drive_file_id: str, class_name: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_INSERT_SQL, (drive_file_id, class_name))
        self._conn.commit()

    def processed_ids(self) -> set[str]:
        with self._conn.cursor() as cur:
            cur.execute(_GET_ALL_SQL)
            rows = cur.fetchall()
        return {row[0] for row in rows}

    def close(self) -> None:
        self._conn.close()


def connect_processed_files_store(db_url: str) -> ProcessedFilesStore:
    import psycopg

    conn = psycopg.connect(db_url)
    return ProcessedFilesStore(conn)
