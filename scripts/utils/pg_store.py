from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from scripts.models.pipeline import EmbeddingRecord, SearchResult

logger = logging.getLogger(__name__)

_UPSERT_SQL = """
INSERT INTO embeddings (
    id, student_id, student_name, class_name, chunk_type, text,
    embedding, start_time, end_time, speaker, metadata
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    text = EXCLUDED.text,
    embedding = EXCLUDED.embedding,
    metadata = EXCLUDED.metadata
"""

_SEARCH_SQL = """
SELECT id, student_id, student_name, class_name, chunk_type, text,
       start_time, end_time, speaker, metadata,
       embedding <=> %s::vector AS distance
FROM embeddings
WHERE student_id = %s
ORDER BY distance
LIMIT %s
"""

_SEARCH_SQL_WITH_TYPES = """
SELECT id, student_id, student_name, class_name, chunk_type, text,
       start_time, end_time, speaker, metadata,
       embedding <=> %s::vector AS distance
FROM embeddings
WHERE student_id = %s AND chunk_type = ANY(%s)
ORDER BY distance
LIMIT %s
"""

_DELETE_CLASS_SQL = "DELETE FROM embeddings WHERE class_name = %s"

_GET_STUDENT_SQL = """
SELECT id, student_id, student_name, class_name, chunk_type, text,
       start_time, end_time, speaker, metadata
FROM embeddings
WHERE student_id = %s
"""


class PgVectorStore:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def upsert_chunks(self, records: Sequence[EmbeddingRecord]) -> int:
        if not records:
            return 0

        with self._conn.cursor() as cur:
            for rec in records:
                cur.execute(
                    _UPSERT_SQL,
                    (
                        rec.id,
                        rec.student_id,
                        rec.student_name,
                        rec.class_name,
                        rec.chunk_type,
                        rec.text,
                        rec.embedding,
                        rec.start_time,
                        rec.end_time,
                        rec.speaker,
                        json.dumps(rec.metadata),
                    ),
                )
        self._conn.commit()
        return len(records)

    def delete_class_chunks(self, class_name: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(_DELETE_CLASS_SQL, (class_name,))
            count = cur.rowcount
        self._conn.commit()
        return count

    def search(
        self,
        query_embedding: list[float],
        student_id: str,
        top_k: int = 5,
        chunk_types: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        types = list(chunk_types or [])
        if types:
            sql = _SEARCH_SQL_WITH_TYPES
            params: tuple[Any, ...] = (query_embedding, student_id, types, top_k)
        else:
            sql = _SEARCH_SQL
            params = (query_embedding, student_id, top_k)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[SearchResult] = []
        for row in rows:
            (
                chunk_id, sid, sname, cname, ctype, text,
                start_time, end_time, speaker, metadata_raw, distance,
            ) = row
            metadata: dict[str, Any] = {}
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except ValueError:
                    pass
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    student_id=sid,
                    student_name=sname,
                    class_name=cname,
                    chunk_type=ctype,
                    text=text,
                    distance=float(distance),
                    start_time=start_time,
                    end_time=end_time,
                    speaker=speaker,
                    metadata=metadata,
                )
            )
        return results

    def get_student_chunks(self, student_id: str) -> list[SearchResult]:
        with self._conn.cursor() as cur:
            cur.execute(_GET_STUDENT_SQL, (student_id,))
            rows = cur.fetchall()

        results: list[SearchResult] = []
        for row in rows:
            (chunk_id, sid, sname, cname, ctype, text, start_time, end_time, speaker, metadata_raw) = row
            metadata: dict[str, Any] = {}
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except ValueError:
                    pass
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    student_id=sid,
                    student_name=sname,
                    class_name=cname,
                    chunk_type=ctype,
                    text=text,
                    distance=0.0,
                    start_time=start_time,
                    end_time=end_time,
                    speaker=speaker,
                    metadata=metadata,
                )
            )
        return results

    def close(self) -> None:
        self._conn.close()


def connect_pg_store(db_url: str) -> PgVectorStore:
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(db_url)
    register_vector(conn)
    return PgVectorStore(conn)
