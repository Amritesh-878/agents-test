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

# Fixed SELECT/ORDER/LIMIT around a WHERE clause assembled from static fragments only
# (never user text), so the query stays fully parameterized — values go through %s.
_SEARCH_SQL_HEAD = """
SELECT id, student_id, student_name, class_name, chunk_type, text,
       start_time, end_time, speaker, metadata,
       embedding <=> %s::vector AS distance
FROM embeddings
WHERE """
_SEARCH_SQL_TAIL = """
ORDER BY distance
LIMIT %s
"""

_DELETE_CLASS_SQL = "DELETE FROM embeddings WHERE class_name = %s"

_DELETE_STUDENT_MATERIAL_SQL = (
    "DELETE FROM embeddings "
    "WHERE class_name = %s AND student_id = %s AND chunk_type = 'material'"
)

_GET_STUDENT_SQL = """
SELECT id, student_id, student_name, class_name, chunk_type, text,
       start_time, end_time, speaker, metadata
FROM embeddings
WHERE student_id = %s
"""

_GET_STUDENT_NAME_SQL = "SELECT student_name FROM embeddings WHERE student_id = %s LIMIT 1"

_LIST_STUDENTS_SQL = """
SELECT DISTINCT student_id, student_name
FROM embeddings
ORDER BY student_name, student_id
"""

_LIST_STUDENT_CLASSES_SQL = """
SELECT DISTINCT class_name
FROM embeddings
WHERE student_id = %s
ORDER BY class_name
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

    def delete_student_material_chunks(self, class_name: str, student_id: str) -> int:
        """Purge one student's ``material`` chunks for a class before re-upserting.

        Scoped to chunk_type='material' so spoken/chat/class_context/missed chunks
        are never touched by a materials re-ingest.
        """
        with self._conn.cursor() as cur:
            cur.execute(_DELETE_STUDENT_MATERIAL_SQL, (class_name, student_id))
            count = cur.rowcount
        self._conn.commit()
        return count

    def search(
        self,
        query_embedding: list[float],
        student_id: str,
        top_k: int = 5,
        chunk_types: Sequence[str] | None = None,
        class_name: str | None = None,
    ) -> list[SearchResult]:
        types = list(chunk_types or [])
        # Build the WHERE clause from static fragments; every value is bound via %s.
        conditions = ["student_id = %s"]
        filter_params: list[Any] = [student_id]
        if types:
            conditions.append("chunk_type = ANY(%s)")
            filter_params.append(types)
        if class_name:
            conditions.append("class_name = %s")
            filter_params.append(class_name)
        sql = _SEARCH_SQL_HEAD + " AND ".join(conditions) + _SEARCH_SQL_TAIL
        params: tuple[Any, ...] = (query_embedding, *filter_params, top_k)

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

    def get_student_name(self, student_id: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(_GET_STUDENT_NAME_SQL, (student_id,))
            row = cur.fetchone()
        if row is None:
            return None
        name = row[0]
        return name if isinstance(name, str) and name else None

    def list_students(self) -> list[tuple[str, str]]:
        """Return distinct ``(student_id, student_name)`` pairs present in the store.

        Used by the demo UI to populate the teacher's student picker. Names may
        repeat across ids, so the id is the stable selector.
        """
        with self._conn.cursor() as cur:
            cur.execute(_LIST_STUDENTS_SQL)
            rows = cur.fetchall()
        return [(str(sid), str(name)) for sid, name in rows]

    def list_student_classes(self, student_id: str) -> list[str]:
        """Return the distinct ``class_name`` (session) values for one student, ordered.

        Used by the demo to populate the per-session picker so a student can scope a
        question to a single class instead of all their sessions at once.
        """
        with self._conn.cursor() as cur:
            cur.execute(_LIST_STUDENT_CLASSES_SQL, (student_id,))
            rows = cur.fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def close(self) -> None:
        self._conn.close()


def connect_pg_store(db_url: str) -> PgVectorStore:
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(db_url)
    register_vector(conn)
    return PgVectorStore(conn)
