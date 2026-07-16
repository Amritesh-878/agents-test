from __future__ import annotations

import json
import logging
from datetime import datetime
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

_LEXICAL_SEARCH_HEAD = """
SELECT id, student_id, student_name, class_name, chunk_type, text,
       start_time, end_time, speaker, metadata
FROM embeddings
WHERE """
_LEXICAL_SEARCH_TAIL = """
ORDER BY ts_rank_cd(to_tsvector('simple', text), websearch_to_tsquery('simple', %s)) DESC
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

_LIST_STUDENT_CLASS_PAIRS_SQL = """
SELECT DISTINCT student_id, student_name, class_name
FROM embeddings
ORDER BY student_name, student_id, class_name
"""

_COUNT_CHUNKS_SQL = "SELECT COUNT(*) FROM embeddings"

_LOG_QUERY_SQL = """
INSERT INTO query_log (student_id, grade, answer_source, scoped_class, question_len)
VALUES (%s, %s, %s, %s, %s)
"""

_QUERY_STATS_HEAD = """
SELECT grade, answer_source, COUNT(*)
FROM query_log
"""
_QUERY_STATS_TAIL = """
GROUP BY grade, answer_source
ORDER BY grade, answer_source
"""


def db_error_types() -> tuple[type[BaseException], ...]:
    try:
        import psycopg
    except ImportError:
        return ()
    return (psycopg.Error,)


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

    def search_lexical(
        self,
        query_text: str,
        *,
        student_id: str,
        chunk_types: Sequence[str] | None = None,
        limit: int = 25,
        class_name: str | None = None,
    ) -> list[SearchResult]:
        types = list(chunk_types or [])
        conditions = ["student_id = %s"]
        filter_params: list[Any] = [student_id]
        if types:
            conditions.append("chunk_type = ANY(%s)")
            filter_params.append(types)
        if class_name:
            conditions.append("class_name = %s")
            filter_params.append(class_name)
        conditions.append("to_tsvector('simple', text) @@ websearch_to_tsquery('simple', %s)")
        filter_params.append(query_text)
        sql = _LEXICAL_SEARCH_HEAD + " AND ".join(conditions) + _LEXICAL_SEARCH_TAIL
        params: tuple[Any, ...] = (*filter_params, query_text, limit)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
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
                    distance=None,
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

    def list_student_class_pairs(self) -> list[tuple[str, str, str]]:
        with self._conn.cursor() as cur:
            cur.execute(_LIST_STUDENT_CLASS_PAIRS_SQL)
            rows = cur.fetchall()
        return [
            (str(sid), str(name), str(cname))
            for sid, name, cname in rows
            if cname
        ]

    def list_student_classes(self, student_id: str) -> list[str]:
        """Return the distinct ``class_name`` (session) values for one student, ordered.

        Used by the demo to populate the per-session picker so a student can scope a
        question to a single class instead of all their sessions at once.
        """
        with self._conn.cursor() as cur:
            cur.execute(_LIST_STUDENT_CLASSES_SQL, (student_id,))
            rows = cur.fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def count_chunks(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(_COUNT_CHUNKS_SQL)
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def log_query(
        self,
        student_id: str,
        grade: str,
        answer_source: str,
        scoped_class: str | None,
        question_len: int,
    ) -> bool:
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    _LOG_QUERY_SQL,
                    (student_id, grade, answer_source, scoped_class, question_len),
                )
            self._conn.commit()
        except db_error_types() as exc:
            logger.warning("Query telemetry insert failed for %r: %s", student_id, exc)
            return False
        return True

    def fetch_query_stats(
        self,
        *,
        student_ids: Sequence[str] | None = None,
        since: datetime | None = None,
    ) -> list[tuple[str, str, int]]:
        clauses: list[str] = []
        params: list[Any] = []
        if student_ids is not None:
            if not student_ids:
                return []
            clauses.append("student_id = ANY(%s)")
            params.append(list(student_ids))
        if since is not None:
            clauses.append("ts >= %s")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"{_QUERY_STATS_HEAD}{where}{_QUERY_STATS_TAIL}"
        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [(str(grade), str(source), int(count)) for grade, source, count in rows]

    def close(self) -> None:
        self._conn.close()


def connect_pg_store(db_url: str) -> PgVectorStore:
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(db_url)
    register_vector(conn)
    return PgVectorStore(conn)
