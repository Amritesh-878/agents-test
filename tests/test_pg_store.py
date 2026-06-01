from __future__ import annotations

from unittest.mock import MagicMock

from scripts.models.pipeline import EmbeddingRecord
from scripts.utils.pg_store import PgVectorStore


def make_store() -> tuple[PgVectorStore, MagicMock]:
    mock_conn = MagicMock()
    store = PgVectorStore(mock_conn)
    return store, mock_conn


def make_record(
    id: str = "abc123",
    student_id: str = "2301",
    text: str = "hello world",
    chunk_type: str = "spoken",
    class_name: str = "CS101",
) -> EmbeddingRecord:
    return EmbeddingRecord(
        id=id,
        student_id=student_id,
        student_name="Anshi",
        class_name=class_name,
        chunk_type=chunk_type,
        text=text,
        embedding=[0.1, 0.2, 0.3],
    )


# --- upsert_chunks ---


def test_upsert_calls_execute_for_each_record() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    records = [make_record("r1"), make_record("r2")]
    count = store.upsert_chunks(records)

    assert count == 2
    assert mock_cursor.execute.call_count == 2
    mock_conn.commit.assert_called_once()


def test_upsert_empty_returns_zero() -> None:
    store, mock_conn = make_store()
    assert store.upsert_chunks([]) == 0
    mock_conn.cursor.assert_not_called()


# --- delete_class_chunks ---


def test_delete_class_chunks_executes_delete() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 5
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    count = store.delete_class_chunks("CS101")

    assert count == 5
    mock_cursor.execute.assert_called_once()
    sql, params = mock_cursor.execute.call_args.args
    assert "DELETE" in sql
    assert params == ("CS101",)
    mock_conn.commit.assert_called_once()


# --- search ---


def test_search_returns_search_results() -> None:
    import json

    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("id1", "2301", "Anshi", "CS101", "spoken", "hello", 0.0, 5.0, "Anshi", json.dumps({}), 0.1),
    ]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    results = store.search([0.1, 0.2, 0.3], "2301", top_k=3)

    assert len(results) == 1
    assert results[0].chunk_id == "id1"
    assert results[0].student_id == "2301"
    assert results[0].distance == 0.1


def test_search_pushes_chunk_type_filter_into_sql() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    store.search([0.1, 0.2, 0.3], "2301", top_k=5, chunk_types=["spoken"])

    sql, params = mock_cursor.execute.call_args.args
    assert "chunk_type = ANY(%s)" in sql
    # types array is passed as a parameter (not interpolated), preserving the LIMIT.
    assert params == ([0.1, 0.2, 0.3], "2301", ["spoken"], 5)


def test_search_without_chunk_types_uses_plain_sql() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    store.search([0.1], "2301", top_k=5)

    sql, params = mock_cursor.execute.call_args.args
    assert "chunk_type = ANY" not in sql
    assert params == ([0.1], "2301", 5)


def test_search_empty_returns_empty_list() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    results = store.search([0.1], "9999", top_k=5)
    assert results == []


# --- get_student_chunks ---


def test_get_student_chunks_no_distance() -> None:
    import json

    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("id1", "2301", "Anshi", "CS101", "missed", "missed text", None, None, None, json.dumps({})),
    ]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    chunks = store.get_student_chunks("2301")
    assert len(chunks) == 1
    assert chunks[0].distance == 0.0
    assert chunks[0].chunk_type == "missed"


# --- close ---


def test_close_calls_connection_close() -> None:
    store, mock_conn = make_store()
    store.close()
    mock_conn.close.assert_called_once()


# --- connect_pg_store (import check) ---


def test_connect_pg_store_is_callable() -> None:
    from scripts.utils.pg_store import connect_pg_store

    assert callable(connect_pg_store)
