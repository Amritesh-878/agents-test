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


# --- delete_student_material_chunks ---


def test_delete_student_material_chunks_scoped_to_material() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 3
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    count = store.delete_student_material_chunks("CS101", "2301")

    assert count == 3
    sql, params = mock_cursor.execute.call_args.args
    assert "chunk_type = 'material'" in sql
    assert "student_id = %s" in sql
    assert params == ("CS101", "2301")
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


def test_search_pushes_class_name_filter_into_sql() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    store.search([0.1, 0.2, 0.3], "2301", top_k=5, class_name="CS101")

    sql, params = mock_cursor.execute.call_args.args
    assert "class_name = %s" in sql
    # class_name is bound as a parameter (between student_id and the LIMIT).
    assert params == ([0.1, 0.2, 0.3], "2301", "CS101", 5)


def test_search_combines_chunk_type_and_class_filters() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    store.search([0.1], "2301", top_k=5, chunk_types=["spoken"], class_name="CS101")

    sql, params = mock_cursor.execute.call_args.args
    assert "chunk_type = ANY(%s)" in sql
    assert "class_name = %s" in sql
    assert params == ([0.1], "2301", ["spoken"], "CS101", 5)


def test_search_empty_class_name_is_not_a_filter() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    store.search([0.1], "2301", top_k=5, class_name=None)

    sql, params = mock_cursor.execute.call_args.args
    assert "class_name = %s" not in sql
    assert params == ([0.1], "2301", 5)


def test_search_empty_returns_empty_list() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    results = store.search([0.1], "9999", top_k=5)
    assert results == []


def test_search_lexical_returns_results_with_no_distance() -> None:
    import json

    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("id1", "2301", "Anshi", "CS101", "spoken", "worksheet problems", 0.0, 5.0, "Anshi", json.dumps({})),
    ]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    results = store.search_lexical("worksheet", student_id="2301", limit=5)

    assert len(results) == 1
    assert results[0].chunk_id == "id1"
    assert results[0].distance is None
    sql = mock_cursor.execute.call_args.args[0]
    assert "websearch_to_tsquery" in sql
    assert "ts_rank_cd" in sql


def test_search_lexical_unfiltered_params() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    store.search_lexical("worksheet", student_id="2301", limit=5)

    sql, params = mock_cursor.execute.call_args.args
    assert "chunk_type = ANY" not in sql
    assert "class_name = %s" not in sql
    assert params == ("2301", "worksheet", "worksheet", 5)


def test_search_lexical_pushes_chunk_type_filter() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    store.search_lexical("worksheet", student_id="2301", chunk_types=["spoken", "chat"], limit=5)

    sql, params = mock_cursor.execute.call_args.args
    assert "chunk_type = ANY(%s)" in sql
    assert params == ("2301", ["spoken", "chat"], "worksheet", "worksheet", 5)


def test_search_lexical_pushes_class_name_filter() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    store.search_lexical("worksheet", student_id="2301", class_name="CS101", limit=5)

    sql, params = mock_cursor.execute.call_args.args
    assert "class_name = %s" in sql
    assert params == ("2301", "CS101", "worksheet", "worksheet", 5)


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


# --- get_student_name ---


def test_get_student_name_returns_name() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("Bhagyashree",)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    assert store.get_student_name("2302") == "Bhagyashree"
    sql, params = mock_cursor.execute.call_args.args
    assert "LIMIT 1" in sql
    assert params == ("2302",)


def test_get_student_name_missing_returns_none() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    assert store.get_student_name("9999") is None


# --- list_students ---


def test_list_students_returns_distinct_pairs() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [("2504", "A_Disha"), ("2302", "Bhagyashree")]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    students = store.list_students()

    assert students == [("2504", "A_Disha"), ("2302", "Bhagyashree")]
    sql = mock_cursor.execute.call_args.args[0]
    assert "DISTINCT" in sql
    assert "student_id" in sql and "student_name" in sql


def test_list_students_empty_returns_empty_list() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    assert store.list_students() == []


# --- list_student_classes ---


def test_list_student_classes_returns_ordered_class_names() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [("Economics.02_Supply",), ("Math.01_Time",)]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    classes = store.list_student_classes("2302")

    assert classes == ["Economics.02_Supply", "Math.01_Time"]
    sql, params = mock_cursor.execute.call_args.args
    assert "DISTINCT" in sql and "class_name" in sql
    assert params == ("2302",)


def test_list_student_classes_empty_returns_empty_list() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    assert store.list_student_classes("9999") == []


# --- close ---


def test_close_calls_connection_close() -> None:
    store, mock_conn = make_store()
    store.close()
    mock_conn.close.assert_called_once()


# --- connect_pg_store (import check) ---


def test_connect_pg_store_is_callable() -> None:
    from scripts.utils.pg_store import connect_pg_store

    assert callable(connect_pg_store)
