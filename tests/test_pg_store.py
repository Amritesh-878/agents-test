from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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


def test_list_student_class_pairs_returns_triples() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("2301", "anshi", "English.04_AY26-27_Cornell Notetaking_29 Jun"),
        ("2302", "Bhagyashree", "Economics.02_AY2025-26_ Supply Function_16 April"),
    ]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    pairs = store.list_student_class_pairs()

    assert pairs == [
        ("2301", "anshi", "English.04_AY26-27_Cornell Notetaking_29 Jun"),
        ("2302", "Bhagyashree", "Economics.02_AY2025-26_ Supply Function_16 April"),
    ]
    sql = mock_cursor.execute.call_args.args[0]
    assert "DISTINCT" in sql and "class_name" in sql


def test_list_student_class_pairs_skips_null_class() -> None:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [("2301", "anshi", None)]
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    assert store.list_student_class_pairs() == []


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


def test_close_calls_connection_close() -> None:
    store, mock_conn = make_store()
    store.close()
    mock_conn.close.assert_called_once()


def test_connect_pg_store_is_callable() -> None:
    from scripts.utils.pg_store import connect_pg_store

    assert callable(connect_pg_store)


def _cursor_store() -> tuple[PgVectorStore, MagicMock, MagicMock]:
    store, mock_conn = make_store()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return store, mock_conn, mock_cursor


def test_log_query_inserts_a_row_and_commits() -> None:
    store, mock_conn, mock_cursor = _cursor_store()

    assert store.log_query("2409", "high", "groq", "English.03_x", 42) is True

    sql, params = mock_cursor.execute.call_args[0]
    assert "INSERT INTO query_log" in sql
    assert params == ("2409", "high", "groq", "English.03_x", 42)
    mock_conn.commit.assert_called_once()


def test_log_query_accepts_an_unscoped_turn() -> None:
    store, _, mock_cursor = _cursor_store()
    store.log_query("2409", "low", "fallback", None, 10)
    assert mock_cursor.execute.call_args[0][1] == ("2409", "low", "fallback", None, 10)


def test_log_query_never_stores_question_text() -> None:
    store, _, mock_cursor = _cursor_store()
    store.log_query("2409", "high", "groq", None, 63)
    sql, params = mock_cursor.execute.call_args[0]
    assert "question_len" in sql
    assert "text" not in sql.lower()
    assert all(not isinstance(p, str) or len(p) < 40 for p in params)


def test_log_query_degrades_to_a_warning_on_a_raising_connection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import psycopg

    store, mock_conn = make_store()
    mock_conn.cursor.side_effect = psycopg.OperationalError("connection gone")

    with caplog.at_level("WARNING"):
        assert store.log_query("2409", "high", "groq", None, 12) is False

    assert "telemetry" in caplog.text.casefold()
    mock_conn.commit.assert_not_called()


def test_log_query_degrades_when_the_insert_raises(caplog: pytest.LogCaptureFixture) -> None:
    import psycopg

    store, _, mock_cursor = _cursor_store()
    mock_cursor.execute.side_effect = psycopg.errors.UndefinedTable("no query_log table")

    with caplog.at_level("WARNING"):
        assert store.log_query("2409", "high", "groq", None, 12) is False
    assert "telemetry" in caplog.text.casefold()


def test_log_query_does_not_swallow_non_db_errors() -> None:
    store, mock_conn = make_store()
    mock_conn.cursor.side_effect = RuntimeError("a real bug")
    with pytest.raises(RuntimeError, match="a real bug"):
        store.log_query("2409", "high", "groq", None, 12)


def test_fetch_query_stats_unfiltered() -> None:
    store, _, mock_cursor = _cursor_store()
    mock_cursor.fetchall.return_value = [("high", "groq", 3), ("low", "fallback", 1)]

    rows = store.fetch_query_stats()

    sql, params = mock_cursor.execute.call_args[0]
    assert "WHERE" not in sql
    assert "GROUP BY grade, answer_source" in sql
    assert params == ()
    assert rows == [("high", "groq", 3), ("low", "fallback", 1)]


def test_fetch_query_stats_filters_by_student_ids() -> None:
    store, _, mock_cursor = _cursor_store()
    mock_cursor.fetchall.return_value = []

    store.fetch_query_stats(student_ids=["2409", "2410"])

    sql, params = mock_cursor.execute.call_args[0]
    assert "student_id = ANY(%s)" in sql
    assert params == (["2409", "2410"],)


def test_fetch_query_stats_filters_by_since() -> None:
    from datetime import UTC, datetime

    store, _, mock_cursor = _cursor_store()
    mock_cursor.fetchall.return_value = []
    since = datetime(2026, 7, 1, tzinfo=UTC)

    store.fetch_query_stats(since=since)

    sql, params = mock_cursor.execute.call_args[0]
    assert "ts >= %s" in sql
    assert params == (since,)


def test_fetch_query_stats_combines_filters() -> None:
    from datetime import UTC, datetime

    store, _, mock_cursor = _cursor_store()
    mock_cursor.fetchall.return_value = []
    since = datetime(2026, 7, 1, tzinfo=UTC)

    store.fetch_query_stats(student_ids=["2409"], since=since)

    sql, params = mock_cursor.execute.call_args[0]
    assert "student_id = ANY(%s)" in sql and "ts >= %s" in sql and " AND " in sql
    assert params == (["2409"], since)


def test_fetch_query_stats_with_an_empty_student_filter_skips_the_query() -> None:
    store, mock_conn = make_store()
    assert store.fetch_query_stats(student_ids=[]) == []
    mock_conn.cursor.assert_not_called()
