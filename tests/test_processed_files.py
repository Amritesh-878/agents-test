from __future__ import annotations

from unittest.mock import MagicMock

from scripts.utils.processed_files import ProcessedFilesStore


def make_store() -> tuple[ProcessedFilesStore, MagicMock]:
    mock_conn = MagicMock()
    store = ProcessedFilesStore(mock_conn)
    return store, mock_conn


def _bind_cursor(mock_conn: MagicMock) -> MagicMock:
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_cursor


# --- is_processed ---


def test_is_processed_true_when_row_exists() -> None:
    store, mock_conn = make_store()
    mock_cursor = _bind_cursor(mock_conn)
    mock_cursor.fetchone.return_value = (1,)

    assert store.is_processed("file-123") is True
    sql, params = mock_cursor.execute.call_args.args
    assert "processed_files" in sql
    assert params == ("file-123",)


def test_is_processed_false_when_no_row() -> None:
    store, mock_conn = make_store()
    mock_cursor = _bind_cursor(mock_conn)
    mock_cursor.fetchone.return_value = None

    assert store.is_processed("file-404") is False


# --- mark_processed ---


def test_mark_processed_inserts_and_commits() -> None:
    store, mock_conn = make_store()
    mock_cursor = _bind_cursor(mock_conn)

    store.mark_processed("file-1", "Economics.02")

    mock_cursor.execute.assert_called_once()
    sql, params = mock_cursor.execute.call_args.args
    assert "INSERT INTO processed_files" in sql
    assert params == ("file-1", "Economics.02")
    mock_conn.commit.assert_called_once()


def test_mark_processed_uses_parameters_not_fstring() -> None:
    store, mock_conn = make_store()
    mock_cursor = _bind_cursor(mock_conn)

    store.mark_processed("'; DROP TABLE embeddings; --", "X")

    sql, params = mock_cursor.execute.call_args.args
    # The malicious value travels as a bound parameter, never interpolated.
    assert "DROP TABLE" not in sql
    assert params[0] == "'; DROP TABLE embeddings; --"


# --- processed_ids ---


def test_processed_ids_returns_set() -> None:
    store, mock_conn = make_store()
    mock_cursor = _bind_cursor(mock_conn)
    mock_cursor.fetchall.return_value = [("a",), ("b",), ("c",)]

    assert store.processed_ids() == {"a", "b", "c"}


def test_processed_ids_empty() -> None:
    store, mock_conn = make_store()
    mock_cursor = _bind_cursor(mock_conn)
    mock_cursor.fetchall.return_value = []

    assert store.processed_ids() == set()


# --- close / connect ---


def test_close_calls_connection_close() -> None:
    store, mock_conn = make_store()
    store.close()
    mock_conn.close.assert_called_once()


def test_connect_processed_files_store_is_callable() -> None:
    from scripts.utils.processed_files import connect_processed_files_store

    assert callable(connect_processed_files_store)
