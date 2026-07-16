from __future__ import annotations

from unittest.mock import MagicMock

from scripts.migrate_db import (
    build_dimension_migration_ddl,
    build_ddl,
    build_trigram_cleanup_ddl,
    get_ddl_statements,
    run_migration,
)
from scripts.models.pipeline import MigrationResult


def test_ddl_has_extension() -> None:
    stmts = get_ddl_statements()
    kinds = [kind for kind, _, _ in stmts]
    assert "extension" in kinds


def test_ddl_has_table() -> None:
    stmts = get_ddl_statements()
    kinds = [kind for kind, _, _ in stmts]
    assert "table" in kinds


def test_ddl_has_indexes() -> None:
    stmts = get_ddl_statements()
    kinds = [kind for kind, _, _ in stmts]
    assert "index" in kinds


def test_ddl_is_idempotent() -> None:
    stmts = get_ddl_statements()
    for kind, name, sql in stmts:
        upper = sql.upper()
        assert "IF NOT EXISTS" in upper, f"DDL for {kind} '{name}' is not idempotent: {sql[:60]}"


def test_ddl_extension_sql_correct() -> None:
    stmts = get_ddl_statements()
    ext_stmts = [(n, s) for k, n, s in stmts if k == "extension"]
    assert any("vector" in sql.lower() for _, sql in ext_stmts)


def test_ddl_table_has_embedding_column() -> None:
    stmts = get_ddl_statements()
    table_stmts = [s for k, _, s in stmts if k == "table"]
    assert any("embedding" in sql.lower() for sql in table_stmts)


def test_ddl_table_has_required_columns() -> None:
    stmts = get_ddl_statements()
    table_sql = next(s for k, _, s in stmts if k == "table")
    for col in ("id", "student_id", "class_name", "text", "metadata"):
        assert col in table_sql.lower(), f"Column '{col}' missing from table DDL"


def test_ddl_hnsw_index_present() -> None:
    stmts = get_ddl_statements()
    index_stmts = [s for k, _, s in stmts if k == "index"]
    assert any("hnsw" in sql.lower() for sql in index_stmts)


def test_ddl_has_processed_files_table() -> None:
    stmts = get_ddl_statements()
    names = [name for kind, name, _ in stmts if kind == "table"]
    assert "processed_files" in names


def test_ddl_processed_files_keyed_by_drive_file_id() -> None:
    stmts = get_ddl_statements()
    sql = next(s for k, n, s in stmts if k == "table" and n == "processed_files")
    lowered = sql.lower()
    assert "drive_file_id text primary key" in lowered
    assert "class_name" in lowered
    assert "processed_at" in lowered


def test_migration_result_model() -> None:
    result = MigrationResult(
        extensions_created=["vector"],
        tables_created=["embeddings"],
        indexes_created=["idx_a", "idx_b"],
        success=True,
    )
    assert result.success
    assert "vector" in result.extensions_created


def test_validate_inputs_empty_url() -> None:
    import pytest

    from scripts.migrate_db import MigrateArgs, validate_inputs

    with pytest.raises(ValueError, match="Database URL"):
        validate_inputs(MigrateArgs(db_url=""))


def test_build_ddl_defaults_to_384() -> None:
    table_sql = next(s for k, _, s in build_ddl() if k == "table" and "embedding vector" in s)
    assert "vector(384)" in table_sql


def test_build_ddl_uses_requested_dimension() -> None:
    table_sql = next(s for k, _, s in build_ddl(768) if k == "table" and "embedding vector" in s)
    assert "vector(768)" in table_sql
    assert "vector(384)" not in table_sql


def test_dimension_migration_alters_column_and_rebuilds_index() -> None:
    sqls = [s for _, _, s in build_dimension_migration_ddl(768)]
    assert any("DROP INDEX IF EXISTS idx_embeddings_hnsw" in s for s in sqls)
    assert any("ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector(768)" in s for s in sqls)
    assert any("CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw" in s for s in sqls)


def _mock_conn() -> tuple[MagicMock, MagicMock]:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cur


def test_run_migration_applies_dimension_migration_when_dim_differs() -> None:
    conn, cur = _mock_conn()
    run_migration(conn, embedding_dim=768)
    executed = " ".join(str(call.args[0]) for call in cur.execute.call_args_list)
    assert "vector(768)" in executed
    assert "ALTER COLUMN embedding TYPE vector(768)" in executed


def test_run_migration_default_dim_has_no_column_alter() -> None:
    conn, cur = _mock_conn()
    run_migration(conn)
    executed = " ".join(str(call.args[0]) for call in cur.execute.call_args_list)
    assert "vector(384)" in executed
    assert "ALTER COLUMN" not in executed


def test_build_ddl_no_longer_creates_trigram_machinery() -> None:
    joined = " ".join(sql for _, _, sql in build_ddl()).lower()
    assert "pg_trgm" not in joined
    assert "gin_trgm_ops" not in joined


def test_trigram_cleanup_drops_index_idempotently() -> None:
    sqls = [sql for _, _, sql in build_trigram_cleanup_ddl()]
    assert any("DROP INDEX IF EXISTS idx_embeddings_text_trgm" in sql for sql in sqls)


def test_run_migration_drops_stale_trigram_index() -> None:
    conn, cur = _mock_conn()
    run_migration(conn)
    executed = " ".join(str(call.args[0]) for call in cur.execute.call_args_list)
    assert "DROP INDEX IF EXISTS idx_embeddings_text_trgm" in executed


# --- query_log telemetry table (TASK-022) ---


def test_ddl_creates_the_query_log_table() -> None:
    tables = [name for kind, name, _ in get_ddl_statements() if kind == "table"]
    assert "query_log" in tables


def test_query_log_ddl_has_the_specced_columns() -> None:
    sql = next(s for k, n, s in get_ddl_statements() if k == "table" and n == "query_log")
    lowered = sql.lower()
    for column in ("id", "ts", "student_id", "grade", "answer_source", "scoped_class", "question_len"):
        assert column in lowered


def test_query_log_ddl_stores_no_question_text() -> None:
    sql = next(s for k, n, s in get_ddl_statements() if k == "table" and n == "query_log")
    columns = {
        line.strip().split()[0].lower()
        for line in sql.splitlines()
        if line.startswith("    ") and line.strip()
    }
    assert columns == {
        "id",
        "ts",
        "student_id",
        "grade",
        "answer_source",
        "scoped_class",
        "question_len",
    }


def test_query_log_ddl_is_created_alongside_embeddings() -> None:
    tables = [name for kind, name, _ in get_ddl_statements() if kind == "table"]
    assert "embeddings" in tables and "query_log" in tables


def test_run_migration_reports_query_log() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    result = run_migration(conn)

    assert isinstance(result, MigrationResult)
    assert "query_log" in result.tables_created
