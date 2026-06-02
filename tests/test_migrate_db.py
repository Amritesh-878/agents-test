from __future__ import annotations

from scripts.migrate_db import get_ddl_statements
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
