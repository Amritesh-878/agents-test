from __future__ import annotations

import logging

import pytest

import scripts.utils.db_url as db_url_mod
from scripts.utils.db_url import resolve_db_url


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db_url_mod, "load_dotenv", lambda *a, **k: False)


def test_flag_value_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://env/db")
    assert resolve_db_url("postgresql://flag/db") == "postgresql://flag/db"


def test_flag_value_warns_about_leak(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with caplog.at_level(logging.WARNING):
        resolve_db_url("postgresql://postgres:pw@localhost/db")
    assert any("DATABASE_URL" in record.message for record in caplog.records)


def test_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://env/db")
    assert resolve_db_url(None) == "postgresql://env/db"


def test_empty_flag_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://env/db")
    assert resolve_db_url("   ") == "postgresql://env/db"


def test_returns_empty_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert resolve_db_url(None) == ""
