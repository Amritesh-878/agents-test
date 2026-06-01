from __future__ import annotations

from pathlib import Path

import pytest

from scripts.auth import AuthError, AuthService, load_credentials


def write_csv(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# --- load_credentials ---


def test_load_credentials_success(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", "student_id,password\n2302,alpha\n2504,beta\n")
    creds = load_credentials(path)
    assert creds == {"2302": "alpha", "2504": "beta"}


def test_load_credentials_tolerant_headers(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", " Student_ID , Password \n2302,alpha\n")
    assert load_credentials(path) == {"2302": "alpha"}


def test_load_credentials_skips_blank_lines(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", "student_id,password\n2302,alpha\n\n2504,beta\n")
    assert load_credentials(path) == {"2302": "alpha", "2504": "beta"}


def test_load_credentials_skips_comment_lines(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "c.csv",
        "student_id,password\n2302,alpha\n# a docs comment line,ignored\n",
    )
    assert load_credentials(path) == {"2302": "alpha"}


def test_load_credentials_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AuthError, match="not found"):
        load_credentials(tmp_path / "nope.csv")


def test_load_credentials_missing_columns(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", "id,secret\n2302,alpha\n")
    with pytest.raises(AuthError, match="columns"):
        load_credentials(path)


def test_load_credentials_duplicate_id(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", "student_id,password\n2302,alpha\n2302,beta\n")
    with pytest.raises(AuthError, match="Duplicate student_id"):
        load_credentials(path)


def test_load_credentials_empty_id(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", "student_id,password\n,alpha\n")
    with pytest.raises(AuthError, match="missing a student_id"):
        load_credentials(path)


def test_load_credentials_empty_password(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", "student_id,password\n2302,\n")
    with pytest.raises(AuthError, match="empty password"):
        load_credentials(path)


# --- authenticate ---


def make_service() -> AuthService:
    return AuthService({"2302": "alpha", "2504": "beta"})


def test_authenticate_correct() -> None:
    assert make_service().authenticate("2302", "alpha") is True


def test_authenticate_wrong_password() -> None:
    assert make_service().authenticate("2302", "wrong") is False


def test_authenticate_unknown_id() -> None:
    assert make_service().authenticate("9999", "alpha") is False


def test_authenticate_unknown_id_does_not_raise_on_non_ascii() -> None:
    # The dummy-compare path must also encode to bytes; a non-ASCII guess here
    # would raise TypeError if compare_digest saw a str.
    assert make_service().authenticate("9999", "गुप्त") is False


def test_authenticate_non_ascii_password() -> None:
    # compare_digest raises TypeError on non-ASCII str — proves the bytes encoding.
    service = AuthService({"2302": "pä55wörd-गुप्त"})
    assert service.authenticate("2302", "pä55wörd-गुप्त") is True
    assert service.authenticate("2302", "pä55wörd") is False


def test_from_csv(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", "student_id,password\n2302,alpha\n")
    service = AuthService.from_csv(path)
    assert service.authenticate("2302", "alpha") is True
