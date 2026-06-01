from __future__ import annotations

import csv
import hmac
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Compared against on an unknown student_id so an attacker can't tell which ids
# exist from response timing. Its actual value is irrelevant.
_DUMMY_PASSWORD = "x" * 32


class AuthError(RuntimeError):
    pass


def _normalize_header(name: str) -> str:
    return name.strip().lower()


def load_credentials(path: Path) -> dict[str, str]:
    """Load a ``student_id -> password`` map from a credentials CSV.

    Mirrors :func:`scripts.match_identity.load_roster`: stdlib ``csv`` with
    tolerant (case-insensitive, trimmed) headers. The ``student_id`` is the trust
    boundary, so this fails loudly on a missing file, missing columns, duplicate
    ``student_id``, empty id, or empty password; fully-blank lines are skipped.
    """
    if not path.exists():
        raise AuthError(f"Credentials file not found: {path}")

    credentials: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = {_normalize_header(h) for h in (reader.fieldnames or [])}
        if "student_id" not in headers or "password" not in headers:
            raise AuthError("Credentials CSV must have 'student_id' and 'password' columns.")
        for row in reader:
            norm = {_normalize_header(k): (v or "").strip() for k, v in row.items() if k}
            student_id = norm.get("student_id", "")
            password = norm.get("password", "")
            if not student_id and not password:
                continue  # fully-blank line
            if not student_id:
                raise AuthError("Credentials row is missing a student_id.")
            if not password:
                raise AuthError(f"Credentials row for '{student_id}' has an empty password.")
            if student_id in credentials:
                raise AuthError(f"Duplicate student_id in credentials: {student_id}")
            credentials[student_id] = password
    return credentials


class AuthService:
    """Verify student login against an in-memory credentials map.

    Credential handling is isolated here so a later swap to hashed passwords or a
    Postgres-backed store touches only this file.
    """

    def __init__(self, credentials: dict[str, str]) -> None:
        self._credentials = credentials

    @classmethod
    def from_csv(cls, path: Path) -> AuthService:
        return cls(load_credentials(path))

    def authenticate(self, student_id: str, password: str) -> bool:
        stored = self._credentials.get(student_id)
        # Always run one constant-time compare so timing doesn't reveal which ids
        # exist. hmac.compare_digest raises TypeError on non-ASCII str, so both
        # sides are encoded to bytes first. The password is never logged.
        expected = stored if stored is not None else _DUMMY_PASSWORD
        match = hmac.compare_digest(expected.encode("utf-8"), password.encode("utf-8"))
        return match and stored is not None
