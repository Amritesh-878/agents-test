from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import hmac
import logging
import secrets
from pathlib import Path
from typing import Callable, Literal, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Role = Literal["admin", "teacher", "student"]

PBKDF2_ITERATIONS = 600_000

_ROLES: tuple[Role, ...] = ("admin", "teacher", "student")
_HASH_SCHEME = "pbkdf2_sha256"
_SALT_BYTES = 16
_SECTION_SEPARATOR = ";"
_REQUIRED_COLUMNS = ("username", "role", "scope", "password_hash")
_LEGACY_COLUMNS = ("student_id", "password")

_DUMMY_PASSWORD_HASH = (
    "pbkdf2_sha256$600000$7452a8249a63266009da69063cb430b1$"
    "d09a81a918d771dd2a8adb01c6e40a93c835d7dfb42d60fcecaeb9c1392f8b9c"
)


class AuthError(RuntimeError):
    pass


class Principal(BaseModel):
    username: str
    role: Role
    student_id: str | None = None
    sections: list[str] = Field(default_factory=list)


class CredentialRecord(BaseModel):
    username: str
    role: Role
    scope: str
    password_hash: str


def hash_password(
    password: str, *, iterations: int = PBKDF2_ITERATIONS, salt: bytes | None = None
) -> str:
    salt_bytes = secrets.token_bytes(_SALT_BYTES) if salt is None else salt
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations)
    return f"{_HASH_SCHEME}${iterations}${salt_bytes.hex()}${digest.hex()}"


def _parse_password_hash(encoded: str) -> tuple[int, bytes, bytes] | None:
    parts = encoded.split("$")
    if len(parts) != 4 or parts[0] != _HASH_SCHEME:
        return None
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        digest = bytes.fromhex(parts[3])
    except ValueError:
        return None
    if iterations < 1 or not salt or not digest:
        return None
    return iterations, salt, digest


def verify_password(password: str, encoded: str) -> bool:
    parsed = _parse_password_hash(encoded)
    if parsed is None:
        return False
    iterations, salt, expected = parsed
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


def parse_role(value: str) -> Role | None:
    for role in _ROLES:
        if value == role:
            return role
    return None


def parse_sections(scope: str) -> list[str]:
    return [s.strip() for s in scope.split(_SECTION_SEPARATOR) if s.strip()]


def build_credential_record(
    username: str, role_value: str, scope: str, password_hash: str
) -> CredentialRecord:
    if not username:
        raise AuthError("Credentials row is missing a username.")
    role = parse_role(role_value)
    if role is None:
        raise AuthError(
            f"Credentials row for '{username}' has an invalid role '{role_value}'; "
            f"expected one of: {', '.join(_ROLES)}."
        )
    if not password_hash:
        raise AuthError(f"Credentials row for '{username}' has an empty password_hash.")
    if _parse_password_hash(password_hash) is None:
        raise AuthError(
            f"Credentials row for '{username}' has a malformed password_hash; expected "
            f"'{_HASH_SCHEME}$<iterations>$<salt_hex>$<hash_hex>'."
        )
    if role == "student" and not scope:
        raise AuthError(f"Credentials row for student '{username}' needs a scope (their student_id).")
    if role == "teacher" and not parse_sections(scope):
        raise AuthError(
            f"Credentials row for teacher '{username}' needs a scope "
            f"('{_SECTION_SEPARATOR}'-separated section labels)."
        )
    if role == "admin" and scope:
        raise AuthError(f"Credentials row for admin '{username}' must have an empty scope.")
    return CredentialRecord(username=username, role=role, scope=scope, password_hash=password_hash)


def principal_from_record(record: CredentialRecord) -> Principal:
    if record.role == "student":
        return Principal(username=record.username, role="student", student_id=record.scope)
    if record.role == "teacher":
        return Principal(
            username=record.username, role="teacher", sections=parse_sections(record.scope)
        )
    return Principal(username=record.username, role="admin")


def _normalize_header(name: str) -> str:
    return name.strip().lower()


def load_credentials(path: Path) -> dict[str, CredentialRecord]:
    if not path.exists():
        raise AuthError(f"Credentials file not found: {path}")

    credentials: dict[str, CredentialRecord] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = {_normalize_header(h) for h in (reader.fieldnames or [])}
        if all(column in headers for column in _LEGACY_COLUMNS):
            raise AuthError(
                f"Credentials file {path} uses the old plaintext format "
                f"({','.join(_LEGACY_COLUMNS)}). Plaintext passwords are no longer accepted; "
                "re-create each user with 'python -m scripts.auth add-user --credentials "
                "<path> --username <u> --role <r> [--scope <s>]'."
            )
        missing = [column for column in _REQUIRED_COLUMNS if column not in headers]
        if missing:
            raise AuthError(
                f"Credentials CSV must have {', '.join(repr(c) for c in _REQUIRED_COLUMNS)} "
                f"columns; missing: {', '.join(missing)}."
            )
        for row in reader:
            norm = {_normalize_header(k): (v or "").strip() for k, v in row.items() if k}
            username = norm.get("username", "")
            role_value = norm.get("role", "")
            scope = norm.get("scope", "")
            password_hash = norm.get("password_hash", "")
            if username.startswith("#"):
                continue
            if not any((username, role_value, scope, password_hash)):
                continue
            record = build_credential_record(username, role_value, scope, password_hash)
            if record.username in credentials:
                raise AuthError(f"Duplicate username in credentials: {record.username}")
            credentials[record.username] = record
    return credentials


def write_credentials(path: Path, credentials: dict[str, CredentialRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(_REQUIRED_COLUMNS))
        for record in sorted(credentials.values(), key=lambda r: r.username):
            writer.writerow([record.username, record.role, record.scope, record.password_hash])


def add_user(path: Path, username: str, role_value: str, scope: str, password: str) -> CredentialRecord:
    record = build_credential_record(username, role_value, scope, hash_password(password))
    credentials = load_credentials(path) if path.exists() else {}
    credentials[record.username] = record
    write_credentials(path, credentials)
    return record


class AuthService:
    def __init__(self, credentials: dict[str, CredentialRecord]) -> None:
        self._credentials = credentials

    @classmethod
    def from_csv(cls, path: Path) -> AuthService:
        return cls(load_credentials(path))

    def authenticate(self, username: str, password: str) -> Principal | None:
        record = self._credentials.get(username)
        encoded = _DUMMY_PASSWORD_HASH if record is None else record.password_hash
        verified = verify_password(password, encoded)
        if record is None or not verified:
            return None
        return principal_from_record(record)


def allowed_student_ids(
    principal: Principal, pairs: Sequence[tuple[str, str, str]]
) -> set[str] | None:
    if principal.role == "admin":
        return None
    if principal.role == "student":
        return {principal.student_id} if principal.student_id else set()
    from scripts.demo_backend import students_by_section  # deferred: demo_backend -> chat -> auth cycle

    grouped = students_by_section(pairs)
    allowed: set[str] = set()
    for section in principal.sections:
        allowed.update(student_id for student_id, _ in grouped.get(section, []))
    return allowed


def can_access_student(
    principal: Principal, student_id: str, pairs: Sequence[tuple[str, str, str]]
) -> bool:
    allowed = allowed_student_ids(principal, pairs)
    return True if allowed is None else student_id in allowed


def read_new_password(prompt_fn: Callable[[str], str]) -> str:
    password = prompt_fn("Password: ")
    if not password:
        raise AuthError("Password must not be empty.")
    if password != prompt_fn("Confirm password: "):
        raise AuthError("Passwords do not match.")
    return password


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.auth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add-user")
    add.add_argument("--credentials", type=Path, required=True)
    add.add_argument("--username", required=True)
    add.add_argument("--role", required=True, choices=list(_ROLES))
    add.add_argument("--scope", default="")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--credentials", type=Path, required=True)
    verify.add_argument("--username", required=True)

    return parser


def describe_scope(principal: Principal) -> str:
    if principal.role == "teacher":
        return _SECTION_SEPARATOR.join(principal.sections)
    return principal.student_id or ""


def main(
    argv: Sequence[str] | None = None,
    *,
    prompt_fn: Callable[[str], str] = getpass.getpass,
    output_fn: Callable[[str], None] = print,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "add-user":
            password = read_new_password(prompt_fn)
            record = add_user(args.credentials, args.username, args.role, args.scope, password)
            output_fn(f"Saved {record.role} '{record.username}' to {args.credentials}.")
            return 0

        principal = AuthService.from_csv(args.credentials).authenticate(
            args.username, prompt_fn("Password: ")
        )
        if principal is None:
            output_fn("Authentication failed.")
            return 1
        output_fn(
            f"OK: {principal.username} role={principal.role} "
            f"scope={describe_scope(principal) or '(none)'}"
        )
        return 0
    except AuthError as exc:
        output_fn(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
