from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

Role = Literal["teacher", "student"]

ROLL_EMAIL_RE = re.compile(r"_(\d{4})@")

_SECTION_SEPARATOR = ";"
_TEACHER_SECTION_COLUMNS = ("email", "sections")


class AuthError(RuntimeError):
    pass


class Principal(BaseModel):
    username: str
    role: Role
    student_id: str | None = None
    sections: list[str] = Field(default_factory=list)


def parse_sections(scope: str) -> list[str]:
    return [s.strip() for s in scope.split(_SECTION_SEPARATOR) if s.strip()]


def normalize_email(email: str) -> str:
    return email.strip().lower()


def extract_roll(email: str) -> str | None:
    match = ROLL_EMAIL_RE.search(email)
    return match.group(1) if match else None


def _normalize_header(name: str) -> str:
    return name.strip().lower()


def load_teacher_sections(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise AuthError(f"Teacher sections file not found: {path}")

    mapping: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = {_normalize_header(h) for h in (reader.fieldnames or [])}
        missing = [c for c in _TEACHER_SECTION_COLUMNS if c not in headers]
        if missing:
            raise AuthError(
                f"Teacher sections CSV must have "
                f"{', '.join(repr(c) for c in _TEACHER_SECTION_COLUMNS)} columns; "
                f"missing: {', '.join(missing)}."
            )
        for row in reader:
            norm = {_normalize_header(k): (v or "").strip() for k, v in row.items() if k}
            email = normalize_email(norm.get("email", ""))
            sections_value = norm.get("sections", "")
            if email.startswith("#"):
                continue
            if not email and not sections_value:
                continue
            if not email:
                raise AuthError("Teacher sections row is missing an email.")
            sections = parse_sections(sections_value)
            if not sections:
                raise AuthError(
                    f"Teacher sections row for '{email}' has no sections "
                    f"('{_SECTION_SEPARATOR}'-separated section labels required)."
                )
            if email in mapping:
                raise AuthError(f"Duplicate email in teacher sections: {email}")
            mapping[email] = sections
    return mapping


def principal_from_identity(
    email: str,
    lms_role: str,
    *,
    teacher_sections: dict[str, list[str]],
) -> Principal | None:
    normalized = normalize_email(email)
    if not normalized:
        logger.warning("Identity has an empty email; access denied.")
        return None
    if lms_role == "teacher":
        sections = teacher_sections.get(normalized)
        if not sections:
            logger.warning(
                "Teacher %r has no section mapping; access denied. Add a row to the "
                "teacher-sections CSV.",
                normalized,
            )
            return None
        return Principal(username=normalized, role="teacher", sections=list(sections))
    if lms_role == "student":
        roll = extract_roll(normalized)
        if roll is None:
            logger.warning(
                "Student email %r has no 4-digit roll; access denied.", normalized
            )
            return None
        return Principal(username=normalized, role="student", student_id=roll)
    logger.warning(
        "Unsupported LMS role %r for %r; access denied.", lms_role, normalized
    )
    return None


def allowed_student_ids(
    principal: Principal, pairs: Sequence[tuple[str, str, str]]
) -> set[str]:
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
    return student_id in allowed_student_ids(principal, pairs)
