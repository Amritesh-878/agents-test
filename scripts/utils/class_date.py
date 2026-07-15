from __future__ import annotations

import re

MONTH_ABBREVS: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

CLASS_DATE_RE = re.compile(
    r"(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*",
    re.IGNORECASE,
)

_FILENAME_DATE_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})$")


def parse_class_date(text: str) -> tuple[int, int] | None:
    match = CLASS_DATE_RE.search(text)
    if match is None:
        return None
    return (MONTH_ABBREVS[match.group(2).lower()], int(match.group(1)))


def parse_filename_date(stem: str) -> tuple[int, int] | None:
    match = _FILENAME_DATE_RE.search(stem)
    if match is None:
        return None
    return (int(match.group(2)), int(match.group(3)))
