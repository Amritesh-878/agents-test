from __future__ import annotations

import re
from datetime import date

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

_ACADEMIC_YEAR_RE = re.compile(r"AY(\d{2}|\d{4})-\d{2}", re.IGNORECASE)

_FILENAME_DATE_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})$")


def parse_class_date(text: str) -> date | None:
    day_month = CLASS_DATE_RE.search(text)
    year_match = _ACADEMIC_YEAR_RE.search(text)
    if day_month is None or year_match is None:
        return None
    month = MONTH_ABBREVS[day_month.group(2).lower()]
    day = int(day_month.group(1))
    first_year = int(year_match.group(1))
    if first_year < 100:
        first_year += 2000
    year = first_year if month >= 6 else first_year + 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_filename_date(stem: str) -> date | None:
    match = _FILENAME_DATE_RE.search(stem)
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None
