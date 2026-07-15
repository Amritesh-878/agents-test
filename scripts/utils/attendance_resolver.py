from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from scripts.utils.class_date import parse_class_date, parse_filename_date

logger = logging.getLogger(__name__)

_REFERENCE_YEAR = 2001


def _day_before(month: int, day: int) -> tuple[int, int] | None:
    try:
        previous = date(_REFERENCE_YEAR, month, day) - timedelta(days=1)
    except ValueError:
        return None
    return (previous.month, previous.day)


def _matches_for_day(
    candidates: list[tuple[Path, tuple[int, int]]], target: tuple[int, int]
) -> list[Path]:
    return [path for path, day_month in candidates if day_month == target]


def _select_unique(matches: list[Path], class_name: str, attendance_dir: Path) -> Path | None:
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous attendance for {class_name!r} in {attendance_dir}: "
            f"{sorted(p.name for p in matches)}"
        )
    return matches[0] if matches else None


def resolve_attendance_file(class_name: str, attendance_dir: Path) -> Path | None:
    target = parse_class_date(class_name)
    if target is None:
        logger.warning("No date parsed from class name %r; attendance not resolved", class_name)
        return None

    candidates: list[tuple[Path, tuple[int, int]]] = []
    for path in sorted(attendance_dir.glob("*.csv")):
        parsed = parse_filename_date(path.stem)
        if parsed is not None:
            candidates.append((path, parsed))

    same_day = _select_unique(_matches_for_day(candidates, target), class_name, attendance_dir)
    if same_day is not None:
        return same_day

    fallback = _day_before(*target)
    if fallback is not None:
        prev_day = _select_unique(
            _matches_for_day(candidates, fallback), class_name, attendance_dir
        )
        if prev_day is not None:
            logger.warning(
                "No same-day attendance for %r; using day-before file %s",
                class_name,
                prev_day.name,
            )
            return prev_day

    logger.warning(
        "No attendance file for %r in %s; proceeding attendance-less",
        class_name,
        attendance_dir,
    )
    return None
