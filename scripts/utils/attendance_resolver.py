from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

from scripts.utils.class_date import parse_class_date, parse_filename_date

logger = logging.getLogger(__name__)


def resolve_attendance_file(class_name: str, attendance_dir: Path) -> Path | None:
    class_date = parse_class_date(class_name)
    if class_date is None:
        logger.warning("No date parsed from class name %r; attendance not resolved", class_name)
        return None

    # the daily export pulls day D's meeting but names the file D-1 (exporter main.py)
    target = class_date - timedelta(days=1)
    matches = [
        path
        for path in sorted(attendance_dir.glob("*.csv"))
        if parse_filename_date(path.stem) == target
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous attendance for {class_name!r} in {attendance_dir}: "
            f"{sorted(p.name for p in matches)}"
        )
    if matches:
        return matches[0]

    logger.warning(
        "No attendance file for %r (expected filename date %s) in %s; proceeding attendance-less",
        class_name,
        target.strftime("%Y_%m_%d"),
        attendance_dir,
    )
    return None
