from __future__ import annotations

import argparse
import csv
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel

from scripts.models.identity import (
    AttendanceRecord,
    IdentityMap,
    IdentityMapEntry,
    PerStudentAudioFile,
    RosterEntry,
    ZoomFileManifest,
)

logger = logging.getLogger(__name__)

_ROLL_NO_PATTERN = re.compile(r"_\s*(\d{4})$")
_PAREN_PATTERN = re.compile(r"\s*\(.*?\)\s*$")


class MatchArgs(BaseModel):
    manifest_path: Path
    roster_path: Path | None = None
    attendance_path: Path | None = None
    teacher: list[str]
    short_duration_threshold: float = 5.0


def parse_args(argv: Sequence[str] | None = None) -> MatchArgs:
    parser = argparse.ArgumentParser(
        description="Match per-student M4A files to roster/attendance identities."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        dest="manifest_path",
        help="Path to manifest.json produced by ingest_zip.py.",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        dest="roster_path",
        default=None,
        help="Path to master roster CSV (Name, RollNo, Email).",
    )
    parser.add_argument(
        "--attendance",
        type=Path,
        dest="attendance_path",
        default=None,
        help="Path to Zoom attendance CSV.",
    )
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        dest="teacher",
        metavar="NAME",
        help="Teacher display name. Repeat for multiple teachers.",
    )
    parser.add_argument(
        "--short-duration",
        type=float,
        default=5.0,
        dest="short_duration_threshold",
        help="Attendance duration (minutes) below which to tag 'short_duration'. Default: 5.",
    )
    namespace = parser.parse_args(argv)
    return MatchArgs.model_validate(vars(namespace))


def validate_inputs(args: MatchArgs) -> None:
    if not args.manifest_path.exists():
        raise ValueError(f"Manifest file not found: {args.manifest_path}")
    if args.roster_path is not None and not args.roster_path.exists():
        raise ValueError(f"Roster file not found: {args.roster_path}")
    if args.attendance_path is not None and not args.attendance_path.exists():
        raise ValueError(f"Attendance file not found: {args.attendance_path}")
    if not args.teacher:
        raise ValueError("At least one --teacher name is required.")


def parse_attendance_name(raw_name: str) -> tuple[str, str | None]:
    """Parse attendance CSV name → (clean_name, roll_no_4digit).

    Strips parenthetical info, then extracts a trailing _DDDD roll number.
    Handles optional space between underscore and digits (e.g. 'Name_ 2302').
    """
    name = _PAREN_PATTERN.sub("", raw_name).strip()
    match = _ROLL_NO_PATTERN.search(name)
    if match:
        roll_no = match.group(1)
        clean_name = name[: match.start()].strip()
        return clean_name, roll_no
    return name, None


def _find_column(headers: list[str], *patterns: str) -> str | None:
    for pattern in patterns:
        for key in headers:
            if pattern.lower() in key.lower():
                return key
    return None


def load_roster(path: Path) -> list[RosterEntry]:
    entries: list[RosterEntry] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm = {k.strip(): v.strip() for k, v in row.items() if k}
            name = norm.get("Name", "").strip()
            roll_no = (
                norm.get("RollNo")
                or norm.get("Roll No")
                or norm.get("roll_no")
                or ""
            ).strip()
            email = (norm.get("Email") or norm.get("email") or "").strip()
            if name and roll_no:
                entries.append(RosterEntry(name=name, roll_no=roll_no, email=email))
    return entries


def _parse_duration_minutes(raw: str) -> float:
    """Parse a duration value to minutes. Accepts plain numbers or H:MM:SS."""
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    parts = raw.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 60 + m + s / 60
    if len(parts) == 2:
        m, s = int(parts[0]), float(parts[1])
        return m + s / 60
    return 0.0


def load_attendance(path: Path) -> list[AttendanceRecord]:
    records: list[AttendanceRecord] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        name_col = _find_column(headers, "name")
        dur_col = _find_column(headers, "duration", "dura")
        guest_col = _find_column(headers, "guest")
        for row in reader:
            raw_name = (row.get(name_col, "") if name_col else "").strip()
            if not raw_name:
                continue
            clean_name, roll_no = parse_attendance_name(raw_name)
            duration_minutes = 0.0
            if dur_col:
                duration_minutes = _parse_duration_minutes(row.get(dur_col, "0") or "0")
            guest: bool | None = None
            if guest_col:
                guest_raw = (row.get(guest_col, "") or "").strip().lower()
                guest = guest_raw == "yes"
            records.append(
                AttendanceRecord(
                    name=clean_name,
                    roll_no=roll_no,
                    duration_minutes=duration_minutes,
                    guest=guest,
                )
            )
    return records


def fuzzy_match_score(name_a: str, name_b: str) -> float:
    return SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()


def _best_teacher_score(display_name: str, teacher_names: list[str]) -> tuple[float, str]:
    best_score = 0.0
    best_name = ""
    for teacher in teacher_names:
        score = fuzzy_match_score(display_name, teacher)
        if score > best_score:
            best_score = score
            best_name = teacher
    return best_score, best_name


def _entry_tags(
    attendance_record: AttendanceRecord | None,
    short_duration_threshold: float,
    extra: list[str] | None = None,
) -> list[str]:
    tags: list[str] = list(extra or [])
    if attendance_record is not None and attendance_record.duration_minutes < short_duration_threshold:
        tags.append("short_duration")
    return tags


def _build_matched_entry(
    audio_file: PerStudentAudioFile,
    roster_entry: RosterEntry,
    attendance_record: AttendanceRecord | None,
    method: Literal["roll_no", "fuzzy_name"],
    confidence: float,
    short_duration_threshold: float,
) -> IdentityMapEntry:
    return IdentityMapEntry(
        audio_file=audio_file.filename,
        roll_no_4digit=audio_file.roll_no_4digit,
        matched_name=roster_entry.name,
        matched_roll_no=roster_entry.roll_no,
        matched_email=roster_entry.email or None,
        match_method=method,
        match_confidence=confidence,
        is_teacher=False,
        is_unmatched=False,
        attendance_duration_minutes=(
            attendance_record.duration_minutes if attendance_record else None
        ),
        tags=_entry_tags(attendance_record, short_duration_threshold),
    )


def _build_unmatched_entry(audio_file: PerStudentAudioFile) -> IdentityMapEntry:
    return IdentityMapEntry(
        audio_file=audio_file.filename,
        roll_no_4digit=audio_file.roll_no_4digit,
        matched_name=None,
        matched_roll_no=None,
        matched_email=None,
        match_method="none",
        match_confidence=0.0,
        is_teacher=False,
        is_unmatched=True,
        attendance_duration_minutes=None,
        tags=["unmatched"],
    )


def match_files(
    manifest: ZoomFileManifest,
    roster: list[RosterEntry],
    attendance: list[AttendanceRecord],
    teacher_names: list[str],
    short_duration_threshold: float = 5.0,
) -> IdentityMap:
    # Check for duplicate roll numbers in roster
    roll_nos = [r.roll_no for r in roster]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for rn in roll_nos:
        if rn in seen:
            duplicates.add(rn)
        seen.add(rn)
    if duplicates:
        raise ValueError(f"Duplicate roll numbers in roster: {sorted(duplicates)}")

    roster_by_roll: dict[str, RosterEntry] = {r.roll_no: r for r in roster}
    attendance_by_roll: dict[str, AttendanceRecord] = {
        a.roll_no: a for a in attendance if a.roll_no is not None
    }

    matched_roll_nos: set[str] = set()
    roll_to_audio: dict[str, str] = {}
    roll_to_display: dict[str, str] = {}
    matched_entries: list[IdentityMapEntry] = []
    unmatched_entries: list[IdentityMapEntry] = []
    teacher_audio_file: str | None = None

    for audio_file in manifest.per_student_m4as:
        roll_no_4digit = audio_file.roll_no_4digit
        display_name = audio_file.display_name or ""

        # Step 1: Roll number match (primary, deterministic)
        # Tries roster first; falls back to attendance-only when no roster is loaded.
        if roll_no_4digit is not None:
            roster_entry = roster_by_roll.get(roll_no_4digit)
            attendance_record = attendance_by_roll.get(roll_no_4digit)
            # Two DISTINCT per-student M4As resolving to the same 4-digit roll would
            # co-mingle both students under one id (build_student_context silently
            # drops the second). Refuse to ingest an ambiguous identity. This guard
            # fires in BOTH the roster and the no-roster/attendance-only paths.
            if roster_entry is not None or attendance_record is not None:
                if roll_no_4digit in roll_to_audio:
                    raise ValueError(
                        f"Two audio files resolve to the same roll {roll_no_4digit}: "
                        f"{roll_to_audio[roll_no_4digit]} and {audio_file.filename}"
                    )
                roll_to_audio[roll_no_4digit] = audio_file.filename
            if roster_entry is not None:
                matched_roll_nos.add(roll_no_4digit)
                matched_entries.append(
                    _build_matched_entry(
                        audio_file,
                        roster_entry,
                        attendance_record,
                        "roll_no",
                        1.0,
                        short_duration_threshold,
                    )
                )
                continue
            if attendance_record is not None:
                # No roster — synthesise a minimal entry from attendance data.
                synthetic = RosterEntry(
                    name=attendance_record.name,
                    roll_no=attendance_record.roll_no or roll_no_4digit,
                    email="",
                )
                matched_roll_nos.add(roll_no_4digit)
                matched_entries.append(
                    _build_matched_entry(
                        audio_file,
                        synthetic,
                        attendance_record,
                        "roll_no",
                        0.9,
                        short_duration_threshold,
                    )
                )
                continue
            if not roster and not attendance:
                # No roster AND no attendance: the M4A filename's 4-digit roll is the
                # only identity source, and per-student M4As are ground truth. Trust it
                # so present students still get a roll-keyed context (student_id = roll)
                # and spoken attribution, instead of being dropped as "unmatched".
                prior = roll_to_audio.get(roll_no_4digit)
                if prior is not None:
                    # Same roll already taken. Same display name = one student recorded
                    # across multiple files (rejoin / multi-session zip): keep the first,
                    # ignore the rest. Distinct names sharing a roll is a real identity
                    # collision and must fail loud (audit #4).
                    prior_name = roll_to_display.get(roll_no_4digit, "")
                    if prior_name.strip().casefold() != display_name.strip().casefold():
                        raise ValueError(
                            f"Two audio files resolve to the same roll {roll_no_4digit}: "
                            f"{prior} ({prior_name}) and {audio_file.filename} ({display_name})"
                        )
                    logger.warning(
                        "Additional audio file for %s (roll %s) ignored: %s",
                        display_name or roll_no_4digit,
                        roll_no_4digit,
                        audio_file.filename,
                    )
                    continue
                roll_to_audio[roll_no_4digit] = audio_file.filename
                roll_to_display[roll_no_4digit] = display_name
                synthetic = RosterEntry(
                    name=display_name or roll_no_4digit,
                    roll_no=roll_no_4digit,
                    email="",
                )
                matched_roll_nos.add(roll_no_4digit)
                matched_entries.append(
                    _build_matched_entry(
                        audio_file,
                        synthetic,
                        None,
                        "roll_no",
                        0.8,
                        short_duration_threshold,
                    )
                )
                continue

        # Step 2: Teacher fuzzy match (raised to 0.75 to avoid short-name collisions)
        teacher_score, _matched_teacher = _best_teacher_score(display_name, teacher_names)
        if teacher_score >= 0.75:
            if teacher_audio_file is None:
                teacher_audio_file = audio_file.filename
                logger.info("Teacher identified: %s (score=%.2f)", audio_file.filename, teacher_score)
            else:
                logger.warning(
                    "Additional teacher file found (ignored): %s", audio_file.filename
                )
            continue

        # Step 3: Unmatched
        logger.warning("No match for audio file: %s", audio_file.filename)
        unmatched_entries.append(_build_unmatched_entry(audio_file))

    # Roster students with no audio file
    roster_students_without_audio = [
        r.roll_no for r in roster if r.roll_no not in matched_roll_nos
    ]

    return IdentityMap(
        teacher_name=", ".join(teacher_names),
        teacher_audio_file=teacher_audio_file,
        entries=matched_entries,
        unmatched_entries=unmatched_entries,
        roster_students_without_audio=roster_students_without_audio,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as exc:
        logger.error("Input validation failed: %s", exc)
        raise SystemExit(2) from exc

    manifest = ZoomFileManifest.model_validate_json(
        args.manifest_path.read_text(encoding="utf-8")
    )
    roster = load_roster(args.roster_path) if args.roster_path else []
    attendance = load_attendance(args.attendance_path) if args.attendance_path else []

    identity_map = match_files(
        manifest=manifest,
        roster=roster,
        attendance=attendance,
        teacher_names=args.teacher,
        short_duration_threshold=args.short_duration_threshold,
    )

    output_path = args.manifest_path.parent / "identity_map.json"
    output_path.write_text(identity_map.model_dump_json(indent=2), encoding="utf-8")

    print(f"Identity map written to {output_path}")
    print(f"  Matched students:   {len(identity_map.entries)}")
    print(f"  Unmatched files:    {len(identity_map.unmatched_entries)}")
    print(f"  Teacher file:       {identity_map.teacher_audio_file}")
    print(f"  No-audio students:  {len(identity_map.roster_students_without_audio)}")


if __name__ == "__main__":
    main()
