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
        headers = list(reader.fieldnames or [])
        name_col = _find_column(headers, "student name", "name")
        roll_col = _find_column(headers, "student id", "rollno", "roll no", "roll")
        email_col = _find_column(headers, "email")
        if name_col is None or roll_col is None:
            return entries
        for row in reader:
            name = (row.get(name_col) or "").strip()
            roll_no = (row.get(roll_col) or "").strip()
            email = (row.get(email_col) or "").strip() if email_col else ""
            if name and roll_no:
                entries.append(RosterEntry(name=name, roll_no=roll_no, email=email))
    return entries


def _parse_duration_minutes(raw: str) -> float:
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


_NAME_MATCH_THRESHOLD = 0.6
_NAME_TOKEN_EQ_RATIO = 0.88
_FUZZY_NAME_CONFIDENCE = 0.7
_SECTION_PREFIX_RE = re.compile(r"^[A-Za-z]_")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _normalize_display_tokens(display_name: str) -> list[str]:
    s = _SECTION_PREFIX_RE.sub("", display_name or "")
    s = s.replace("_", " ")
    s = _CAMEL_BOUNDARY_RE.sub(" ", s)
    s = re.sub(r"[^A-Za-z\s]", " ", s)
    return [t for t in s.lower().split() if t]


def _normalize_plain_tokens(name: str) -> list[str]:
    return [t for t in re.sub(r"[^A-Za-z\s]", " ", name or "").lower().split() if t]


def _token_eq(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a, b).ratio() >= _NAME_TOKEN_EQ_RATIO


def _name_score(file_tokens: list[str], roster_name: str) -> float:
    if not file_tokens:
        return 0.0
    roster_tokens = _normalize_plain_tokens(roster_name)
    if not roster_tokens:
        return 0.0
    matched = sum(1 for ft in file_tokens if any(_token_eq(ft, rt) for rt in roster_tokens))
    return matched / len(file_tokens)


def resolve_attendance_rolls(
    records: list[AttendanceRecord], roster: list[RosterEntry]
) -> list[AttendanceRecord]:
    resolved: list[AttendanceRecord] = []
    for record in records:
        if record.roll_no is not None:
            resolved.append(record)
            continue
        tokens = _normalize_plain_tokens(record.name)
        if not tokens:
            resolved.append(record)
            continue
        candidates = [e for e in roster if _name_score(tokens, e.name) >= _NAME_MATCH_THRESHOLD]
        if len(candidates) == 1:
            resolved.append(record.model_copy(update={"roll_no": candidates[0].roll_no}))
        elif len(candidates) > 1:
            logger.warning(
                "Attendance row %r name-matches multiple roster entries (%s); left unresolved",
                record.name,
                ", ".join(f"{c.name}/{c.roll_no}" for c in candidates),
            )
            resolved.append(record)
        else:
            resolved.append(record)
    return resolved


def _roster_name_match(
    display_name: str, roster: list[RosterEntry]
) -> tuple[Literal["none", "unique", "ambiguous"], RosterEntry | None]:
    file_tokens = _normalize_display_tokens(display_name)
    if not file_tokens:
        return ("none", None)
    scored = [(e, _name_score(file_tokens, e.name)) for e in roster]
    above = [(e, s) for e, s in scored if s >= _NAME_MATCH_THRESHOLD]
    if not above:
        return ("none", None)
    best = max(s for _, s in above)
    top = [e for e, s in above if abs(s - best) < 1e-9]
    if len(top) == 1:
        return ("unique", top[0])
    return ("ambiguous", None)


def resolve_chat_sender(
    sender: str,
    roster: list[RosterEntry],
    roster_by_roll: dict[str, RosterEntry],
) -> str | None:
    clean_name, roll = parse_attendance_name(sender)
    kind, entry = _resolve_roster_identity(roll, clean_name, roster, roster_by_roll)
    if kind in ("roll", "name") and entry is not None:
        return entry.roll_no
    return None


def _resolve_roster_identity(
    roll_no_4digit: str | None,
    display_name: str,
    roster: list[RosterEntry],
    roster_by_roll: dict[str, RosterEntry],
) -> tuple[Literal["roll", "name", "ambiguous", "none"], RosterEntry | None]:
    status, entry = _roster_name_match(display_name, roster)
    roll_entry = roster_by_roll.get(roll_no_4digit) if roll_no_4digit else None
    file_tokens = _normalize_display_tokens(display_name)
    if roll_entry is not None:
        if _name_score(file_tokens, roll_entry.name) >= _NAME_MATCH_THRESHOLD:
            return ("roll", roll_entry)
        if status == "unique" and entry is not None and entry.roll_no != roll_no_4digit:
            return ("name", entry)
        if status == "ambiguous":
            return ("ambiguous", None)
        return ("roll", roll_entry)
    if status == "unique" and entry is not None:
        return ("name", entry)
    if status == "ambiguous":
        return ("ambiguous", None)
    return ("none", None)


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


def _build_unmatched_entry(
    audio_file: PerStudentAudioFile, extra_tags: list[str] | None = None
) -> IdentityMapEntry:
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
        tags=["unmatched", *(extra_tags or [])],
    )


def match_files(
    manifest: ZoomFileManifest,
    roster: list[RosterEntry],
    attendance: list[AttendanceRecord],
    teacher_names: list[str],
    short_duration_threshold: float = 5.0,
) -> IdentityMap:
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

        if roster:
            kind, entry = _resolve_roster_identity(
                roll_no_4digit, display_name, roster, roster_by_roll
            )
            if kind == "ambiguous":
                logger.warning(
                    "Ambiguous roster name match for %s; left unmatched for review",
                    audio_file.filename,
                )
                unmatched_entries.append(_build_unmatched_entry(audio_file, ["name_ambiguous"]))
                continue
            if kind in ("roll", "name") and entry is not None:
                adopted_roll = entry.roll_no
                if adopted_roll in roll_to_audio:
                    same_student = (
                        kind == "name"
                        or roll_to_display.get(adopted_roll, "").casefold()
                        == display_name.casefold()
                    )
                    if same_student:
                        logger.warning(
                            "Additional audio for roster student %s (roll %s) ignored: %s",
                            entry.name, adopted_roll, audio_file.filename,
                        )
                        continue
                    raise ValueError(
                        f"Two audio files resolve to the same roll {adopted_roll}: "
                        f"{roll_to_audio[adopted_roll]} and {audio_file.filename}"
                    )
                roll_to_audio[adopted_roll] = audio_file.filename
                roll_to_display[adopted_roll] = display_name
                matched_roll_nos.add(adopted_roll)
                method: Literal["roll_no", "fuzzy_name"] = (
                    "roll_no" if kind == "roll" else "fuzzy_name"
                )
                matched_entries.append(
                    _build_matched_entry(
                        audio_file,
                        entry,
                        attendance_by_roll.get(adopted_roll),
                        method,
                        1.0 if kind == "roll" else _FUZZY_NAME_CONFIDENCE,
                        short_duration_threshold,
                    )
                )
                continue

        elif roll_no_4digit is not None:
            roster_entry = roster_by_roll.get(roll_no_4digit)
            attendance_record = attendance_by_roll.get(roll_no_4digit)
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
                prior = roll_to_audio.get(roll_no_4digit)
                if prior is not None:
                    prior_name = roll_to_display.get(roll_no_4digit, "")
                    if prior_name.strip().casefold() != display_name.strip().casefold():
                        logger.warning(
                            "Roll collision %s: keeping %s (%s); flagging %s (%s) unmatched",
                            roll_no_4digit, prior, prior_name, audio_file.filename, display_name,
                        )
                        unmatched_entries.append(
                            _build_unmatched_entry(audio_file, ["roll_collision"])
                        )
                        continue
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

        logger.warning("No match for audio file: %s", audio_file.filename)
        unmatched_entries.append(_build_unmatched_entry(audio_file))

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
