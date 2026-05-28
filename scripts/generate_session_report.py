from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from scripts.embed_and_store import is_quality_text

logger = logging.getLogger(__name__)

_NOISE_WORDS = {
    "yeah", "okay", "ok", "yes", "no", "like", "right", "so", "um",
    "uh", "hmm", "ah", "oh", "hm", "well", "just", "get", "got",
    "yep", "alright", "actually", "now", "also",
}

_ROLL_RE = re.compile(r"_\s*(\d{4})$")
_PAREN_RE = re.compile(r"\s*\(.*?\)\s*$")


class ReportArgs(BaseModel):
    class_dir: Path
    attendance: Path | None = None
    output: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> ReportArgs:
    parser = argparse.ArgumentParser(
        description="Generate a single-page session engagement report."
    )
    parser.add_argument("--class-dir", required=True, type=Path, dest="class_dir")
    parser.add_argument(
        "--attendance", type=Path, default=None,
        help="Zoom attendance CSV — shows all attendees, not just those with M4A recordings.",
    )
    parser.add_argument("--output", type=Path, default=None)
    namespace = parser.parse_args(argv)
    return ReportArgs.model_validate(vars(namespace))


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse repeated spaces, fix spaced-out characters, strip trailing noise."""
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1 \2", text)
    # Strip trailing Hindi particles and English fillers that leak into sentences
    # Apply repeatedly to handle "या, या" double-patterns
    for _ in range(3):
        prev = text
        text = re.sub(r"[,\s]+(या|हैं|है|ह)[,\s.।]*$", "", text).strip()
        text = re.sub(r"[,\s]+(the|a|an|and|or|so|but|like)[,\s.]*$", "", text, flags=re.I).strip()
        if text == prev:
            break
    return text


def _english_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    eng = sum(1 for w in words if re.match(r"^[a-zA-Z']+$", w))
    return eng / len(words)


def _is_meaningful(text: str, min_chars: int = 15, min_english: float = 0.45) -> bool:
    cleaned = _clean(text)
    if not is_quality_text(cleaned, min_chars=min_chars):
        return False
    if _english_ratio(cleaned) < min_english:
        return False
    words = [w.strip(".,!?") for w in cleaned.lower().split()]
    real = [w for w in words if w not in _NOISE_WORDS and w.isalpha() and len(w) > 2]
    return len(real) >= 2


def _engagement_level(quality_count: int) -> str:
    if quality_count >= 10:
        return "Active"
    if quality_count >= 3:
        return "Moderate"
    if quality_count >= 1:
        return "Passive"
    return "Silent"


# ---------------------------------------------------------------------------
# Attendance CSV loading
# ---------------------------------------------------------------------------

def _load_attendance_csv(path: Path) -> list[dict]:
    """Return list of {name, roll_no, duration_minutes, has_roll} dicts."""
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])

        def find_col(*patterns: str) -> str | None:
            for p in patterns:
                for h in headers:
                    if p.lower() in h.lower():
                        return h
            return None

        name_col = find_col("name")
        dur_col = find_col("duration", "dura")
        for row in reader:
            raw = (row.get(name_col, "") if name_col else "").strip()
            if not raw:
                continue
            # Strip parenthetical
            clean_name = _PAREN_RE.sub("", raw).strip()
            # Extract roll number
            m = _ROLL_RE.search(clean_name)
            roll_no = m.group(1) if m else None
            display = clean_name[:m.start()].strip() if m else clean_name
            # Skip bots / notetakers
            if any(x in raw.lower() for x in ["otter", "read.ai", "notetaker", "notes"]):
                continue
            duration = 0.0
            if dur_col:
                try:
                    duration = float(row.get(dur_col, "0") or "0")
                except ValueError:
                    pass
            rows.append({
                "display": display,
                "raw_name": raw,
                "roll_no": roll_no,
                "duration": duration,
                "has_roll": roll_no is not None,
            })
    return sorted(rows, key=lambda r: -r["duration"])


# ---------------------------------------------------------------------------
# Topic extraction from quality segments
# ---------------------------------------------------------------------------

_DEFN_RE = re.compile(
    r"(?<!\bthat )\b(is a |is the |are |means |refers to|defined as|called an? |represents|is used to)\b",
    re.I,
)


def _extract_topic_sentences(segments: list[dict], n: int = 5) -> list[str]:
    """Extract definitional sentences from class segments.

    Student segments are prioritised because student answers in Q&A classes
    contain the clearest concept definitions (e.g. 'Supply schedule is a table
    representation of price and quantity').  Teacher segments fill any gaps.
    """
    seen: set[str] = set()
    student_defns: list[tuple[float, str]] = []
    teacher_defns: list[tuple[float, str]] = []

    for seg in segments:
        source = seg.get("source", "")
        txt = _clean(seg.get("text", ""))
        if not _is_meaningful(txt, min_chars=25, min_english=0.65):
            continue
        if not _DEFN_RE.search(txt):
            continue
        key = txt[:60].lower()
        if key in seen:
            continue
        seen.add(key)

        words = txt.split()
        real = sum(1 for w in words if w.lower() not in _NOISE_WORDS and w.isalpha() and len(w) > 3)
        score = (real / max(len(words), 1)) * _english_ratio(txt)

        bucket = student_defns if source == "per_student" else teacher_defns
        bucket.append((score, txt))

    student_defns.sort(reverse=True)
    teacher_defns.sort(reverse=True)

    result = [t for _, t in student_defns[:n]]
    result += [t for _, t in teacher_defns[: max(0, n - len(result))]]
    return result[:n]


def _session_type(segments: list[dict]) -> str:
    for seg in segments[:15]:
        txt = seg.get("text", "").lower()
        if any(w in txt for w in ["revise", "revision", "recap", "last week"]):
            return "Revision Session"
        if any(w in txt for w in ["test", "exam", "quiz"]):
            return "Assessment"
    return "Class Session"


def _build_timeline(segments: list[dict], duration_sec: float, n_slots: int = 6) -> list[tuple[float, str]]:
    """One representative clean sentence per time slot across the session."""
    slot_size = duration_sec / n_slots
    result: list[tuple[float, str]] = []
    seen: set[str] = set()
    for i in range(n_slots):
        slot_start = i * slot_size
        slot_end = (i + 1) * slot_size
        slot_segs = [
            s for s in segments
            if slot_start <= s.get("start", 0) < slot_end
            and _is_meaningful(s.get("text", ""), min_chars=35, min_english=0.6)
        ]
        if not slot_segs:
            continue
        # Pick the one with highest English density
        best = max(slot_segs, key=lambda s: _english_ratio(s.get("text", "")))
        txt = _clean(best["text"])[:200]
        key = txt[:50].lower()
        if key not in seen:
            seen.add(key)
            result.append((best["start"] / 60, txt))
    return result


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(class_dir: Path, attendance_path: Path | None = None) -> str:
    import json

    ctx = json.loads((class_dir / "student_contexts.json").read_text(encoding="utf-8"))
    merged = json.loads((class_dir / "transcript_merged.json").read_text(encoding="utf-8"))

    class_name = merged.get("class_name", class_dir.name)
    duration_sec = merged.get("duration_seconds", 0.0)
    duration_min = round(duration_sec / 60, 1)
    teacher = merged.get("teacher_name", "Unknown")
    all_segments = merged.get("segments", [])
    present = ctx.get("present_students", {})
    absent = ctx.get("absent_students", {})

    session_type = _session_type(all_segments)

    # Load full attendance list from CSV if provided
    attendance_rows: list[dict] = []
    if attendance_path and attendance_path.exists():
        attendance_rows = _load_attendance_csv(attendance_path)

    # Students with M4A recordings
    recorded_rolls: set[str] = {
        s.get("roll_no", "") for s in present.values() if s.get("roll_no")
    }

    # Build student participation table from M4A data
    m4a_students: list[dict] = []
    student_first_names: set[str] = set()
    for key, s in present.items():
        name = s.get("name", key)
        for part in re.split(r"[_\s]", name.lower()):
            if len(part) > 3:
                student_first_names.add(part)
        spoken_segs = s.get("spoken_segments", [])
        quality_segs = [seg for seg in spoken_segs if _is_meaningful(seg.get("text", ""))]
        quotes: list[str] = []
        for seg in quality_segs:
            cleaned = _clean(seg.get("text", ""))
            if len(cleaned) > 20 and _english_ratio(cleaned) > 0.5:
                quotes.append(cleaned[:180])
            if len(quotes) >= 3:
                break
        att_min = s.get("attendance_duration_minutes")
        att_display = "Full session" if att_min and att_min > duration_min else (
            f"{round(att_min)} min" if att_min else "—"
        )
        m4a_students.append({
            "name": name,
            "roll": s.get("roll_no", "—"),
            "att": att_display,
            "level": _engagement_level(len(quality_segs)),
            "quality": len(quality_segs),
            "quotes": quotes,
        })
    m4a_students.sort(key=lambda r: -r["quality"])

    # Topics — pull definitional sentences from ALL segments (student answers often clearest)
    topic_sentences = _extract_topic_sentences(all_segments, n=5)

    # Timeline
    timeline = _build_timeline(all_segments, duration_sec)

    # Build report
    lines: list[str] = []

    # Header
    lines += [
        "# Session Report",
        f"## {class_name.replace('_', ' ')}",
        "",
        "| | |",
        "|--|--|",
        f"| **Session type** | {session_type} |",
        f"| **Duration** | {duration_min} minutes |",
        f"| **Teacher** | {teacher} |",
        f"| **Students in this class** | {len(m4a_students)} |",
        "",
        "---",
        "",
    ]

    # Students — definitive class list from M4A recordings
    if m4a_students:
        lines += [
            "## Students",
            "",
            "_Confirmed from per-student audio recordings in this Zoom class export._",
            "",
            "| Student | Roll | Attendance | Engagement | Verbal Contributions |",
            "|---------|------|-----------|------------|---------------------|",
        ]
        for r in m4a_students:
            lines.append(
                f"| {r['name']} | {r['roll']} | {r['att']} | **{r['level']}** | {r['quality']} verified segments |"
            )
        lines += [""]

        # Key quotes
        active = [r for r in m4a_students if r["quotes"]]
        if active:
            lines += ["### What Students Said", ""]
            for r in active:
                lines.append(f"**{r['name']}** ({r['level']})")
                for q in r["quotes"]:
                    lines.append(f"> {q}")
                lines.append("")

        lines += ["---", ""]
    elif absent:
        lines += [
            "## Students",
            "",
            "_No students had audio recordings in this class._",
            "",
            "---",
            "",
        ]

    # Absent students (from roster, if any)
    if absent:
        lines += ["## Absent Students (Enrolled, No Recording)", ""]
        for key, s in absent.items():
            lines.append(f"- **{s.get('name', key)}** (roll {s.get('roll_no', '—')})")
        lines += ["", "---", ""]

    # What was covered
    if topic_sentences:
        lines += [
            "## What Was Covered",
            "",
            "_Key concepts from the session transcript:_",
            "",
        ]
        for sent in topic_sentences:
            lines.append(f"- {sent}")
        lines += ["", "---", ""]

    # Session timeline
    if timeline:
        lines += [
            "## Session Flow",
            "",
            "_Teacher content at regular intervals throughout the class:_",
            "",
        ]
        for t_min, txt in timeline:
            lines.append(f"**{t_min:.0f} min** — {txt}")
            lines.append("")
        lines += ["---", ""]

    # Zoom meeting attendance — clearly scoped as full-meeting data
    if attendance_rows:
        avg_dur = sum(r["duration"] for r in attendance_rows) / len(attendance_rows)
        is_multiclass = avg_dur > duration_min * 1.5

        scope_note = (
            "_⚠️ This CSV covers the full Zoom meeting for the day, which includes multiple classes. "
            "Durations shown are total time in the meeting, not time in this specific class._"
            if is_multiclass
            else f"_{len(attendance_rows)} participants from the Zoom attendance export._"
        )

        lines += [
            "## Zoom Meeting Attendance",
            "",
            scope_note,
            "",
            "| Student | Roll | Duration in meeting | Recording in this class |",
            "|---------|------|-------------------|------------------------|",
        ]
        for row in attendance_rows:
            dur = f"{round(row['duration'])} min" if row["duration"] else "—"
            roll = row["roll_no"] or "—"
            has_audio = "✓ Audio recorded" if row["roll_no"] in recorded_rolls else ""
            lines.append(f"| {row['display']} | {roll} | {dur} | {has_audio} |")
        lines += [""]

    # Footer
    lines += [
        "---",
        "",
        "*Generated from Zoom cloud recording transcripts. "
        "Students are confirmed from per-student M4A files in the class zip — "
        "this is the ground truth for who participated in this specific class. "
        "Verbal engagement is measured from isolated microphone audio only.*",
    ]

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    if not args.class_dir.exists():
        logger.error("Class directory not found: %s", args.class_dir)
        raise SystemExit(1)

    report = generate_report(args.class_dir, args.attendance)
    out = args.output or args.class_dir / "session_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
