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
_DEFN_RE = re.compile(
    r"(?<!\bthat )\b(is a |is the |are |means |refers to|defined as|called an? |represents|is used to)\b",
    re.I,
)
_HOMEWORK_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|next class|next week|"
    r"homework|make sure|please make|assignment|submit|send me|by end of|when you come)\b",
    re.I,
)
_QUESTION_RE = re.compile(r"\?\s*$")


class ReportArgs(BaseModel):
    class_dir: Path
    attendance: Path | None = None
    output: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> ReportArgs:
    parser = argparse.ArgumentParser(
        description="Generate a session engagement report in the approved ISL format."
    )
    parser.add_argument("--class-dir", required=True, type=Path, dest="class_dir")
    parser.add_argument(
        "--attendance", type=Path, default=None,
        help="Zoom attendance CSV (full-day meeting export). Shown as appendix only.",
    )
    parser.add_argument("--output", type=Path, default=None)
    namespace = parser.parse_args(argv)
    return ReportArgs.model_validate(vars(namespace))


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1 \2", text)
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
# Attendance CSV
# ---------------------------------------------------------------------------

def _load_attendance_csv(path: Path) -> list[dict]:
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
            if any(x in raw.lower() for x in ["otter", "read.ai", "notetaker", "notes"]):
                continue
            clean_name = _PAREN_RE.sub("", raw).strip()
            m = _ROLL_RE.search(clean_name)
            roll_no = m.group(1) if m else None
            display = clean_name[:m.start()].strip() if m else clean_name
            duration = 0.0
            if dur_col:
                try:
                    duration = float(row.get(dur_col, "0") or "0")
                except ValueError:
                    pass
            rows.append({"display": display, "roll_no": roll_no, "duration": duration})
    return sorted(rows, key=lambda r: -r["duration"])


# ---------------------------------------------------------------------------
# Dialogue reconstruction
# ---------------------------------------------------------------------------

def _build_dialogue(
    segments: list[dict],
    teacher_name: str,
    student_names: dict[str, str],  # source label -> display name
) -> list[dict]:
    """Return time-sorted list of {time, speaker, text, source} dicts, quality-filtered."""
    result: list[dict] = []
    seen: set[str] = set()

    for seg in segments:
        txt = _clean(seg.get("text", ""))
        if not _is_meaningful(txt, min_chars=20, min_english=0.62):
            continue
        key = txt[:60].lower()
        if key in seen:
            continue
        seen.add(key)

        source = seg.get("source", "")
        speakers = seg.get("speakers", [])
        t = seg.get("start", 0.0)

        if source == "session_fallback":
            speaker = teacher_name
        elif source == "per_student" and speakers:
            speaker = student_names.get(speakers[0], speakers[0])
        else:
            continue

        result.append({"time": t, "speaker": speaker, "text": txt[:250], "source": source})

    return sorted(result, key=lambda d: d["time"])


def _split_into_parts(
    dialogue: list[dict],
    duration_sec: float,
    n_parts: int = 5,
) -> list[tuple[str, list[dict]]]:
    """Split dialogue into N time-based parts, return (label, entries) pairs."""
    if not dialogue or duration_sec <= 0:
        return []

    slot = duration_sec / n_parts
    parts: list[tuple[str, list[dict]]] = []
    labels = ["Part 1", "Part 2", "Part 3", "Part 4", "Part 5", "Part 6"]

    for i in range(n_parts):
        start = i * slot
        end = (i + 1) * slot
        entries = [d for d in dialogue if start <= d["time"] < end]
        # Deduplicate within a part
        seen: set[str] = set()
        unique: list[dict] = []
        for e in entries:
            k = e["text"][:50].lower()
            if k not in seen:
                seen.add(k)
                unique.append(e)
        if unique:
            start_min = round(start / 60)
            end_min = round(end / 60)
            label = f"{labels[i]} ({start_min}–{end_min} min)"
            parts.append((label, unique[:8]))  # cap at 8 exchanges per part

    return parts


def _detect_homework(segments: list[dict]) -> list[str]:
    """Find segments where homework or next-class instructions are given."""
    results: list[str] = []
    seen: set[str] = set()
    for seg in reversed(segments):  # check from end of class
        txt = _clean(seg.get("text", ""))
        key = txt[:80].lower()
        if key in seen:
            continue
        if len(txt) > 20 and _HOMEWORK_RE.search(txt) and _english_ratio(txt) > 0.55:
            seen.add(key)
            results.append(txt[:300])
        if len(results) >= 2:
            break
    return results


def _extract_definitions(segments: list[dict]) -> list[str]:
    """Pull definitional sentences from student and teacher segments."""
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
        if score < 0.15:
            continue
        if source == "per_student":
            student_defns.append((score, txt))
        else:
            teacher_defns.append((score, txt))

    student_defns.sort(reverse=True)
    teacher_defns.sort(reverse=True)
    result = [t for _, t in student_defns[:4]]
    result += [t for _, t in teacher_defns[:max(0, 4 - len(result))]]
    return result[:5]


def _session_type(segments: list[dict]) -> str:
    for seg in segments[:15]:
        txt = seg.get("text", "").lower()
        if any(w in txt for w in ["revise", "revision", "recap", "last week"]):
            return "Revision Session"
        if any(w in txt for w in ["test", "exam", "quiz"]):
            return "Assessment"
    return "Class Session"


def _overview_sentence(segments: list[dict]) -> str:
    """First meaningful teacher sentence — sets session context."""
    for seg in segments:
        if seg.get("source") != "session_fallback":
            continue
        txt = _clean(seg.get("text", ""))
        if _is_meaningful(txt, min_chars=40, min_english=0.6):
            return txt[:300]
    return ""


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def generate_report(class_dir: Path, attendance_path: Path | None = None) -> str:
    import json

    ctx = json.loads((class_dir / "student_contexts.json").read_text(encoding="utf-8"))
    merged = json.loads((class_dir / "transcript_merged.json").read_text(encoding="utf-8"))

    class_name = merged.get("class_name", class_dir.name)
    duration_sec = merged.get("duration_seconds", 0.0)
    duration_min = round(duration_sec / 60, 1)
    teacher = merged.get("teacher_name", "Unknown")
    all_segs = merged.get("segments", [])
    present = ctx.get("present_students", {})
    absent = ctx.get("absent_students", {})

    session_type = _session_type(all_segs)

    # Build student map: roll_no/name -> display name
    student_names: dict[str, str] = {}
    for s in present.values():
        student_names[s.get("name", "")] = s.get("name", "")

    # Student participation data
    m4a_students: list[dict] = []
    for key, s in present.items():
        name = s.get("name", key)
        spoken_segs = s.get("spoken_segments", [])
        quality_segs = [sg for sg in spoken_segs if _is_meaningful(sg.get("text", ""))]

        # Richer quotes — up to 8 clean quality contributions (English-dominant only)
        quotes: list[str] = []
        for sg in quality_segs:
            cleaned = _clean(sg.get("text", ""))
            if len(cleaned) > 20 and _english_ratio(cleaned) > 0.65:
                quotes.append(cleaned[:250])
            if len(quotes) >= 8:
                break

        # Engagement metrics
        defns_answered = sum(
            1 for sg in quality_segs
            if _DEFN_RE.search(_clean(sg.get("text", "")))
        )
        qs_asked = sum(
            1 for sg in quality_segs
            if _QUESTION_RE.search(_clean(sg.get("text", "")))
        )

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
            "defns": defns_answered,
            "qs_asked": qs_asked,
            "quotes": quotes,
        })
    m4a_students.sort(key=lambda r: -r["quality"])

    # Dialogue reconstruction
    dialogue = _build_dialogue(all_segs, teacher, student_names)
    parts = _split_into_parts(dialogue, duration_sec, n_parts=min(5, max(3, int(duration_min // 10))))

    # Key definitions and overview
    definitions = _extract_definitions(all_segs)
    overview = _overview_sentence(all_segs)
    homework = _detect_homework(all_segs)

    # Attendance CSV
    attendance_rows: list[dict] = []
    recorded_rolls: set[str] = {s.get("roll_no", "") for s in present.values() if s.get("roll_no")}
    if attendance_path and attendance_path.exists():
        attendance_rows = _load_attendance_csv(attendance_path)

    # ---------------------------------------------------------------------------
    # Build markdown
    # ---------------------------------------------------------------------------
    lines: list[str] = []

    # Header
    student_str = (
        m4a_students[0]["name"] if len(m4a_students) == 1
        else f"{len(m4a_students)} students"
    )
    lines += [
        "# Session Report",
        f"## {class_name.replace('_', ' ')}",
        "",
        "| | |",
        "|--|--|",
        f"| **Session type** | {session_type} |",
        f"| **Duration** | {duration_min} minutes |",
        f"| **Teacher** | {teacher} |",
        f"| **{'Student' if len(m4a_students) == 1 else 'Students'}** | {student_str} |",
        f"| **Format** | {'1-on-1 Q&A' if len(m4a_students) == 1 else 'Group class'} |",
        "",
        "---",
        "",
    ]

    # Session overview
    if overview:
        lines += [
            "## Session Overview",
            "",
        ]
        # Auto-generate a brief narrative
        overview_clean = overview[:200]
        if m4a_students:
            active = [s for s in m4a_students if s["level"] in ("Active", "Moderate")]
            engagement_note = (
                f"{m4a_students[0]['name']} was actively engaged throughout, answering questions and working through problems live."
                if len(active) == 1 and len(m4a_students) == 1
                else f"{len(active)} of {len(m4a_students)} students actively contributed."
                if active
                else "Students were primarily in listening mode."
            )
        else:
            engagement_note = ""

        lines += [
            f"Opening: *\"{overview_clean}\"*",
            "",
            engagement_note,
            "",
            "---",
            "",
        ]

    # Key concepts / definitions
    if definitions:
        lines += [
            "## Key Concepts Covered",
            "",
            "_Definitions and explanations from the session transcript:_",
            "",
        ]
        for d in definitions:
            lines.append(f"- {d}")
        lines += ["", "---", ""]

    # Session dialogue — the main body
    if parts:
        lines += ["## Session — Detailed Dialogue", ""]
        for label, exchanges in parts:
            lines += [f"### {label}", ""]
            for ex in exchanges:
                spk = f"**{ex['speaker']}**"
                lines.append(f"{spk}: {ex['text']}")
                lines.append("")
            lines += []
        lines += ["---", ""]

    # Homework
    if homework:
        lines += ["## Homework / Next Steps", ""]
        for h in homework:
            lines.append(f"> {h}")
        lines += ["", "---", ""]

    # Student engagement summary
    lines += [
        "## Student Engagement Summary",
        "",
    ]
    if m4a_students:
        lines += [
            "| Student | Roll | Attendance | Engagement | Verbal Contributions | Definitions | Questions Asked |",
            "|---------|------|-----------|------------|---------------------|-------------|-----------------|",
        ]
        for r in m4a_students:
            lines.append(
                f"| {r['name']} | {r['roll']} | {r['att']} | **{r['level']}** | "
                f"{r['quality']} segments | {r['defns']} | {r['qs_asked']} |"
            )
        lines += [""]

        # All student quotes
        active = [r for r in m4a_students if r["quotes"]]
        if active:
            lines += ["### What Students Said", ""]
            for r in active:
                lines.append(f"**{r['name']}** ({r['level']})")
                for q in r["quotes"]:
                    lines.append(f"> {q}")
                lines.append("")
    else:
        lines += [
            "_No students had isolated audio recordings in this class export._",
            "",
        ]
    lines += ["---", ""]

    # Absent students
    if absent:
        lines += ["## Absent Students (Enrolled, No Recording)", ""]
        for key, s in absent.items():
            lines.append(f"- **{s.get('name', key)}** (roll {s.get('roll_no', '—')})")
        lines += ["", "---", ""]

    # Attendance appendix
    if attendance_rows:
        avg_dur = sum(r["duration"] for r in attendance_rows) / len(attendance_rows)
        is_multiclass = avg_dur > duration_min * 1.5
        scope_note = (
            "_⚠️ This CSV covers the full Zoom meeting for the day — it includes multiple classes. "
            "Durations are total time in the meeting, not time in this specific class. "
            "Only students with M4A files in the class zip were confirmed in this class._"
            if is_multiclass
            else f"_{len(attendance_rows)} participants from the Zoom attendance export._"
        )
        lines += [
            "## Zoom Meeting Attendance",
            "",
            scope_note,
            "",
            "| Student | Roll | Duration in meeting | In this class |",
            "|---------|------|-------------------|---------------|",
        ]
        for row in attendance_rows:
            dur = f"{round(row['duration'])} min" if row["duration"] else "—"
            roll = row["roll_no"] or "—"
            in_class = "✓ Audio recorded" if row["roll_no"] in recorded_rolls else "Other class"
            lines.append(f"| {row['display']} | {roll} | {dur} | {in_class} |")
        lines += [""]

    # Footer
    lines += [
        "---",
        "",
        "*Generated from Zoom cloud recording transcripts (WhisperX dual-language, small model). "
        "Students confirmed from per-student M4A files in the class zip. "
        "Verbal contributions extracted from isolated microphone audio. "
        "Transcript quality may vary — Hindi/Hinglish speech and background noise affect accuracy.*",
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
