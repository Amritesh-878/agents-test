from __future__ import annotations

import argparse
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
    "yeah", "yep", "alright", "actually",
}


class ReportArgs(BaseModel):
    class_dir: Path
    output: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> ReportArgs:
    parser = argparse.ArgumentParser(
        description="Generate a single-page session engagement report."
    )
    parser.add_argument("--class-dir", required=True, type=Path, dest="class_dir")
    parser.add_argument("--output", type=Path, default=None)
    namespace = parser.parse_args(argv)
    return ReportArgs.model_validate(vars(namespace))


def _clean_text(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1 \2", text)
    return text


def _is_meaningful(text: str, min_chars: int = 15) -> bool:
    cleaned = _clean_text(text)
    if not is_quality_text(cleaned, min_chars=min_chars):
        return False
    words = [w.strip(".,!?") for w in cleaned.lower().split() if w.strip(".,!?")]
    real_words = [w for w in words if w not in _NOISE_WORDS and w.isalpha() and len(w) > 2]
    return len(real_words) >= 2


def _engagement_level(spoken_count: int, quality_count: int) -> str:
    if quality_count >= 10:
        return "Active"
    if quality_count >= 4:
        return "Moderate"
    if quality_count >= 1:
        return "Passive"
    return "Silent"


def _clean_topics(raw_topics: list[str], student_names: set[str] | None = None) -> list[str]:
    skip = set(_NOISE_WORDS) | {"day", "ताद", "पूँच्य", "अगर", "telling", "called", "part", "second"}
    names_lower = {n.lower() for n in (student_names or set())}
    result = []
    for t in raw_topics:
        t_lower = t.lower()
        if t_lower in skip:
            continue
        if any(c.isdigit() for c in t):
            continue
        # Skip if the topic is or contains a student name
        if any(name in t_lower for name in names_lower if len(name) > 3):
            continue
        result.append(t)
    return result[:6]


def generate_report(class_dir: Path) -> str:
    import json

    ctx_path = class_dir / "student_contexts.json"
    merge_path = class_dir / "transcript_merged.json"
    imap_path = class_dir / "identity_map.json"

    if not ctx_path.exists() or not merge_path.exists():
        raise ValueError(f"Missing student_contexts.json or transcript_merged.json in {class_dir}")

    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    merged = json.loads(merge_path.read_text(encoding="utf-8"))
    # identity_map available for future extensions (e.g. unmatched student list)
    _ = json.loads(imap_path.read_text(encoding="utf-8")) if imap_path.exists() else {}

    class_name = merged.get("class_name", class_dir.name)
    duration_min = round(merged.get("duration_seconds", 0) / 60, 1)
    teacher = merged.get("teacher_name", "Unknown")
    all_segments = merged.get("segments", [])

    # Session type detection
    session_type = "Class Session"
    for seg in all_segments[:10]:
        txt = seg.get("text", "").lower()
        if "revise" in txt or "revision" in txt or "recap" in txt:
            session_type = "Revision Session"
            break
        if "test" in txt or "exam" in txt or "quiz" in txt:
            session_type = "Assessment"
            break

    # Student engagement
    present = ctx.get("present_students", {})
    absent = ctx.get("absent_students", {})

    # Collect student first names for topic filtering
    student_first_names: set[str] = set()
    for s in present.values():
        name = s.get("name", "")
        # e.g. "A_Jagruti" → "jagruti"; "Bhagyashree" → "bhagyashree"
        parts = re.split(r"[_\s]", name.lower())
        student_first_names.update(p for p in parts if len(p) > 3)

    # Topics (cleaned, student names filtered out)
    raw_topics = ctx.get("topics", [])
    topics = _clean_topics(raw_topics, student_first_names)

    student_rows: list[dict] = []
    for key, s in present.items():
        spoken_segs = s.get("spoken_segments", [])
        att_min = s.get("attendance_duration_minutes")
        # Cap attendance at session duration — Zoom sometimes reports inflated values
        if att_min and att_min > duration_min:
            att_display: int | str = "Full session"
        else:
            att_display = round(att_min) if att_min else "—"
        quality_segs = [seg for seg in spoken_segs if _is_meaningful(seg.get("text", ""))]
        best_quotes = [
            _clean_text(seg["text"])[:160]
            for seg in quality_segs
            if len(_clean_text(seg.get("text", ""))) > 20
        ][:3]
        student_rows.append({
            "name": s.get("name", key),
            "roll": s.get("roll_no", "—"),
            "att_display": att_display,
            "spoken": len(spoken_segs),
            "quality": len(quality_segs),
            "level": _engagement_level(len(spoken_segs), len(quality_segs)),
            "quotes": best_quotes,
        })

    student_rows.sort(key=lambda r: r["quality"], reverse=True)

    # Teacher content samples (meaningful fallback segments)
    teacher_segs = [
        _clean_text(s["text"])
        for s in all_segments
        if s.get("source") == "session_fallback"
        and _is_meaningful(s.get("text", ""), min_chars=30)
    ]

    # Deduplicate teacher segments (same text appears often due to transcript chunking)
    seen: set[str] = set()
    unique_teacher: list[str] = []
    for t in teacher_segs:
        key = t[:60].lower()
        if key not in seen:
            seen.add(key)
            unique_teacher.append(t)

    # Sample across session timeline
    total_dur = merged.get("duration_seconds", 1)
    timeline_samples: list[tuple[float, str]] = []
    checkpoints = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    for frac in checkpoints:
        target_t = frac * total_dur
        candidates = [
            s for s in all_segments
            if s.get("source") == "session_fallback"
            and _is_meaningful(s.get("text", ""), min_chars=25)
        ]
        if candidates:
            closest = min(candidates, key=lambda s: abs(s["start"] - target_t))
            txt = _clean_text(closest["text"])[:180]
            t_min = closest["start"] / 60
            key = txt[:50].lower()
            if key not in seen:
                seen.add(key)
                timeline_samples.append((t_min, txt))

    # Build the markdown report
    lines: list[str] = []

    lines += [
        f"# Session Report — {class_name}",
        "",
        f"**Type:** {session_type}  |  **Duration:** {duration_min} min  |  **Teacher:** {teacher}",
        f"**Students with recording:** {len(present)}  |  **Absent (no audio):** {len(absent)}",
        "",
        "---",
        "",
    ]

    # Topics
    if topics:
        lines += [
            "## Topics Covered",
            "",
            ", ".join(f"**{t}**" for t in topics),
            "",
            "---",
            "",
        ]

    # Student engagement
    lines += [
        "## Student Engagement",
        "",
        "| Student | Roll | Attendance | Engagement | Meaningful Contributions |",
        "|---------|------|-----------|------------|--------------------------|",
    ]
    for r in student_rows:
        att_str = f"{r['att_display']} min" if isinstance(r["att_display"], int) else str(r["att_display"])
        lines.append(
            f"| {r['name']} | {r['roll']} | {att_str} | **{r['level']}** | {r['quality']} spoken segments |"
        )
    for key, s in absent.items():
        lines.append(f"| {s.get('name', key)} | {s.get('roll_no', '—')} | Absent | — | — |")

    lines += ["", "---", ""]

    # Key student contributions
    active = [r for r in student_rows if r["quotes"]]
    if active:
        lines += ["## Key Student Contributions", ""]
        for r in active:
            lines.append(f"**{r['name']}** ({r['level']})")
            for q in r["quotes"]:
                lines.append(f"> \"{q}\"")
            lines.append("")
        lines += ["---", ""]

    # Session timeline
    if timeline_samples:
        lines += ["## Session Timeline (Teacher Content)", ""]
        for t_min, txt in timeline_samples[:6]:
            lines.append(f"**{t_min:.0f} min** — {txt}")
            lines.append("")
        lines += ["---", ""]

    # Absent students summary
    if absent:
        lines += ["## Absent Students", ""]
        for key, s in absent.items():
            name = s.get("name", key)
            roll = s.get("roll_no", "—")
            topic_str = ", ".join(s.get("topics_discussed", [])[:4]) or "see class topics"
            lines.append(f"- **{name}** (roll {roll}) — missed class. Topics covered: {topic_str}")
        lines += [""]

    # Footer
    lines += [
        "---",
        "",
        "*Generated from per-student M4A transcripts + session recording. "
        "Engagement levels are based on spoken transcript segments after quality filtering.*",
    ]

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    if not args.class_dir.exists():
        logger.error("Class directory not found: %s", args.class_dir)
        raise SystemExit(1)

    report = generate_report(args.class_dir)

    out = args.output or args.class_dir / "session_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Report written to {out}")
    print()
    print(report)


if __name__ == "__main__":
    main()
