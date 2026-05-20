# TASK-006: Per-Student Context Builder

Archived on 2026-05-20 after implementation in worktree `task-006-context-builder`.
See `MASTER_PLAN.md` for the completion handoff and validation notes.

## Overview

Build per-student context from the diarized transcript plus Zoom attendance data, with support for the approved fallback when the real attendance CSV does not include Join Time and Leave Time.

## Execution Snapshot

- Depends on: TASK-005
- Produces: `scripts/build_context.py`, `data/sample_attendance.csv`, `output/student_contexts.json`, `output/student_context_review.md`, `output/student_context_segments.csv`
- Primary validation: `python scripts/build_context.py --transcript output/transcript_diarized.json --attendance <attendance.csv>`
- Complexity: Medium

## Completed Scope

1. Added exact join/leave parsing for full Zoom attendance exports and the approved duration-only fallback for the real CSV in this repo.
2. Added machine-readable and human-readable outputs so speaker mapping, transcript segments, and per-student context can be reviewed without reverse-engineering raw JSON.
3. Added bounded real-data runtime validation using the supplied attendance CSV and a 60-second clip cut from the supplied Zoom archive.

## Approved Fallback Notes

- The real attendance CSV only provides name, email, total duration, and guest fields.
- Attendance windows are therefore estimated unless explicit join and leave timestamps are present.
- Missed segments are marked approximate whenever they depend on estimated attendance windows.
- Speaker mapping is surfaced for manual review through both JSON metadata and Markdown/CSV review artifacts.

## Notes

- The bounded runtime validation clip diarized as a single speaker, so only one participant was auto-mapped in that pass.
- `output/student_context_review.md` and `output/student_context_segments.csv` are the primary review artifacts for personalized-learning suitability checks.
