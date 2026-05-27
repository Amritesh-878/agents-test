# TASK-013: Identity Matching (Roster + Attendance + Audio Filenames)

## Overview

Match per-student M4A audio files to real student identities using the master roster CSV, Zoom attendance CSV, and 4-digit roll numbers extracted from filenames. Identify the teacher. Handle unmatched files.

## Execution Snapshot

- Depends on: TASK-011, TASK-012
- Produces: `scripts/match_identity.py`, `output/<class_name>/identity_map.json`
- Primary validation: `python scripts/match_identity.py --manifest output/<class>/manifest.json --roster roster.csv --attendance attendance.csv --teacher "Name"`
- Complexity: Medium

## Goals

1. **Roll Number Matching**: Match M4A filename roll numbers to roster and attendance CSV roll numbers
2. **Teacher Identification**: Match teacher name from CLI flag to M4A files using fuzzy name matching
3. **Unmatched Handling**: Tag files with no match; tag short-duration entries; include all in output

---

## Reasoning

### Why roll number as primary key?

**Current Problems:**

- The old pipeline used a "first speaker after join time" heuristic to guess speaker identity
- This was inaccurate and required manual review for every session

**Solution:**

- Zoom M4A filenames contain the student's roll number (first 4 digits of the numeric portion)
- Attendance CSV has `Name_RollNo` format in the name column
- Master roster has a dedicated RollNo column
- Matching on roll number is deterministic — no guessing needed

### Why fuzzy name matching for teacher?

**Current Problems:**

- Teacher has no roll number in their filename
- Teacher's Zoom display name may not exactly match the name in the attendance CSV

**Solution:**

- Teacher name provided via `--teacher` CLI flag
- Use `difflib.SequenceMatcher` to find best match among M4A filenames and attendance entries
- No external dependency needed

---

## Files to Change

### Files to CREATE

1. `scripts/match_identity.py` — **MAJOR** — Identity matching CLI script
2. `tests/test_match_identity.py` — **MAJOR** — Tests for all matching paths

### Files to MODIFY

1. `scripts/models/identity.py` — **MINOR** — Add `IdentityMapEntry`, `IdentityMap` models

---

## Implementation Approach

### Roll Number Extraction from Attendance CSV

The attendance CSV name column follows `Name_RollNo` format:

```
"Anshi_2301" → name="Anshi", roll_no="2301"
"A_Disha_2504" → name="A_Disha", roll_no="2504"
"Amritesh Praveen" → name="Amritesh Praveen", roll_no=None (no underscore+digits pattern)
```

Algorithm:
```
1. Check if name matches pattern: .*_\d{4}$
2. If yes: split on last underscore, extract name and roll_no
3. If no: roll_no = None (likely teacher or non-student)
```

### Matching Pipeline

```
for each per_student_m4a in manifest.per_student_m4as:
  # Step 1: Try roll number match
  if m4a.roll_no_4digit is not None:
    roster_match = roster.find_by_roll(m4a.roll_no_4digit)
    attendance_match = attendance.find_by_roll(m4a.roll_no_4digit)
    if roster_match:
      result = IdentityMapEntry(method="roll_no", confidence=1.0, ...)
      continue

  # Step 2: Try teacher match
  if fuzzy_match(m4a.display_name, teacher_names) > 0.7:
    result = IdentityMapEntry(is_teacher=True, method="fuzzy_name", ...)
    continue

  # Step 3: Unmatched
  result = IdentityMapEntry(method="none", is_unmatched=True, ...)
```

### Fuzzy Name Matching

```python
from difflib import SequenceMatcher

def fuzzy_match_score(name_a: str, name_b: str) -> float:
    return SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
```

### Tagging

- `short_duration`: attendance duration < 5 minutes (configurable)
- `unmatched`: no roster/attendance match found
- `teacher`: matched to teacher name
- `no_audio`: roster student with no corresponding M4A file

### Pydantic Models

```
IdentityMapEntry
  - audio_file: str
  - roll_no_4digit: str | None
  - matched_name: str | None
  - matched_roll_no: str | None
  - matched_email: str | None
  - match_method: Literal["roll_no", "fuzzy_name", "none"]
  - match_confidence: float
  - is_teacher: bool
  - is_unmatched: bool
  - attendance_duration_minutes: float | None
  - tags: list[str]

IdentityMap
  - teacher_name: str
  - teacher_audio_file: str | None
  - entries: list[IdentityMapEntry]
  - unmatched_entries: list[IdentityMapEntry]
  - roster_students_without_audio: list[str]
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] Matches M4A files to roster entries by 4-digit roll number
- [ ] Identifies teacher M4A via fuzzy name match against `--teacher` flag
- [ ] Handles `Name_RollNo` attendance CSV format (e.g., `Anshi_2301`)
- [ ] Tags short-duration entries without filtering them out
- [ ] Lists roster students with no matching M4A file (absent/no audio)
- [ ] Writes valid `identity_map.json`
- [ ] Handles multiple `--teacher` names (space-separated or repeated flag)

### Code Quality

- [ ] argparse CLI with `--manifest`, `--roster`, `--attendance`, `--teacher`
- [ ] Complete type hints
- [ ] No bare except
- [ ] `ruff` and `mypy` pass clean

---

## Testing Requirements

### Unit Tests

1. **Roll Number Matching**
   - Standard match: M4A roll "2301" → roster entry with roll "2301"
   - No match: M4A roll "9999" → unmatched
   - Attendance CSV parsing: "Anshi_2301" → roll="2301"
   - Attendance with no roll: "Amritesh Praveen" → roll=None

2. **Teacher Matching**
   - Exact name match
   - Fuzzy match: "Amritesh" in M4A vs "Amritesh Praveen" in --teacher
   - No teacher M4A found → teacher_audio_file=None

3. **Tagging**
   - Short duration (3 min) → tagged
   - Normal duration (60 min) → not tagged
   - Unmatched M4A → tagged

4. **Edge Cases**
   - Empty roster CSV
   - Empty attendance CSV
   - M4A file with no parseable name/number
   - Duplicate roll numbers in roster (error)

Target: ~20 tests.

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Roll number collision (different students, same 4 digits) | Unlikely with 4-digit roll numbers in a single class; log and flag if detected |
| Teacher fuzzy match picks wrong person | Confidence threshold at 0.7; log match details for review |
| Attendance CSV has inconsistent name format | Regex-based extraction with fallback to None; log all extraction attempts |

---

## Related Tasks

- **TASK-012**: Provides `manifest.json` with M4A file list and parsed roll numbers
- **TASK-014**: Uses `identity_map.json` to label transcripts with student names
- **TASK-015**: Uses `identity_map.json` for speaker attribution in merge
- **TASK-016**: Uses `identity_map.json` + roster for context building

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~3 files (2 created, 1 modified)
- **No GPU needed** — CSV parsing and string matching only
