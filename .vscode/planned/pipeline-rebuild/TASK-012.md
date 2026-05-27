# TASK-012: Zip Extraction, File Discovery, and Filename Parsing

## Overview

Accept a Zoom .zip file (or directory of zips for batch mode), extract contents, discover and classify all files (session MP4, mixed M4A, per-student M4As, recording.conf), and parse student roll numbers from M4A filenames.

## Execution Snapshot

- Depends on: TASK-011
- Produces: `scripts/ingest_zip.py`, `output/<class_name>/raw/`, `output/<class_name>/manifest.json`
- Primary validation: `python scripts/ingest_zip.py --input <test.zip> --output-dir output/`
- Complexity: Medium

## Goals

1. **Zip Extraction**: Auto-extract Zoom .zip to `output/<class_name>/raw/`
2. **File Discovery**: Classify extracted files by type (MP4, mixed M4A, per-student M4As, recording.conf)
3. **Filename Parsing**: Extract student display name and 4-digit roll number from M4A filenames like `audioAnshi_23013186578705.m4a`

---

## Reasoning

### Why auto-extract?

**Current Problems:**

- The current pipeline requires manual extraction and pointing at individual files
- Batch processing of multiple classes needs automated extraction

**Solution:**

- Pipeline accepts `.zip` files directly
- Extracts to `output/<class_name>/raw/` where class_name comes from the zip filename
- For batch mode, accepts a directory and globs for `*.zip`

### Why parse filenames?

**Current Problems:**

- The current pipeline has no way to identify which audio belongs to which student
- Pyannote gives anonymous speaker IDs (SPEAKER_00, SPEAKER_01)

**Solution:**

- Zoom M4A filenames encode the student's display name and a number whose first 4 digits are their roll number
- Parse this directly — no ML or probabilistic matching needed for students

---

## Files to Change

### Files to CREATE

1. `scripts/ingest_zip.py` — **MAJOR** — CLI script for zip extraction, file discovery, and manifest generation
2. `tests/test_ingest_zip.py` — **MAJOR** — Tests for extraction, discovery, and filename parsing

### Files to MODIFY

1. `scripts/models/identity.py` — **MINOR** — Add `ZoomFileManifest`, `PerStudentAudioFile` models if not already present from TASK-011

---

## Implementation Approach

### Filename Parsing Algorithm

M4A filename pattern: `audio<DisplayName>_<Number>.m4a`

```
input: "audioAnshi_23013186578705.m4a"

1. Strip extension: "audioAnshi_23013186578705"
2. Strip "audio" prefix: "Anshi_23013186578705"
3. Split on last underscore: display_name="Anshi", number="23013186578705"
4. Extract roll number: first 4 digits of number = "2301"
```

Edge cases:
- Names with underscores: `audioA_Disha_25043186578705.m4a` — split on the LAST underscore that precedes a digit sequence
- Names with no number: teacher or malformed — flag as `roll_no=None`
- Short numbers: fewer than 4 digits — use what's available, flag for review

### File Classification

```
for each file in extracted zip:
  if extension == ".mp4":
    classify as session_mp4 (expect exactly 1)
  elif extension == ".m4a":
    if filename starts with "audio":
      classify as per_student_m4a (parse name + roll)
    else:
      classify as mixed_m4a (expect exactly 1)
  elif filename == "recording.conf":
    parse as key=value metadata
  elif filename == "zoomver.tag":
    read version string
```

### recording.conf Parsing

Parse as key=value or INI format. Extract any useful metadata:
- Meeting ID, date, duration, host name
- If none of these are found, log contents and continue without error

### Batch Mode

```
if --input is a .zip file:
  process single zip
elif --input is a directory:
  glob for *.zip
  process each sequentially
  write batch_manifest.json listing all class sessions
```

### Pydantic Models

```
PerStudentAudioFile
  - path: Path
  - filename: str
  - display_name: str | None
  - extracted_number: str | None
  - roll_no_4digit: str | None

ZoomFileManifest
  - class_name: str
  - raw_dir: Path
  - session_mp4: Path | None
  - mixed_m4a: Path | None
  - per_student_m4as: list[PerStudentAudioFile]
  - recording_conf: dict[str, str] | None
  - zoomver_tag: str | None
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] Extracts .zip to `output/<class_name>/raw/`
- [ ] Discovers session MP4, mixed M4A, per-student M4As correctly
- [ ] Parses display names and 4-digit roll numbers from M4A filenames
- [ ] Handles batch mode (directory of zips)
- [ ] Writes valid `manifest.json` with `ZoomFileManifest` schema
- [ ] Handles nested directories inside zip (Zoom often wraps in a top-level folder)
- [ ] Logs warning for unexpected file types or missing expected files

### Code Quality

- [ ] argparse CLI with `--input`, `--output-dir`
- [ ] Complete type hints on all functions
- [ ] No bare except
- [ ] `ruff check --fix .` and `mypy .` pass clean

---

## Testing Requirements

### Unit Tests

1. **Filename Parsing**
   - Standard: `audioAnshi_23013186578705.m4a` → name="Anshi", roll="2301"
   - Underscore in name: `audioA_Disha_25043186578705.m4a` → name="A_Disha", roll="2504"
   - No number: `audioTeacher.m4a` → name="Teacher", roll=None
   - Short number: `audio_Test_12.m4a` → handle gracefully

2. **File Classification**
   - Zip with standard Zoom layout
   - Zip with nested top-level directory
   - Zip with missing MP4 (log warning, continue)
   - Zip with multiple M4A files correctly classified

3. **Manifest Generation**
   - Valid manifest JSON round-trips through Pydantic
   - Per-student M4A list populated correctly
   - recording.conf parsed when present, None when absent

4. **Batch Mode**
   - Directory with 2 zips → 2 manifests
   - Directory with no zips → clear error message

Target: ~20 tests.

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Zoom zip format varies between versions | Test with real exports; handle nested dirs; log unexpected structures |
| M4A filename format changes | Conservative parsing with fallback to None; log all extractions |
| Large zip extraction fills disk | Log file sizes; warn if total > 10GB; process one at a time |

---

## Related Tasks

- **TASK-011**: Provides shared models package
- **TASK-013**: Consumes `manifest.json` for identity matching
- **TASK-014**: Consumes `manifest.json` for file paths to transcribe

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~3 files (2 created, 1 modified)
- **No GPU needed** — file I/O and string parsing only
