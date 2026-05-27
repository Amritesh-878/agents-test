# TASK-011: Cleanup and Foundation Reset

## Overview

Remove all pyannote diarization code and old tests. Create a shared Pydantic models package (`scripts/models/`) to eliminate circular imports and the `sys.path.insert` hack. Update dependencies.

## Execution Snapshot

- Depends on: None (first task)
- Produces: `scripts/models/` package, cleaned `requirements.txt`, updated `scripts/validate_env.py`
- Primary validation: `ruff check --fix . && mypy . && pytest`
- Complexity: Medium

## Goals

1. **Remove pyannote**: Delete `scripts/diarize.py` and all references to pyannote diarization
2. **Delete old tests**: Clean break — remove all 12 test files (62 tests total)
3. **Shared models package**: Create `scripts/models/` with transcript, identity, and pipeline config models
4. **Update dependencies**: Remove `pyannote.audio`, add `psycopg[binary]>=3.1` and `pgvector>=0.2`

---

## Reasoning

### Why delete all old tests?

**Current Problems:**

- The entire pipeline architecture is changing (diarization replaced by per-student M4A identity)
- Old tests validate the wrong behavior (pyannote merge, heuristic speaker mapping)
- Keeping them creates false confidence — tests pass but the code they test is being replaced

**Solution:**

- Clean break: delete all test files, write new ones per-task
- Each new task (TASK-012 through TASK-018) writes its own tests for new behavior

### Why a shared models package?

**Current Problems:**

- Each script defines its own Pydantic models inline (e.g., `TranscriptSegment` in `transcribe.py`, imported by `merge.py`)
- Cross-script imports require `sys.path.insert(0, ...)` hack at the top of each file
- New pipeline has more scripts sharing more models — the hack doesn't scale

**Solution:**

- `scripts/models/` package with `transcript.py`, `identity.py`, `pipeline.py`
- All scripts import from `scripts.models.*` using standard package imports
- Models are the single source of truth for data contracts between pipeline steps

---

## Files to Change

### Files to DELETE Entirely

1. `scripts/diarize.py`
2. `tests/test_diarize.py`
3. `tests/test_validate_env.py`
4. `tests/test_extract_audio.py`
5. `tests/test_transcribe.py`
6. `tests/test_merge.py`
7. `tests/test_build_context.py`
8. `tests/test_chunker.py`
9. `tests/test_chunk_and_embed.py`
10. `tests/test_retrieval.py`
11. `tests/test_chat.py`
12. `tests/test_evaluate.py`

### Files to CREATE

#### Shared Models Package

1. `scripts/models/__init__.py` — Package init, re-exports key models
2. `scripts/models/transcript.py` — `TranscriptWord`, `TranscriptSegment`, `TranscriptDocument`, `DualLanguageWord`
3. `scripts/models/identity.py` — `RosterEntry`, `AttendanceRecord`, `AudioFileIdentity`, `StudentIdentity`, `MatchResult`
4. `scripts/models/pipeline.py` — `PipelineConfig`, `ClassSession`

#### Tests

5. `tests/test_models.py` — Validation, serialization, edge cases for all shared models

### Files to MODIFY

1. `scripts/validate_env.py` — **MAJOR** — Remove pyannote validation checks, add pgvector/psycopg availability check, keep CUDA + WhisperX checks
2. `requirements.txt` — **MINOR** — Remove `pyannote.audio==3.1.1`, add `psycopg[binary]>=3.1`, `pgvector>=0.2`

---

## Implementation Approach

### `scripts/models/transcript.py`

**Purpose:** Shared transcript data models used by TASK-014 (transcription), TASK-015 (merge), and TASK-016 (context builder).

**Key Models:**

```
TranscriptWord
  - start: float
  - end: float
  - word: str
  - score: float | None  # word-level confidence from WhisperX alignment

TranscriptSegment
  - start: float
  - end: float
  - text: str
  - words: list[TranscriptWord]
  - language: str | None  # detected or specified language

TranscriptDocument
  - language: str | None
  - model: str
  - segments: list[TranscriptSegment]

DualLanguageWord
  - start: float
  - end: float
  - word: str
  - score: float  # word-level probability
  - source_language: str  # "hi" or "en"
```

### `scripts/models/identity.py`

**Purpose:** Student identity models used by TASK-012 (file discovery), TASK-013 (matching), and TASK-016 (context builder).

**Key Models:**

```
RosterEntry
  - name: str
  - roll_no: str
  - email: str

AttendanceRecord
  - name: str
  - roll_no: str | None  # extracted from Name_RollNo pattern
  - duration_minutes: float
  - guest: bool | None
  - tags: list[str]  # e.g., "short_duration"

AudioFileIdentity
  - filename: str
  - extracted_number: str | None  # full number from filename
  - roll_no_4digit: str | None    # first 4 digits
  - display_name: str | None      # name portion from filename

StudentIdentity
  - name: str
  - roll_no: str | None
  - email: str | None
  - source: Literal["roster", "attendance", "audio_file"]

MatchResult
  - audio_file: str
  - matched_student: StudentIdentity | None
  - method: Literal["roll_no", "fuzzy_name", "none"]
  - confidence: float
```

### `scripts/models/pipeline.py`

**Purpose:** Pipeline configuration and session metadata.

**Key Models:**

```
PipelineConfig
  - input_path: Path
  - output_dir: Path
  - teacher_names: list[str]
  - roster_path: Path | None
  - attendance_path: Path | None
  - db_url: str | None
  - single_language: str | None  # override for dual-language

ClassSession
  - class_name: str
  - zip_path: Path | None
  - raw_dir: Path
  - output_dir: Path
  - teacher_name: str
```

### Preservation Strategy

Old scripts (`merge.py`, `build_context.py`, `transcribe.py`, `chunk_and_embed.py`) are NOT deleted in this task because `retrieval.py`, `chat.py`, and `evaluate.py` still import from them. They become orphaned after TASK-018 updates all downstream imports. Final cleanup (rename to `.txt`) happens at the end of TASK-018.

---

## Acceptance Criteria

### Functional Requirements

- [ ] `scripts/diarize.py` deleted
- [ ] All 12 test files deleted
- [ ] `scripts/models/` package created with 3 modules + `__init__.py`
- [ ] `pyannote.audio` removed from `requirements.txt`
- [ ] `psycopg[binary]` and `pgvector` added to `requirements.txt`
- [ ] `scripts/validate_env.py` no longer references pyannote

### Code Quality

- [ ] All new models have complete type hints
- [ ] `ruff check --fix .` passes with 0 errors
- [ ] `mypy .` passes with 0 errors
- [ ] `pytest` passes with 0 errors (new test_models.py)

---

## Testing Requirements

### Unit Tests

1. **Model Construction** — All models construct with valid data
2. **Validation Errors** — Required fields raise `ValidationError` when missing
3. **Serialization** — `model_dump()` and `model_validate()` round-trip correctly
4. **Edge Cases** — Empty strings, None values, boundary values for floats
5. **DualLanguageWord** — Score comparison, source_language values
6. **RosterEntry** — Roll number formats (4-digit, leading zeros)
7. **AttendanceRecord** — Roll number extraction from `Name_RollNo` patterns

Target: ~15 tests.

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Deleting old tests breaks CI if any exist | Verify no CI pipeline references these tests before deleting |
| Old scripts fail to import after pyannote removal | Old scripts are preserved; only diarize.py is deleted. Downstream scripts don't import from diarize.py directly. |
| New models package causes import errors | Test imports in test_models.py; verify `from scripts.models.transcript import TranscriptWord` works |

---

## Related Tasks

- **TASK-012**: First consumer of `scripts/models/identity.py` (`AudioFileIdentity`)
- **TASK-014**: First consumer of `scripts/models/transcript.py` (`DualLanguageWord`)
- **TASK-018**: Final cleanup of old deprecated scripts

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~16 files (12 deleted, 4 created, 2 modified)
- **No GPU needed** — structural cleanup only
