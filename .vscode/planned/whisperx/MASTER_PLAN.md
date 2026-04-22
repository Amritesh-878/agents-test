# Master Implementation Plan

**Project:** WhisperX Transcription & Diarization Pipeline (Phase 1 — Ingestion Test)

**Date Created:** 2026-04-22

**Planning Note:** This file is a working copy of the root `MASTER_PLAN.md`. The root file remains the source of truth; keep both aligned when task status or scope changes.

---

## 🎉 PROJECT COMPLETION SUMMARY

**Status:** 🔄 **IN PROGRESS** (2026-04-22)

**Overview of all tasks:**

| Phase | Task                                     | Status | Build | Tests |
| ----- | ---------------------------------------- | ------ | ----- | ----- |
| 1     | TASK-001: Environment Setup              | ✅     | ✅    | ✅    |
| 1     | TASK-002: Audio Extraction (ffmpeg)      | ⏳     | ❌    | ❓    |
| 2     | TASK-003: WhisperX Transcription         | ⏳     | ❌    | ❓    |
| 2     | TASK-004: Speaker Diarization (pyannote) | ⏳     | ❌    | ❓    |
| 3     | TASK-005: Transcript Merge & Export      | ⏳     | ❌    | ❓    |
| 3     | TASK-006: Per-Student Context Builder    | ⏳     | ❌    | ❓    |

**Current Verification (2026-04-22):**

- ✅ Build: TASK-001 bootstrap files lint and type-check cleanly on Python 3.11
- ✅ Tests: 4 pytest checks passing for TASK-001 helpers; 1/6 tasks fully completed
- 🔄 Integration: Not started
- ✅ Environment: CUDA, Hugging Face token access, pyannote gated model access, and WhisperX ASR model loading validated in the worktree

**Deliverables:**

- Clean diarized transcript JSON from a 1-hour Zoom class recording
- Per-student context object (attendance window, spoken segments, missed segments)
- CLI script runnable locally on Windows with CUDA support

---

## Table of Contents

1. [Implementation Order](#implementation-order)
2. [Dependency Graph](#dependency-graph)
3. [Task Status Tracker](#task-status-tracker)
4. [Phase Summaries](#phase-summaries)
5. [Handoff Notes](#handoff-notes)
6. [Critical Dependencies](#critical-dependencies)

---

## Implementation Order

### Rationale

**Why this order?**

Phase 1 sets up the environment and confirms GPU/CUDA works before touching any ML models. Phase 2 runs transcription and diarization as independent outputs that both depend on the extracted audio from Phase 1; on the local RTX 3050 4GB machine they should be executed sequentially to avoid VRAM contention. Phase 3 merges the two outputs and builds the per-student context object which is the final deliverable for RAG ingestion later.

Each task is a standalone Python script so they can be tested and debugged independently before wiring together.

---

## Phase 1: Environment & Audio Extraction

### 1️⃣ TASK-001: Environment Setup & CUDA Validation

**Why first:**

- Must confirm PyTorch + CUDA works on RTX 3050 before installing whisperX
- whisperX + pyannote have specific dependency constraints (Python 3.10/3.11)
- HuggingFace token and model agreement gating must be validated early

**Scope:**

- `requirements.txt` — all pinned dependencies
- `scripts/validate_env.py` — CUDA check + HF token check + model access check
- `README.md` — setup instructions
- ~3 files

**Success Criteria:**

- [x] `torch.cuda.is_available()` returns True
- [x] HuggingFace token loads pyannote models without 403
- [x] `validate_env.py` exits with no errors

---

### 2️⃣ TASK-002: Audio Extraction via ffmpeg

**Why second:**

- Both whisperX and pyannote need a clean audio file, not raw MP4
- Extracting to 16kHz mono WAV is the standard input format for both
- Validates ffmpeg PATH setup before running ML steps

**Scope:**

- `scripts/extract_audio.py` — ffmpeg subprocess wrapper
- Input: `.mp4` Zoom recording
- Output: `output/audio.wav` (16kHz, mono)
- ~1 file

**Success Criteria:**

- [ ] Extracts WAV from Zoom MP4 without errors
- [ ] Output is 16kHz mono (verified with ffprobe)
- [ ] Handles long recordings (1hr+) without memory issues

---

## Phase 2: Transcription & Diarization

### 3️⃣ TASK-003: WhisperX Transcription

**Why third:**

- Depends on audio WAV from TASK-002
- Must run before diarization merge in TASK-005
- Use `small` model to fit 4GB VRAM

**Scope:**

- `scripts/transcribe.py` — whisperX transcription + word-level alignment
- Output: `output/transcript_raw.json`
- ~1 file

**Success Criteria:**

- [ ] Transcribes 1hr recording without OOM error on RTX 3050
- [ ] Output JSON has word-level timestamps
- [ ] Indian English accent quality is manually verified on 5-min sample

---

### 4️⃣ TASK-004: Speaker Diarization (pyannote)

**Why fourth:**

- Depends on audio WAV from TASK-002
- Runs independently from transcription, merged in TASK-005
- pyannote speaker-diarization-3.1 requires HF token + accepted agreements

**Scope:**

- `scripts/diarize.py` — pyannote diarization pipeline
- Output: `output/diarization.json` (SPEAKER_00, SPEAKER_01... with timestamps)
- ~1 file

**Success Criteria:**

- [ ] Diarization runs without 403 HuggingFace error
- [ ] Output has speaker-labeled time segments
- [ ] Correctly segments 2-5 speakers in test recording

---

## Phase 3: Merge & Student Context

### 5️⃣ TASK-005: Transcript + Diarization Merge

**Why fifth:**

- Depends on TASK-003 (transcript) and TASK-004 (diarization)
- Merges word timestamps with speaker labels
- Output is the "clean diarized transcript" — primary deliverable

**Scope:**

- `scripts/merge.py` — timestamp-based merge of whisperX + pyannote outputs
- Output: `output/transcript_diarized.json`
- ~1 file

**Success Criteria:**

- [ ] Each transcript segment has a speaker label
- [ ] No orphaned segments (all words assigned to a speaker)
- [ ] Output readable as `[timestamp] SPEAKER_X: text`

---

### 6️⃣ TASK-006: Per-Student Context Builder

**Why last:**

- Depends on TASK-005 (diarized transcript)
- Requires Zoom attendance CSV as second input (join/leave timestamps)
- Produces the final per-student JSON for RAG ingestion

**Scope:**

- `scripts/build_context.py` — maps speaker labels → student names via attendance CSV
- `data/sample_attendance.csv` — sample Zoom attendance report format
- Output: `output/student_contexts.json`
- ~2 files

**Success Criteria:**

- [ ] Each student has: name, attendance window, spoken segments, missed segments
- [ ] Missed segments correctly computed from transcript timestamps vs attendance
- [ ] Output schema documented and ready for RAG ingestion in Phase 2

---

## Dependency Graph

```
┌─────────────────────────────────┐
│  TASK-001: Environment Setup    │  ⬅️ START HERE
└─────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  TASK-002: Audio Extraction     │
└─────────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────────┐
│  TASK-003   │  │   TASK-004      │
│ Transcribe  │  │   Diarize       │
│ (whisperX)  │  │   (pyannote)    │
└─────────────┘  └─────────────────┘
       │                │
       └───────┬────────┘
               ▼
┌─────────────────────────────────┐
│  TASK-005: Merge Outputs        │
└─────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  TASK-006: Student Context      │
└─────────────────────────────────┘
```

---

## Task Status Tracker

| Phase | TODO | Title                  | Status         | Notes                                  |
| ----- | ---- | ---------------------- | -------------- | -------------------------------------- |
| 1     | 001  | Environment Setup      | ✅ Completed   | Live validation passed in the worktree |
| 1     | 002  | Audio Extraction       | ⏳ Not Started | Blocked by TASK-001                    |
| 2     | 003  | WhisperX Transcription | ⏳ Not Started | Blocked by TASK-002                    |
| 2     | 004  | Speaker Diarization    | ⏳ Not Started | Blocked by TASK-002                    |
| 3     | 005  | Transcript Merge       | ⏳ Not Started | Blocked by TASK-003, TASK-004          |
| 3     | 006  | Per-Student Context    | ⏳ Not Started | Blocked by TASK-005                    |

**Status Legend:**

- 🔄 In Progress
- ⏳ Blocked / Waiting
- ✅ Completed
- ❌ Failed / Needs Rework

---

## Phase Summaries

### Phase 1: Environment & Audio

**Phase Goal:** Confirm all tooling works on the local Windows + RTX 3050 machine before touching ML models.

**What gets built:**

- `requirements.txt` with pinned versions
- `scripts/validate_env.py` — CUDA + HF token validation
- `scripts/extract_audio.py` — ffmpeg WAV extractor

**What stays the same:**

- Nothing exists yet — greenfield project

**Impact:** Prevents wasted time debugging ML errors caused by environment issues.

---

### Phase 2: Transcription & Diarization

**Phase Goal:** Generate word-level transcript and speaker-labeled time segments independently.

**What gets built:**

- `scripts/transcribe.py` — whisperX pipeline
- `scripts/diarize.py` — pyannote pipeline

**Impact:** Produces the two raw outputs that get merged in Phase 3.

---

### Phase 3: Merge & Student Context

**Phase Goal:** Combine transcript + diarization and produce the per-student context JSON ready for RAG.

**What gets built:**

- `scripts/merge.py` — timestamp merge
- `scripts/build_context.py` — student context builder

**Impact:** Final deliverable of Phase 1. Output feeds directly into the RAG pipeline in Phase 2 of the broader project.

---

## Handoff Notes

### TODO-001 Handoff

**Status:** 🔄 In Progress

**Prerequisites met:**

- [x] Python 3.10 and 3.11 are installed locally
- [x] CUDA is visible to PyTorch on the RTX 3050 Laptop GPU
- [ ] Worktree `.env` contains a valid HF token with accepted pyannote agreements

```
Completed by: GPT-5.4
Build status: ✅ PASS

### What was done:
- Added a Windows-focused bootstrap README, `.env.example`, `.gitignore`, and pinned `requirements.txt`
- Implemented `scripts/validate_env.py` with Python version, CUDA, Hugging Face, pyannote, and WhisperX checks plus GPU cleanup
- Added pytest coverage for the pure validation helpers and confirmed `ruff`, `mypy`, and `pytest` pass under Python 3.11
- Pinned `numpy==1.26.4` to avoid the NumPy 2.x breakage in the current pyannote and WhisperX stack

### Tests passing: ✅ 4 tests

### Warnings to next implementor:
- TASK-002 remains blocked until `.env` is created in this worktree and `python scripts/validate_env.py` passes without token-related failures
- The local machine should use Python 3.11 for the project environment; the system default Python 3.14 is outside the supported range

### Breaking changes:
- None
```

---

### TODO-002 Handoff

**Status:** ⏳ Not Started

**Prerequisites from TODO-001:**

- [ ] Environment validated (CUDA + HF token working)
- [ ] ffmpeg in system PATH (already done)

```
[Fill in after completion]
```

---

### TODO-003 Handoff

**Status:** ⏳ Not Started

**Prerequisites from TODO-002:**

- [ ] `output/audio.wav` exists at 16kHz mono

```
[Fill in after completion]
```

---

### TODO-004 Handoff

**Status:** ⏳ Not Started

**Prerequisites from TODO-002:**

- [ ] `output/audio.wav` exists at 16kHz mono
- [ ] HF token in `.env`

```
[Fill in after completion]
```

---

### TODO-005 Handoff

**Status:** ⏳ Not Started

**Prerequisites from TODO-003 + TODO-004:**

- [ ] `output/transcript_raw.json` exists
- [ ] `output/diarization.json` exists

```
[Fill in after completion]
```

---

### TODO-006 Handoff

**Status:** ⏳ Not Started

**Prerequisites from TODO-005:**

- [ ] `output/transcript_diarized.json` exists
- [ ] Zoom attendance CSV available

```
[Fill in after completion]
```

---

## Critical Dependencies

⚠️ **DO NOT SKIP OR REORDER:**

| Violation                        | Consequence                                             |
| -------------------------------- | ------------------------------------------------------- |
| Run TASK-003/004 before TASK-002 | No WAV file to process — both will crash immediately    |
| Run TASK-005 before TASK-003/004 | Nothing to merge — empty output                         |
| Skip TASK-001 CUDA validation    | OOM errors on whisperX will be confusing to debug later |
| Use `large-v3` model on 4GB VRAM | OOM — must use `small` for RTX 3050                     |

---

## Summary

**Key Principle:** Validate environment first, extract audio second, run ML outputs independently but execute them sequentially on the local 4GB GPU, merge last.

Begin with **TASK-001: Environment Setup** when ready. See individual TASK files for implementation details.

**Next Phase (not in scope here):** RAG pipeline — chunk diarized transcript, embed, store in pgvector, build retrieval layer for student chatbot.
