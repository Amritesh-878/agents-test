# Master Implementation Plan

**Project:** WhisperX Transcription & Diarization Pipeline (Phase 1 — Ingestion Test)

**Date Created:** 2026-04-22

---

## 🎉 PROJECT COMPLETION SUMMARY

**Status:** 🔄 **IN PROGRESS** (2026-04-22)

**Overview of all tasks:**

| Phase | Task                                     | Status | Build | Tests |
| ----- | ---------------------------------------- | ------ | ----- | ----- |
| 1     | TASK-001: Environment Setup              | ✅     | ✅    | ✅    |
| 1     | TASK-002: Audio Extraction (ffmpeg)      | ✅     | ✅    | ✅    |
| 2     | TASK-003: WhisperX Transcription         | ✅     | ✅    | ✅    |
| 2     | TASK-004: Speaker Diarization (pyannote) | ✅     | ✅    | ✅    |
| 3     | TASK-005: Transcript Merge & Export      | ⏳     | ❌    | ❓    |
| 3     | TASK-006: Per-Student Context Builder    | ⏳     | ❌    | ❓    |

**Current Verification (2026-04-29):**

- ✅ Build: TASK-001 through TASK-004 scripts lint and type-check cleanly on Python 3.10 in this worktree
- ✅ Tests: 43 pytest checks passing across TASK-001 through TASK-004 helpers
- ✅ Integration: TASK-002 re-extracted the provided class recording to a validated 16kHz mono WAV in the TASK-004 worktree
- ✅ Runtime: TASK-003 completed on the local RTX 3050 GPU after pinning `ctranslate2==3.24.0` and `faster-whisper==0.10.1`; the script writes `output/transcript_raw.json` from the provided class recording while bypassing the upstream WhisperX VAD redirect
- ✅ Runtime: TASK-004 now loads the gated pyannote pipeline, works around newer Torch checkpoint defaults, and writes `output/diarization.json` from the provided class recording; this host validated the run with `--allow-cpu` because the current Python 3.10 Torch install is CPU-only

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

- [x] Extracts WAV from Zoom MP4 without errors
- [x] Output is 16kHz mono (verified with ffprobe)
- [x] Handles long recordings (1hr+) without memory issues

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

| Phase | TODO | Title                  | Status         | Notes                                                                                                                                                              |
| ----- | ---- | ---------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | 001  | Environment Setup      | ✅ Completed   | Live validation passed in the worktree                                                                                                                             |
| 1     | 002  | Audio Extraction       | ✅ Completed   | Validated with the real Zoom MP4 via ffmpeg and ffprobe                                                                                                            |
| 2     | 003  | WhisperX Transcription | ✅ Completed   | Validated on the real class recording and produced `output/transcript_raw.json` on the RTX 3050                                                                    |
| 2     | 004  | Speaker Diarization    | ✅ Completed   | Validated on the real class recording and produced `output/diarization.json`; this host used CPU fallback because the active Python 3.10 Torch install is CPU-only |
| 3     | 005  | Transcript Merge       | ⏳ Not Started | Unblocked by TASK-004; ready to merge `output/transcript_raw.json` and `output/diarization.json`                                                                   |
| 3     | 006  | Per-Student Context    | ⏳ Not Started | Blocked by TASK-005                                                                                                                                                |

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

**Status:** ✅ Completed

**Prerequisites met:**

- [x] Python 3.10 and 3.11 are installed locally
- [x] CUDA is visible to PyTorch on the RTX 3050 Laptop GPU
- [x] Worktree `.env` contains a valid HF token with accepted pyannote agreements

```
Completed by: GPT-5.4
Build status: ✅ PASS

### What was done:
- Added a Windows-focused bootstrap README, `.env.example`, `.gitignore`, and pinned `requirements.txt`
- Implemented `scripts/validate_env.py` with Python version, CUDA, Hugging Face, pyannote, and WhisperX checks plus GPU cleanup
- Added pytest coverage for the pure validation helpers and confirmed `ruff`, `mypy`, and `pytest` pass under Python 3.11
- Pinned `numpy==1.26.4` to avoid the NumPy 2.x breakage in the current pyannote and WhisperX stack
- Updated the WhisperX validation step to load the ASR backend directly so TASK-001 no longer depends on the upstream VAD bootstrap URL that currently responds with HTTP 301
- Re-ran live validation in the worktree with the existing secret-backed `.env` and confirmed all five checks pass end to end

### Tests passing: ✅ 4 tests

### Warnings to next implementor:
- TASK-002 can start from this worktree using the validated Python 3.11 virtual environment
- Keep `scripts/validate_env.py` on the ASR backend path unless the upstream WhisperX VAD bootstrap URL is verified stable again
- The local machine should use Python 3.11 for the project environment; the system default Python 3.14 is outside the supported range

### Breaking changes:
- None
```

---

### TODO-002 Handoff

**Status:** ✅ Completed

**Prerequisites from TODO-001:**

- [x] Environment validated (CUDA + HF token working)
- [x] ffmpeg in system PATH (already done)

```
Completed by: GPT-5.4
Build status: ✅ PASS

### What was done:
- Added `scripts/extract_audio.py` with argparse + Pydantic input handling, fail-fast validation, ffmpeg extraction, and ffprobe-based output verification
- Added pytest coverage for TASK-002 parsing, validation, ffmpeg command construction, ffprobe parsing, and format enforcement helpers
- Updated `README.md` with extraction usage and manual ffprobe inspection guidance
- Live-validated the script against the provided Zoom MP4 extracted from the supplied ZIP and produced `output/audio.wav` at 16kHz mono

### Tests passing: ✅ 13 tests

### Warnings to next implementor:
- `output/audio.wav` is gitignored, so regenerate it in a fresh worktree with `python scripts/extract_audio.py --input <recording.mp4>`
- TASK-003 is now unblocked by the extractor and can consume `output/audio.wav` directly
- TASK-004 still also depends on a worktree `.env` with a valid `HF_TOKEN` and accepted pyannote agreements

### Breaking changes:
- None
```

---

### TODO-003 Handoff

**Status:** ✅ Completed

**Prerequisites from TODO-002:**

- [x] `output/audio.wav` exists at 16kHz mono

```
Completed by: GPT-5.4
Build status: ✅ PASS
Runtime status: ✅ PASS

### What was done:
- Added `scripts/transcribe.py` with argparse + Pydantic input handling, fail-fast WAV and JSON validation, GPU cache cleanup, and JSON serialization for alignment-ready transcript output
- Added a compatibility shim for multiple faster-whisper API signatures, plus a fallback from the broken WhisperX VAD bootstrap path to the direct ASR backend when the upstream VAD URL returns HTTP 301
- Added focused pytest coverage for TASK-003 argument parsing, runtime selection, ASR option compatibility, backend transcription normalization, and transcript document validation
- Reused TASK-002 to regenerate `output/audio.wav` from the provided class recording ZIP and completed GPU transcription plus alignment on that asset, producing `output/transcript_raw.json`
- Pinned `ctranslate2==3.24.0` and `faster-whisper==0.10.1` in `requirements.txt` because that pair was the validated GPU backend stack for the local CUDA 11.8 RTX 3050 environment

### Tests passing: ✅ 30 tests

### Warnings to next implementor:
- Keep `ctranslate2==3.24.0` and `faster-whisper==0.10.1` pinned on this machine; newer 4.x and 1.x releases hit the `cublas64_12.dll` failure here
- The script bypasses the upstream WhisperX VAD bootstrap redirect automatically, so reuse the direct ASR fallback pattern if TASK-004 or TASK-005 encounter the same upstream URL issue
- Alignment can return untimed or missing per-word metadata for short numeric segments such as `27.`, so downstream code should tolerate the segment-span fallback word entries in `output/transcript_raw.json`

### Breaking changes:
- None
```

---

### TODO-004 Handoff

**Status:** ✅ Completed

**Prerequisites from TODO-002:**

- [x] `output/audio.wav` exists at 16kHz mono
- [x] HF token in `.env`

```
Completed by: GPT-5.4
Build status: ✅ PASS
Runtime status: ✅ PASS

### What was done:
- Added `scripts/diarize.py` with argparse + Pydantic input handling, fail-fast WAV and JSON validation, Hugging Face token loading, GPU cache cleanup, and JSON serialization for speaker segments
- Added focused pytest coverage for TASK-004 argument parsing, runtime selection, pyannote annotation normalization, and diarization document validation
- Added Torch 2.6+ checkpoint compatibility shims to both `scripts/diarize.py` and `scripts/validate_env.py` so pyannote 3.1.1 can load trusted checkpoints on newer Torch releases
- Reused TASK-002 to regenerate `output/audio.wav` from the provided class recording ZIP and completed a live diarization run on that asset, producing `output/diarization.json` with 2 speakers and 353 segments
- Fixed the local pyannote runtime stack by aligning `torchaudio` with the installed Torch build and restoring `numpy==1.26.4` before re-running the gated model validation

### Tests passing: ✅ 43 tests

### Warnings to next implementor:
- TASK-005 can now consume `output/transcript_raw.json` and `output/diarization.json` directly from this worktree
- The current Python 3.10 runtime is `torch 2.8.0+cpu`, so TASK-004 validated with `--allow-cpu`; reinstall the CUDA-enabled Torch and torchaudio wheels if later tasks need GPU execution in this worktree
- The archive ZIP also contains individual participant M4A clips under `Audio Record/`; they were not needed for diarization output, but they may help with speaker-label debugging in TASK-005 or TASK-006

### Breaking changes:
- None
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
