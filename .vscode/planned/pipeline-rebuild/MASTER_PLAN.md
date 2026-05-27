# Master Implementation Plan — Pipeline Rebuild

**Project:** Adira Academy Learning Assistant — Phase 1 Pipeline Rebuild

**Date Created:** 2026-05-27

---

## Overview

Rebuild the Phase 1 ingestion pipeline to use Zoom cloud recording per-student M4A files for speaker identity instead of pyannote diarization. Add dual-language WhisperX transcription for Hinglish (Hindi+English), migrate storage from ChromaDB to PostgreSQL + pgvector, and support batch processing of multiple class recordings.

**Status:** Planned

**Tasks:**

| Phase | Task                                          | Status  | Depends On         |
| ----- | --------------------------------------------- | ------- | ------------------ |
| 1     | TASK-011: Cleanup and Foundation Reset        | Done    | —                  |
| 2     | TASK-012: Zip Extraction + File Discovery     | Done    | 011                |
| 2     | TASK-013: Identity Matching                   | Done    | 011, 012           |
| 2     | TASK-014: Dual-Language WhisperX              | Done    | 011, 012           |
| 3     | TASK-015: Transcript Merge                    | Done    | 013, 014           |
| 3     | TASK-016: Student Context Builder             | Planned | 013, 015           |
| 4     | TASK-017: pgvector Migration + Embedding      | Planned | 011, 016           |
| 4     | TASK-018: Orchestrator + Retrieval/Chat       | Planned | 011–017            |

---

## Why This Rebuild?

The current pipeline ignores the per-student M4A audio files that Zoom cloud recording already provides. Instead, it runs pyannote speaker diarization on the mixed session audio — which is inaccurate and unnecessary. The per-student M4As contain:

1. **Student identity** baked into filenames (name + roll number)
2. **Clean isolated audio** per student (better transcription quality)
3. **Ground-truth speaker attribution** (no probabilistic diarization needed)

This rebuild also addresses:
- **Hinglish**: Classes are in Hindi+English. The current single-language transcription mangles one language.
- **ChromaDB limitations**: Moving to PostgreSQL + pgvector for production-grade vector storage.
- **Batch processing**: Need to process multiple class recordings, not just one.
- **Master roster**: Every enrolled student gets a chatbot, even if absent.

---

## Design Decisions

| Decision | Choice |
|----------|--------|
| Input format | Zoom .zip per class, auto-extracted |
| Batch mode | Multiple zips, sequential (VRAM safety on RTX 3050 4GB) |
| Attendance CSV | `Name (ori Email)`, `Total dura`, `Guest` — students as `Name_RollNo` |
| Master roster | Separate CSV: Name, RollNo, Email |
| Teacher ID | CLI flag `--teacher "Name"` |
| Identity match key | 4-digit roll number (primary), fuzzy name (fallback) |
| Source audio | Extract WAV from MP4 via ffmpeg |
| Language | Dual WhisperX (hi + en), word-level probability merge |
| Per-student text | Canonical (replaces mixed audio). Mixed is fallback. |
| Timestamp alignment | Detect session-aligned vs join-offset automatically |
| Multi-speaker | Tag segment with all overlapping speakers |
| Absent students | Summary only (date, duration, teacher, TF-IDF topics) |
| Topic extraction | TF-IDF keyword-based (no LLM, no API) |
| Vector DB | PostgreSQL + pgvector, local native install |
| Embedding | sentence-transformers (unchanged) |
| LLM | Groq (unchanged) |
| DB schema | Separate migration script |
| Scripts | Individual per step + orchestrator |
| Output | `output/<class_name>/` with standard filenames |
| Review artifacts | Keep .md and .csv human-review files |
| Tests | Clean break — delete all 62 old tests, write ~155 new |

---

## Dependency Graph

```
TASK-011 (Cleanup + Foundation)
    |
    +---> TASK-012 (Zip Extraction + File Discovery)
    |         |
    |         +---> TASK-013 (Identity Matching)
    |         |         |
    |         |         +---> TASK-015 (Transcript Merge) ----+
    |         |                                                |
    |         +---> TASK-014 (Dual-Language Transcription) ---+
    |                                                         |
    |                              +--------------------------+
    |                              v
    |                      TASK-016 (Student Context Builder)
    |                              |
    |                              v
    +---> TASK-017 (pgvector Migration + Embedding)
                                   |
                                   v
                           TASK-018 (Orchestrator + Retrieval/Chat Update)
```

TASK-013 and TASK-014 can proceed in parallel (both depend on TASK-012, not on each other). TASK-015 requires both.

---

## Output Directory Structure

```
output/<class_name>/
    raw/                          # Extracted zip contents
    manifest.json                 # TASK-012: file discovery
    identity_map.json             # TASK-013: student-audio matching
    transcripts/
        session.json              # TASK-014: full-session dual-language
        audio<Name>_<ID>.json     # TASK-014: per-student dual-language
    transcript_merged.json        # TASK-015: speaker-attributed merged
    transcript_review.md          # TASK-015: human review
    student_contexts.json         # TASK-016: present + absent contexts
    student_context_review.md     # TASK-016: human review
    student_context_segments.csv  # TASK-016: flat review
    rag_chunks.jsonl              # TASK-017: chunks
    rag_chunk_review.csv          # TASK-017: review
    pipeline_report.json          # TASK-018: orchestrator report
```

---

## Critical Risks

| Risk | Mitigation |
|------|------------|
| Dual-language merge worse than single | `--single-language` fallback flag; compare on 5-min clip first |
| Per-student M4A timestamps not session-aligned | Alignment detection in TASK-015; log mode detected |
| RTX 3050 4GB OOM on dual-language | One file at a time, GPU cleanup, WhisperX small model |
| Roll number extraction edge cases | Conservative regex; log all extractions; manual override CSV |
| pgvector install on Windows | `psycopg[binary]`; documented steps; Docker fallback |
| Hinglish TF-IDF low quality | Combined stopwords; bigram/trigram; top 10 only, flagged |

---

## Verification

After each task: `ruff check --fix . && mypy . && pytest` — 0 errors, 0 warnings, 100% pass.

End-to-end after TASK-018:
1. `python scripts/run_pipeline.py --input <zips_dir> --teacher "Name" --roster roster.csv --attendance attendance.csv --db-url postgresql://...`
2. Verify all output directories populated
3. `python scripts/chat.py` against pgvector — verify retrieval
4. Verify absent students have summary-only context with topics
