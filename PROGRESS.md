# Adira Academy Learning Assistant — Progress Report

**Date:** 2026-05-28  
**Phase:** 1 Pipeline Rebuild (v2) — In Progress  
**Author:** Amritesh Praveen  

---

## What We're Building

A personalized RAG (Retrieval-Augmented Generation) chatbot for ISL students. After every Zoom class, each enrolled student gets their own chatbot that knows:
- What **they** said during class (from their isolated microphone recording)
- What they **missed** (teacher explanations they were present for but didn't engage with)
- The **full class context** (topics covered, examples used)
- A **summary** if they were absent

Students can ask the chatbot "What was the time and work problem we solved today?" or "I missed the last 20 minutes — what did I miss?" and get grounded, transcript-backed answers.

---

## Architecture Overview

```
Zoom .zip export
     │
     ├── Session MP4 (mixed audio of everyone)
     ├── Audio Record/
     │     ├── audioStudentName_RollNo<id>.m4a  ← per-student isolated mic
     │     └── audioNisha<id>.m4a               ← teacher's mic
     └── recording.conf, chat.txt
          │
          ▼
  [ingest_zip]     Extract + classify files + parse roll numbers from filenames
          │
          ▼
  [match_identity] Match M4A files to students via roll number → attendance CSV
          │
          ▼
  [transcribe_dual] WhisperX (small, CUDA) — dual language: Hindi + English
                    Word-level probability merge: picks higher-confidence word at each position
                    Handles Hinglish code-switching ("yeh function ka return type string hai")
          │
          ▼
  [merge_transcripts] Combine session + per-student transcripts
                      Duration-based alignment detection
                      Cluster overlapping speech events → speaker-attributed segments
          │
          ▼
  [build_student_context] Per-student: spoken / present / missed segments + TF-IDF topics
                          Absent students: class summary with topics (still get a chatbot)
          │
          ▼
  [embed_and_store] sentence-transformers (all-MiniLM-L6-v2, 384-dim) → PostgreSQL + pgvector
                    Chunk types: spoken, missed, class_context
                    Stable SHA-1 chunk IDs, stale chunks purged before upsert
          │
          ▼
  [chat.py]  Groq (llama-3.1-8b-instant) + pgvector retrieval → student-scoped chatbot
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Transcription | WhisperX + faster-whisper (CUDA), small model, no alignment models |
| Dual-language | Hindi + English word-level probability merge |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim) |
| Vector DB | PostgreSQL 17 + pgvector (HNSW index, cosine distance) |
| LLM | Groq `llama-3.1-8b-instant` |
| Data models | Pydantic v2 |
| Code quality | ruff, mypy, pytest — 0 errors, 211 tests passing |
| Python | 3.11, CUDA 11.8, RTX 3050 4GB |

---

## What Was Built (Phase 1 v2 Rebuild — TASK-011 to TASK-018)

The original Phase 1 pipeline (TASK-001–006) used pyannote speaker diarization on the mixed session audio. That was inaccurate and unnecessary — Zoom cloud recording already provides per-student isolated M4A files with the student's name and roll number baked into the filename.

### TASK-011 — Cleanup and Foundation Reset
- Deleted pyannote diarization code and all 62 old tests
- Created shared Pydantic models package (`scripts/models/`) used across all pipeline steps
- Replaced pyannote with pgvector in requirements
- Stripped HuggingFace token checks from `validate_env.py`

### TASK-012 — Zip Extraction + File Discovery
- `scripts/ingest_zip.py`: accepts Zoom `.zip` (single or batch directory)
- Extracts to `output/<class>/raw/`, classifies all files by type
- Parses 4-digit roll numbers from M4A filenames using last-underscore algorithm
  - `audioA_Disha_250471031110282.m4a` → name=`A_Disha`, roll=`2504`
  - Handles names with underscores, short numbers, teacher files without roll numbers
- Writes `manifest.json` per class

### TASK-013 — Identity Matching
- `scripts/match_identity.py`: matches M4A files to student identities
- **Primary key**: 4-digit roll number (deterministic, no ML needed)
- **Fallback**: attendance-only matching when no roster CSV provided
- **Teacher detection**: SequenceMatcher fuzzy match at threshold ≥ 0.75
- Detects short-duration entries, duplicate roll numbers, unmatched files
- Writes `identity_map.json`

### TASK-014 — Dual-Language WhisperX Transcription
- `scripts/transcribe_dual.py`: runs WhisperX twice per audio file (hi + en)
- Merges word-level by probability — Hindi preferred on ties (primary class language)
- Scores: `None` / negative coerced to 0.0
- Re-segments merged words by 1.5s gap threshold
- WAV loading via `soundfile` (no ffmpeg dependency at transcription time)
- GPU cleanup between every run for RTX 3050 4GB safety

### TASK-015 — Transcript Merge with Speaker Attribution
- `scripts/merge_transcripts.py`: speaker-attributes the full session timeline
- Duration-based alignment detection (Zoom cloud recordings are always session-aligned)
- Clusters overlapping speech events; primary speaker = longest overlap
- Sequential non-overlapping events within one session segment → split into separate segments
- Fills gaps between student events with session-transcript fallback
- Writes `transcript_merged.json` + `transcript_review.md`

### TASK-016 — Student Context Builder
- `scripts/build_student_context.py`: roster-driven (every enrolled student gets a context)
- **Present students**: spoken segments, present segments, missed segments, attendance window
- **Absent students**: class summary with auto-extracted topics (still get a chatbot)
- Works without a roster CSV by building context from identity_map entries directly
- TF-IDF topic extraction (Hindi + English combined stopwords, 3+ char tokens, bigrams)
- Writes `student_contexts.json` + review artifacts

### TASK-017 — pgvector Migration + Embedding
- `scripts/migrate_db.py`: idempotent DDL (CREATE IF NOT EXISTS) — extension, table, HNSW index
- `scripts/utils/pg_store.py`: `PgVectorStore` class — upsert, delete-by-class, search, get-all
- `scripts/embed_and_store.py`: chunks student contexts → embeds → upserts to pgvector
  - Chunk types: `spoken` (student's own words), `missed` (content they missed), `class_context`
  - SHA-1 stable chunk IDs (deterministic re-runs)
  - Stale chunks purged by class_name before upsert
- Writes `rag_chunks.jsonl` + `rag_chunk_review.csv`

### TASK-018 — Orchestrator + Retrieval/Chat Update
- `scripts/run_pipeline.py`: single CLI orchestrates all steps sequentially
  - Batch mode: processes a directory of `.zip` files, continues on individual failures
  - `--skip-transcribe` flag for re-running merge/context/embed after transcription is done
- `scripts/retrieval.py`: replaced ChromaDB with pgvector (`retrieve_from_pgvector`)
  - Keeps identical `RetrievedChunk` + `RetrievalResult` contracts
- `scripts/chat.py`: swapped `chroma_dir` for `db_url`, all session/Groq logic unchanged
- `scripts/evaluate.py`: updated to pgvector-backed retrieval
- Deprecated old scripts renamed to `.txt`: `merge.py`, `build_context.py`, `chunk_and_embed.py`

---

## Bugs Found and Fixed on Real Data

Testing with real Zoom exports from ISL classes revealed 8 bugs:

| # | Bug | Fix |
|---|-----|-----|
| 1 | Session-level mixed M4A (`audio<id>.m4a`) misclassified as per-student file | Check for `Audio Record/` subdirectory before classifying |
| 2 | Teacher "Nisha" fuzzy-matched to student "A_Disha" (both contain "isha") | Raised teacher threshold from 0.6 → 0.75 |
| 3 | WhisperX VAD bootstrap URL returns HTTP 301 | Use `whisperx.asr.WhisperModel` directly (bypasses VAD download) |
| 4 | `Wav2Vec2Processor.sampling_rate` removed in newer transformers | Dropped alignment step entirely — use faster-whisper word timestamps directly |
| 5 | `whisperx.load_audio` shells out to ffmpeg (not in CMD PATH) | Load WAV with `soundfile` instead (no ffmpeg needed at transcription time) |
| 6 | Windows CMD cp1252 encoding rejects `→`, `✓`, `✗`, `📋`, `🎤` in print/log | Replaced all non-ASCII chars in print/log statements with ASCII equivalents |
| 7 | Alignment detection false positive: text-matching returned 987.5s offset for all students | Duration-based detection first: if both recordings span ±5% of same duration → session_aligned |
| 8 | Whisper small model hallucinates repeated `अपने` on silent/muted student M4As | Hallucination filter: skip segments where one word appears in >70% of positions |

---

## Real Data Observations

**Test class:** Math.01 — Linear Equation Scaffolding: Time and Work (Apr 8, ~17 minutes)  
**Students transcribed:** 7 (A_Disha, A_Jagruti, A_Kalyani, A_Saisha, A_Sanaya, A_Shravani, A_Sonakshi)  
**Teacher:** Nisha  

**Session transcript sample** (shows real class content was captured):
```
[500s] "If you got M and A"
[514s] "Okay. So M. Tell me. How many days? Mohit is finish"
[530s] "10 days. Okay. I am. I'm sorry."
[545s] "Sam. She's only half past. So she's very slow. 20 days."
```

This confirms the dual-language pipeline is capturing the actual math class content (variables M and A, time-and-work problems, student names being called).

**Observations:**
- Session transcript: 57 segments, dominant language English (expected for Hinglish — English captures the structure)
- Per-student M4As: most students were mostly silent (listening mode) — only a few words each
- Silent students trigger Whisper hallucination (`अपने` repeated) → filtered out post-fix
- Topics detected include math vocabulary alongside Hindi filler words

---

## Current Status

| Component | Status |
|-----------|--------|
| Pipeline code (TASK-011–018) | Complete, 211 tests |
| Database schema (pgvector) | Deployed on local PostgreSQL 17 |
| First real class run | In progress (transcription re-running with hallucination filter) |
| Chatbot (chat.py) | Ready, pending first successful embed |
| Evaluation framework | Built (eval_qa.json format) |
| Multi-class batch | Ready (4 class zips available to test) |

---

## What's Next

1. **Verify first full run** — check that 7 student contexts are embedded in pgvector and chatbot answers real questions about the Time & Work class
2. **Run all 4 classes** — Economics, Math Part 04, CTD, and Math Time & Work
3. **Identify answer quality gaps** — run the eval framework against real student questions
4. **Improvements identified so far:**
   - Students who were mostly silent get very sparse contexts (expected limitation of per-student M4A approach)
   - Topic extraction quality improves with longer classes (more content for TF-IDF)
   - Roster CSV would improve absent-student context and attendance window accuracy

---

## Repository

**GitHub:** https://github.com/Amritesh-878/agents-test  
**Branch:** main  
**Commits:** 13 commits since pipeline rebuild started (2026-05-27)
