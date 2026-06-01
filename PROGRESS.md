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
| First real class run | Complete — Math Apr8, 14 chunks in pgvector |
| Chatbot (chat.py) | Ready, pending first successful embed |
| Evaluation framework | Built (eval_qa.json format) |
| Multi-class batch | Ready (4 class zips available to test) |

---

## Security & Audit Remediation

Audit fixes landed so far (see `AUDIT_AND_FIX_PLAN.md`):

- **#1 DB password in chat traces — CLOSED.** `db_url` dropped from session records.
- **#2 secrets via CLI flag — CLOSED.** `DATABASE_URL` env is primary; `--db-url` warns.
- **#7 dead `chromadb` dep — CLOSED.** Removed from `requirements.txt`.
- **#8 chunk-type filter after LIMIT — CLOSED.** Filter pushed into search SQL.
- **#9 model/connection reload per turn — CLOSED.** Embedder + connection reused.
- **#12–#18 cleanups — CLOSED.**
- **Per-student login (#3 access half) — PARTIAL.** `scripts.chat` now authenticates
  against a credentials CSV and scopes retrieval to the logged-in `student_id`; no CLI
  input can redirect it. **Scope is honest and limited:**
  - **#3 is closed for `chat.py` ONLY.** `scripts.retrieval`'s CLI is intentionally left
    as an *unauthenticated local dev tool* that still trusts `--student-id`. Do not expose it.
  - **#4 — CLOSED (both halves), within the filename-trust model.** *Access half:* the
    login CSV rejects duplicate `student_id`s. *Data half:* `match_identity.match_files`
    now FAILS LOUD when two distinct per-student M4As resolve to the same 4-digit roll
    (in both the roster and no-roster/attendance-only paths), so ambiguous identities are
    refused at ingest instead of silently co-mingled. **Caveat:** this closes #4 by
    *refusing* ambiguous input; it does NOT introduce a stable `student_uid` / roster
    reconciliation. Roll numbers are still trusted from filenames, so the deeper "stop
    trusting the filename, use a real uid" hardening remains future work.

- **#6 zip-bomb guard — CLOSED (decompression-bomb half).** `ingest_zip.extract_zip`
  now calls `check_zip_safety` before `extractall`, rejecting archives over the
  entry-count (`MAX_ZIP_ENTRY_COUNT`) or uncompressed-size (`MAX_ZIP_UNCOMPRESSED_BYTES`)
  caps. Classic path-traversal Zip Slip was already mitigated by CPython's zipfile.

- **#5 Groq egress — DISCLOSED, not resolved.** The chat banner and README now state
  that questions + retrieved transcript excerpts are sent to Groq (external US LLM API).
  This is only the *notice*; the retention/consent policy decision remains the owner's
  and is not yet made. Do not treat #5 as fully resolved.

- **#10 dead alignment path — RESOLVED (documented).** `detect_alignment`'s unused
  offset params/constants (`aligned_tolerance`, `similarity_threshold`) were removed and
  the hard "Zoom per-student M4As are always session-aligned, offset is always 0.0"
  assumption is now documented in the function and at the `merge_all` call site. Runtime
  behavior is unchanged (still `session_aligned`); a non-session-aligned Drive source
  would require restoring a validated offset check.

Still open / not started this pass: #11 (inert "missed" trust flag).

---

## What's Next

1. **Run all 4 classes** — Economics (in progress), Math Part 04, CTD
2. **Add roster CSV** — improves absent-student context and attendance window accuracy
3. **Use teacher M4A as primary context source** — cleanest audio, should produce better chunks than mixed session MP4
4. **Evaluate with real student questions** — run eval framework once more classes are loaded

---

## Real Data Results — Math.01 Time & Work (Apr 8, 17 min)

**First end-to-end run with real Zoom exports. 11 bugs found and fixed in the process.**

### Pipeline Run Stats
```
Class:     Math.01_A — Linear Equation Scaffolding: Time and Work
Date:      April 8, 2026 (17 min 7 sec session)
Students:  7 per-student M4As (Disha, Jagruti, Kalyani, Saisha, Sanaya, Shravani, Sonakshi)
Teacher:   Nisha (identified at score=1.00 from filename)
Chunks:    14 stored in pgvector (after quality filter)
Duration:  ~18 min total pipeline (transcription on RTX 3050 CUDA)
```

### Session Transcript — Real Output Examples

The dual-language WhisperX correctly captured the Hinglish class content (English dominant for this class):

**Good segments — real teacher content:**
```
[23s]  "खेल today we हैं। will be going तो forward with the scaffolding, we will
        build our foundation of time and work, we have already built enough foundation"

[53s]  "this is already done, but it includes, it is just scaffolded in a..."

[83s]  "yes, so we already know how this is to be done, only thing..."

[443s] "You are telling me after you told me the second part Jagruti Kalyani
        after you told me day 1 day 2"

[472s] "day 2. You are telling me the second part Jagruti Kalyani day 1"

[786s] "You are telling me the second part Jagruti Kalyani day 1, day 2."
```

**Bad segments — Whisper hallucinating on silence/noise:**
```
[0s]   "you"                          ← student muted, model outputs filler word
[30s]  "you"                          ← same
[141s] "yes, yes, yes, yes, yes, yes" ← looping on background noise
[146s] "अवाईवोद अवाईवोद अवाईवोद..."  ← complete model failure on noise
[368s] "अगर अगर अगर अगर अगर..."       ← another repeated hallucination
```

**The quality filter correctly blocked all hallucinated segments from entering pgvector.**

### Alignment Detection — Before and After

| Student | Run 1 (text matching) | Run 2 (duration-based) |
|---------|----------------------|------------------------|
| A_Disha | join_offset **+987.5s** ❌ | session_aligned 0.0s ✅ |
| A_Jagruti | join_offset **+987.5s** ❌ | session_aligned 0.0s ✅ |
| A_Kalyani | join_offset **+987.5s** ❌ | session_aligned 0.0s ✅ |
| A_Saisha | join_offset **+987.5s** ❌ | session_aligned 0.0s ✅ |
| A_Sanaya | join_offset **+987.5s** ❌ | session_aligned 0.0s ✅ |
| A_Shravani | session_aligned (uncertain) | session_aligned 0.0s ✅ |
| A_Sonakshi | join_offset **+987.5s** ❌ | session_aligned 0.0s ✅ |

The 987.5s false offset was pushing all student events past the session end, causing the merge to produce only 4 segments. After the fix: 30 session segments used correctly.

### Chunk Quality — Before and After Quality Filter

**Before (62 chunks — garbled content in pgvector):**
```
[class_context] "अवाईवोद अवाईवोद अवाईवोद अवाईवोद अवाईवोद अवाईवोद..."  ← GARBAGE
[class_context] "you खेल today we हैं। will be going forward with scaffolding..."  ← GOOD
[spoken]        "तादी सुभ़ पूँच्य। तादी सुभ़ पूँच्य। तादी सुभ़..."  ← GARBAGE
```

**After (14 chunks — quality filter active):**
```
[class_context] "today we will be going forward with the scaffolding, we will
                 build our foundation of time and work, we have already built
                 enough foundation..."  ← GOOD

[class_context] "You are telling me the second part Jagruti Kalyani day 1,
                 day 2. You are telling me the second part Jagruti Kalyani..."  ← GOOD
```

### Retrieval Test — Real Question Against pgvector

**Query:** "What was the time and work problem covered in class today?"
**Student:** A_Disha (roll 2504)

```
[class_context] score=0.552
  "today we will be going forward with the scaffolding, we will build our
   foundation of time and work, we have already built enough foundation..."

[class_context] score=0.552
  "You are telling me the second part Jagruti Kalyani day 1, day 2..."

[class_context] score=0.534
  "I am going to paste you the whiteboard itself, so that you all can see..."
```

Retrieval is working semantically — the top results are genuinely relevant to the question.

### Topics Extracted (TF-IDF)
```
['day', 'yes', 'second', 'telling', 'jagruti', 'jagruti kalyani',
 'kalyani day', 'day day', 'second jagruti', 'kalyani']
```

Topics reflect real class content (Jagruti and Kalyani being quizzed on day 1/day 2 time & work problems). Quality limited by short class duration (17 min) and noisy transcript.

### Key Finding: Small Model on Mixed Audio = Noisy Transcripts

The **WhisperX small model** struggles with:
- Mixed session audio (all students' mics combined → echo + overlap)
- Students who are muted/listening → hallucinated filler words ("you", "yes yes yes")
- Hinglish code-switching under noise → garbled transliterations

**What's salvageable:** The teacher's actual explanation sentences are captured correctly and retrievable. Roughly 5-7 clean segments per 17-minute class survived the quality filter.

**Improvement path:** Use the **teacher's isolated M4A** (audioNisha.m4a) as the primary class context source. Her mic records only her voice → much cleaner audio → better transcription → more chunks per class.

---

## Real Data Results — Economics.02 Supply Function (Apr 16)

**This run confirms the pipeline scales to longer classes and produces significantly better RAG content.**

### Pipeline Run Stats
```
Class:     Economics.02 — Supply Function
Date:      April 16, 2026 (51 min 7 sec session)
Students:  1 per-student M4A (Bhagyashree, roll 2302)
Teacher:   Nisha (score=1.00)
Chunks:    192 stored in pgvector (vs 14 for 17-min Math class)
Duration:  ~32 min pipeline (51-min audio x 3 files x 2 languages on RTX 3050)
```

### Chunk Counts vs Math Class

| Class | Duration | Students | Chunks | Quality |
|-------|----------|----------|--------|---------|
| Math Time & Work | 17 min | 7 (mostly silent) | 14 | Limited — short class, students in listening mode |
| Economics Supply Function | 51 min | 1 (active) | 192 | Good — longer class, student actively spoke |

**Finding: Low Math chunk count was class-specific, not a pipeline bug.** A 51-minute Economics class with an active student produces 14x more chunks. Longer sessions with active student participation = richer RAG context.

### Session Transcript — Real Output Examples

**Good segments — real economics content captured:**
```
[class_context]
  "is a constant. Yeah. A is also called an intercept. Okay. That's no direct
   relation. Like in this function, it is constant. It is not related to x. Yeah.
   What happens is when exchanges, y w..."

[class_context]
  "that is like a price and quantity of supply. price में and quantity, how you
   express it is, it is a table representation of quantity, supply different"

[class_context]
  "So let us revise the concepts which we have discussed in the class last week.
   So this is going to be the last class or you will take at least one class,
   more likely I have to show you some simulation"

[class_context]
  "quantity supply is zero. Can you derive it? Okay. We also have to find beta
   right? Yes. What does the supply function talking to you about?"
```

**Bhagyashree's spoken chunk (she answered in class):**
```
[spoken]
  "Determinant of a supply is what changes supply. The factor."
```

### Topics Extracted (TF-IDF)
```
['yeah', 'okay', 'like', 'supply', 'price', 'beta', 'function', 'yes', 'right', 'called']
```

Real domain terms: **supply, price, beta, function** (intercept was called beta in class). The `yeah`, `okay`, `like` are filler words — expected for a conversational class. This is a clear improvement over Math's topics which were mostly student names.

### Retrieval Test — Real Question

**Query:** "What is the supply function and how is the intercept defined?"
**Student:** Bhagyashree (roll 2302)

```
[class_context] score=0.701
  "We also have to find beta right? Yes. What does the supply function talking
   to you about or any function A? There is any quantity what is it talking
   about? Like if you solve the right hand side of the equation you should
   be able to get the quantity supply..."

[class_context] score=0.699
  "quantity supply is zero. Can you derive it? Okay. We also have to find beta
   right? Yes. What does the supply function talking to you about? Like if you
   solve the right hand side of the equation you should be able to find the
   quantity supply..."

[spoken]  score=0.693
  "Determinant of a supply is what changes supply. The factor."
          ← Bhagyashree's OWN ANSWER from class retrieved here
```

**Retrieval scores: 0.69–0.70** (vs 0.55 for Math). The chatbot calling Groq with this context would produce a highly grounded, personalized answer. Crucially, the student's own spoken contribution is surfaced back to her.

### Confirmed: Pipeline Works, Quality Scales with Class Length

| Metric | Math (17 min) | Economics (51 min) |
|--------|--------------|-------------------|
| Chunks embedded | 14 | 192 |
| Top retrieval score | 0.55 | 0.70 |
| Student spoken chunks | 0 useful | 1 ("Determinant of supply...") |
| Topics quality | Student names + noise | Real domain terms (supply, price, beta) |
| Alignment | All session_aligned ✅ | session_aligned ✅ |

---

## Bugs Found and Fixed on Real Data (Complete List)

| # | Where Found | Bug | Fix | Commit |
|---|-------------|-----|-----|--------|
| 1 | Ingest | Session M4A misclassified as per-student | Check for `Audio Record/` directory | `fad4147` |
| 2 | Identity | Teacher "Nisha" matched student "A_Disha" (shared "isha") | Threshold 0.6 → 0.75 | `240aa39` |
| 3 | Transcription | WhisperX VAD bootstrap URL returns HTTP 301 | Use `whisperx.asr.WhisperModel` directly | `90a4667` |
| 4 | Transcription | `Wav2Vec2Processor.sampling_rate` removed in newer transformers | Drop alignment step, use faster-whisper word timestamps | `93b61d7` |
| 5 | Transcription | `whisperx.load_audio` shells out to ffmpeg (not in CMD PATH) | Load WAV with `soundfile` (no ffmpeg needed) | `a236b84` |
| 6 | All scripts | Windows CMD cp1252 rejects `→`, `✓`, `✗`, `📋`, `🎤` in print/log | Replace all non-ASCII with ASCII equivalents | `a3c02ef` |
| 7 | Merge | Alignment text-matching returned false 987.5s offset for all students | Duration-based detection: if ±5% same duration → session_aligned | `f7d90f9` |
| 8 | Transcription | Whisper hallucinates `अपने अपने...` on silent/muted student M4As | Skip segments where one word >70% of total | `f7d90f9` |
| 9 | Context builder | 0 students embedded when no roster CSV provided | Also iterate `identity_map.entries` (attendance-matched students) | `f7d90f9` |
| 10 | Embed | Garbled chunks (phrase repetition, replacement chars) in pgvector | Pre-embed quality filter: trigram repetition + `�` ratio check | `3e56131` |
| 11 | Embed | Short/garbled segments diluting good content in class_context chunks | Filter individual segments before concatenating for chunking | `b9418c3` |

---

## Repository

**GitHub:** https://github.com/Amritesh-878/agents-test  
**Branch:** main  
**Commits:** 17 commits since pipeline rebuild started (2026-05-27)
