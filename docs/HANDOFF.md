# Handoff Notes — Adira Academy Learning Assistant

**For:** The next agent building the Google Drive ingestion pipeline.
**Date:** 2026-05-28
**Updated:** 2026-06-02 — security/audit fix-plan steps 1–6 implemented and the per-student
login system built. See `PROGRESS.md` → "Security & Audit Remediation" for the full status.
**Repo:** https://github.com/Amritesh-878/agents-test (branch `main`)

---

## 1. The End Goal (read this first)

A **personalized chatbot for each student**. After every Zoom class, a student can ask:
- "What did we cover today?"
- "I was absent — what did I miss?"
- "What did the teacher say about supply curves?"

The chatbot answers **only from that student's own class data** (their spoken words, what they were present for, what they missed, and class-wide context), grounded strictly in transcripts — never hallucinated.

This already works end-to-end for the zip-based pipeline. **Your job is to put a Google Drive ingestion front-end on it** so classes flow in automatically instead of being run by hand.

---

## 2. What Already Exists and Works (do not rebuild)

The full processing pipeline is **done, tested (263 tests passing), and validated on two real classes.** It lives in `scripts/` and is orchestrated by `scripts/run_pipeline.py`.

### The pipeline (all working)
```
Zoom .zip  →  ingest_zip  →  match_identity  →  transcribe_dual  →
merge_transcripts  →  build_student_context  →  embed_and_store (pgvector)  →  chat
```

| Script | What it does | Status |
|--------|--------------|--------|
| `ingest_zip.py` | Extracts Zoom .zip, classifies files, parses roll numbers from M4A filenames | ✅ Works |
| `match_identity.py` | Matches per-student M4A → student via roll number; identifies teacher | ✅ Works |
| `transcribe_dual.py` | WhisperX dual-language (Hindi+English) per audio file, GPU | ✅ Works |
| `merge_transcripts.py` | Speaker-attributed merge of session + per-student transcripts | ✅ Works |
| `build_student_context.py` | Per-student spoken/present/missed segments + TF-IDF topics | ✅ Works |
| `migrate_db.py` | Idempotent pgvector schema (extension, table, HNSW index) | ✅ Works |
| `embed_and_store.py` | Chunks + embeds (all-MiniLM-L6-v2) → pgvector, quality-filtered | ✅ Works |
| `retrieval.py` | Student-scoped pgvector similarity search (now an **unauthenticated local dev tool** — still takes `--student-id`) | ✅ Works |
| `auth.py` | `AuthService` — CSV-backed, constant-time login used by `chat.py` | ✅ Works |
| `chat.py` | CLI chatbot: **login-gated** (id+password vs credentials CSV), Groq llama-3.1-8b + pgvector retrieval | ✅ Works |
| `generate_session_report.py` | Per-class engagement report (Nisha-approved format) | ✅ Works |
| `run_pipeline.py` | Orchestrates all steps, batch mode, per-class failure isolation | ✅ Works |

> **Security hardening since the audit (steps 1–6, all landed — see `AUDIT_AND_FIX_PLAN.md` / `PROGRESS.md`):**
> per-student **login** gates `chat.py` (student_id derived from auth, never CLI); `match_identity`
> now **fails loud** if two M4As share a 4-digit roll; `ingest_zip` has a **zip-bomb guard**
> (entry-count + uncompressed-size caps) before `extractall`; `--db-url` warns and `DATABASE_URL`
> env is preferred; Groq third-party egress is disclosed in the chat banner. **Relevant to your
> phase:** untrusted Drive zips now pass through the zip-bomb guard, and colliding-roll zips will
> fail the ingest step loudly rather than silently co-mingle students.

### Verified results on real data

**Economics.02 — Supply Function (51 min, 1 student, Bhagyashree):**
- 192 chunks embedded; retrieval score **0.70** on "What is the supply function?"
- Student's own answer retrieved: *"Determinant of a supply is what changes supply."*
- Full Q&A arc captured (definitions → determinants → MC=MR → deriving Qs = -15 + 3P → graphing)

**Math.01 — Time & Work (17 min, 7 students):**
- 14 chunks; students were silent listeners (correctly detected as "Silent" engagement)
- Confirms chunk count scales with class length + student participation, not a bug

### Stack
- Python 3.11, WhisperX + faster-whisper (CUDA 11.8, RTX 3050 4GB), `small` model
- PostgreSQL 17 + pgvector (HNSW, cosine), sentence-transformers all-MiniLM-L6-v2 (384-dim)
- Groq llama-3.1-8b-instant, Pydantic v2, ruff + mypy + pytest

### Run it (current manual flow)
```powershell
# Prefer setting DATABASE_URL in .env over passing --db-url (a secret-bearing flag warns).
python -m scripts.migrate_db            # reads DATABASE_URL from .env
python -m scripts.run_pipeline --input "path\to\class.zip" --output-dir output/ `
  --teacher "Nisha" --attendance "attendance.csv"
# chat.py is login-gated: it prompts for student id + password (verified against the
# credentials CSV) and scopes retrieval to the logged-in id. No --student-id flag.
python -m scripts.chat --credentials data/credentials.csv
```

---

## 3. Your Task — Google Drive Ingestion Pipeline

> **STATUS (2026-06-02): ingestion code BUILT and gated green (289 tests).** The
> runtime-independent Python — `processed_files` migration + `ProcessedFilesStore`,
> `scripts/drive_sync.py` (`DriveSyncService` + `GoogleDriveClient`), roster wiring, and
> pinned `google-api-python-client` / `google-auth` — is done. See `PROGRESS.md` →
> "Google Drive Ingestion Front-End" for specifics. **Runtime decided + built:**
> `.github/workflows/drive-sync.yml` runs the full pipeline on a **self-hosted runner on
> the RTX 3050 box** (owner-approved interim; the GPU half moves to AWS later). Only
> runner provisioning + repo secrets remain an owner setup task. The design notes below
> are retained as the original spec.

### Confirmed design decisions (from project owner)

| Decision | Answer |
|----------|--------|
| **What's in Drive** | Zoom **.zip** exports — same format the existing pipeline already handles |
| **Auth** | **Service account** (JSON key). Drive folder shared with the service account email |
| **"Docs" output** | The **pgvector RAG store** is the deliverable (powers the chatbot). No separate doc export needed |
| **Trigger + dedup** | **Scheduled poll** + a `processed_files` Postgres table keyed by Drive file ID. Idempotent |
| **Query scope** | Chatbot answers across **all of a student's classes** (current `student_id`-only filter already does this) |
| **Roster (absent students)** | **In scope.** Owner will provide tools/data to build the master roster CSV. Absent-student chatbot depends on it |
| **Runtime** | **GitHub Actions** ⚠️ (see critical note below) |

### The build, concretely

1. **`scripts/drive_sync.py`** (new)
   - Authenticate with a service-account JSON key (`google-api-python-client`, `google-auth`)
   - List the configured Drive folder; filter to `*.zip`
   - For each zip whose Drive file ID is **not** in the `processed_files` table:
     - Download to a temp dir
     - Call the existing `run_pipeline.process_single_class()` (or shell `run_pipeline.py`)
     - On success, insert `{drive_file_id, class_name, processed_at}` into `processed_files`
   - Continue on per-file failure (don't let one bad zip stop the batch — `run_pipeline` already isolates failures)

2. **DB migration addition** — add a `processed_files` table to `migrate_db.py`:
   ```sql
   CREATE TABLE IF NOT EXISTS processed_files (
     drive_file_id TEXT PRIMARY KEY,
     class_name TEXT NOT NULL,
     processed_at TIMESTAMPTZ DEFAULT NOW()
   );
   ```

3. **Roster ingestion** — owner provides tooling/data. Plan for a `roster.csv` (Name, RollNo, Email) either in the Drive folder or a fixed path, passed to `run_pipeline.py --roster`. This unblocks the absent-student feature (`build_student_context.py` already supports `--roster`; without it, absent students get no context object).

4. **Tests** — follow the existing pattern: mock the Drive API client (like `test_pg_store.py` mocks psycopg). Test dedup logic, download flow, processed_files insertion, failure isolation. Keep ruff + mypy clean, all tests passing.

### ⚠️ CRITICAL: GitHub Actions has no GPU

Standard GitHub Actions runners are **CPU-only**. WhisperX dual-language transcription is the slow, GPU-bound step (~20–35 min/class on the RTX 3050; **hours** on CPU). You **must** resolve this before the Actions plan is viable. Options, best first:

1. **Self-hosted GitHub Actions runner on the RTX 3050 machine** (recommended) — reconciles "GitHub Actions" with "needs a GPU." The workflow triggers on schedule, runs on the self-hosted runner that has CUDA + the local Postgres. Cleanest fit.
2. **Split the pipeline** — run lightweight steps (Drive sync, dedup, embedding, reporting) on GitHub-hosted runners; run only transcription on a self-hosted GPU runner or a separate GPU box.
3. **GPU cloud runner** — GitHub's larger GPU runners or an external GPU service. Cost + complexity.
4. **CPU-only with `--allow-cpu`** — works today but impractically slow for production volume.

**Recommendation:** self-hosted runner on the existing machine. It also means Postgres stays `localhost` and secrets stay on that box.

### Secrets needed in GitHub Actions
- `GOOGLE_SERVICE_ACCOUNT_JSON` (Drive auth)
- `DATABASE_URL` (Postgres connection)
- `GROQ_API_KEY` (only if the workflow also runs chat/eval)
- `HF_TOKEN` (WhisperX model downloads)

These map to the current `.env` keys. **Never commit them** — `.env` is gitignored; use GitHub repository secrets.

---

## 4. Known Limitations & Quality Backlog (context, not blockers)

1. **Transcript quality ceiling.** WhisperX `small` on *mixed session audio* is noisy on Hinglish + background noise. The biggest single win: **use the teacher's isolated M4A** (e.g. `audioNisha*.m4a`) as the primary class-context source instead of the mixed session MP4. Her mic is clean → far better transcripts. Not yet implemented.
2. **Whisper hallucination on silence** — muted students produce repeated junk (`अपने अपने…`). Filtered by `is_hallucinated_segment` (transcribe) and `is_quality_text` (embed), but multi-word garble can still slip through.
3. **Attendance CSV is full-day, multi-class.** The Zoom export covers the entire day's meeting, not one class. **Per-student M4A files in the zip are the ground truth** for who was in a given class. Reports treat the CSV as an appendix only.
4. **No roster yet** → absent-student chatbot is blocked until the roster CSV exists (now in your scope).
5. **Identity still trusts the filename roll.** The colliding-roll guard refuses ambiguous input, but there is no stable `student_uid` / roster reconciliation yet — a misnamed/spoofed M4A can still misattribute a recording. Deeper hardening (#4 in the audit plan) is future work.

> Resolved since the audit (no longer limitations): `chromadb` removed from `requirements.txt`;
> dead v1 `scripts/transcribe.py` retired to `scripts/transcribe.py.txt`.

---

## 5. Pointers
- `README.md` — setup + usage for the current pipeline
- `PROGRESS.md` — detailed results, real output examples, full bug history
- `AUDITCONTEXT.md` — companion doc for the code/security auditor
- `AUDIT_AND_FIX_PLAN.md` — audit findings + fix order + the per-student login design. **Steps 1–6 are now implemented** (login built); see its status banner and `PROGRESS.md` for what's closed vs. still open before touching security/auth
- `.vscode/planned/pipeline-rebuild/` — TASK-011 through TASK-018 specs (how the current pipeline was designed)
- Data models: `scripts/models/` (transcript, identity, context, pipeline)

**Commit style for this repo:** short one-line messages, no body, no Co-Authored-By line.
