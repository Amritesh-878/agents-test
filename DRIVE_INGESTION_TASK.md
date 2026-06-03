# Task — Google Drive Ingestion Pipeline

**For:** the next agent. **Created:** 2026-06-02. **Source of design:** `HANDOFF.md` §3.

Read `HANDOFF.md` (especially §2 "what already exists" and §3 "Your Task — Google Drive
Ingestion Pipeline") and `CLAUDE.md`. Also skim `AUDIT_AND_FIX_PLAN.md`'s status banner and
`PROGRESS.md` → "Security & Audit Remediation" so you know what's already hardened. The full
zip→transcript→context→pgvector→chat pipeline is DONE (263 tests green) and orchestrated by
`scripts/run_pipeline.py` — do NOT rebuild it. Your job is to put a Google Drive ingestion
front-end on it so classes flow in automatically.

## §0 Ground rules (non-negotiable, from `CLAUDE.md`)

- Gate after EVERY change, in this order: `python -m ruff check --fix .` → `python -m mypy .`
  → `python -m pytest`. 0 errors / 0 warnings / 100% passing is the only acceptable result.
  Never run tests before lint+typecheck pass. Don't comment out code/tests or disable rules
  to pass — fix the cause.
- Type hints everywhere; `from __future__ import annotations`. Domain logic = classes,
  utilities = plain functions. No bare `except:`.
- One logical change per commit, short ONE-LINE messages, no body, no `Co-Authored-By` line.
- Add/expand tests for every behavioral change. Mock the Drive API client the way
  `tests/test_pg_store.py` mocks psycopg — no real network/Drive calls in tests.
- Secrets come from `.env` / env vars, never CLI flags. Reuse
  `scripts/utils/db_url.resolve_db_url` for `DATABASE_URL`. The service-account JSON path and
  Drive folder id also come from env.
- Pre-release alpha: breaking changes allowed, no backward-compat shims.
- Be honest about scope in docstrings/PROGRESS — don't mark anything "done" if it isn't.

## Design (confirmed in `HANDOFF.md` §3 — don't re-litigate)

Drive holds Zoom `.zip` exports; service-account JSON auth; scheduled poll + a
`processed_files` Postgres table keyed by Drive file id for idempotent dedup; the pgvector RAG
store is the deliverable; roster is in scope.

## Step 0 — Runtime recommendation (PLAN FIRST, do not build CI yet)

GitHub Actions runners are CPU-only and WhisperX transcription is GPU-bound. Evaluate the
options (self-hosted runner on the RTX 3050 box; split pipeline lightweight-on-Actions +
GPU-transcription-self-hosted; GPU cloud; CPU-only) and write a short recommendation with
trade-offs. PRESENT it and STOP for owner approval before writing any GitHub Actions workflow
YAML or runner provisioning. The Python ingestion code (Steps 1–4) is runtime-independent —
build it regardless of this decision.

## Step 1 — `processed_files` migration

Add to `scripts/migrate_db.py` (idempotent `CREATE TABLE IF NOT EXISTS`), keyed by
`drive_file_id` PRIMARY KEY, with `class_name` + `processed_at`. Add dedup access following the
`pg_store` raw-SQL pattern (parameterized, no f-string SQL). Tests for the DDL + insert/exists.

## Step 2 — `scripts/drive_sync.py` (new)

`DriveSyncService` class:

- Authenticate with a service-account JSON key (`google-api-python-client` + `google-auth`),
  path from env.
- List the configured Drive folder; filter to `*.zip`.
- For each zip whose Drive file id is NOT already in `processed_files`: download to a temp dir,
  call `run_pipeline.process_single_class()`, and on success record `{drive_file_id,
  class_name, processed_at}`. Clean up the temp download.
- Continue on per-file failure (one bad zip must not stop the batch). EXPECT and handle two
  real failure modes the pipeline now raises: the **zip-bomb guard** (`ingest_zip` rejects
  oversized / too-many-entry archives) and the **colliding-roll guard** (`match_identity`
  fails loud when two M4As share a 4-digit roll). A failed file must NOT be marked processed.
- Tests: mock the Drive client + `process_single_class`. Cover dedup-skip, download+process+
  record happy path, `processed_files` insertion, and failure isolation (failed file not
  recorded, batch continues).

## Step 3 — Roster ingestion

Wire a `roster.csv` (Name, RollNo, Email) path (env or fixed) through to `run_pipeline`'s
`--roster` so absent-student context works. Tests for the wiring.

## Step 4 — Dependencies + docs

Pin `google-api-python-client` and `google-auth` in `requirements.txt` (verify mypy stays
clean; prefer targeted inline ignores over project-wide if stubs are missing). Update README (a
Drive-sync usage section) and PROGRESS/HANDOFF to reflect what's built, with honest scope (what
is automated vs. still manual, and the runtime decision status).

## Done criteria

Run the full gate one final time, then STOP and give a commit-by-commit summary with the gate
result. Do NOT touch the existing pipeline internals beyond what's needed to call them, and do
NOT build the Actions workflow until Step 0 is approved.

## Secrets (map to current `.env` keys; use GitHub repository secrets in CI, never commit)

- `GOOGLE_SERVICE_ACCOUNT_JSON` (Drive auth)
- `DATABASE_URL` (Postgres)
- `GROQ_API_KEY` (only if the workflow also runs chat/eval)
- `HF_TOKEN` (WhisperX model downloads)
