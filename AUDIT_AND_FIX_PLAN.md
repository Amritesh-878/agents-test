# Audit & Fix Plan — Adira Academy Learning Assistant

> **STATUS (2026-06-02): steps 1–6 implemented and verified (263 tests green).** Findings
> #1, #2, #7, #8, #9, #12–#18 are **closed**; #3 closed for `chat.py` (retrieval.py left as an
> unauthenticated dev tool); #4 closed **within the filename-trust model** (colliding-roll guard
> + login dup-id check), stable-`student_uid` hardening still future; #6 closed (decompression-bomb
> half); #5 **disclosed, not policy-resolved**; #10/#11 resolved. The §4 per-student login system
> is **built**. Remaining/out-of-scope: #4 stable-uid, #5 retention/consent decision, Google Drive
> ingestion (see `HANDOFF.md`). The findings/plan below are the **original** text and are not
> rewritten — see `PROGRESS.md` → "Security & Audit Remediation" for per-finding status.

**Audience:** the engineering agent who will implement these fixes.
**Author:** security/code audit pass, 2026-06-02.
**Status of repo:** pre-release **alpha, local-only**, processing **real student PII**.

> Read this whole file first. It is self-contained: project context, audit findings,
> fix order, and the new per-student login design are all here. Companion docs you
> should also skim: [`AUDITCONTEXT.md`](AUDITCONTEXT.md), [`README.md`](README.md),
> [`HANDOFF.md`](HANDOFF.md), [`PROGRESS.md`](PROGRESS.md), and the rules in
> [`CLAUDE.md`](CLAUDE.md) (these OVERRIDE defaults — follow them exactly).

---

## 0. Ground rules for the implementing agent

These come from [`CLAUDE.md`](CLAUDE.md) and the repo's established conventions. **Non-negotiable:**

- **Quality gate, in this exact order, after every change:**
  ```sh
  python -m ruff check --fix .   # must end at 0 errors
  python -m mypy .               # must be 0 errors, 0 warnings
  python -m pytest               # must be 100% passing
  ```
  Never run tests before lint + typecheck pass clean. 0 errors / 100% passing is the
  *only* acceptable result. Do **not** comment out code or tests, skip tests, or
  disable rules project-wide to make checks pass — fix the underlying issue.
- **Type hints everywhere** (params + returns). `from __future__ import annotations`
  is already used across the codebase; keep it. Use `Optional[T]` / `| None` for
  nullables, `isinstance()` for narrowing — no bare `except:`.
- **Domain logic = classes; utilities = plain functions.** No modules-as-namespaces,
  no static-method utility classes.
- **Tests are mandatory**, even when hard. Mock heavy deps the way existing tests do
  (e.g. `test_pg_store.py` mocks psycopg; transcription/embedding GPU paths are mocked).
  Write whatever fixtures/mocks are needed to mimic real usage.
- **Commit style:** short **one-line** messages, no body, **no `Co-Authored-By` line**.
- **Do not edit [`AGENTS.md`](AGENTS.md)** without explicit permission.
- Pre-release alpha: **breaking changes are allowed.** Do not add backward-compat shims.
- Prefer `git mv` to preserve history; to "delete" a file the repo convention is to
  rename it to `.txt`.
- End every response with a summary of file changes.

---

## 1. What this project is (so you have full context)

**Adira Academy Learning Assistant** turns Zoom cloud class recordings into a
**personalized RAG chatbot for each student**. After a class, a student asks things like
"what did we cover today?", "I was absent — what did I miss?", "what did the teacher say
about supply curves?" and gets answers grounded **only in that student's own class data**
(their spoken words, what they were present for / missed, and class-wide context) — never
hallucinated.

### Data flow (untrusted input → storage → query)
```
Zoom .zip (per-student isolated M4A mics + mixed session MP4 + attendance)
  → scripts/ingest_zip.py         extract zip, classify files, parse 4-digit roll from M4A filename
  → scripts/match_identity.py     map M4A → student via roll number; detect teacher (fuzzy ≥0.75)
  → scripts/transcribe_dual.py    WhisperX small, CUDA, run twice (hi+en), word-prob merge
  → scripts/merge_transcripts.py  speaker-attributed merged timeline (per-student is canonical)
  → scripts/build_student_context.py  per-student spoken/present/missed segments + TF-IDF topics
  → scripts/embed_and_store.py    all-MiniLM-L6-v2 (384-dim) → PostgreSQL/pgvector (raw SQL, psycopg3)
  → scripts/retrieval.py / chat.py  query embed → pgvector similarity → Groq llama-3.1-8b-instant
```

### Stack
- Python 3.11, Pydantic v2 (all data contracts), argparse CLIs.
- WhisperX + faster-whisper (CUDA 11.8, RTX 3050 4GB), `small` model, no alignment models.
- PostgreSQL 17 + pgvector (HNSW, cosine), `psycopg[binary]>=3.1`, `pgvector>=0.2`.
- sentence-transformers `all-MiniLM-L6-v2` (384-dim).
- Groq SDK (`llama-3.1-8b-instant`) — **external** LLM API.
- ffmpeg via `subprocess` (argv list, no shell).
- ruff + mypy + pytest (216 tests, currently all green).
- Secrets in `.env` (gitignored): `HF_TOKEN`, `GROQ_API_KEY`, `DATABASE_URL`.

### Threat model / deployment reality
Runs on **one Windows machine** with local Postgres. **Not yet network-exposed.** The
next phase (see [`HANDOFF.md`](HANDOFF.md)) adds a Google Drive ingestion front-end on
GitHub Actions (ideally a **self-hosted runner** on the GPU box). Most security risk
becomes *live* at that point. Findings below are tagged **🔴 blocking-for-deploy**,
**🟡 fix-now correctness**, or **🟢 nice-to-have**.

### Key data model facts you must know
- The **`student_id`** used for all RAG isolation is currently the **first 4 digits of
  the number in the M4A filename** (`scripts/ingest_zip.py` `parse_m4a_filename`,
  `candidate_number[:4]`). It is trusted from the filename, with no verification.
- Retrieval is scoped by `WHERE student_id = %s` in SQL
  ([`scripts/utils/pg_store.py`](scripts/utils/pg_store.py)). There is a *post-hoc*
  leakage guard in [`scripts/retrieval.py`](scripts/retrieval.py) that raises
  `RetrievalError` if a returned row's `student_id` ≠ the requested one — but it does
  **not** authenticate the caller (see Finding #3).
- `student_id` is also assigned in `embed_and_store.chunk_student_context` as
  `ctx.roll_no or ctx.name.lower().replace(" ", "_")`.

---

## 2. Findings (verified against code, with file:line)

### 🔴 Blocking before any deployment / network exposure

| # | Finding | Location |
|---|---------|----------|
| 1 | **DB password persisted in plaintext chat traces.** `ChatSessionRecord.db_url` = full `postgresql://user:PASSWORD@host/db`; written to `output/chat_sessions/<id>.json` every turn. | [`scripts/chat.py:58`](scripts/chat.py:58), [`:333`](scripts/chat.py:333), `write_session_record` [`:269`](scripts/chat.py:269) |
| 2 | **Secrets passed as CLI flags.** `--db-url` with password is visible in process list / shell history / echoed logs. Env fallback already exists. | [`scripts/migrate_db.py:58`](scripts/migrate_db.py:58), [`scripts/run_pipeline.py:43`](scripts/run_pipeline.py:43) |
| 3 | **No authorization on retrieval/chat.** `student_id` is a caller-supplied arg; the "leakage" guard only checks returned rows match the *requested* id, not that the caller is *entitled* to it. Full IDOR once networked. **This is what the new login system fixes — see §4.** | [`scripts/retrieval.py:183`](scripts/retrieval.py:183), [`:216`](scripts/retrieval.py:216), [`scripts/chat.py:300`](scripts/chat.py:300) |
| 4 | **`student_id` = 4-digit filename prefix → collision + spoofing.** No uniqueness check in the no-roster/attendance-only path (the roster dup-guard only runs with `--roster`). Shared 4-digit prefixes co-mingle two students under one id. Roll is also trusted from the filename (rename → misattribution). | [`scripts/ingest_zip.py:80`](scripts/ingest_zip.py:80), [`scripts/match_identity.py:263`](scripts/match_identity.py:263), [`scripts/embed_and_store.py:124`](scripts/embed_and_store.py:124) |
| 5 | **Undisclosed third-party PII egress to Groq.** Verbatim student transcript chunks sent to Groq (US LLM API) per question; no consent/disclosure/retention stance. | [`scripts/chat.py:284`](scripts/chat.py:284), context at [`:216`](scripts/chat.py:216) |
| 6 | **No zip-bomb / size / entry-count guard before extract.** CPython's `zipfile` sanitizes `..`/abs paths (classic Zip Slip mostly mitigated), but a malicious/huge Drive zip can exhaust disk. Trusted today; untrusted in the Drive phase. | [`scripts/ingest_zip.py:170`](scripts/ingest_zip.py:170) |
| 7 | **`chromadb==1.5.9` dead dependency** still pinned (replaced by pgvector) → needless CVE/supply-chain surface. | `requirements.txt:5` |

### 🟡 Correctness bugs to fix now (degrade the product today)

| # | Finding | Location |
|---|---------|----------|
| 8 | **Chunk-type filter applied AFTER `LIMIT top_k` → silent empty results.** SQL `LIMIT top_k` has no type predicate; Python filters afterward. `--chunk-type spoken --top-k 5` can return 0 even when `spoken` chunks exist. Push the filter into SQL `WHERE`. | [`scripts/retrieval.py:208`](scripts/retrieval.py:208) vs [`scripts/utils/pg_store.py:21`](scripts/utils/pg_store.py:21) |
| 9 | **Embedding model reloaded from disk on every query; new PG connection per chat turn.** `embed_query` constructs `SentenceTransformer(...)` each call (~80 MB load); chat opens+closes a connection per turn. Load once, reuse. | [`scripts/retrieval.py:176`](scripts/retrieval.py:176), [`:198`](scripts/retrieval.py:198) |
| 10 | **`detect_alignment` is a constant function** (both branches return `session_aligned, 0.0`); the entire offset machinery is dead. Correct for Zoom, but a silent-mis-merge trap for any non-session-aligned source in the Drive phase. Delete the dead path + document the hard assumption, or restore a *validated* check. | [`scripts/merge_transcripts.py:57-83`](scripts/merge_transcripts.py:57), [`:97`](scripts/merge_transcripts.py:97) |
| 11 | **"Missed" detection is inert without per-class attendance** (`window_end = class_duration` → nothing missed) and assumes a contiguous block from t=0 (late-join/rejoin mis-split). Surface a trust flag instead of silently emitting empty "missed". | [`scripts/build_student_context.py:90-103`](scripts/build_student_context.py:90) |

### 🟢 Nice-to-have / code quality

| # | Finding | Location |
|---|---------|----------|
| 12 | Dynamic `__import__("scripts.merge_transcripts", ...)` — replace with a top-level import. | [`scripts/run_pipeline.py:177`](scripts/run_pipeline.py:177) |
| 13 | Broad `except Exception` in `_timed_step` hides real defects (validation errors become a "step failed" string). Narrow it or log full tracebacks at debug. | [`scripts/run_pipeline.py:75`](scripts/run_pipeline.py:75) |
| 14 | `assert isinstance(conn, psycopg.Connection)` — asserts stripped under `-O`; use a typed connection. | [`scripts/migrate_db.py:84`](scripts/migrate_db.py:84) |
| 15 | Lazy `import json` inside methods/loops violates the stdlib-imports-at-top rule. (Lazy `torch`/`whisperx`/`sentence_transformers`/`psycopg` imports are justified — leave those.) | [`scripts/utils/pg_store.py:48,84,122`](scripts/utils/pg_store.py:48) |
| 16 | `_wav_tmp` WAV cache grows unbounded (no cleanup). | [`scripts/transcribe_dual.py:356`](scripts/transcribe_dual.py:356) |
| 17 | Dead v1 code: `scripts/transcribe.py` and the `*.py.txt` files — remove or leave, but they add noise. | `scripts/transcribe.py`, `scripts/*.py.txt` |
| 18 | Stray 0-byte `outputpipeline_run.log` at repo root is not covered by `.gitignore` (it's not under `output/`). Delete it or add to `.gitignore`. | `outputpipeline_run.log` |

### Verified-safe (do NOT "fix" — regressions risk)
- **SQL is fully parameterized** (`%s` placeholders incl. `embedding <=> %s::vector`). No f-string SQL.
- **ffmpeg subprocess is injection-safe** — argv list, no `shell=True`, `-nostdin`, paths as separate args.
- **Idempotent re-runs** — `delete_class_chunks(class_name)` before upsert + stable SHA-1 ids.

---

## 3. Recommended fix order

Do these as **separate commits** (one logical fix per commit, one-line message). Run the
full quality gate (§0) after each. Add/expand tests for each behavioral change.

1. **Quick credential wins (🔴 #1, #2, #7).** Stop persisting `db_url` in chat traces
   (drop the field or redact to host/db); make `--db-url` optional everywhere with
   `DATABASE_URL` env as the primary source and warn if a secret-bearing flag is used;
   remove `chromadb` from `requirements.txt`. Small and mechanical.
2. **Core correctness (🟡 #8, #9).** Push chunk-type filtering into SQL; load the
   embedding model + DB connection once and reuse across chat turns. These directly
   affect answer quality and chat latency *today*.
3. **The per-student login / true personalization system (🔴 #3, and the foundation for #4).**
   This is the big one — full design in §4 below. It makes `student_id` an
   *authenticated* principal instead of a trusted CLI arg.
4. **Identity integrity (🔴 #4).** Once login exists, reconcile the 4-digit filename roll
   against an authoritative roster `student_uid`, enforce uniqueness in *all* code paths,
   and fail loudly on ambiguous/duplicate rolls.
5. **Deploy-gating items (🔴 #5, #6).** Add a disclosure/consent note + retention decision
   for Groq egress; add zip-bomb guards (max uncompressed size, max entry count) before
   `extractall`. Schedule these before the Drive/Actions phase.
6. **Correctness polish (🟡 #10, #11).** Resolve the dead alignment path; turn the inert
   "missed" path into an explicit trust flag.
7. **Cleanups (🟢 #12–#18)** as you touch the surrounding code.

---

## 4. New feature: per-student login → truly personalized, isolated access

**Goal (owner request):** a login system so each student authenticates and can *only* ever
see their own data — no path by which one student reads another's transcripts, voice, or
chat history. This is the proper fix for Finding #3 and the access-control half of #4.

### Core principle
**`student_id` must stop being a caller-supplied argument and become a value derived from
an authenticated session.** Today anyone can pass `--student-id 2302` and read Bhagyashree's
data. After this change, the identity is established by *logging in*, and every retrieval is
scoped to the logged-in student — the student id is never trusted from CLI/HTTP input.

### Credentials store — a CSV for now (id + password)

For this phase, **login is backed by a simple credentials CSV** — no `students`/sessions DB
table yet. Keep it dead simple; the owner will maintain the file by hand for now.

**File format** (path passed via `--credentials` flag or a fixed default, e.g.
`data/credentials.csv` — keep it **gitignored**, it holds secrets):
```csv
student_id,password
2302,somePassword
2504,anotherPassword
```
- `student_id` is the same value used as the RAG partition key in `embeddings.student_id`
  (today the 4-digit roll). The login row's `student_id` is what scopes every retrieval.
- `password` is the student's login secret. **For now it may be stored as written in the
  CSV** (owner's call for the local alpha). *Hardening note for later:* prefer storing a
  hash (argon2id/bcrypt) per row instead of plaintext, and move the store into the
  `students` Postgres table — but that is **out of scope for this pass**; the CSV is the
  agreed interim.
- The `student_id` in the CSV is the trust boundary, so it must be unique. Loading the CSV
  must **fail loudly on duplicate `student_id`** (ties into Finding #4).

> **Pre-step (do first, separate commit):** pin `scikit-learn` in `requirements.txt`.
> `scripts/utils/topics.py` imports it but it is unpinned, so a clean checkout can't
> reproduce the green gate. `chore: pin scikit-learn used by topics extraction`.

### Build (CLI auth — matches the project's current shape, no web server)
1. New `scripts/auth.py` with an `AuthService` **class** (domain logic → class, per
   `CLAUDE.md`):
   - `load_credentials(path: Path) -> dict[str, str]` — parse the CSV with the stdlib
     `csv` module (mirror `match_identity.load_roster`); reject duplicate `student_id`,
     empty id, empty password, and missing file; skip fully-blank lines.
   - `authenticate(student_id: str, password: str) -> bool` — constant-time verify with
     `hmac.compare_digest`. **Encode both sides to `bytes` (`.encode("utf-8")`) before the
     compare** — `compare_digest` raises `TypeError` on non-ASCII `str`. On an unknown
     `student_id`, run a **dummy compare against a constant** so timing doesn't leak which
     ids exist. **Never log the password.**
   - Keep credential handling isolated here so swapping the CSV for a hashed/DB store later
     touches only this file.
2. **Rework `scripts/chat.py`:** remove `--student-id` / `--student-name` as *trusted*
   inputs. The user supplies a `student_id` + password at login (prompt the password via
   `getpass`, injectable for tests; never a flag/env), authenticate via `AuthService`, and
   only on success derive the `student_id` used for retrieval. Loop until correct or a clean
   `Ctrl-C`/EOF exit — and **add a short `time.sleep` backoff (~1s) on each failed attempt**
   so the loop isn't an unthrottled password-guessing oracle. Resolve the display name
   eagerly from the embeddings store (new `pg_store.get_student_name`, fallback to the id).
   The student can no longer point the chatbot at anyone else's data.
3. **`scripts/retrieval.py`:** keep the existing `RetrievalError` cross-student guard as
   defense-in-depth. **Note in scope:** retrieval.py's CLI stays an *unauthenticated local
   dev tool* (it still takes `--student-id`). Add a one-line docstring saying so. This means
   Finding #3 is closed **for `chat.py` only** — see the corrected framing below.
4. Add a `--credentials` flag (default `data/credentials.csv`) and add that path to
   `.gitignore`. Ship a `data/credentials.example.csv` with obviously-fake rows. **Document
   in the example file + README that `student_id` must EXACTLY match the embeddings
   partition id** — which is the 4-digit roll, or a name-slug
   (`name.lower().replace(" ", "_")`) for roster-less students. A mismatched id authenticates
   fine but silently retrieves nothing.
5. Tests: load CSV (success + duplicate-id / empty-id / empty-password / missing-file
   rejection + blank-line skip), `authenticate` (correct / wrong / unknown-id, and a
   non-ASCII password to prove the bytes-encoding fix), and a chat test proving a logged-in
   student's retrieval is scoped to their `student_id` with **no `parse_args` path to set a
   different id**. No psycopg needed for the auth tests — it's a CSV.

### ⚠️ What this pass does and does NOT close (corrected framing)
- **Finding #3 — partially closed.** `chat.py` is gated; `retrieval.py` is intentionally
  left as an unauthenticated local dev tool. Mark #3 "partially resolved," not closed.
- **Finding #4 — only the *access* half.** Rejecting duplicate ids in the CSV stops two
  *credential rows* sharing an id. It does **NOT** un-merge data the ingest pipeline may have
  already co-mingled under a colliding 4-digit prefix — logging in as a collided id still
  surfaces both students' chunks. The **data-integrity half of #4 (ingest-time uniqueness)
  remains open** and is still step 4 of the fix order. Do not mark #4 done.

### Later (when networked — context only, not this pass)
Move credentials into a `students` Postgres table with **hashed** passwords, add real
sessions/JWT (store token hashes, short expiries), rate-limit login, enforce HTTPS, and keep
Postgres reachable only from the app. Also revisit the Groq egress disclosure (#5) once real
students log in.

### Acceptance criteria for "truly personalized"
- [ ] In `chat.py`, no CLI input can set the queried `student_id`; it is established only by
      a successful login against the credentials CSV.
- [ ] Credentials CSV is gitignored; password encoded to bytes and compared in constant
      time; unknown id runs a dummy compare; failed attempts back off; nothing sensitive
      logged.
- [ ] Loading the CSV fails loudly on duplicate / empty `student_id` and empty password.
- [ ] A test demonstrates student A, when logged in to `chat.py`, cannot retrieve student
      B's chunks by any input (incl. no `parse_args` path to another id).
- [ ] Chat session traces no longer contain secrets (#1, already done).
- [ ] Docstrings/PROGRESS note state the *scope honestly*: #3 closed for `chat.py` only
      (retrieval.py left as an unauthenticated dev tool); #4 only the access half — ingest
      collision still open.

---

## 5. How to run / verify (for reproduction)
```powershell
python -m scripts.migrate_db --db-url "postgresql://postgres:<pw>@localhost:5432/adira"
python -m scripts.run_pipeline --input "class.zip" --output-dir output/ --teacher "Nisha" `
  --attendance "attendance.csv" --db-url "postgresql://..."
python -m scripts.chat --student-id 2302 --student-name "Bhagyashree" --db-url "postgresql://..."
# quality gate:
python -m ruff check --fix .   # 0 errors
python -m mypy .               # 0 errors
python -m pytest               # 100% passing
```
> Note: after the §4 login work lands, the `chat` invocation changes (login instead of
> `--student-id`). Update [`README.md`](README.md) accordingly in the same commit.

---

## 6. Out of scope for this pass (context only)
- Transcript quality ceiling (WhisperX small on mixed audio). Biggest functional win is
  using the teacher's isolated M4A as the primary class-context source — not yet implemented.
- GPU on GitHub Actions: standard runners are CPU-only; use a **self-hosted runner** on the
  RTX 3050 box (also keeps Postgres on `localhost` and secrets on one machine).
- Google Drive ingestion front-end + `processed_files` dedup table — see [`HANDOFF.md`](HANDOFF.md).
