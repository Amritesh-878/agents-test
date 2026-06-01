# Audit Context — Adira Academy Learning Assistant

**Purpose:** Full project context for a code/security auditor. Use this to perform a detailed analysis: bug finding, design flaws, security vulnerabilities, data-privacy issues, and production-readiness gaps.

**Audit scope requested:** Security vulnerabilities • Code quality & correctness • Data privacy / PII • Production readiness.

**Deployment reality (threat model):** Pre-release **alpha, local only**. Runs on one Windows machine with local PostgreSQL. It processes **real student data** (names, roll numbers, voice recordings, transcripts) but is **not yet network-exposed**. Flag anything that **must be fixed before any deployment**, and separate "blocking-for-deploy" from "nice-to-have."

**Repo:** https://github.com/Amritesh-878/agents-test (branch `main`, 41 commits)

---

## 1. What This System Does

An educational pipeline that turns Zoom cloud class recordings into a **per-student personalized RAG chatbot**. Students ask what happened in a class / what they missed; the bot answers grounded in their own transcript data.

**Data flow (untrusted input → storage → query):**
```
Zoom .zip (UNTRUSTED archive)
  → ingest_zip.py        : zipfile.extractall to output/<class>/raw/
  → match_identity.py    : parse roll numbers from filenames; read attendance CSV; read roster CSV
  → transcribe_dual.py   : ffmpeg (subprocess) → WhisperX (GPU) → JSON transcripts
  → merge_transcripts.py : speaker attribution
  → build_student_context.py : per-student segments + TF-IDF topics
  → embed_and_store.py   : sentence-transformers → PostgreSQL/pgvector (raw SQL via psycopg3)
  → retrieval.py / chat.py : query embedding → pgvector similarity → Groq LLM (external API)
```

PII at every stage: student names, 4-digit roll numbers, isolated voice recordings (M4A), and verbatim transcripts of what each student said.

---

## 2. Tech Stack & Dependencies

- **Python 3.11**, Pydantic v2 (all data contracts), argparse CLIs
- **WhisperX 3.1.1** + faster-whisper 0.10.1 + ctranslate2 3.24.0 (CUDA 11.8, RTX 3050 4GB)
- **PostgreSQL 17 + pgvector** via `psycopg[binary]>=3.1` and `pgvector>=0.2`
- **sentence-transformers 2.7.0** (all-MiniLM-L6-v2, 384-dim), transformers 4.41.2
- **Groq SDK 0.9.0** (llama-3.1-8b-instant) — external LLM API
- **ffmpeg** — invoked via `subprocess` for audio extraction
- Tooling: ruff 0.4.8, mypy 1.10.0, pytest 8.2.2 (216 tests, all passing, mypy/ruff clean)
- ⚠️ `chromadb==1.5.9` still pinned in `requirements.txt` but **no longer used** (dead dependency, v1 leftover — review for removal / CVE surface)

Secrets via `.env` (gitignored): `HF_TOKEN`, `GROQ_API_KEY`, `DATABASE_URL`.

---

## 3. Files to Audit (by risk priority)

### HIGH-RISK (untrusted input / data egress / secrets)

| File | Why it matters |
|------|----------------|
| `scripts/ingest_zip.py` | Extracts an **untrusted .zip** with `zipfile.extractall`. **Audit for Zip Slip / path traversal** (malicious entry names like `../../etc/...`). Also reads `recording.conf`, `zoomver.tag`. |
| `scripts/utils/pg_store.py` | All raw SQL against Postgres. Verify **parameterization** (it uses `%s` placeholders — confirm no f-string SQL). Check `embedding <=> %s::vector` cast path. Connection lifecycle, no pooling. |
| `scripts/migrate_db.py` | DDL execution; reads `DATABASE_URL` from env/flag. Idempotent CREATE IF NOT EXISTS. |
| `scripts/retrieval.py` | **Cross-student data isolation.** Filters by `student_id`; there is a leakage guard (`RetrievalError` if a returned chunk's student_id mismatches). Verify the guard is sound and the filter can't be bypassed. |
| `scripts/chat.py` | Sends retrieved transcript context to **Groq (external API)** — data egress of student PII to a third party. Loads `GROQ_API_KEY`. Writes session traces to `output/chat_sessions/`. |
| `scripts/extract_audio.py` / `transcribe_dual.py` | Build `ffmpeg` command lines via `subprocess`. Audit for **command injection** through file paths / filenames derived from zip contents. |
| `scripts/match_identity.py` | Parses attendance & roster CSVs and filenames. Regex-based roll-number extraction. Untrusted filename → identity mapping. |

### MEDIUM-RISK (logic correctness, heuristics)

| File | Why it matters |
|------|----------------|
| `scripts/merge_transcripts.py` | Alignment detection was **hardcoded to `session_aligned`** after text-matching produced false 987.5s offsets on real data. Audit whether this assumption (Zoom per-student M4As always start at session start) is safe, and the clustering/overlap math. |
| `scripts/transcribe_dual.py` | `is_hallucinated_segment` (>70% single-word repetition) and dual-language word-probability merge. Heuristic — audit false-positive/negative behavior. Drops WhisperX alignment entirely (uses faster-whisper word timestamps) due to a transformers API break. |
| `scripts/embed_and_store.py` | `is_quality_text` filter (trigram repetition, replacement-char ratio). SHA-1 chunk IDs (`stable_chunk_id`) — collision/idempotency reasoning. |
| `scripts/build_student_context.py` | Attendance-window logic uses **total duration only** (no join/leave timestamps) → assumes contiguous attendance. Missed-segment calculation correctness. |
| `scripts/generate_session_report.py` | Text-cleaning regexes, English-ratio heuristics. Output-only (no storage), lower risk. |
| `scripts/run_pipeline.py` | Orchestrator. Uses `__import__(...)` dynamically in one place (merge step). Broad `except Exception` for per-class failure isolation — audit error swallowing. |

### LOWER-RISK
- `scripts/models/` — Pydantic models (transcript, identity, context, pipeline). Data contracts; check validation strictness.
- `scripts/transcribe.py` — **v1, unused.** Dead code.
- `scripts/*.py.txt` — deprecated v1 scripts (merge, build_context, chunk_and_embed) kept as `.txt`. Dead.

---

## 4. Specific Things to Probe

### Security
1. **Zip Slip** in `ingest_zip.extract_zip` — `zf.extractall(raw_dir)` on attacker-controlled archive names. Is there path containment?
2. **Command injection** — do any ffmpeg subprocess calls interpolate untrusted filenames into a shell? (Check `shell=` usage and arg-list construction in `extract_audio.py` / `transcribe_dual.py`.)
3. **SQL injection** — confirm every query in `pg_store.py` uses parameters, not string formatting. Check the vector cast and any dynamic `class_name`/`student_id` paths.
4. **Secret handling** — `.env` loading via python-dotenv; `DATABASE_URL` may contain a password and is passed as a CLI flag (visible in process list / shell history / logs). Are secrets ever logged? (`run_pipeline` logs step errors — could a DSN leak?)
5. **SSRF / external calls** — WhisperX model downloads (HuggingFace), Groq API, (future) Google Drive. Pinned/verified?
6. **Coming attack surface (flag proactively):** the next phase moves orchestration to **GitHub Actions** with a **Google service-account JSON** + DB creds + Groq key as CI secrets, and a Drive-folder polling trigger. Note the implications: secret sprawl in CI, a service account with Drive access, and whether Postgres becomes network-reachable.

### Data Privacy / PII
7. Student voice + verbatim transcripts are sensitive. **Per-student transcript JSON and chat session traces are written to `output/`** (gitignored, but plaintext on disk, unencrypted). Acceptable for alpha?
8. **Third-party egress:** `chat.py` sends retrieved student context to Groq. Is students' data leaving to a US LLM provider acceptable / disclosed? Any retention concern?
9. **Cross-student leakage:** retrieval filters by `student_id` only. If two students share the first 4 digits of a roll number, or roll numbers collide, could one student's data surface for another? (`match_identity` has a duplicate-roll guard for roster — verify it's enforced everywhere.)
10. **Identity from filenames:** roll numbers parsed from M4A filenames are trusted as ground truth. A misnamed/spoofed file could misattribute a recording to the wrong student.

### Code Quality / Correctness
11. The **alignment hardcode** (`merge_transcripts.detect_alignment` returns `session_aligned` even on duration mismatch). Correct for Zoom, but is the fallback path dead/misleading?
12. **Heuristic thresholds** scattered as module constants (teacher fuzzy match 0.75, hallucination 0.70, quality english-ratio 0.55–0.65, gap 1.5s). Are these justified / tested at boundaries?
13. **Broad `except Exception`** in `run_pipeline._timed_step` and `process_single_class` — does it mask real bugs?
14. **`__import__("scripts.merge_transcripts", ...)`** dynamic import in `run_pipeline._merge` — code smell; why not a top import?
15. Windows-specific assumptions: cp1252 console encoding caused crashes (fixed by removing non-ASCII from logs); hardcoded ffmpeg/CUDA paths in earlier batch files (since removed). Cross-platform portability.
16. Test coverage: 216 tests, but transcription/embedding GPU paths are mocked. Integration coverage of the real pipeline is manual only.

### Production Readiness
17. **No GPU on GitHub Actions** — transcription is GPU-bound; the stated CI runtime can't run it on standard runners. (Documented in `HANDOFF.md`.)
18. No connection pooling; one psycopg connection per invocation. Concurrency story for multi-student chat?
19. No auth/authorization layer on `chat.py` — any caller can pass any `student_id`. Becomes critical the moment it's network-exposed.
20. Idempotency: stable SHA-1 chunk IDs + delete-by-class before upsert. Verify re-runs don't duplicate or orphan data.
21. Observability: logging only; no metrics/tracing. Failure recovery is per-class continue-on-error.

---

## 5. Known Issues Already Identified (not yet fixed)

- `chromadb` dead dependency in `requirements.txt`.
- `scripts/transcribe.py` and three `*.py.txt` files are dead v1 code.
- Transcript quality ceiling on mixed audio (WhisperX small) — biggest functional weakness; teacher-M4A-as-primary-source is the proposed fix, not implemented.
- Absent-student feature blocked on missing roster CSV.
- Attendance windows are estimates (duration-only, no join/leave times).

---

## 6. How to Run / Reproduce

See `README.md` for setup. Quick reproduction for the auditor:
```powershell
python -m scripts.migrate_db --db-url "postgresql://postgres:<pw>@localhost:5432/adira"
python -m scripts.run_pipeline --input "class.zip" --output-dir output/ --teacher "Nisha" \
  --attendance "attendance.csv" --db-url "postgresql://..."
python -m pytest          # 216 tests
python -m ruff check .    # lint
python -m mypy .          # types
```

**Companion docs:** `README.md` (usage), `PROGRESS.md` (results + full bug history), `HANDOFF.md` (next-phase plan). Pipeline design specs in `.vscode/planned/pipeline-rebuild/`.
