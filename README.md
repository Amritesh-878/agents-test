# Adira Academy Learning Assistant

Personalized RAG chatbot backend for ISL students. After every Zoom class, each enrolled
student gets their own AI assistant grounded in what actually happened in class — what they
said, what the teacher taught, what the slides say, and what they missed. Teachers get
section-scoped access to their own students' assistants.

This repository is the **backend only**. The product UI is the Adira LMS
(`Adira25/AdiraLMS-Frontend` / `AdiraLMS-Backend`); the Streamlit app here is a testing
harness, not the product.

```
                        ┌──────────────  this repo  ──────────────┐
Browser ── Firebase ──> Django LMS ──[X-Service-Token]──> FastAPI service ──> pgvector + Groq
        (Google login)  (verifies ID token,               (scripts/api.py:     (per-student
                         knows email + role)               identity → answer)   chunks)
```

---

## LMS Integration Guide

### The contract in one paragraph

The LMS verifies the Firebase ID token and is the **only** caller of this service. Every
request carries the already-verified `email` and `lms_role` plus the shared secret in the
`X-Service-Token` header. This service maps the identity to a `Principal`
(student → their roll number, parsed from the email; teacher → their sections, from a config
CSV), authorizes the target student, runs retrieval + generation, and returns the answer
**with its grounding sources**. No Firebase, no tokens, no passwords live on this side.

### Identity rules (who becomes what)

| LMS identity | Becomes | How |
|---|---|---|
| `lms_role=student`, email like `bhagyashree_2302@islorg.com` | student principal for roll `2302` | 4-digit roll parsed from the email localpart (`_(\d{4})@`) |
| `lms_role=student`, email without a roll | **denied** (403) | never guess a store key |
| `lms_role=teacher`, email listed in the teacher-sections CSV | teacher principal scoped to those sections | `data/teacher_sections.csv` |
| `lms_role=teacher`, email not listed | **denied** (403) | operator adds the row; never guess |
| anything else (`observer`, `admin`, unknown, mis-cased) | **denied** (403) | fail closed |

There is **no admin role**. A student can only ever query themselves (a smuggled
`student_id` in the request body is ignored). A teacher must name a `student_id` and can
only reach students who have data in one of their sections.

**Teacher sections config:** copy `data/teacher_sections.example.csv` to
`data/teacher_sections.csv` (keep it untracked) and fill one row per teacher:

```csv
email,sections
arista@islorg.com,English.03;English.04
```

Section labels must match what the store derives from class names — the part before `_AY…`
(e.g. `English.03`, `Economics.02`, `Math.01 A`). `GET /students` shows them per student.

### Running the service

The service owns the heavy runtime (embedding model, cross-encoder reranker, pgvector
connection) — run it on the machine that has the database (and GPU, if available), not
inside Django:

```powershell
$env:CHATBOT_SERVICE_TOKEN = "<shared secret, same value in Django settings>"
$env:TEACHER_SECTIONS_CSV  = "data\teacher_sections.csv"
.venv\Scripts\uvicorn.exe scripts.api:create_app --factory --host 0.0.0.0 --port 8000
```

Startup **fails loudly** if `CHATBOT_SERVICE_TOKEN` is empty — a blank secret never means
"open".

**Environment variables** (all read from `.env` / process env):

| Variable | Used by | Purpose |
|---|---|---|
| `DATABASE_URL` | everything | Postgres + pgvector connection |
| `GROQ_API_KEY` | answering | Groq LLM API key |
| `CHATBOT_SERVICE_TOKEN` | API service | shared secret the LMS must send in `X-Service-Token` |
| `TEACHER_SECTIONS_CSV` | API service | path to the teacher→sections config |
| `HF_TOKEN` | transcription only | HuggingFace token for WhisperX models |
| `DEMO_ACCESS_CODE` | Streamlit harness only | optional gate for the test UI; unrelated to the API |
| `GOOGLE_SERVICE_ACCOUNT_JSON`, `GOOGLE_DRIVE_FOLDER_ID` | drive sync only | Zoom-zip ingestion from Drive |

### HTTP API

All endpoints except `/healthz` require the `X-Service-Token` header (constant-time
compared; `401` on mismatch).

#### `POST /ask` — answer a question as (or for) a student

```json
{
  "email": "bhagyashree_2302@islorg.com",
  "lms_role": "student",
  "question": "What are the steps for making a prediction about a story?",
  "class_name": null,
  "student_id": null
}
```

- `class_name` (optional): restrict to one session, exactly as returned by `/sessions`.
  Omit/null = search all the student's sessions. Best for "what did we cover today?".
- `student_id`: **teachers only** — the student they are asking about (`400` if a teacher
  omits it, `403` if it's not their student). Ignored for students.

Response:

```json
{
  "student_id": "2302",
  "student_name": "Bhagyashree",
  "question": "...",
  "answer": "To make predictions about a story ...",
  "grade": "high",
  "answer_source": "groq",
  "sources": [
    {"rank": 1, "chunk_type": "material", "score": 0.71,
     "speaker": "", "start": 0.0, "end": 0.0,
     "text": "State your prediction. Use tentative language such as ..."}
  ]
}
```

**Interpreting the response — important for the UI:**

- `grade` is the retrieval-confidence tier: `high` / `medium` → a grounded LLM answer;
  `low` → the system **refused** rather than guess.
- `answer_source`: `groq` = LLM-generated grounded answer; `fallback` = deterministic
  refusal text (no LLM call was made). A refusal is correct behavior for
  off-corpus questions ("what is photosynthesis") — render it normally, don't retry it.
- `sources` is the trust surface: the exact class records the answer is based on
  (`chunk_type`: `material` = slides/notes, `class_context` = teacher's speech,
  `spoken` = the student's own words, `chat` = their typed chat). Show it (expandable) —
  teachers grade answers by whether the sources support them.

Errors: `401` bad/missing service token · `403` identity denied / not their student ·
`400` teacher without `student_id` · `503` + `{"retry_after_seconds": 20}` Groq rate limit
(retry the same request after the delay) · `502` upstream generation failure.

#### `GET /students?email=...&lms_role=teacher` — the teacher's student picker

Returns exactly the students in that teacher's sections (students get `403`). A student's
`sections` lists only the sections **shared with the requesting teacher** — a teacher never
learns which other subjects a student takes:

```json
[
  {"student_id": "2301", "student_name": "anshi", "sections": ["English.04"]},
  {"student_id": "2302", "student_name": "Bhagyashree", "sections": ["English.04"]}
]
```

#### `GET /sessions?email=...&lms_role=...&student_id=...` — the session picker

Same authorization as `/ask` (`student_id` needed for teachers only). Returns raw
`class_name` (send this back in `/ask`) plus a human label:

```json
[
  {"class_name": "English.04_AY26-27_Cornell Notetaking_29 Jun",
   "label": "Cornell Notetaking — 29 Jun"}
]
```

#### `GET /healthz` — unauthenticated liveness probe → `{"status": "ok"}`

### Django-side glue (paste into the LMS repo)

```python
import requests
from django.conf import settings
from rest_framework.response import Response
from rest_framework.views import APIView


class ChatbotAskView(APIView):
    def post(self, request):
        upstream = requests.post(
            f"{settings.CHATBOT_SERVICE_URL}/ask",
            json={
                "email": request.user.email,
                "lms_role": request.user.role,
                "question": request.data.get("question", ""),
                "class_name": request.data.get("class_name"),
                "student_id": request.data.get("student_id"),
            },
            headers={"X-Service-Token": settings.CHATBOT_SERVICE_TOKEN},
            timeout=60,
        )
        return Response(upstream.json(), status=upstream.status_code)
```

Same pattern for `/students` and `/sessions` (GET with query params). Add
`CHATBOT_SERVICE_URL` and `CHATBOT_SERVICE_TOKEN` to Django settings from env.

**Calling from Python instead of HTTP** (if you ever co-locate): the API is a thin wrapper
over four calls — `principal_from_identity` (`scripts/auth.py`), `can_access_student` /
`allowed_student_ids` (same), and `answer_for_student` (`scripts/demo_backend.py`).

### Security & privacy notes

- Every chunk in the store is keyed by `student_id`; retrieval is scoped to one student at
  the SQL layer. Cross-student leakage is structurally prevented, not just filtered.
- The corpus is **minors' classroom data**. This repo must stay private; `data/` holds PII
  and is gitignored; never expose the service to the public internet — it is LMS-to-service
  only, on a private network or localhost.
- Questions + retrieved excerpts are sent to **Groq** (external US LLM API) to generate
  answers. Surface this notice to users; retention/consent policy is the school's call.
- The free Groq tier allows ~8k tokens/minute — bursts return the `503` retry envelope.

---

## Where Is What

| Path | What it is |
|---|---|
| `scripts/api.py` | **The LMS-facing FastAPI service** (`create_app` factory) |
| `scripts/auth.py` | Identity mapping (email/role → `Principal`) + authorization helpers |
| `scripts/demo_backend.py` | `answer_for_student` — the full ask pipeline behind one function; section grouping |
| `scripts/chat.py` | Answer flow internals (router tiers, prompts, refusals) + a dev-only CLI chat |
| `scripts/retrieval.py` | Hybrid retrieval (dense + full-text, RRF fusion) + cross-encoder rerank + isolation guards |
| `scripts/utils/retrieval_grade.py` | The confidence router (high/medium/low tiers) |
| `scripts/utils/pg_store.py` | pgvector store (schema, upserts, queries) |
| `scripts/run_pipeline.py` | Recording pipeline: Zoom zip → transcripts → contexts → embeddings |
| `scripts/transcribe_dual.py` | WhisperX dual-language transcription (GPU) |
| `scripts/build_student_context.py` | Presence decisions (audio / chat / attendance) + per-student context assembly |
| `scripts/ingest_materials.py` | Slides/PDF/notes → `material` chunks per enrolled student |
| `scripts/drive_sync.py` | Automated ingestion of new Zoom zips from Google Drive |
| `scripts/evaluate.py` | Golden-set evaluation harness (`--baseline` snapshots) |
| `scripts/migrate_db.py` | Creates the Postgres schema (pgvector) |
| `app.py` | Streamlit **testing harness** (teacher review rounds) — not the product UI |
| `data/eval_qa.json` | Golden QA set the eval runs against |
| `data/teacher_sections.example.csv` | Template for the teacher→sections config |
| `docs/EVAL.md`, `docs/PROGRESS.md` | Teacher review sheets, progress log |

---

## Setup

**1. Virtualenv + PyTorch (CUDA) + deps:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

**2. Configure `.env`** (copy `.env.example`): `DATABASE_URL`, `GROQ_API_KEY`, `HF_TOKEN`
(+ the service/Drive variables from the table above as needed).

**3. Create the database and schema:**

```powershell
# In psql: CREATE DATABASE adira;
python -m scripts.migrate_db --db-url "postgresql://postgres:<password>@localhost:5432/adira"
```

**4. Validate the environment:** `python -m scripts.validate_env --skip-pgvector`

Prerequisites: Python 3.11 · CUDA-capable GPU for transcription (4 GB VRAM is enough for
the `small` Whisper model; embedding/serving also runs on CPU) · ffmpeg on PATH ·
PostgreSQL 17 + pgvector.

---

## Feeding It Data

### 1. Class recordings (Zoom zips)

```powershell
python -m scripts.run_pipeline `
  --input "path\to\ClassName.zip" `          # or a folder of zips
  --output-dir output\ `
  --teacher "Arista" `                        # repeatable for display-name variants
  --roster "data\Student Info\C1.csv" `       # cohort badge tracker: STUDENT NAME,STUDENT ID
  --attendance "path\to\participants_....csv" `
  --db-url "..."
```

Key flags: `--attendance-dir <folder>` resolves each class's full-day attendance CSV by the
date in the class name (instead of one `--attendance` file) · `--skip-transcribe` re-runs
merge/context/embed against existing transcripts · `--skip-embed` stops before the DB write
(used for cloud transcription runs) · `--model small` fits 4 GB VRAM.

Presence precedence per student: own audio > chat-only > attendance-based (roster +
day-attendance ≥ 5 min, name-validated) > absent summary. Class names are store keys —
never rename session folders/zips.

### 2. Class materials (slides / PDFs / notes)

```powershell
python -m scripts.ingest_materials `
  --materials-dir "data\...\Class Material\<class_name>" `
  --class-name "<class_name>" `
  --identity-map "output\<class_name>\identity_map.json"   # or repeatable --student-id 2301
```

Materials become authoritative `material` chunks per enrolled student. Re-running replaces
a class's material chunks; transcript chunks are never touched. Note for the LMS future:
materials already live in the LMS's storage, so this step is expected to read from there
eventually — the extractors (`pptx/pdf/docx/txt/md`) are the stable part.

### 3. Automated Drive ingestion (optional)

`python -m scripts.drive_sync` polls a Drive folder of Zoom zips and runs the same pipeline,
idempotently (a `processed_files` table skips what's done). See the env table for its two
variables; share the Drive folder with the service account's email.

---

## How Answers Are Made (and why refusals are a feature)

1. **Hybrid retrieval**: dense vectors (all-MiniLM, 384-dim) + Postgres full-text, fused
   with RRF into a 25-candidate pool — always scoped to one `student_id`, optionally to one
   `class_name`.
2. **Rerank**: a cross-encoder (`ms-marco-MiniLM-L-6-v2`) reorders the pool; top-k survives.
3. **Route**: rerank confidence grades the turn `high` / `medium` / `low`. High and medium
   produce grounded LLM answers (medium uses a softer prompt). **Low produces a
   deterministic refusal with zero LLM involvement** — the bot never invents answers for
   topics that were not taught.
4. Generic meta-questions ("what did we cover?", "what did I say?") are recognized and kept
   answerable even when they don't lexically match any chunk; content-specific traps
   ("what did we cover **about GDP**?") stay refusable.
5. Every answer carries its retrieved sources; the model is instructed to answer only from
   them.

Model provenance is stamped on every chunk and checked at query time — a store/query
embedder mismatch fails loudly instead of returning garbage.

---

## Testing Surfaces (not the product)

- **Streamlit harness** (`streamlit run app.py`): class → student → session pickers, chat
  with a Grounding expander per answer. Optional `DEMO_ACCESS_CODE` gate for tunneling to a
  reviewer. Used for the teacher review rounds (`docs/EVAL.md`).
- **CLI chat** (`python -m scripts.chat`): dev tool; prompts for a student id (no auth) and
  scopes retrieval to it.
- **Evaluation**: `python -m scripts.evaluate` runs the golden set
  (`data/eval_qa.json` — quote-containment recall@k/MRR + refusal cases);
  `--baseline --label <name>` writes a provenance-stamped snapshot
  (store row count, dataset hash, reranker) so before/after comparisons are honest.

---

## Output Structure (per processed class)

```
output/<class_name>/
    manifest.json              # discovered files + parsed roll numbers
    identity_map.json          # M4A -> student matching results
    transcripts/               # session + per-student dual-language transcripts
    transcript_merged.json     # speaker-attributed merged timeline
    transcript_review.md       # human-readable merge summary
    student_contexts.json      # per-student present/missed/spoken segments + topics
    rag_chunks.jsonl           # all embedded chunks
```

---

## Development

```powershell
.venv\Scripts\ruff.exe check --fix .    # lint (must be clean before tests)
.venv\Scripts\mypy.exe .                # typecheck
.venv\Scripts\python.exe -m pytest      # full suite
```

The bar is 0 lint errors, 0 type errors, 100% tests passing (600+ tests). No comments or
docstrings in new code; full type hints everywhere.

---

## Stack

| Component | Technology |
|---|---|
| Transcription | WhisperX + faster-whisper (CUDA), Hindi + English dual-pass, per-segment language pick |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim), provenance-stamped |
| Retrieval | pgvector (HNSW, cosine) + Postgres FTS, RRF fusion, cross-encoder rerank |
| Routing | Confidence tiers (high/medium/low) over rerank scores; deterministic low-tier refusal |
| LLM | Groq `openai/gpt-oss-20b` |
| Service | FastAPI (`scripts/api.py`), shared-secret service auth |
| Data validation | Pydantic v2 |
| Quality | ruff, mypy, pytest |

---

## Known Limitations

- **Whisper hallucinates on silence** — the pipeline dedups looped segments at context-build
  time, but transcripts of quiet students remain thin.
- **Indirect instruction questions** ("what were we asked to do while watching the videos")
  can be refused even when the content exists — a measured cross-encoder blind spot; a
  stronger reranker is a candidate experiment.
- **Attendance is day-level** (no join/leave per session) — attendance-based presence is a
  tagged approximation.
- **Relative time** ("yesterday") isn't resolved; ask by topic or pick a session.
- Refusal messages list raw class names (functional, not pretty — wording is coupled to the
  eval's refusal detection).
