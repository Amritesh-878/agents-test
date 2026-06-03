# Adira Academy Learning Assistant

Personalized RAG chatbot for ISL students. After every Zoom class, each enrolled student gets their own AI assistant that knows what happened in class — what they said, what they missed, and what topics were covered. Students who were absent get a summary chatbot too.

> **Status:** Phase 1 Pipeline v2 — functional end-to-end on real Zoom exports.  
> See [`PROGRESS.md`](docs/PROGRESS.md) for detailed results and real output examples.

---

## How It Works

```
Zoom .zip  ->  ingest  ->  identify  ->  transcribe  ->  merge  ->  context  ->  pgvector  ->  chat
```

1. **Ingest** — extract the Zoom zip, discover the session MP4, per-student M4As, and attendance
2. **Identify** — match each M4A to a student using the 4-digit roll number baked into the filename
3. **Transcribe** — WhisperX (Hindi + English dual-language) on every audio file via CUDA
4. **Merge** — speaker-attribute the full session timeline using per-student timestamps as ground truth
5. **Context** — build per-student context: what they said (spoken), what they attended (present), what they missed (missed)
6. **Embed** — sentence-transformers + pgvector; stale chunks purged before each upsert
7. **Chat** — Groq LLM answers student questions grounded strictly in their retrieved context

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11 |
| CUDA / PyTorch | CUDA 11.8, `torch==2.1.0+cu118` |
| GPU | NVIDIA RTX 3050 4GB (or better) |
| ffmpeg | In system PATH (install via WinGet or https://ffmpeg.org) |
| PostgreSQL | 17 with pgvector extension |

---

## Setup

**1. Create virtualenv and install PyTorch first:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

**2. Configure `.env`:**

```powershell
Copy-Item .env.example .env
# Edit .env and fill in:
#   HF_TOKEN=hf_...          (HuggingFace token for WhisperX models)
#   GROQ_API_KEY=gsk_...     (Groq API key for chat)
#   DATABASE_URL=postgresql://postgres:<password>@localhost:5432/adira
```

**3. Create the PostgreSQL database and run the migration:**

```powershell
# In psql: CREATE DATABASE adira;
python -m scripts.migrate_db --db-url "postgresql://postgres:<password>@localhost:5432/adira"
```

**4. Validate the environment:**

```powershell
python -m scripts.validate_env --skip-pgvector   # checks Python, CUDA, WhisperX
```

---

## Running the Pipeline

### Single class (one Zoom zip):

```powershell
python -m scripts.run_pipeline `
  --input "path\to\ClassName.zip" `
  --output-dir output\ `
  --teacher "Teacher Name" `
  --attendance "path\to\attendance.csv" `
  --db-url "postgresql://postgres:<password>@localhost:5432/adira"
```

### Batch mode (directory of zips):

```powershell
python -m scripts.run_pipeline `
  --input "path\to\zips_folder" `
  --output-dir output\ `
  --teacher "Teacher Name" `
  --attendance "path\to\attendance.csv" `
  --db-url "postgresql://postgres:<password>@localhost:5432/adira"
```

### Skip re-transcription (re-run merge/context/embed only):

```powershell
python -m scripts.run_pipeline ... --skip-transcribe
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--teacher` | required | Teacher's Zoom display name (repeat for multiple) |
| `--attendance` | optional | Zoom attendance CSV path |
| `--roster` | optional | Master roster CSV (Name, RollNo, Email) — enables absent student contexts |
| `--model` | `small` | WhisperX model size (small fits in 4GB VRAM) |
| `--skip-transcribe` | false | Skip transcription, re-run merge/context/embed |

---

## Google Drive Ingestion (automated front-end)

Instead of running each zip by hand, `scripts.drive_sync` polls a Google Drive folder
of Zoom `.zip` exports and feeds any new ones through the same pipeline above. It is
**idempotent**: a `processed_files` Postgres table (keyed by Drive file id) records every
zip that completes, so re-runs skip what's already ingested.

**What it does per run:**
1. Lists the configured Drive folder and filters to `*.zip`.
2. For each zip whose Drive file id is **not** in `processed_files`: downloads it to a
   temp dir, runs `process_single_class`, and on success records
   `{drive_file_id, class_name, processed_at}`.
3. A file that **fails** — including the pipeline's zip-bomb guard (oversized /
   too-many-entry archives) and colliding-roll guard (two M4As sharing a 4-digit roll) —
   is left **unrecorded** so a fixed re-upload can be retried, and one bad zip never
   stops the batch.

**Auth + config come from the environment** (secrets are never CLI flags). Add to `.env`:

```powershell
#   GOOGLE_SERVICE_ACCOUNT_JSON=path\to\service-account.json   (Drive auth)
#   GOOGLE_DRIVE_FOLDER_ID=<id of the shared Drive folder>
#   ROSTER_CSV=data\roster.csv                                 (optional; absent-student context)
#   DATABASE_URL=postgresql://postgres:<password>@localhost:5432/adira
```

The Drive folder must be **shared with the service account's email**. The roster path is
resolved from `--roster`, then `ROSTER_CSV`, then `data/roster.csv` if present.

```powershell
python -m scripts.drive_sync `
  --output-dir output\ `
  --teacher "Nisha" `
  --attendance "path\to\attendance.csv"
```

### Scheduled run (GitHub Actions, self-hosted runner)

`.github/workflows/drive-sync.yml` runs this sync on a schedule. Because WhisperX
transcription is GPU-bound and GitHub-hosted runners are CPU-only, it targets a
**self-hosted Actions runner on the RTX 3050 box** — Postgres stays `localhost` and
secrets stay on one machine. (The GPU half is planned to move to AWS in a later stage;
this self-hosted runner is the interim runtime.)

**One-time runner provisioning** (per [Setup](#setup)): register a self-hosted runner on
the GPU box with labels `self-hosted, windows`, with the project `.venv` (CUDA torch +
WhisperX) and local Postgres already in place. The workflow refreshes only the pinned app
deps — it does not reinstall the heavy GPU wheels.

**Repository secrets** (Settings → Secrets and variables → Actions):

| Secret | Purpose |
|--------|---------|
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Full JSON key content (reuse the one from the Zoom→Drive project); the Drive folder is shared with its email |
| `GOOGLE_DRIVE_FOLDER_ID` | Id of the Drive folder holding the Zoom `.zip` exports |
| `DATABASE_URL` | `postgresql://…@localhost:5432/adira` |
| `HF_TOKEN` | WhisperX model downloads |
| `GROQ_API_KEY` | Only if a later step runs chat/eval |

Plus a repository **variable** `TEACHER_NAME` (the teacher's Zoom display name). The
schedule is daily at 02:00 UTC and is also runnable on demand via **Run workflow**.

---

## Output Structure

```
output/<class_name>/
    manifest.json              # discovered files + parsed roll numbers
    identity_map.json          # M4A -> student matching results
    transcripts/
        session.json           # full-session dual-language transcript
        audio<Name>_<id>.json  # per-student transcripts
        _wav_tmp/              # cached WAV conversions (speeds up re-runs)
    transcript_merged.json     # speaker-attributed merged timeline
    transcript_review.md       # human-readable merge summary
    student_contexts.json      # per-student present/missed/spoken segments + topics
    student_context_review.md  # review table
    rag_chunks.jsonl           # all embedded chunks
    rag_chunk_review.csv       # flat chunk review for spot-checking
```

---

## Chatting with a Student

Students no longer pass `--student-id`. Instead they **log in**: the chatbot prompts
for a student id and password (entered with no echo), verifies them against a
credentials CSV, and scopes every retrieval to the logged-in id. There is no CLI flag
that can point the chatbot at another student's data.

> **Privacy notice (third-party egress):** to generate answers, each question and the
> retrieved transcript excerpts are sent to **Groq**, an external US LLM API. This notice
> is surfaced in the chat banner. A retention/consent policy is the owner's call and is
> not yet decided — treat student transcript content as leaving the machine when chatting.

```powershell
# DATABASE_URL in .env is preferred over passing --db-url (which leaks the password
# into process listings / shell history).
python -m scripts.chat --credentials data/credentials.csv
```

You are then prompted:
```
Student id: 2302
Password:
Chat ready for Bhagyashree (2302). Session: output/chat_sessions/...
You: What was the supply function formula we used today?
Assistant: In today's class, Nisha introduced the supply function where A is the
intercept (a constant, not related to x)...
```

**Single question (still logs in first, interactively):**
```powershell
python -m scripts.chat --credentials data/credentials.csv --question "What did I miss today?"
```

### Credentials CSV (login)

`--credentials` defaults to `data/credentials.csv` (gitignored — it holds secrets).
Copy `data/credentials.example.csv` and fill in real rows:

```csv
student_id,password
2302,somePassword
2504,anotherPassword
```

- **`student_id` must EXACTLY match the embeddings partition id** — the 4-digit roll
  number from the M4A filename, or for roster-less students the name slug
  (`name.lower().replace(" ", "_")`). A mismatched id authenticates fine but retrieval
  returns nothing, because no chunks are stored under that id.
- `student_id` must be unique; loading fails loudly on a duplicate id, an empty id, or
  an empty password. Passwords are stored as plaintext for this local alpha and compared
  in constant time (hashing/DB-backed credentials are a later hardening step).

> **Scope note:** this login gates `scripts.chat` only. `scripts.retrieval`'s CLI remains
> an *unauthenticated local dev tool* that still takes `--student-id` — do not expose it.

---

## Input File Formats

### Zoom .zip
Standard Zoom cloud recording export. The pipeline expects:
- `<session>.mp4` — mixed session video
- `Audio Record/audio<Name>_<number>.m4a` — per-student isolated mic recordings
- `recording.conf` — session metadata (optional)

Roll numbers are parsed from M4A filenames: `audioA_Disha_25041234567890.m4a` → roll=`2504`.

### Attendance CSV
Zoom participants export format:
```
Name (original name),Email,Total duration (minutes),Guest
A_Disha_2504,,121,Yes
Amritesh Praveen,,489,Yes
```
Students appear as `Name_RollNo`. Teacher entries have no roll number suffix.

### Roster CSV (optional, recommended)
```
Name,RollNo,Email
Disha Agarwal,2504,disha@example.com
```
Without a roster, present students still get context (from attendance matching) but absent students won't appear.

---

## Session Engagement Report

Generate a single-page markdown report showing session type, topics covered, student engagement, and key student quotes:

```powershell
python -m scripts.generate_session_report   --class-dir "output\<class_name>"   --output "output\<class_name>\session_report.md"
```

**What the report includes:**
- Session type (Class / Revision / Assessment — auto-detected from transcript)
- Topics covered (TF-IDF, noise-filtered, student names excluded)
- Engagement table: per-student attendance, engagement level (Active / Moderate / Passive / Silent), and contribution count
- Key student quotes (verbatim from their isolated microphone recording)
- Session timeline: what was discussed at each segment of the class

**Example output** (Economics.02 — Supply Function, 51 min):
```
Type: Revision Session | Duration: 51.1 min | Teacher: Nisha

Topics: supply, price, beta, function

| Student      | Roll | Attendance   | Engagement | Contributions |
|--------------|------|-------------|------------|---------------|
| Bhagyashree  | 2302 | Full session | Active     | 75 segments   |

Bhagyashree (Active):
> "Market supply is a total quantity of goods that all the producers
>  in the market are willing to supply at different price."
> "Supply schedule is a table representation of price and quantity."
> "Determinant of a supply is what changes supply. The factor."
```

---

## Development

**Run tests:**
```powershell
python -m pytest
```

**Lint + typecheck:**
```powershell
python -m ruff check --fix .
python -m mypy .
```

**Test suite:** 305 tests, 0 errors, 0 warnings.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Transcription | WhisperX + faster-whisper (CUDA), `whisperx.asr.WhisperModel` |
| Dual-language | Hindi + English, one language selected per segment by confidence |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector DB | PostgreSQL 17 + pgvector, HNSW index, cosine distance |
| LLM | Groq `llama-3.1-8b-instant` |
| Data validation | Pydantic v2 |
| Quality | ruff, mypy, pytest |

---

## Known Limitations

- **WhisperX small model** hallucinates on silence (muted students). The pipeline filters these segments before embedding, but some mixed garbled+real segments may still appear in chunks.
- **Short classes / silent students** produce fewer chunks. A 17-minute class where students are in listening mode yields ~14 chunks. A 51-minute class with an active student yields ~192 chunks.
- **No join/leave timestamps in attendance** — attendance window uses total duration only, so missed-segment detection assumes students attended a contiguous block.
- **Roster CSV optional** — without it, absent students get no context object.

---

## Repository

- **Progress & real output examples:** [`PROGRESS.md`](docs/PROGRESS.md)
- **Pipeline rebuild plan:** [`.vscode/planned/pipeline-rebuild/MASTER_PLAN.md`](.vscode/planned/pipeline-rebuild/MASTER_PLAN.md)
- **GitHub:** https://github.com/Amritesh-878/agents-test
