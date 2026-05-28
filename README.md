# Adira Academy Learning Assistant

Personalized RAG chatbot for ISL students. After every Zoom class, each enrolled student gets their own AI assistant that knows what happened in class — what they said, what they missed, and what topics were covered. Students who were absent get a summary chatbot too.

> **Status:** Phase 1 Pipeline v2 — functional end-to-end on real Zoom exports.  
> See [`PROGRESS.md`](PROGRESS.md) for detailed results and real output examples.

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

```powershell
python -m scripts.chat `
  --student-id <roll_no> `
  --student-name "<Student Name>" `
  --db-url "postgresql://postgres:<password>@localhost:5432/adira"
```

Example:
```
You: What was the supply function formula we used today?
Assistant: In today's class, Nisha introduced the supply function where A is the
intercept (a constant, not related to x). You then derived the quantity supply
by solving the right-hand side of the equation...
```

**Single question (non-interactive):**
```powershell
python -m scripts.chat --student-id 2302 --student-name "Bhagyashree" `
  --db-url "postgresql://..." --question "What did I miss today?"
```

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

**Test suite:** 216 tests, 0 errors, 0 warnings.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Transcription | WhisperX + faster-whisper (CUDA), `whisperx.asr.WhisperModel` |
| Dual-language | Hindi + English word-level probability merge |
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

- **Progress & real output examples:** [`PROGRESS.md`](PROGRESS.md)
- **Pipeline rebuild plan:** [`.vscode/planned/pipeline-rebuild/MASTER_PLAN.md`](.vscode/planned/pipeline-rebuild/MASTER_PLAN.md)
- **GitHub:** https://github.com/Amritesh-878/agents-test
