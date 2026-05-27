# TASK-018: Orchestrator Script + Retrieval/Chat pgvector Update

## Overview

Create a single orchestrator script that runs the full pipeline end-to-end (zip → extract → transcribe → merge → context → embed), and update the retrieval and chat scripts to use pgvector instead of ChromaDB.

## Execution Snapshot

- Depends on: TASK-011 through TASK-017
- Produces: `scripts/run_pipeline.py`, updated `scripts/retrieval.py`, updated `scripts/chat.py`
- Primary validation: `python scripts/run_pipeline.py --input <zips_dir> --teacher "Name" --roster roster.csv --attendance attendance.csv --db-url postgresql://...`
- Complexity: Medium

## Goals

1. **Orchestrator**: Single CLI that runs TASK-012 through TASK-017 in sequence
2. **Batch Support**: Process a directory of zips, one class at a time
3. **pgvector Retrieval**: Replace ChromaDB queries with pgvector SQL in `retrieval.py`
4. **Chat Update**: Update `chat.py` to use new retrieval layer
5. **Final Cleanup**: Deprecate old scripts (rename to `.txt`)

---

## Reasoning

### Why an orchestrator?

**Current Problems:**

- Running 6+ scripts manually in sequence is error-prone
- Each script has its own CLI args that must be passed correctly
- Batch processing of multiple classes needs automated sequencing

**Solution:**

- `run_pipeline.py` accepts top-level args and orchestrates all steps
- Each step is called as a function (not subprocess) for better error handling
- Batch mode iterates zips sequentially, continuing on individual failures

### Why update retrieval/chat now?

**Current Problems:**

- `retrieval.py` and `chat.py` still import from ChromaDB
- With pgvector in place, these are the last scripts using the old storage layer

**Solution:**

- Replace ChromaDB queries with `pg_store.PgVectorStore.search()`
- Keep the same `RetrievalResult` and `RetrievedChunk` Pydantic contracts
- The chat layer sees no change — only the storage backend swaps out

---

## Files to Change

### Files to CREATE

1. `scripts/run_pipeline.py` — **MAJOR** — Orchestrator CLI script
2. `tests/test_run_pipeline.py` — **MAJOR** — Orchestrator tests

### Files to MODIFY

1. `scripts/retrieval.py` — **MAJOR** — Replace ChromaDB with pgvector queries
2. `scripts/chat.py` — **MINOR** — Update imports from retrieval
3. `scripts/evaluate.py` — **MINOR** — Update if retrieval interface changed
4. `tests/test_retrieval.py` — **MAJOR** — New tests for pgvector retrieval (rewrite)
5. `tests/test_chat.py` — **MINOR** — Update for new retrieval imports
6. `tests/test_evaluate.py` — **MINOR** — Update if needed

### Files to DEPRECATE (rename to .txt)

1. `scripts/merge.py` → `scripts/merge.py.txt`
2. `scripts/build_context.py` → `scripts/build_context.py.txt`
3. `scripts/chunk_and_embed.py` → `scripts/chunk_and_embed.py.txt`

---

## Implementation Approach

### Orchestrator (`scripts/run_pipeline.py`)

```python
def run_pipeline(config: PipelineConfig) -> PipelineReport:
    results = []

    if config.input_path.is_dir():
        zips = sorted(config.input_path.glob("*.zip"))
    else:
        zips = [config.input_path]

    for zip_path in zips:
        class_name = zip_path.stem
        output_dir = config.output_dir / class_name
        report = process_single_class(zip_path, output_dir, config)
        results.append(report)

    return PipelineReport(sessions=results)

def process_single_class(zip_path, output_dir, config) -> ClassSessionReport:
    steps = [
        ("ingest_zip", run_ingest_zip),
        ("match_identity", run_match_identity),
        ("transcribe_dual", run_transcribe_dual),
        ("merge_transcripts", run_merge_transcripts),
        ("build_context", run_build_student_context),
        ("embed_and_store", run_embed_and_store),
    ]

    step_results = {}
    for step_name, step_fn in steps:
        try:
            result = step_fn(output_dir, config)
            step_results[step_name] = StepResult(success=True, ...)
        except Exception as e:
            step_results[step_name] = StepResult(success=False, error=str(e))
            break  # Stop this class, continue to next in batch

    return ClassSessionReport(class_name=class_name, step_results=step_results)
```

### Error Handling Strategy

- If a step fails for one class, log the error and skip to the next zip in batch mode
- Never crash the entire batch for one bad zip
- Each step function validates its inputs before running
- Pipeline report summarizes successes and failures

### pgvector Retrieval (`scripts/retrieval.py`)

Replace ChromaDB queries:

```python
# OLD (ChromaDB)
collection = chroma_client.get_collection(name)
results = collection.query(query_embeddings=[embedding], n_results=top_k, where={"student_id": sid})

# NEW (pgvector)
from scripts.utils.pg_store import PgVectorStore
store = PgVectorStore(db_url)
results = store.search(query_embedding=embedding, student_id=sid, top_k=top_k)
```

Keep the `RetrievedChunk` and `RetrievalResult` Pydantic models unchanged. The `store.search()` returns `SearchResult` objects that are mapped to `RetrievedChunk` at the boundary.

### Chat Update (`scripts/chat.py`)

Minimal changes:
- Update import: `from scripts.retrieval import retrieve_context` (function signature unchanged)
- The `retrieve_context` function now internally uses pgvector instead of ChromaDB
- Chat prompts, Groq integration, and session handling unchanged

### Deprecation of Old Scripts

Per CLAUDE.md convention ("prefer changing file extension to .txt to preserve files marked for deletion"):

```
git mv scripts/merge.py scripts/merge.py.txt
git mv scripts/build_context.py scripts/build_context.py.txt
git mv scripts/chunk_and_embed.py scripts/chunk_and_embed.py.txt
```

Also deprecate old utility references that are no longer imported.

### Pydantic Models

```
PipelineReport
  - input_path: str
  - sessions: list[ClassSessionReport]
  - total_duration_seconds: float
  - total_classes: int
  - successful_classes: int
  - failed_classes: int

ClassSessionReport
  - class_name: str
  - zip_file: str
  - output_dir: str
  - step_results: dict[str, StepResult]
  - success: bool
  - error: str | None

StepResult
  - step_name: str
  - success: bool
  - duration_seconds: float
  - output_files: list[str]
  - error: str | None
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] Orchestrator processes single zip end-to-end
- [ ] Orchestrator processes directory of zips in batch mode
- [ ] Failed class doesn't crash entire batch
- [ ] Pipeline report summarizes all class results
- [ ] `retrieval.py` queries pgvector instead of ChromaDB
- [ ] `chat.py` works with pgvector-backed retrieval
- [ ] Old scripts renamed to `.txt`
- [ ] No remaining imports from ChromaDB in active scripts

### Code Quality

- [ ] argparse CLI with `--input`, `--output-dir`, `--teacher`, `--roster`, `--attendance`, `--db-url`
- [ ] Complete type hints
- [ ] No bare except
- [ ] `ruff` and `mypy` pass clean
- [ ] All tests pass (old + new)

---

## Testing Requirements

### Unit Tests

1. **Orchestrator**
   - Single zip → all steps called in order
   - Batch mode (2 zips) → both processed
   - Step failure → logged, next zip started
   - All steps succeed → report shows all green

2. **pgvector Retrieval**
   - Search returns top-k results sorted by distance
   - Student ID filtering works
   - Empty results → empty list, no crash
   - Connection error → clear error message

3. **Chat Integration**
   - Chat with pgvector retrieval produces responses
   - Session recording works with new retrieval

4. **Deprecation**
   - Old scripts renamed to .txt
   - No active imports from deprecated files

Target: ~15 tests.

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Retrieval contract changes break chat | Keep RetrievedChunk and RetrievalResult models identical; only storage backend changes |
| Orchestrator timeout on long classes | Add per-step timeout with configurable limits; log progress |
| Old script imports in evaluate.py | Audit all imports; update evaluate.py to use new modules |

---

## Related Tasks

- **TASK-011 through TASK-017**: All prior tasks feed into the orchestrator
- This is the final task in the rebuild

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~10 files (2 created, 6 modified, 3 renamed)
- **GPU needed** only for the transcription step within the orchestrator
- **End-to-end validation**: After this task, the entire pipeline runs from `run_pipeline.py` with a single command
