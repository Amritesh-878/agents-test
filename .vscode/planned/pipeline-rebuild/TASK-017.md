# TASK-017: PostgreSQL + pgvector Migration Script and Embedding Pipeline

## Overview

Create the PostgreSQL database migration script (schema, pgvector extension, indexes) and replace the ChromaDB embedding pipeline with a pgvector-based one. Reuse the existing chunking logic.

## Execution Snapshot

- Depends on: TASK-011, TASK-016
- Produces: `scripts/migrate_db.py`, `scripts/embed_and_store.py`, `scripts/utils/pg_store.py`
- Primary validation: `python scripts/migrate_db.py --db-url postgresql://...` then `python scripts/embed_and_store.py --contexts output/<class>/student_contexts.json --transcript output/<class>/transcript_merged.json --db-url postgresql://...`
- Complexity: Medium

## Goals

1. **Migration Script**: Idempotent schema creation (pgvector extension, embeddings table, indexes)
2. **Embedding Pipeline**: Chunk student contexts, embed with sentence-transformers, upsert to pgvector
3. **Helper Module**: Reusable PostgreSQL + pgvector connection/query utilities
4. **Chunker Integration**: Reuse existing `scripts/utils/chunker.py` (vector-DB-agnostic)

---

## Reasoning

### Why pgvector over ChromaDB?

**Current Problems:**

- ChromaDB is an in-process embedded database — limited concurrency, no multi-process access
- No built-in schema migrations or versioning
- Not suitable for production deployment or multi-user chatbot access

**Solution:**

- PostgreSQL is production-grade with decades of reliability
- pgvector extension adds native vector similarity search
- HNSW indexes for fast approximate nearest neighbor queries
- Standard SQL for filtering, joins, and metadata queries
- Same embedding model (sentence-transformers) — only the storage layer changes

### Why a separate migration script?

**Current Problems:**

- ChromaDB auto-created collections on first use — convenient but not production-safe
- No way to version the schema or roll back changes

**Solution:**

- `migrate_db.py` as a dedicated setup step run once before the pipeline
- Idempotent SQL: `CREATE EXTENSION IF NOT EXISTS`, `CREATE TABLE IF NOT EXISTS`
- Clear separation between schema management and data ingestion

---

## Files to Change

### Files to CREATE

1. `scripts/migrate_db.py` — **MAJOR** — Database migration CLI script
2. `scripts/embed_and_store.py` — **MAJOR** — Chunking + embedding + pgvector upsert CLI script
3. `scripts/utils/pg_store.py` — **MINOR** — PostgreSQL + pgvector helper module
4. `tests/test_migrate_db.py` — **MINOR** — Migration tests (mock psycopg)
5. `tests/test_embed_and_store.py` — **MAJOR** — Embedding pipeline tests
6. `tests/test_pg_store.py` — **MINOR** — Helper tests

### Files to MODIFY

1. `requirements.txt` — **MINOR** — Ensure `psycopg[binary]>=3.1`, `pgvector>=0.2` present (may already be added in TASK-011)
2. `.env.example` — **MINOR** — Add `DATABASE_URL=postgresql://user:pass@localhost:5432/adira`

---

## Implementation Approach

### Database Schema (`scripts/migrate_db.py`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    student_name TEXT NOT NULL,
    class_name TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
    start_time FLOAT,
    end_time FLOAT,
    speaker TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_student_class
    ON embeddings (student_id, class_name);

CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

The migration script:
1. Connects to PostgreSQL via `DATABASE_URL` env var or `--db-url` flag
2. Executes idempotent DDL
3. Reports results: tables created, indexes created, success status

### Helper Module (`scripts/utils/pg_store.py`)

```python
class PgVectorStore:
    def __init__(self, db_url: str) -> None:
        self.conn = psycopg.connect(db_url)
        register_vector(self.conn)  # pgvector extension

    def upsert_chunks(self, chunks: list[EmbeddingRecord]) -> int:
        # INSERT ... ON CONFLICT (id) DO UPDATE
        pass

    def delete_class_chunks(self, class_name: str) -> int:
        # DELETE FROM embeddings WHERE class_name = $1
        pass

    def search(self, query_embedding: list[float], student_id: str,
               top_k: int = 5) -> list[SearchResult]:
        # SELECT *, embedding <=> $1 AS distance
        # FROM embeddings WHERE student_id = $2
        # ORDER BY distance LIMIT $3
        pass

    def close(self) -> None:
        self.conn.close()
```

### Embedding Pipeline (`scripts/embed_and_store.py`)

1. Load student contexts + merged transcript
2. Reuse `scripts/utils/chunker.py` to chunk into `ChunkRecord` objects
3. For each chunk, compute embedding via sentence-transformers (`all-MiniLM-L6-v2`)
4. Build `EmbeddingRecord` objects with stable chunk IDs (SHA-1 hash of content)
5. Delete stale chunks for this class_name
6. Upsert new chunks
7. Write review artifacts: `rag_chunks.jsonl`, `rag_chunk_review.csv`

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [chunk.text for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)
```

### Chunker Integration

`scripts/utils/chunker.py` is vector-DB-agnostic. It produces `ChunkRecord` objects with:
- `chunk_type`: "spoken", "missed", "class_context"
- `student_id`, `text`, `duration_seconds`, `source_segment_refs`

Only the storage backend changes (ChromaDB → pgvector). The chunking logic itself stays the same. May need to add a new `chunk_type` for absent student summaries.

### Pydantic Models

```
EmbeddingRecord
  - id: str                     # SHA-1 of student_id + chunk_type + text
  - student_id: str
  - student_name: str
  - class_name: str
  - chunk_type: str
  - text: str
  - embedding: list[float]      # 384-dim vector
  - start_time: float | None
  - end_time: float | None
  - speaker: str | None
  - metadata: dict[str, Any]
  - created_at: datetime | None

SearchResult
  - chunk_id: str
  - student_id: str
  - text: str
  - distance: float
  - chunk_type: str
  - metadata: dict[str, Any]

MigrationResult
  - tables_created: list[str]
  - indexes_created: list[str]
  - extensions_created: list[str]
  - success: bool
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] `migrate_db.py` creates pgvector extension, embeddings table, and indexes
- [ ] Migration is idempotent (safe to run multiple times)
- [ ] `embed_and_store.py` chunks, embeds, and upserts to pgvector
- [ ] Stale chunks deleted before new insert (by class_name)
- [ ] Stable chunk IDs (same input → same ID)
- [ ] `pg_store.py` search returns correct nearest neighbors
- [ ] Review artifacts generated (rag_chunks.jsonl, rag_chunk_review.csv)
- [ ] Handles absent student summaries (chunk_type for summaries)

### Code Quality

- [ ] argparse CLI on both scripts
- [ ] Complete type hints
- [ ] No bare except
- [ ] `ruff` and `mypy` pass clean

---

## Testing Requirements

### Unit Tests (mock psycopg — no real DB needed for tests)

1. **Migration**
   - DDL SQL correctness (extension, table, indexes)
   - Idempotent execution (no error on second run)
   - Connection error handling

2. **Embedding Pipeline**
   - Chunk-to-EmbeddingRecord conversion
   - Stable chunk ID generation (deterministic hash)
   - Stale chunk deletion before upsert
   - Empty input → no upsert, no crash

3. **PgVectorStore**
   - Upsert SQL generation
   - Search SQL generation (cosine distance)
   - Delete by class_name
   - Connection lifecycle (connect/close)

4. **Chunker Integration**
   - New context models chunk correctly
   - Absent student summary chunks with correct type

Target: ~20 tests.

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| pgvector not installed on user's PostgreSQL | `migrate_db.py` checks for extension and gives clear error with install instructions |
| psycopg binary not available on Windows | Use `psycopg[binary]` which ships pre-built wheels for Windows |
| HNSW index build slow on large datasets | Index built on first query, not on INSERT; acceptable for class-sized datasets (~1000 chunks) |
| Embedding model download on first run | sentence-transformers handles caching; document first-run behavior |

---

## Related Tasks

- **TASK-011**: Added `psycopg` and `pgvector` to requirements
- **TASK-016**: Provides `student_contexts.json` and `transcript_merged.json`
- **TASK-018**: Updates `retrieval.py` and `chat.py` to use `pg_store.py`

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~8 files (6 created, 2 modified)
- **No GPU needed for migration/store** — GPU only for sentence-transformers embedding (uses CPU fallback if needed)
- **PostgreSQL must be running locally** — document install steps in README
