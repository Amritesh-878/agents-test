# TASK-007: Chunk and Embed Diarized Transcript into ChromaDB

## Overview

Convert the completed Phase 1 outputs into a canonical RAG chunk catalog, store those chunks in ChromaDB, and emit review artifacts that keep speaker mapping, attendance accuracy, and student-specific provenance easy to inspect.

## Execution Snapshot

- Depends on: TASK-006
- Produces: `scripts/chunk_and_embed.py`, `scripts/utils/chunker.py`, `data/chroma/`, `output/rag_chunks.jsonl`, `output/rag_chunk_review.csv`, `output/rag_chunk_review.md`
- Primary validation: `python scripts/chunk_and_embed.py --transcript output/transcript_diarized.json --contexts output/student_contexts.json --chroma-dir data/chroma --chunk-debug output/rag_chunks.jsonl --review-csv output/rag_chunk_review.csv --review-markdown output/rag_chunk_review.md`
- Complexity: Medium

## Goals

1. **Traceable Chunking**: Split the diarized transcript into coherent chunks while preserving exact provenance back to source transcript segments and student context entries.
2. **Student-Scoped Metadata**: Tag every embedded chunk with the student and context fields needed for safe retrieval, including whether the chunk came from `missed`, `spoken`, or shared class context.
3. **Inspectable Storage**: Persist embeddings locally in ChromaDB and also write human-readable and machine-readable review artifacts so chunk quality can be audited before retrieval work begins.

---

## Reasoning

### Why treat inspectability as part of ingestion instead of a later debug task?

**Current Problems:**

- Embeddings and vector stores are opaque unless a parallel review artifact is written during ingestion.
- Phase 1 already marks speaker mappings and attendance windows as estimated in some cases, and that uncertainty would be lost if Phase 2 stores only raw text plus a student id.
- Retrieval and chatbot evaluation will be unreliable if there is no way to inspect which source transcript segments produced each stored chunk.

**Solution:**

- Define a canonical chunk record with source segment ids, timestamp range, speaker metadata, mapped participant, mapping confidence, and attendance accuracy flags.
- Write every chunk record both to ChromaDB and to an inspectable export such as JSONL plus review CSV or Markdown.
- Preserve low-confidence and manual-review-required flags all the way into the stored metadata.

### Why chunk from both `transcript_diarized.json` and `student_contexts.json`?

**Current Problems:**

- The diarized transcript is the source of truth for speaker-timestamp alignment, but the student context output carries the per-student personalization cues the chatbot actually needs.
- If chunks are built only from the raw transcript, retrieval loses whether a segment is something the student missed, said themselves, or was present for.
- If chunks are built only from the student contexts, it becomes harder to trace back to the original diarized transcript and speaker review artifacts.

**Solution:**

- Use `transcript_diarized.json` as the canonical source for segment identity, timestamps, and speaker provenance.
- Use `student_contexts.json` to project those segments into student-specific chunk records with chunk types such as `missed`, `spoken`, and `class_context`.
- Store both the student projection metadata and the original transcript provenance on every chunk.

---

## Files to Change

### Files to CREATE

1. `scripts/chunk_and_embed.py` - CLI entry point for chunking, embedding, and ChromaDB upsert.
2. `scripts/utils/chunker.py` - pure functions or small Pydantic-backed helpers that convert segment lists into bounded chunks.
3. `tests/test_chunk_and_embed.py` - ingestion and metadata contract tests.
4. `tests/test_chunker.py` - chunking behavior tests.

### Files to MODIFY

#### Dependencies and Documentation

1. `requirements.txt` - **MAJOR** - add pinned dependencies for ChromaDB and sentence-transformers.
2. `README.md` - **MINOR** - document Phase 2 ingestion commands and review artifacts.

---

## Implementation Approach

### `scripts/utils/chunker.py`

**Purpose:** Convert diarized transcript segments into chunk records that are semantically coherent and still traceable to the original transcript.

**Key Responsibilities:**

- Group nearby segments into chunks without crossing unsafe provenance boundaries.
- Preserve a list of source segment references for every output chunk.
- Respect chunk size targets while keeping speaker or student-context transitions understandable.
- Return data structures that are directly serializable into review artifacts and Chroma metadata.

**Integration Points:**

- Reads parsed segment models derived from `output/transcript_diarized.json`.
- Feeds chunk records into `scripts/chunk_and_embed.py`.

**Considerations:**

- Do not merge segments that would hide a transition between `missed`, `spoken`, and shared class context.
- Carry forward `manual_review_required`, `mapping_confidence`, and attendance accuracy metadata from Phase 1 outputs.
- Keep the module focused and testable; avoid folding Chroma or embedding code into it.

---

### `scripts/chunk_and_embed.py`

**Purpose:** Build student-scoped chunk records, embed them, upsert them into ChromaDB, and write parallel review artifacts.

**Key Responsibilities:**

- Load `transcript_diarized.json` and `student_contexts.json` with Pydantic validation.
- Build chunk records for `missed`, `spoken`, and `class_context` coverage per student.
- Upsert documents into a local ChromaDB collection under `data/chroma/`.
- Write inspectable exports such as JSONL, CSV, and Markdown summaries.

**Integration Points:**

- Consumes the final Phase 1 review outputs as upstream truth.
- Establishes the metadata contract consumed by TASK-008 retrieval.

**Considerations:**

- Prefer deterministic chunk ids so re-runs are idempotent.
- Include only JSON-serializable metadata fields compatible with ChromaDB.
- Write review outputs that make it easy to answer: which student got this chunk, why, from which transcript spans, and under what confidence assumptions?

---

## Acceptance Criteria

### Functional Requirements

- [ ] ChromaDB collection is created under `data/chroma/` and can be rebuilt without duplicate records.
- [ ] Every chunk includes `student_id`, `student_name`, `chunk_type`, timestamps, source speaker, mapped speaker, mapping confidence, attendance accuracy, and source segment references.
- [ ] Inspectable review artifacts are emitted to `output/` and let a reviewer trace any chunk back to the source transcript and student context.
- [ ] Chunking logic preserves enough context for retrieval without producing empty or trivially tiny chunks.

### Code Quality

- [ ] All new functions and methods have complete type hints.
- [ ] Pydantic models define the canonical chunk schema instead of loose dicts.
- [ ] Re-runs are deterministic and idempotent.
- [ ] No compilation errors or warnings.

---

## Testing Requirements

### Unit Tests

1. **Chunk Boundary Behavior**
   - Merges short adjacent transcript units without hiding provenance transitions.
   - Splits long transcript regions into bounded chunks while keeping source references intact.
   - Preserves speaker and chunk-type boundaries when required.

2. **Metadata Contract**
   - Emits required provenance fields for `missed`, `spoken`, and shared context chunks.
   - Carries forward low-confidence speaker mapping and estimated attendance flags.
   - Generates deterministic chunk ids on repeated runs.

### Integration Tests

1. Ingestion run against a small fixture writes Chroma records plus review artifacts with matching chunk counts.
2. Re-running ingestion against the same fixture does not duplicate stored records.

---

## Risks and Mitigation

| Risk                                             | Mitigation                                                         |
| ------------------------------------------------ | ------------------------------------------------------------------ |
| ChromaDB ids collide or duplicate on re-run      | Use deterministic ids and upsert semantics instead of blind add    |
| Chunking obscures speaker or student provenance  | Keep source segment references and provenance flags on every chunk |
| Review artifacts drift from stored metadata      | Generate all exports from the same validated chunk record objects  |
| First-run embedding download adds setup friction | Document the model choice and expected download size in the README |

---

## Related TODOs

- **TASK-006**: Upstream dependency - provides the inspectable diarized transcript and student context artifacts.
- **TASK-008**: Downstream dependency - consumes the chunk schema and Chroma metadata contract.

---

## Handoff Template

**Status:** ⏳ Not Started

```
When implemented, confirm that a reviewer can pick any row in output/rag_chunk_review.csv,
find the same chunk id in output/rag_chunks.jsonl and ChromaDB, and trace it back to the
original transcript span plus the student-context classification that created it.
```

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~6 files
- **No GPU needed:** Embedding should be planned for CPU-safe local execution unless profiling proves otherwise
