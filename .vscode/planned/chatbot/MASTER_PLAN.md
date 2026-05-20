# Master Implementation Plan

**Project:** WhisperX Personalized RAG + CLI Chatbot (Phase 2)

**Date Updated:** 2026-05-20

**Planning Note:** This worktree bootstraps the Phase 2 planning files that were missing from the branch point. The canonical planning intent still matches the source repo's `.vscode/planned/chatbot/` files.

---

## Status Summary

| Phase | Task                                | Status         | Build | Tests |
| ----- | ----------------------------------- | -------------- | ----- | ----- |
| 2     | TASK-007: Chunk + Embed into Chroma | ✅ Completed   | ✅    | ✅    |
| 2     | TASK-008: Retrieval Layer           | ⏳ Not Started | ❓    | ❓    |
| 2     | TASK-009: CLI Chatbot (Groq + RAG)  | ⏳ Not Started | ❓    | ❓    |
| 2     | TASK-010: RAG Evaluation            | ⏳ Not Started | ❓    | ❓    |

---

## Current Verification

- ✅ `scripts/chunk_and_embed.py` builds deterministic student-scoped chunk records from `transcript_diarized.json` and `student_contexts.json`
- ✅ `scripts/utils/chunker.py` preserves chunk-type and source-speaker boundaries while merging nearby transcript units into bounded chunks
- ✅ Review artifacts are written in both machine-readable and human-readable form: `output/rag_chunks.jsonl`, `output/rag_chunk_review.csv`, and `output/rag_chunk_review.md`
- ✅ Local ChromaDB persistence is idempotent under `data/chroma/` via deterministic ids plus upsert/delete sync semantics
- ✅ Validation passed in this worktree with `ruff check --fix .`, `mypy .`, and `pytest` (`75 passed`, `0 warnings`)
- ✅ Runtime validation against the verified Phase 1 outputs succeeded twice with the collection count remaining stable at `2116`

---

## Task Status Tracker

| Phase | TODO | Title                            | Status         | Notes                                                                                   |
| ----- | ---- | -------------------------------- | -------------- | --------------------------------------------------------------------------------------- |
| 2     | 007  | Chunk + Embed into ChromaDB      | ✅ Completed   | Canonical chunk schema, Chroma ingestion, and inspectable review artifacts are in place |
| 2     | 008  | Student-Scoped Retrieval Layer   | ⏳ Not Started | Consume the TASK-007 collection and preserve chunk ids plus source segment references   |
| 2     | 009  | CLI Chatbot (Groq + RAG)         | ⏳ Not Started | Depend on the stable retrieval contract from TASK-008                                   |
| 2     | 010  | RAG Evaluation with Golden Truth | ⏳ Not Started | Use the saved retrieval and chat traces for source-linked failed-case review            |

---

## Handoff Notes

### TODO-007 Handoff

**Status:** ✅ Completed

Completed by: GPT-5.4
Build status: ✅ PASS

### What was done:

- Added `scripts/chunk_and_embed.py` with deterministic chunk ids, Chroma upsert/delete sync, and JSONL/CSV/Markdown review exports
- Added `scripts/utils/chunker.py` with Pydantic-backed chunk projection and bounded merge logic that preserves provenance boundaries
- Added `tests/test_chunker.py` and `tests/test_chunk_and_embed.py` covering chunk boundaries, metadata contract, review artifact output, and idempotent Chroma storage
- Updated `requirements.txt`, `.gitignore`, and `README.md` for the Phase 2 ingestion workflow and local embedding stack

### Tests passing: ✅ 75 tests

### Warnings to next implementor:

- `source_segment_refs_json`, `source_segment_ids_json`, and `source_segment_indices_json` are stored in Chroma metadata as JSON strings because Chroma metadata values must remain primitive
- `student_id` is derived deterministically from email when present, otherwise from a slugified student name; TASK-008 should treat that as the stable student filter key
- `class_context` chunks are student-specific complements of `missed` and `spoken`, so retrieval should filter by both `student_id` and `chunk_type`
- Low-confidence speaker mapping and estimated attendance remain visible on every chunk and must stay surfaced in retrieval debug outputs

### Breaking changes:

- None. This task adds the new Phase 2 ingestion slice without changing Phase 1 schemas.

---

### TODO-008 Handoff

**Status:** ⏳ Not Started

**Prerequisites from TASK-007:**

- [x] Canonical chunk schema defined and covered by tests
- [x] Review artifacts available for spot-checking chunk quality
- [x] ChromaDB collection name and metadata contract stabilized as `student_transcript_chunks`

```
Retrieval debug output is a required feature, not a developer convenience.
Do not hide chunk ids, chunk_type, source_segment_refs_json, or mapping-confidence flags behind the TASK-008 library API.
```
