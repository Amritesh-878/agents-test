# Master Implementation Plan

**Project:** WhisperX Personalized RAG + CLI Chatbot (Phase 2)

**Date Updated:** 2026-05-20

**Planning Note:** This worktree bootstraps the Phase 2 planning files that were missing from the branch point. The canonical planning intent still matches the source repo's `.vscode/planned/chatbot/` files.

---

## Status Summary

| Phase | Task                                | Status         | Build | Tests |
| ----- | ----------------------------------- | -------------- | ----- | ----- |
| 2     | TASK-007: Chunk + Embed into Chroma | ✅ Completed   | ✅    | ✅    |
| 2     | TASK-008: Retrieval Layer           | ✅ Completed   | ✅    | ✅    |
| 2     | TASK-009: CLI Chatbot (Groq + RAG)  | ✅ Completed   | ✅    | ✅    |
| 2     | TASK-010: RAG Evaluation            | ⏳ Not Started | ❓    | ❓    |

---

## Current Verification

- ✅ `scripts/chunk_and_embed.py` builds deterministic student-scoped chunk records from `transcript_diarized.json` and `student_contexts.json`
- ✅ `scripts/utils/chunker.py` preserves chunk-type and source-speaker boundaries while merging nearby transcript units into bounded chunks
- ✅ Review artifacts are written in both machine-readable and human-readable form: `output/rag_chunks.jsonl`, `output/rag_chunk_review.csv`, and `output/rag_chunk_review.md`
- ✅ Local ChromaDB persistence is idempotent under `data/chroma/` via deterministic ids plus upsert/delete sync semantics
- ✅ `scripts/retrieval.py` returns strict student-scoped ranked chunks, a prompt-ready context string, and deterministic debug JSON under `output/retrieval_debug/`
- ✅ `scripts/chat.py` now provides a Groq-backed student chat loop with `context`, `sources`, `help`, and `quit` commands plus JSON session traces under `output/chat_sessions/`
- ✅ Session traces persist the prompt question, full `RetrievalResult`, prompt messages, model id, answer text, and trust flags so TASK-010 can inspect the same evidence used at generation time
- ✅ Validation passed in this worktree with `ruff check --fix .`, `mypy .`, and `pytest` (`90 passed`, `0 warnings`)
- ✅ Runtime validation against the verified Phase 1 outputs succeeded twice with the collection count remaining stable at `2116`
- ✅ Runtime retrieval validation succeeded against the TASK-007 Chroma store with student id `a-disha-2504`, returning 5 ranked chunks and writing `output/retrieval_debug/sample_query.json`
- ✅ Runtime chat validation succeeded for student `a-disha-2504`, writing `output/chat_sessions/20260520T151934Z-a-disha-2504.json` with a grounded answer that acknowledged low-confidence estimated context

---

## Task Status Tracker

| Phase | TODO | Title                            | Status         | Notes                                                                                   |
| ----- | ---- | -------------------------------- | -------------- | --------------------------------------------------------------------------------------- |
| 2     | 007  | Chunk + Embed into ChromaDB      | ✅ Completed   | Canonical chunk schema, Chroma ingestion, and inspectable review artifacts are in place |
| 2     | 008  | Student-Scoped Retrieval Layer   | ✅ Completed   | Strict `student_id` filtering, provenance-rich results, and debug exports are in place  |
| 2     | 009  | CLI Chatbot (Groq + RAG)         | ✅ Completed   | Groq-backed chat loop, debug commands, and inspectable session traces are in place      |
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

**Status:** ✅ Completed

Completed by: GPT-5.4
Build status: ✅ PASS

### What was done:

- Added `scripts/retrieval.py` with a strict `student_id` Chroma filter path, Pydantic `RetrievedChunk` and `RetrievalResult` models, prompt-ready context formatting, and optional JSON debug export
- Added `tests/test_retrieval.py` covering student scoping, provenance preservation, chunk-type filters, empty results, and debug serialization against a live temporary Chroma collection
- Updated `README.md` with retrieval CLI usage, debug artifact output, and student-scope guarantees
- Runtime-validated retrieval against the TASK-007 Chroma store by querying student `a-disha-2504` and writing `output/retrieval_debug/sample_query.json`

### Tests passing: ✅ 83 tests

### Warnings to next implementor:

- Retrieval uses `student_id` as the only student-scope key; TASK-009 should pass the stable slugified student id rather than student display names
- `source_segment_ids_json`, `source_segment_indices_json`, and `source_segment_refs_json` are decoded back into structured provenance during retrieval and should be reused directly in chat/session traces
- Empty retrieval is a first-class result with warning strings, so TASK-009 should treat `result_count == 0` as a safe fallback path instead of as an exception
- The real validation query for `a-disha-2504` returned `class_context` chunks first for the generic question "What did I miss?"; TASK-009 prompt design should account for mixed chunk types instead of assuming only `missed` chunks

### Breaking changes:

- None. TASK-008 adds the retrieval layer without changing the TASK-007 metadata contract.

**Prerequisites from TASK-007:**

- [x] Canonical chunk schema defined and covered by tests
- [x] Review artifacts available for spot-checking chunk quality
- [x] ChromaDB collection name and metadata contract stabilized as `student_transcript_chunks`

```
Retrieval debug output is a required feature, not a developer convenience.
Do not hide chunk ids, chunk_type, source_segment_refs_json, or mapping-confidence flags behind the TASK-008 library API.
```

---

### TODO-009 Handoff

**Status:** ✅ Completed

Completed by: GPT-5.4
Build status: ✅ PASS

### What was done:

- Added `scripts/chat.py` with an argparse-driven Groq chat loop, bounded conversation history, early `GROQ_API_KEY` validation, and debug commands for `context`, `sources`, `help`, and `quit`
- Added `tests/test_chat.py` covering prompt construction, session trace writing, debug command output, history bounding, fallback behavior for empty retrieval, and missing-key failure
- Updated `requirements.txt` with pinned Groq and `httpx` versions after real runtime validation exposed the SDK compatibility requirement in the project venv
- Bootstrapped the missing `.vscode/planned/chatbot/TASK-009.md` file into this worktree and validated a real single-turn run that wrote `output/chat_sessions/20260520T151934Z-a-disha-2504.json`

### Tests passing: ✅ 90 tests

### Warnings to next implementor:

- Session traces intentionally store full nested `RetrievalResult` objects plus prompt messages so TASK-010 can score answers against the exact evidence sent to Groq
- Real retrieval for broad questions like "What did I miss in class?" can still surface only `class_context` chunks, so TASK-010 should judge evidence sufficiency and trust flags rather than assume every answer will be based on `missed` chunks
- The retained runtime Chroma store was copied into this worktree under `data/chroma/` for validation only; do not commit copied vector data or generated chat sessions

### Breaking changes:

- None. TASK-009 adds the chat layer and inspectable traces without changing the retrieval schema contract.
