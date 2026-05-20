# TASK-008: Student-Scoped Retrieval Layer

**Status:** ✅ Completed

**Completed:** 2026-05-20

## Outcome

TASK-008 now queries the TASK-007 Chroma collection with strict student scoping, returns provenance-rich ranked retrieval results, formats a prompt-ready context string from the same structured objects, and writes deterministic JSON debug artifacts for manual review.

## Delivered Files

- `scripts/retrieval.py`
- `tests/test_retrieval.py`
- `README.md`
- `.vscode/planned/chatbot/MASTER_PLAN.md`

## Validation

- `python -m ruff check --fix .` ✅ PASS
- `python -m mypy .` ✅ PASS
- `python -m pytest` ✅ PASS (`83 passed`, `0 warnings`)
- `python scripts/retrieval.py --student-id a-disha-2504 --query "What did I miss?" --chroma-dir data/chroma --debug-output output/retrieval_debug/sample_query.json` ✅ PASS

## Retrieval Contract for TASK-009

- Use `student_id` as the stable personalization scope key.
- Reuse `RetrievalResult` and `RetrievedChunk` directly instead of flattening provenance away.
- Preserve `chunk_id`, `chunk_type`, `trust_flags`, and `source_segment_refs` in saved chat traces.
- Treat `result_count == 0` plus `warnings` as a supported retrieval outcome.

## Notes for Next Task

- The retrieval debug JSON is deterministic and mirrors the same structured objects that build `context_string`.
- Generic questions may surface `class_context` first; TASK-009 should guide the model with chunk types and trust flags instead of assuming every answer comes only from `missed` content.
- This worktree runtime validation copied the TASK-007 Chroma store into `data/chroma/` for local querying. Do not commit copied Chroma data or generated retrieval debug artifacts.
