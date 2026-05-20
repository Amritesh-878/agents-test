# TASK-007: Chunk and Embed Diarized Transcript into ChromaDB

**Status:** ✅ Completed

**Completed:** 2026-05-20

## Outcome

TASK-007 now converts the Phase 1 diarized transcript plus student context outputs into a deterministic, student-scoped chunk catalog, persists that catalog into local ChromaDB, and writes inspectable review exports for chunk provenance and student relevance review.

## Delivered Files

- `scripts/chunk_and_embed.py`
- `scripts/utils/chunker.py`
- `tests/test_chunk_and_embed.py`
- `tests/test_chunker.py`
- `requirements.txt`
- `README.md`
- `.gitignore`

## Validation

- `python -m ruff check --fix .` ✅ PASS
- `python -m mypy .` ✅ PASS
- `python -m pytest` ✅ PASS (`75 passed`, `0 warnings`)
- Real ingestion run against `verify-main-clean/output/transcript_diarized.json` and `verify-main-clean/output/student_contexts.json` ✅ PASS
- Re-run idempotence check against the same Chroma collection ✅ PASS (`2116` chunks before and after rerun)

## Metadata Contract for TASK-008

Every stored chunk preserves the following review-critical fields:

- `chunk_id`
- `student_id`
- `student_name`
- `chunk_type`
- `start`, `end`, `duration_seconds`
- `source_speaker`
- `source_mapped_student`
- `source_mapping_confidence`
- `source_manual_review_required`
- `attendance_accuracy`
- `attendance_source_mode`
- `attendance_estimated`
- `approximate`
- `source_segment_ids_json`
- `source_segment_indices_json`
- `source_segment_refs_json`

## Notes for Next Task

- Retrieval must preserve student scoping with `student_id` as the primary filter key.
- Retrieval debug output should expose `chunk_id`, `chunk_type`, and `source_segment_refs_json` verbatim so a reviewer can trace any answer back to the original transcript span.
- Low-confidence speaker mapping and estimated attendance flags are intentionally kept on every chunk and must not be dropped in TASK-008.
