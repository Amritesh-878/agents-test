# TASK-010: RAG Evaluation with Golden Truth

**Status:** ✅ Completed

**Completed:** 2026-05-20

## Outcome

TASK-010 now provides a committed golden QA dataset plus a reproducible evaluation CLI that replays student-scoped retrieval and grounded generation, scores the result against real chunk evidence, and writes per-case JSON artifacts alongside aggregate JSON and Markdown reports. The bounded real evaluation run produced one safe-refusal pass and three inspectable failures across retrieval, provenance, and chat-generation stages.

## Delivered Files

- `data/eval_qa.json`
- `scripts/evaluate.py`
- `tests/test_evaluate.py`
- `.vscode/planned/chatbot/TASK-010.md`
- `.vscode/planned/chatbot/MASTER_PLAN.md`

## Validation

- `python -m ruff check --fix .` ✅ PASS
- `python -m mypy .` ✅ PASS
- `python -m pytest` ✅ PASS (`94 passed`, `0 warnings`)
- `python scripts/evaluate.py --eval-file data/eval_qa.json --chroma-dir ../task-007-chunk-embed/data/chroma --output-dir output/evaluation` ✅ PASS (`1 passed`, `3 failed`, reports written)

## Report Contract

- `output/evaluation/summary.json` captures the aggregate pass/fail counts and embeds each case result.
- `output/evaluation/summary.md` renders the high-level evaluation table for manual review.
- `output/evaluation/cases/<case_id>.json` stores the full retrieval and answer checks for every case.
- `output/evaluation/failed_cases/<case_id>.json` mirrors the failing cases so reviewers can inspect failure ownership directly.

## Notes for Next Task

- The golden set is based on the real student context review output, the real chunk export, and the saved TASK-009 session trace rather than synthetic examples.
- Failure ownership is exposed through both a heuristic `failure_stage` and the raw booleans for indexed evidence presence, retrieved evidence presence, trust flags, and answer scoring.
- Runtime evaluation should continue to use an external Chroma store from the validated TASK-007 or TASK-009 worktree rather than committing copied vector data into this branch.
