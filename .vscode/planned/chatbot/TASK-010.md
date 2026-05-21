# TASK-010: RAG Evaluation with Golden Truth

## Overview

Build a reproducible evaluation harness for TASK-007 through TASK-009 using a committed golden QA set grounded in the real class review artifacts. The evaluator must score retrieval evidence, answer quality, and provenance acknowledgement, then write inspectable case reports that make failure ownership visible.

## Execution Snapshot

- Depends on: TASK-007, TASK-008, TASK-009
- Produces: `data/eval_qa.json`, `scripts/evaluate.py`, `tests/test_evaluate.py`, runtime report outputs under `output/evaluation/`
- Primary validation: `python scripts/evaluate.py --eval-file data/eval_qa.json --chroma-dir ../task-007-chunk-embed/data/chroma --output-dir output/evaluation`
- Complexity: Medium

## Goals

1. Ground evaluation in the real Phase 1 review outputs, chunk exports, retrieval behavior, and saved chat traces.
2. Distinguish whether a failed case came from missing indexed evidence, poor top-k retrieval, answer-generation behavior, or low-confidence provenance.
3. Emit reports that are inspectable without rerunning the whole pipeline.

## Deliverables

1. `data/eval_qa.json`
   - Real questions and golden expectations tied to actual chunk ids, source segment ids, review notes, and saved session traces.
2. `scripts/evaluate.py`
   - CLI evaluator that runs retrieval plus grounded generation, writes aggregate reports, and stores per-case JSON artifacts.
3. `tests/test_evaluate.py`
   - Unit coverage for retrieval scoring, answer scoring, failure-stage classification, and report writing.

## Acceptance Criteria

- [x] The evaluator loads a committed golden dataset and validates arguments before any expensive work.
- [x] Each case records whether the expected evidence exists in the indexed chunk universe and whether top-k retrieval actually surfaced it.
- [x] Each case scores answer behavior against expected concepts or safe insufficiency language.
- [x] Aggregate and per-case report files are written under `output/evaluation/`.
- [x] Failed-case artifacts expose enough structure to tell whether the issue is chunking, retrieval, chat generation, or upstream low-confidence provenance.
- [x] `ruff check --fix .`, `mypy .`, and `pytest` pass in this worktree.

## Notes

- Prefer bounded real validation over synthetic examples.
- Do not commit generated evaluation outputs under `output/` unless they are explicit fixtures.
