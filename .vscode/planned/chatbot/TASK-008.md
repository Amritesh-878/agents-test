# TASK-008: Student-Scoped Retrieval Layer

## Overview

Build a retrieval module that takes a student id and a question, fetches the most relevant chunks from ChromaDB, and returns both a prompt-ready context string and an inspectable record of why those chunks were selected.

## Execution Snapshot

- Depends on: TASK-007
- Produces: `scripts/retrieval.py`, `tests/test_retrieval.py`, optional debug exports under `output/retrieval_debug/`
- Primary validation: `python scripts/retrieval.py --student-id <student_id> --query "What did I miss?" --chroma-dir data/chroma --debug-output output/retrieval_debug/sample_query.json`
- Complexity: Medium

## Goals

1. **Student-Safe Retrieval**: Ensure queries are filtered to the correct student scope so missed and spoken segments from other students never leak into the result set.
2. **Structured Ranking Output**: Return a Pydantic-backed retrieval result that exposes scores, chunk provenance, and source references instead of only a plain text context block.
3. **Inspectable Query Traces**: Make every retrieval query easy to review manually by supporting a debug export with the ranked chunk list and the metadata that justified it.

---

## Reasoning

### Why isolate retrieval from the chat loop?

**Current Problems:**

- If retrieval is embedded inside the chatbot, it becomes hard to tell whether a bad answer came from poor chunk selection or poor generation.
- Student-scoping mistakes are one of the most serious correctness failures for this project and need direct tests.
- TASK-010 needs to evaluate retrieval hit rate independently from answer faithfulness.

**Solution:**

- Implement retrieval as a separate module with a stable API and a small CLI debug mode.
- Return a structured `RetrievalResult` model rather than hiding ranked chunks behind a formatted prompt string.
- Save optional debug output per query so failures can be reviewed without rerunning the whole chat flow.

### Why include provenance in `RetrievalResult` instead of only scores and text?

**Current Problems:**

- Manual reviewers need to see whether a result came from `missed`, `spoken`, or shared class context.
- Phase 1 already marks estimated attendance windows and low-confidence speaker mappings, and those constraints matter when judging personalization quality.
- Similarity scores alone are not enough to decide whether a chunk is appropriate for a student's question.

**Solution:**

- Include chunk id, timestamps, chunk type, speaker fields, mapped participant, mapping confidence, attendance accuracy, and source segment references in each retrieved chunk.
- Build the prompt context string from the same structured objects used for the debug export.
- Preserve enough metadata to answer: why did this result appear, and should a reviewer trust it?

---

## Files to Change

### Files to CREATE

1. `scripts/retrieval.py` - retrieval module plus argparse-driven debug entry point.
2. `tests/test_retrieval.py` - retrieval behavior and serialization tests.

### Files to MODIFY

#### Documentation

1. `README.md` - **MINOR** - document retrieval debug usage and expected artifacts.

---

## Implementation Approach

### `scripts/retrieval.py`

**Purpose:** Query the ChromaDB collection created in TASK-007 and return structured, student-scoped retrieval results.

**Key Responsibilities:**

- Load the Chroma collection and embedding model used during ingestion.
- Accept `student_id`, `query`, `top_k`, and optional filters such as chunk types.
- Query Chroma with student-scoped filters and rank the returned chunks.
- Build both a context string for the LLM and an optional debug export for manual review.

**Integration Points:**

- Consumes the chunk schema and metadata contract defined in TASK-007.
- Provides the retrieval API used by `scripts/chat.py` in TASK-009.
- Feeds TASK-010 with retrieval outputs for automated hit-rate measurement.

**Considerations:**

- The CLI mode should use `argparse` and write a JSON artifact when `--debug-output` is provided.
- Empty results should be explicit and well-formed rather than raising opaque errors.
- The same formatting function should drive both the prompt context and any human-readable debug report to avoid drift.

---

## Acceptance Criteria

### Functional Requirements

- [ ] Retrieval filters strictly by student scope and never returns another student's personalized chunks.
- [ ] Retrieved chunks are ranked and returned with the provenance needed for manual inspection.
- [ ] `context_string` is formatted clearly with chunk ids, timestamps, chunk types, and trust-related flags where relevant.
- [ ] CLI debug mode can write a per-query artifact under `output/retrieval_debug/`.
- [ ] Queries for sparse or missing student data return a valid empty or partial result instead of failing unexpectedly.

### Code Quality

- [ ] All new functions and methods have complete type hints.
- [ ] Pydantic models define `RetrievedChunk` and `RetrievalResult`.
- [ ] Query serialization for debug output is deterministic and easy to diff.
- [ ] No compilation errors or warnings.

---

## Testing Requirements

### Unit Tests

1. **Student Filtering**
   - Querying for student A never returns student B's `missed` or `spoken` chunks.
   - Unknown or partially populated students return a safe empty result.

2. **Result Formatting**
   - `context_string` includes timestamps, chunk types, and chunk ids.
   - Structured output preserves mapping-confidence and attendance-accuracy fields.

3. **Debug Export**
   - `--debug-output` writes valid JSON with query metadata and ranked results.
   - Debug exports and in-memory results are derived from the same structured objects.

### Integration Tests

1. Retrieval against a small fixture collection returns expected chunks in rank order.
2. Retrieval hit checks can be reused directly by TASK-010 without additional adapters.

---

## Risks and Mitigation

| Risk                                                     | Mitigation                                                                             |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Cross-student leakage through incorrect metadata filters | Test filter logic explicitly and keep the student scope in one well-defined query path |
| Debug output diverges from actual retrieval objects      | Serialize the same validated result models that feed the LLM prompt                    |
| Over-reliance on scores without context semantics        | Include chunk type and provenance in both ranking review and prompt formatting         |
| Sparse student data produces brittle behavior            | Treat empty retrieval as a first-class supported outcome                               |

---

## Related TODOs

- **TASK-007**: Upstream dependency - defines the chunk schema and Chroma collection.
- **TASK-009**: Downstream dependency - uses retrieval results to build Groq prompts and debug commands.

---

## Handoff Template

**Status:** ⏳ Not Started

```
When implemented, a reviewer should be able to run one retrieval query, open the saved debug JSON,
and understand exactly which chunks were returned, why they belong to that student, and what source
evidence backs them.
```

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~3 files
- **No GPU needed:** Query embedding should be planned for CPU-local execution
