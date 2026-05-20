# TASK-009: CLI Chatbot Using Groq + RAG

## Overview

Build a CLI chatbot that uses the TASK-008 retrieval layer plus Groq generation to answer student questions, while saving enough per-turn evidence to inspect retrieved chunks, source provenance, and student-specific relevance after the session.

## Execution Snapshot

- Depends on: TASK-008
- Produces: `scripts/chat.py`, `tests/test_chat.py`, session artifacts under `output/chat_sessions/`, optional README updates
- Primary validation: `python scripts/chat.py --student-id <student_id> --student-name "<student_name>" --chroma-dir data/chroma --save-session-dir output/chat_sessions`
- Complexity: Medium

## Goals

1. **Grounded Personalized Answers**: Generate responses from the retrieved class context for that student rather than generic model knowledge.
2. **Inspectable Conversation Traces**: Save each turn's query, retrieved chunk set, and answer so manual reviewers can audit why the chatbot responded the way it did.
3. **Usable CLI Workflow**: Support a clean command-line flow for local testing, including debug commands that expose the last retrieved context without modifying code.

---

## Reasoning

### Why require saved session traces instead of only printing answers?

**Current Problems:**

- Manual evaluation becomes unreliable if the reviewer cannot reconstruct which chunks supported a specific answer.
- TASK-010 needs reusable answer traces to evaluate failure cases without manually replaying every conversation.
- Personalized-learning usefulness depends not just on fluency but on whether the answer reflects what the student missed, said, or was present for.

**Solution:**

- Persist structured session logs with user question, retrieval result, answer text, model id, and any provenance warnings.
- Make those logs easy to inspect in JSON or JSONL form under `output/chat_sessions/`.
- Ensure the chat UI exposes the same retrieval information through debug commands such as `context` or `sources`.

### Why surface provenance and uncertainty in the prompt design?

**Current Problems:**

- The current Phase 1 speaker mapping may be estimated or low-confidence, especially when Zoom attendance data lacks exact join and leave timestamps.
- If the chatbot ignores those limits, it may overstate personalized claims that the underlying data cannot support.
- A clean-sounding answer can still be misleading if it hides that a chunk came from estimated attendance or weak speaker mapping.

**Solution:**

- Instruct the model to answer only from retrieved context and to acknowledge when the supporting data is estimated or incomplete.
- Include trust-related flags in the system prompt or retrieval summary where relevant.
- Preserve those same flags in the saved session trace so review does not depend on prompt-memory alone.

---

## Files to Change

### Files to CREATE

1. `scripts/chat.py` - argparse-driven CLI chatbot entry point.
2. `tests/test_chat.py` - prompt-building, command-handling, and session-log tests.

### Files to MODIFY

#### Dependencies and Documentation

1. `requirements.txt` - **MAJOR** - add a pinned Groq SDK dependency if Groq remains the selected provider.
2. `README.md` - **MINOR** - document CLI usage, required `.env` values, and debug commands.

---

## Implementation Approach

### `scripts/chat.py`

**Purpose:** Provide a local CLI for student-specific question answering backed by retrieval and Groq generation.

**Key Responsibilities:**

- Parse CLI args such as `student_id`, `student_name`, `chroma_dir`, and session output path.
- Load `.env` configuration and fail fast if required secrets are missing.
- Retrieve context per user turn using TASK-008 and build a constrained Groq prompt.
- Support manual-inspection commands for the last retrieved context and saved session traces.

**Integration Points:**

- Consumes `RetrievalResult` from `scripts/retrieval.py`.
- Writes session outputs that TASK-010 can reuse directly.

**Considerations:**

- Use `argparse` for CLI behavior and keep the main loop thin.
- Session logs should capture at least the prompt question, retrieved chunk ids, retrieved metadata summary, answer text, and any trust-related warnings.
- History should be bounded so multi-turn support does not silently balloon prompt size.

---

## Acceptance Criteria

### Functional Requirements

- [ ] `python scripts/chat.py --student-id X --student-name Y` starts a working student-scoped chat loop.
- [ ] Each turn uses retrieved class context rather than answering without evidence.
- [ ] Debug commands expose the last retrieved chunks and their provenance without editing code.
- [ ] Session logs are written under `output/chat_sessions/` and contain enough detail for later review.
- [ ] The chatbot clearly acknowledges when the supporting context is estimated, sparse, or missing.
- [ ] The CLI exits cleanly on `quit` and handles missing API keys with a clear error.

### Code Quality

- [ ] All new functions and methods have complete type hints.
- [ ] Session-log payloads are structured and deterministic enough for test assertions.
- [ ] Prompt construction is separated from the interactive loop so it can be tested directly.
- [ ] No compilation errors or warnings.

---

## Testing Requirements

### Unit Tests

1. **Prompt and Trace Construction**
   - Prompt builder includes retrieved context plus trust-related flags where appropriate.
   - Session-log writer captures question, retrieval payload, and answer in the expected schema.

2. **CLI Commands**
   - `context` or `sources` prints the last retrieval payload safely.
   - `quit` exits cleanly.
   - Missing `GROQ_API_KEY` raises a clear, early error.

3. **Conversation State**
   - Conversation history is bounded and preserved across follow-up turns.
   - Empty retrieval results lead to a safe fallback answer path.

### Manual Evaluation

1. Ask a student-specific missed-content question and confirm the answer cites the missed context.
2. Ask a question about something the student said and confirm the relevant spoken chunk appears in the retrieval trace.
3. Ask a question not covered in the transcript and confirm the chatbot says the context is insufficient instead of guessing.
4. Review the saved session file and confirm it is enough to audit the answer after the run.

---

## Risks and Mitigation

| Risk                                                         | Mitigation                                                                         |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Groq answers from prior training instead of provided context | Use a restrictive system prompt and keep retrieval traces for auditing             |
| Session logs miss important provenance fields                | Reuse the structured retrieval models directly instead of flattening them manually |
| Estimated Phase 1 mappings are presented as facts            | Carry confidence and attendance-accuracy flags into the answering workflow         |
| Multi-turn state causes prompt bloat                         | Bound retained history and keep retrieval top-k conservative                       |

---

## Related TODOs

- **TASK-008**: Upstream dependency - provides student-scoped retrieval and debug-ready result models.
- **TASK-010**: Downstream dependency - evaluates chat answers and consumes saved traces.

---

## Handoff Template

**Status:** ⏳ Not Started

```
When implemented, a reviewer should be able to open one saved chat session file and answer three questions:
what the student asked, which chunks were used, and whether the answer made claims beyond the available evidence.
```

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~4 files
- **Configuration:** Add `GROQ_API_KEY` to `.env.example` if that file is introduced later