# TASK-009: CLI Chatbot Using Groq + RAG

**Status:** ✅ Completed

**Completed:** 2026-05-20

## Outcome

TASK-009 now provides a student-scoped CLI chatbot that reuses the TASK-008 retrieval contract, generates grounded answers through Groq, exposes `context` and `sources` debug commands, and writes inspectable JSON session traces under `output/chat_sessions/` for later audit.

## Delivered Files

- `scripts/chat.py`
- `tests/test_chat.py`
- `requirements.txt`
- `.vscode/planned/chatbot/TASK-009.md`
- `.vscode/planned/chatbot/MASTER_PLAN.md`

## Validation

- `python -m ruff check --fix .` ✅ PASS
- `python -m mypy .` ✅ PASS
- `python -m pytest` ✅ PASS (`90 passed`, `0 warnings`)
- `python scripts/chat.py --student-id a-disha-2504 --student-name "A Disha" --chroma-dir data/chroma --save-session-dir output/chat_sessions --question "What did I miss in class?"` ✅ PASS

## Trace Contract for TASK-010

- Each turn stores the question, answer, model id, trust flags, prompt messages, and the full nested `RetrievalResult` used for grounding.
- `context` prints the last prompt-ready retrieval string and `sources` prints provenance-rich JSON for the same last retrieval.
- Empty retrieval remains a supported path and produces a safe fallback answer without calling Groq.

## Notes for Next Task

- The real validation turn for `a-disha-2504` surfaced only low-confidence `class_context` chunks, and the generated answer correctly stated that the evidence was insufficient.
- Groq `0.9.0` required `httpx==0.27.2` in this environment; keep those versions aligned for reproducible runtime validation.
- Do not commit copied `data/chroma/` contents or generated session traces from `output/chat_sessions/`.