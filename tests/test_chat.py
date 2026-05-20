from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import pytest

from scripts.chat import (
    ChatArgs,
    ChatError,
    PromptMessage,
    ChatService,
    ChatTurnRecord,
    build_prompt_messages,
    build_sources_payload,
    load_groq_api_key,
)
from scripts.retrieval import RetrievalResult, RetrievedChunk
from scripts.utils.chunker import SourceSegmentReference


class FakeRetrievalBackend:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = list(results)
        self.queries: list[str] = []

    def retrieve(self, args: ChatArgs, question: str) -> RetrievalResult:
        self.queries.append(question)
        if not self.results:
            raise AssertionError("No retrieval result was queued for the test.")
        return self.results.pop(0)


class FakeLlmBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[PromptMessage], str]] = []

    def generate(self, *, messages: Sequence[PromptMessage], model: str) -> str:
        self.calls.append((list(messages), model))
        if not self.responses:
            raise AssertionError("No fake LLM response was queued for the test.")
        return self.responses.pop(0)


def build_chunk(
    *,
    chunk_id: str = "student-a:missed:0001",
    chunk_type: str = "missed",
    trust_flags: list[str] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        approximate=True,
        attendance_accuracy="estimated",
        attendance_estimated=True,
        attendance_source_mode="duration_only_estimated",
        chunk_id=chunk_id,
        chunk_type=chunk_type,
        distance=0.2,
        duration_seconds=8.0,
        end=8.0,
        participant_kind="student",
        rank=1,
        score=0.833333,
        source_manual_review_required=True,
        source_mapped_student="Student A",
        source_mapping_confidence="low",
        source_segment_count=1,
        source_segment_ids=["seg-0000"],
        source_segment_indices=[0],
        source_segment_refs=[
            SourceSegmentReference(
                end=8.0,
                segment_id="seg-0000",
                segment_index=0,
                source_speaker="SPEAKER_00",
                start=0.0,
                text="You missed the explanation about equivalent fractions.",
            )
        ],
        source_speaker="SPEAKER_00",
        start=0.0,
        student_email="student@example.com",
        student_id="student-a",
        student_manual_review_required=True,
        student_mapped_speaker="SPEAKER_01",
        student_mapping_confidence="low",
        student_name="Student A",
        text="You missed the explanation about equivalent fractions.",
        trust_flags=trust_flags or [
            "attendance_estimated",
            "source_mapping_confidence=low",
            "student_mapping_confidence=low",
        ],
    )


def build_result(
    *,
    query: str,
    chunks: list[RetrievedChunk] | None = None,
    warnings: list[str] | None = None,
) -> RetrievalResult:
    retrieved_chunks = list(chunks or [])
    return RetrievalResult(
        chunk_types=[],
        collection_name="student_transcript_chunks",
        context_string="\n".join(
            [
                "Student retrieval context for student-a",
                f"Query: {query}",
                f"Warnings: {'; '.join(warnings or [])}" if warnings else "Warnings: none",
            ]
        ),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        query=query,
        result_count=len(retrieved_chunks),
        retrieved_chunks=retrieved_chunks,
        student_id="student-a",
        top_k=5,
        warnings=list(warnings or []),
    )


def build_args(tmp_path: Path, *, question: str | None = None, max_history_turns: int = 2) -> ChatArgs:
    return ChatArgs(
        chroma_dir=tmp_path / "chroma",
        groq_model="llama-3.1-8b-instant",
        max_history_turns=max_history_turns,
        question=question,
        save_session_dir=tmp_path / "chat_sessions",
        student_id="student-a",
        student_name="Student A",
        top_k=5,
    )


def fixed_now() -> datetime:
    return datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)


def test_build_prompt_messages_includes_context_and_trust_flags() -> None:
    retrieval_result = build_result(
        query="What did I miss?",
        chunks=[build_chunk()],
        warnings=["Retrieved chunks use estimated attendance windows."],
    )
    history_turn = ChatTurnRecord(
        answer="Earlier grounded answer.",
        answer_source="groq",
        asked_at="2026-05-20T12:00:00Z",
        model="llama-3.1-8b-instant",
        prompt_messages=[],
        question="What homework was assigned?",
        retrieval_result=retrieval_result,
        turn_index=1,
        trust_flags=["attendance_estimated"],
    )

    messages = build_prompt_messages(
        student_id="student-a",
        student_name="Student A",
        question="What did I miss?",
        retrieval_result=retrieval_result,
        history_turns=[history_turn],
        max_history_turns=1,
    )

    assert messages[0].role == "system"
    assert "Answer only from the retrieved transcript context" in messages[0].content
    assert messages[1].content == "What homework was assigned?"
    assert messages[2].content == "Earlier grounded answer."
    assert "Retrieved chunks use estimated attendance windows." in messages[3].content
    assert "source_mapping_confidence=low" in messages[3].content
    assert retrieval_result.context_string in messages[3].content


def test_chat_service_writes_session_trace_with_retrieval_payload(tmp_path: Path) -> None:
    retrieval_result = build_result(query="What did I miss?", chunks=[build_chunk()])
    retrieval_backend = FakeRetrievalBackend([retrieval_result])
    llm_backend = FakeLlmBackend(["You missed the fractions explanation."])
    output_lines: list[str] = []
    service = ChatService(
        build_args(tmp_path),
        retrieval_backend=retrieval_backend,
        llm_backend=llm_backend,
        output_fn=output_lines.append,
        now_provider=fixed_now,
        session_id="session-test",
    )

    turn_record = service.ask_question("What did I miss?")
    payload = json.loads((tmp_path / "chat_sessions" / "session-test.json").read_text(encoding="utf-8"))

    assert turn_record.answer == "You missed the fractions explanation."
    assert payload["student_id"] == "student-a"
    assert payload["turns"][0]["question"] == "What did I miss?"
    assert payload["turns"][0]["retrieval_result"]["retrieved_chunks"][0]["chunk_id"] == "student-a:missed:0001"
    assert payload["turns"][0]["answer"] == "You missed the fractions explanation."
    assert payload["turns"][0]["model"] == "llama-3.1-8b-instant"


def test_context_and_sources_commands_expose_last_retrieval(tmp_path: Path) -> None:
    retrieval_result = build_result(query="What did I miss?", chunks=[build_chunk()])
    retrieval_backend = FakeRetrievalBackend([retrieval_result])
    llm_backend = FakeLlmBackend(["Grounded answer."])
    output_lines: list[str] = []
    service = ChatService(
        build_args(tmp_path),
        retrieval_backend=retrieval_backend,
        llm_backend=llm_backend,
        output_fn=output_lines.append,
        now_provider=fixed_now,
        session_id="session-test",
    )

    service.handle_user_input("context")
    service.handle_user_input("What did I miss?")
    service.handle_user_input("context")
    service.handle_user_input("sources")

    assert output_lines[0] == "No retrieval trace is available yet. Ask a question first."
    assert retrieval_result.context_string in output_lines[-2]
    sources_payload = json.loads(output_lines[-1])
    assert sources_payload == build_sources_payload(retrieval_result)


def test_run_exits_cleanly_on_quit(tmp_path: Path) -> None:
    output_lines: list[str] = []
    inputs = iter(["quit"])
    service = ChatService(
        build_args(tmp_path),
        retrieval_backend=FakeRetrievalBackend([]),
        llm_backend=FakeLlmBackend([]),
        input_fn=lambda prompt: next(inputs),
        output_fn=output_lines.append,
        now_provider=fixed_now,
        session_id="session-test",
    )

    service.run()

    assert output_lines[0].startswith("Chat ready for Student A")
    assert output_lines[-1] == "Ending chat session."


def test_load_groq_api_key_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ChatError, match="GROQ_API_KEY is missing"):
        load_groq_api_key()


def test_conversation_history_is_bounded(tmp_path: Path) -> None:
    retrieval_results = [
        build_result(query="Question one", chunks=[build_chunk(chunk_id="chunk-1")]),
        build_result(query="Question two", chunks=[build_chunk(chunk_id="chunk-2")]),
        build_result(query="Question three", chunks=[build_chunk(chunk_id="chunk-3")]),
    ]
    retrieval_backend = FakeRetrievalBackend(retrieval_results)
    llm_backend = FakeLlmBackend(["Answer one.", "Answer two.", "Answer three."])
    service = ChatService(
        build_args(tmp_path, max_history_turns=1),
        retrieval_backend=retrieval_backend,
        llm_backend=llm_backend,
        output_fn=lambda _: None,
        now_provider=fixed_now,
        session_id="session-test",
    )

    service.ask_question("Question one")
    service.ask_question("Question two")
    service.ask_question("Question three")

    final_messages = llm_backend.calls[-1][0]
    assert len(final_messages) == 4
    assert final_messages[1].content == "Question two"
    assert final_messages[2].content == "Answer two."
    assert "Question one" not in final_messages[3].content


def test_empty_retrieval_uses_safe_fallback_without_calling_llm(tmp_path: Path) -> None:
    retrieval_backend = FakeRetrievalBackend(
        [
            build_result(
                query="What did I miss?",
                chunks=[],
                warnings=["No stored chunks found for the requested student scope."],
            )
        ]
    )
    llm_backend = FakeLlmBackend(["This response should not be used."])
    service = ChatService(
        build_args(tmp_path),
        retrieval_backend=retrieval_backend,
        llm_backend=llm_backend,
        output_fn=lambda _: None,
        now_provider=fixed_now,
        session_id="session-test",
    )

    turn_record = service.ask_question("What did I miss?")

    assert turn_record.answer_source == "fallback"
    assert "do not have enough student-scoped class context" in turn_record.answer
    assert llm_backend.calls == []