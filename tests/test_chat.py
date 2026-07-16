from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from scripts.auth import AuthService
from scripts.chat import (
    ChatArgs,
    ChatError,
    ChatService,
    ChatTurnRecord,
    RetrievalBackend,
    normalize_answer_text,
    parse_args,
    resolve_display_name,
    run_login,
)
from scripts.retrieval import RetrievalResult


def make_args(tmp_path: Path) -> ChatArgs:
    return ChatArgs(
        db_url="postgresql://postgres:sup3rsecret@localhost:5432/adira",
        student_id="2302",
        student_name="Bhagyashree",
        save_session_dir=tmp_path / "chat_sessions",
    )


# --- #1: session traces hold no secrets ---


def test_session_trace_excludes_db_secret(tmp_path: Path) -> None:
    service = ChatService(make_args(tmp_path))
    contents = service.session_path.read_text(encoding="utf-8")
    assert "sup3rsecret" not in contents
    assert "postgresql://" not in contents
    assert "db_url" not in contents


def test_session_record_has_no_db_url_field() -> None:
    from scripts.chat import ChatSessionRecord

    assert "db_url" not in ChatSessionRecord.model_fields


# --- #3/#4 access half: no CLI path to set the queried student_id ---


def test_parse_args_rejects_student_id_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--student-id", "9999"])


def test_parse_args_rejects_student_name_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--student-name", "Someone"])


class _RecordingBackend:
    def __init__(self) -> None:
        self.seen_student_id: str | None = None

    def retrieve(self, args: ChatArgs, question: str) -> RetrievalResult:
        self.seen_student_id = args.student_id
        return RetrievalResult(
            context_string="none",
            embedding_model="m",
            query=question,
            result_count=0,
            student_id=args.student_id,
            top_k=args.top_k,
        )


def test_logged_in_retrieval_is_scoped_to_authed_id(tmp_path: Path) -> None:
    auth = AuthService({"2302": "alpha"})
    student_id = run_login(
        auth,
        id_provider=lambda _: "2302",
        password_provider=lambda _: "alpha",
        sleep_fn=lambda _: None,
    )
    assert student_id == "2302"

    args = make_args(tmp_path).model_copy(update={"student_id": student_id})
    backend = _RecordingBackend()
    service = ChatService(args, retrieval_backend=backend)
    service.ask_question("what did I miss?")

    # The only id ever queried is the authenticated one — there is no input that
    # could redirect retrieval to another student.
    assert backend.seen_student_id == "2302"


# --- run_login behavior ---


def test_run_login_loops_until_correct_with_backoff() -> None:
    auth = AuthService({"2302": "alpha"})
    passwords = iter(["wrong", "alpha"])
    sleeps: list[float] = []

    student_id = run_login(
        auth,
        id_provider=lambda _: "2302",
        password_provider=lambda _: next(passwords),
        output_fn=lambda _: None,
        sleep_fn=sleeps.append,
    )

    assert student_id == "2302"
    assert sleeps == [1.0]  # one backoff after the single failed attempt


def test_run_login_aborts_cleanly_on_eof() -> None:
    auth = AuthService({"2302": "alpha"})

    def raise_eof(_: str) -> str:
        raise EOFError

    with pytest.raises(ChatError, match="aborted"):
        run_login(
            auth,
            id_provider=lambda _: "2302",
            password_provider=raise_eof,
        )


# --- resolve_display_name ---


class _NameStore:
    def __init__(self, name: str | None) -> None:
        self._name = name

    def get_student_name(self, student_id: str) -> str | None:
        return self._name


def test_resolve_display_name_uses_store_name() -> None:
    assert resolve_display_name(_NameStore("Bhagyashree"), "2302") == "Bhagyashree"


def test_resolve_display_name_falls_back_to_id() -> None:
    assert resolve_display_name(_NameStore(None), "2302") == "2302"


# --- RetrievalBackend still does not leak an injected store on close ---


def test_retrieval_backend_close_is_safe_without_store() -> None:
    backend = RetrievalBackend()
    backend.close()  # should not raise even though no store was ever opened


def test_normalize_answer_text_strips_typographic_dashes() -> None:
    # en-dash used as a minus sign (the gpt-oss-20b habit) becomes a plain hyphen
    assert normalize_answer_text("quantity is –12 at price one") == "quantity is -12 at price one"
    # minus sign (U+2212) is also normalized to a hyphen
    assert normalize_answer_text("value −5") == "value -5"
    # em-dash clause separator becomes a comma, no double spaces or space-before-comma
    assert normalize_answer_text("the intercept is negative — about minus three") == (
        "the intercept is negative, about minus three"
    )


def test_normalize_answer_text_straightens_curly_quotes() -> None:
    # curly apostrophe in a contraction becomes straight ASCII (keeps refusal matchers robust)
    assert normalize_answer_text("I don’t have enough evidence.") == "I don't have enough evidence."
    # curly double quotes are straightened too
    assert normalize_answer_text("she said “yes”") == 'she said "yes"'


def test_normalize_answer_text_leaves_plain_text_unchanged() -> None:
    plain = "You said the intercept should be negative. For example, minus three."
    assert normalize_answer_text(plain) == plain


# --- #5: Groq egress disclosure surfaced in the banner ---


def test_banner_includes_groq_egress_notice(tmp_path: Path) -> None:
    from scripts.chat import GROQ_EGRESS_NOTICE

    captured: list[str] = []
    service = ChatService(make_args(tmp_path), output_fn=captured.append)
    service.print_banner()

    assert GROQ_EGRESS_NOTICE in "\n".join(captured)


# --- Finding A: "what did I say" must not surface the teacher's class context ---


def test_is_self_referential_question_matches_own_speech() -> None:
    from scripts.chat import is_self_referential_question

    assert is_self_referential_question("What did I personally say during class today?")
    assert is_self_referential_question("What did I say about the determinants of supply?")
    assert is_self_referential_question("What did I ask the teacher?")
    assert is_self_referential_question("Did I contribute anything today?")
    assert is_self_referential_question("What was my answer to the supply question?")
    # Broadened contribution phrasings beyond the canonical "what did I say".
    assert is_self_referential_question("What numbers did I work out in class?")
    assert is_self_referential_question("What answer did I get for the worksheet?")
    assert is_self_referential_question("Did I submit anything in the chat?")
    assert is_self_referential_question("What did I type during class?")
    assert is_self_referential_question("How did I solve the second question?")


def test_is_self_referential_question_ignores_class_and_passive_questions() -> None:
    from scripts.chat import is_self_referential_question

    # "we"/topic questions are about class content, not the student's own words.
    assert not is_self_referential_question("What did we cover in class today?")
    assert not is_self_referential_question(
        "What's the difference between a supply schedule and a supply curve?"
    )
    assert not is_self_referential_question("What were Jagruti and Kalyani being asked about?")
    # First-person but NOT about the student's own speech (passive / "I missed").
    assert not is_self_referential_question(
        "I missed the part about why quantity supplied can be negative — what was said?"
    )
    assert not is_self_referential_question("I joined late — what was the plan for today's class?")
    # Neutral first-person verbs ask about class content, not the student's contribution.
    assert not is_self_referential_question("What did I learn in class today?")
    assert not is_self_referential_question("What did I do in class today?")


def test_select_retrieval_chunk_types_scopes_self_referential_to_own_contributions() -> None:
    from scripts.chat import select_retrieval_chunk_types

    # Both spoken AND chat are the student's own words — a quiet student who only typed
    # must still surface for "what did I say".
    assert select_retrieval_chunk_types("What did I say today?", ()) == ["spoken", "chat"]
    # Non-self-referential questions stay unfiltered (full class context available).
    assert select_retrieval_chunk_types("What did we cover today?", ()) == []
    # An explicit caller filter always wins, even for a self-referential question.
    assert select_retrieval_chunk_types("What did I say today?", ["missed"]) == ["missed"]


def test_build_prompt_messages_teaches_speaker_attribution() -> None:
    from scripts.chat import build_prompt_messages

    result = RetrievalResult(
        context_string="ctx",
        embedding_model="m",
        query="q",
        result_count=1,
        student_id="2302",
        top_k=5,
    )
    messages = build_prompt_messages(
        student_id="2302",
        student_name="Bhagyashree",
        question="What did I say today?",
        retrieval_result=result,
        history_turns=[],
        max_history_turns=0,
    )
    system = messages[0].content.casefold()
    assert "speaker=teacher" in system
    assert "not the student" in system
    assert "type=spoken" in system


def _result_with_material(question: str) -> RetrievalResult:
    from scripts.retrieval import RetrievedChunk

    material_chunk = RetrievedChunk(
        chunk_id="m1",
        chunk_type="material",
        rank=1,
        rerank_score=0.82,
        score=0.8,
        source_speaker="material",
        source_file="supply_deck.pptx",
        student_id="2302",
        student_name="Bhagyashree",
        text="The determinants of supply shift the whole supply curve.",
    )
    return RetrievalResult(
        context_string="[1] type=material source=supply_deck.pptx\ntext=determinants shift supply",
        embedding_model="m",
        query=question,
        result_count=1,
        retrieved_chunks=[material_chunk],
        student_id="2302",
        top_k=8,
    )


def test_system_prompt_teaches_material_semantics_and_no_world_knowledge() -> None:
    from scripts.chat import build_prompt_messages

    messages = build_prompt_messages(
        student_id="2302",
        student_name="Bhagyashree",
        question="How are the determinants connected to the supply function?",
        retrieval_result=_result_with_material("concept q"),
        history_turns=[],
        max_history_turns=0,
    )
    system = messages[0].content.casefold()
    # Material is named as authoritative and may be synthesized for concept questions.
    assert "type=material" in system
    assert "authoritative" in system
    assert "synthesize" in system
    # The no-world-knowledge binding must be explicit.
    assert "world knowledge" in system or "outside knowledge" in system
    # Attribution requirement present.
    assert "class material" in system


def test_prompt_carries_material_chunks_into_user_context() -> None:
    from scripts.chat import build_prompt_messages

    result = _result_with_material("How do determinants relate to supply?")
    messages = build_prompt_messages(
        student_id="2302",
        student_name="Bhagyashree",
        question="How do determinants relate to supply?",
        retrieval_result=result,
        history_turns=[],
        max_history_turns=0,
    )
    user = messages[-1].content
    assert "type=material" in user
    assert "supply_deck.pptx" in user


class _AttributingBackend:
    """Fake LLM that echoes an attributed answer only if material is in the prompt."""

    def generate(self, *, messages: Sequence[object], model: str) -> str:
        joined = "\n".join(getattr(m, "content", "") for m in messages)
        if "type=material" in joined:
            return "According to the class material, the determinants shift the supply curve."
        return "I do not have enough evidence."


def test_concept_question_produces_attributed_answer(tmp_path: Path) -> None:
    class _MaterialBackend:
        def retrieve(self, args: ChatArgs, question: str) -> RetrievalResult:
            return _result_with_material(question)

    service = ChatService(
        make_args(tmp_path),
        retrieval_backend=_MaterialBackend(),
        llm_backend=_AttributingBackend(),
    )
    turn = service.ask_question("How are the determinants connected to the supply function?")
    assert turn.answer_source == "groq"
    assert "class material" in turn.answer.casefold()


def test_no_support_question_refuses_via_fallback(tmp_path: Path) -> None:
    # result_count == 0 → no LLM call, deterministic refusal fallback (no world knowledge).
    service = ChatService(make_args(tmp_path), retrieval_backend=_RecordingBackend())
    turn = service.ask_question("What is the capital of France?")
    assert turn.answer_source == "fallback"
    assert "not have enough" in turn.answer.casefold()


def test_self_referential_question_excludes_material(tmp_path: Path) -> None:
    store = _CapturingStore()
    backend = RetrievalBackend(store=store, embedder=_FixedEmbedder())  # type: ignore[arg-type]
    backend.retrieve(make_args(tmp_path), "What did I say about determinants today?")
    # Self-referential scope is the student's own words only — material must not appear.
    assert store.search_chunk_types == ["spoken", "chat"]
    assert "material" not in (store.search_chunk_types or [])


class _CapturingStore:
    def __init__(self) -> None:
        self.search_chunk_types: list[str] | None = None
        self.search_top_k: int | None = None
        self.lexical_chunk_types: list[str] | None = None

    def search(
        self,
        query_embedding: list[float],
        student_id: str,
        top_k: int = 5,
        chunk_types: Sequence[str] | None = None,
        class_name: str | None = None,
    ) -> list[object]:
        self.search_chunk_types = list(chunk_types or [])
        self.search_top_k = top_k
        return []

    def search_lexical(
        self,
        query_text: str,
        *,
        student_id: str,
        chunk_types: Sequence[str] | None = None,
        limit: int = 25,
        class_name: str | None = None,
    ) -> list[object]:
        self.lexical_chunk_types = list(chunk_types or [])
        return []


class _FixedEmbedder:
    def encode(self, query: str) -> list[float]:
        return [0.1, 0.2]


def test_retrieval_backend_scopes_self_referential_to_spoken(tmp_path: Path) -> None:
    store = _CapturingStore()
    backend = RetrievalBackend(store=store, embedder=_FixedEmbedder())  # type: ignore[arg-type]
    args = make_args(tmp_path)

    backend.retrieve(args, "What did I say during class today?")
    assert store.search_chunk_types == ["spoken", "chat"]
    assert store.lexical_chunk_types == ["spoken", "chat"]

    backend.retrieve(args, "What did we cover in class today?")
    assert store.search_chunk_types == []
    assert store.lexical_chunk_types == []


def test_retrieval_backend_returns_base_top_k_over_a_hybrid_pool(tmp_path: Path) -> None:
    from scripts.retrieval import HYBRID_POOL_SIZE

    store = _CapturingStore()
    backend = RetrievalBackend(store=store, embedder=_FixedEmbedder())  # type: ignore[arg-type]
    args = make_args(tmp_path)  # base top_k defaults to 5

    general = backend.retrieve(args, "What did we cover in class today?")
    assert general.top_k == args.top_k
    assert store.search_top_k == HYBRID_POOL_SIZE

    self_referential = backend.retrieve(args, "What did I say during class today?")
    assert self_referential.top_k == args.top_k
    assert store.search_top_k == HYBRID_POOL_SIZE


# --- TASK-031: confidence routing ---


class _CountingBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[Sequence[object]] = []

    def generate(self, *, messages: Sequence[object], model: str) -> str:
        self.calls += 1
        self.messages.append(messages)
        return "generated"


def _routed_result(rerank_score: float | None, *, chunk_type: str = "class_context") -> RetrievalResult:
    from scripts.retrieval import RetrievedChunk

    chunk = RetrievedChunk(
        chunk_id="c1",
        chunk_type=chunk_type,  # type: ignore[arg-type]
        rank=1,
        rerank_score=rerank_score,
        source_speaker="teacher",
        student_id="2302",
        student_name="Bhagyashree",
        text="the supply curve shifts with its determinants",
    )
    return RetrievalResult(
        context_string="[1] type=class_context\ntext=determinants shift supply",
        embedding_model="m",
        query="q",
        result_count=1,
        retrieved_chunks=[chunk],
        student_id="2302",
        top_k=5,
    )


def _answer_turn(
    result: RetrievalResult,
    backend: _CountingBackend,
    question: str = "What did we cover about supply?",
) -> ChatTurnRecord:
    from scripts.chat import answer_turn, utc_now

    return answer_turn(
        student_id="2302",
        student_name="Bhagyashree",
        question=question,
        retrieval_result=result,
        llm_backend=backend,
        groq_model="test-model",
        history_turns=[],
        max_history_turns=0,
        student_classes=["Economics.02"],
        now=utc_now(),
        turn_index=1,
    )


def test_high_tier_answers_via_groq() -> None:
    backend = _CountingBackend()
    turn = _answer_turn(_routed_result(0.9), backend)
    assert turn.grade == "high"
    assert turn.answer_source == "groq"
    assert backend.calls == 1
    assert turn.answer == "generated"


def test_medium_tier_uses_soft_landing_prompt() -> None:
    from scripts.chat import MEDIUM_SYSTEM_INSTRUCTION, MEDIUM_USER_INSTRUCTION

    backend = _CountingBackend()
    turn = _answer_turn(_routed_result(0.4), backend)
    assert turn.grade == "medium"
    assert turn.answer_source == "groq"
    assert backend.calls == 1
    assert MEDIUM_SYSTEM_INSTRUCTION in turn.prompt_messages[0].content
    assert MEDIUM_USER_INSTRUCTION in turn.prompt_messages[-1].content


def test_low_tier_makes_no_llm_call_and_refuses_deterministically() -> None:
    backend = _CountingBackend()
    turn = _answer_turn(_routed_result(0.1), backend)
    assert turn.grade == "low"
    assert turn.answer_source == "fallback"
    assert backend.calls == 0
    assert "not have enough" in turn.answer.casefold()
    assert "Economics.02" in turn.answer


def test_overview_question_lifts_low_to_medium_when_chunks_exist() -> None:
    backend = _CountingBackend()
    turn = _answer_turn(_routed_result(0.001), backend, question="What did we cover in class?")
    assert turn.grade == "medium"
    assert turn.answer_source == "groq"
    assert backend.calls == 1


def test_overview_question_with_content_tail_stays_low() -> None:
    backend = _CountingBackend()
    turn = _answer_turn(
        _routed_result(0.001), backend, question="What did we cover about GDP or national income?"
    )
    assert turn.grade == "low"
    assert turn.answer_source == "fallback"
    assert backend.calls == 0


def test_overview_question_with_empty_retrieval_stays_low() -> None:
    backend = _CountingBackend()
    empty = RetrievalResult(
        context_string="none",
        embedding_model="m",
        query="q",
        result_count=0,
        retrieved_chunks=[],
        student_id="2302",
        top_k=5,
    )
    turn = _answer_turn(empty, backend, question="What did we cover in class?")
    assert turn.grade == "low"
    assert backend.calls == 0


def test_is_class_overview_question_accepts_generic_forms() -> None:
    from scripts.chat import is_class_overview_question

    accepted = [
        "What did we cover in class?",
        "what did we do in class today",
        "I was absent. Can you give me a short summary?",
        "I missed the class, what happened in class?",
        "I joined late, what was the plan for class?",
        "Was there any homework or worksheet?",
        "Give me one practice question on today's topic",
        "What topic did we start working on in class?",
        "Can you give me a short summary?",
        "Explain today's topic again in simple words",
    ]
    for question in accepted:
        assert is_class_overview_question(question), question


def test_is_class_overview_question_rejects_content_and_personal_forms() -> None:
    from scripts.chat import is_class_overview_question

    rejected = [
        "What did we cover about GDP or national income?",
        "What did we learn about Shakespeare's Macbeth?",
        "What did we learn about the determinants of supply?",
        "What did I say in class?",
        "what is photosynthesis",
        "Give me a summary of the supply function",
        "Explain the supply function again",
    ]
    for question in rejected:
        assert not is_class_overview_question(question), question


def test_empty_retrieval_routes_low_without_a_model() -> None:
    backend = _CountingBackend()
    empty = RetrievalResult(
        context_string="none",
        embedding_model="m",
        query="q",
        result_count=0,
        student_id="2302",
        top_k=5,
    )
    turn = _answer_turn(empty, backend)
    assert turn.grade == "low"
    assert turn.answer_source == "fallback"
    assert backend.calls == 0


def test_build_low_tier_answer_lists_classes_and_missed_notes() -> None:
    from scripts.chat import build_low_tier_answer
    from scripts.retrieval import RetrievedChunk

    missed_chunk = RetrievedChunk(
        chunk_id="x",
        chunk_type="missed",
        rank=1,
        source_speaker="teacher",
        student_id="2302",
        student_name="Bhagyashree",
        text="the part you missed on elasticity",
    )
    result = RetrievalResult(
        context_string="c",
        embedding_model="m",
        query="q",
        result_count=1,
        retrieved_chunks=[missed_chunk],
        student_id="2302",
        top_k=5,
    )
    text = build_low_tier_answer("Bhagyashree", result, ["Economics.02", "Math.01"])
    assert "not have enough" in text.casefold()
    assert "Economics.02" in text
    assert "Math.01" in text
    assert "missed" in text.casefold()


def test_high_tier_prompt_is_byte_identical_to_default() -> None:
    from scripts.chat import build_prompt_messages

    result = _result_with_material("concept q")
    kwargs = dict(
        student_id="2302",
        student_name="Bhagyashree",
        question="How do determinants relate to supply?",
        retrieval_result=result,
        history_turns=[],
        max_history_turns=0,
    )
    default = build_prompt_messages(**kwargs)  # type: ignore[arg-type]
    high = build_prompt_messages(**kwargs, grade="high")  # type: ignore[arg-type]
    assert [m.model_dump() for m in high] == [m.model_dump() for m in default]


def test_medium_prompt_is_strictly_additive_over_high() -> None:
    from scripts.chat import (
        MEDIUM_SYSTEM_INSTRUCTION,
        MEDIUM_USER_INSTRUCTION,
        build_prompt_messages,
    )

    result = _result_with_material("concept q")
    kwargs = dict(
        student_id="2302",
        student_name="Bhagyashree",
        question="How do determinants relate to supply?",
        retrieval_result=result,
        history_turns=[],
        max_history_turns=0,
    )
    high = build_prompt_messages(**kwargs, grade="high")  # type: ignore[arg-type]
    medium = build_prompt_messages(**kwargs, grade="medium")  # type: ignore[arg-type]

    assert MEDIUM_SYSTEM_INSTRUCTION not in high[0].content
    assert MEDIUM_USER_INSTRUCTION not in high[-1].content
    assert MEDIUM_SYSTEM_INSTRUCTION in medium[0].content
    assert MEDIUM_USER_INSTRUCTION in medium[-1].content
    # the high content is preserved verbatim as the prefix of the medium content
    assert medium[0].content.startswith(high[0].content)
    assert medium[-1].content.startswith(high[-1].content)
