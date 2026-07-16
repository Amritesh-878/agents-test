from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Sequence

import pytest

from scripts.metrics import (
    TierRates,
    build_parser,
    build_tier_rates,
    derive_presence_mode,
    main,
    render_students_markdown,
    render_tier_markdown,
    run_cli,
    section_metrics,
    student_metrics,
    tier_rates,
)
from scripts.models.pipeline import SearchResult

ECONOMICS_CLASS = "Economics.02_AY2025-26_Supply Function_16 April"
POETRY_CLASS = "English.03_AY2025-26_Poem Refrain_05 May"
PROSE_CLASS = "English.04_AY2025-26_Prose Passage_06 May"


def make_chunk(
    *,
    student_id: str = "2409",
    student_name: str = "Shraddha Singh",
    class_name: str = POETRY_CLASS,
    chunk_type: str = "spoken",
    text: str = "some grounded text",
    metadata: dict[str, Any] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=f"{student_id}-{chunk_type}-{len(text)}",
        student_id=student_id,
        student_name=student_name,
        class_name=class_name,
        chunk_type=chunk_type,
        text=text,
        distance=0.25,
        start_time=4.0,
        end_time=12.0,
        speaker="student",
        metadata=metadata if metadata is not None else {},
    )


class FakeStore:
    def __init__(
        self,
        *,
        chunks: Sequence[SearchResult] = (),
        pairs: Sequence[tuple[str, str, str]] = (),
        stats: Sequence[tuple[str, str, int]] = (),
    ) -> None:
        self._chunks = list(chunks)
        self._pairs = list(pairs)
        self._stats = list(stats)
        self.stats_calls: list[tuple[list[str] | None, datetime | None]] = []

    def get_student_chunks(self, student_id: str) -> list[SearchResult]:
        return [c for c in self._chunks if c.student_id == student_id]

    def list_student_class_pairs(self) -> list[tuple[str, str, str]]:
        return list(self._pairs)

    def fetch_query_stats(
        self,
        *,
        student_ids: Sequence[str] | None = None,
        since: datetime | None = None,
    ) -> list[tuple[str, str, int]]:
        self.stats_calls.append((list(student_ids) if student_ids is not None else None, since))
        return list(self._stats)


# --- presence-mode table (one test per row) ---


def test_presence_mode_absent_wins_over_everything() -> None:
    assert derive_presence_mode({"spoken": 3, "class_context": 1}, True) == "absent"


def test_presence_mode_audio_when_spoken_present() -> None:
    assert derive_presence_mode({"spoken": 2, "chat": 5, "class_context": 1}, False) == "audio"


def test_presence_mode_chat_only_when_chat_and_no_spoken() -> None:
    assert derive_presence_mode({"chat": 4, "class_context": 1}, False) == "chat-only"


def test_presence_mode_attendance_only_when_context_without_spoken_or_chat() -> None:
    assert derive_presence_mode({"class_context": 2}, False) == "attendance-only"


def test_presence_mode_materials_only_when_material_alone() -> None:
    assert derive_presence_mode({"material": 6}, False) == "materials-only"


def test_presence_mode_unknown_when_no_chunks() -> None:
    assert derive_presence_mode({}, False) == "unknown"


# --- presence mode end-to-end from store chunks ---


def test_ramsha_shaped_absent_student_is_absent() -> None:
    store = FakeStore(
        chunks=[
            make_chunk(
                student_id="2408",
                student_name="Ramsha Khan",
                chunk_type="class_context",
                metadata={"status": "absent", "topics": ["poem refrain"]},
            )
        ]
    )
    metrics = student_metrics(store, "2408")
    assert [s.presence_mode for s in metrics.sessions] == ["absent"]
    assert metrics.sessions_by_mode == {"absent": 1}


def test_soumya_shaped_attendance_only_student() -> None:
    store = FakeStore(
        chunks=[
            make_chunk(
                student_id="2410",
                student_name="Soumya Nair",
                chunk_type="class_context",
                metadata={"status": "present"},
            )
        ]
    )
    metrics = student_metrics(store, "2410")
    assert [s.presence_mode for s in metrics.sessions] == ["attendance-only"]
    assert metrics.spoken_chunk_total == 0


def test_audio_student_is_audio_with_spoken_totals() -> None:
    store = FakeStore(
        chunks=[
            make_chunk(chunk_type="spoken", text="first"),
            make_chunk(chunk_type="spoken", text="second turn"),
            make_chunk(chunk_type="class_context", metadata={"status": "present"}),
        ]
    )
    metrics = student_metrics(store, "2409")
    assert [s.presence_mode for s in metrics.sessions] == ["audio"]
    assert metrics.spoken_chunk_total == 2


def test_chat_only_student() -> None:
    store = FakeStore(
        chunks=[
            make_chunk(chunk_type="chat", text="typed a question"),
            make_chunk(chunk_type="class_context", metadata={"status": "present"}),
        ]
    )
    metrics = student_metrics(store, "2409")
    assert [s.presence_mode for s in metrics.sessions] == ["chat-only"]
    assert metrics.chat_chunk_total == 1
    assert metrics.spoken_chunk_total == 0


def test_materials_only_student() -> None:
    store = FakeStore(chunks=[make_chunk(chunk_type="material", text="slide text")])
    metrics = student_metrics(store, "2409")
    assert [s.presence_mode for s in metrics.sessions] == ["materials-only"]
    assert metrics.sessions[0].has_materials is True


def test_legacy_metadata_without_embedding_model_key_is_fine() -> None:
    store = FakeStore(chunks=[make_chunk(chunk_type="spoken", metadata={"status": "present"})])
    metrics = student_metrics(store, "2409")
    assert metrics.sessions[0].presence_mode == "audio"


def test_metadata_missing_entirely_is_fine() -> None:
    store = FakeStore(chunks=[make_chunk(chunk_type="spoken", metadata={})])
    assert student_metrics(store, "2409").sessions[0].presence_mode == "audio"


def test_session_carries_section_and_counts() -> None:
    store = FakeStore(
        chunks=[
            make_chunk(chunk_type="spoken"),
            make_chunk(chunk_type="chat", text="typed"),
            make_chunk(chunk_type="material", text="deck"),
        ]
    )
    session = student_metrics(store, "2409").sessions[0]
    assert session.class_name == POETRY_CLASS
    assert session.section == "English.03"
    assert session.chunk_counts == {"spoken": 1, "chat": 1, "material": 1}
    assert session.has_materials is True


def test_student_name_resolves_from_chunks() -> None:
    store = FakeStore(chunks=[make_chunk(student_id="2409", student_name="Shraddha Singh")])
    assert student_metrics(store, "2409").student_name == "Shraddha Singh"


def test_unknown_student_has_no_sessions() -> None:
    metrics = student_metrics(FakeStore(chunks=[make_chunk()]), "9999")
    assert metrics.sessions == []
    assert metrics.sessions_on_record == 0
    assert metrics.student_name == "9999"


def test_chunks_without_class_name_are_skipped() -> None:
    store = FakeStore(chunks=[make_chunk(class_name="")])
    assert student_metrics(store, "2409").sessions == []


# --- dual-subject + section scoping ---


def dual_subject_store() -> FakeStore:
    return FakeStore(
        chunks=[
            make_chunk(student_id="2405", student_name="Esha Patel", class_name=ECONOMICS_CLASS),
            make_chunk(
                student_id="2405",
                student_name="Esha Patel",
                class_name=PROSE_CLASS,
                chunk_type="chat",
                text="typed in english",
            ),
            make_chunk(student_id="2402", student_name="Bhavna Rao", class_name=POETRY_CLASS),
        ],
        pairs=[
            ("2405", "Esha Patel", ECONOMICS_CLASS),
            ("2405", "Esha Patel", PROSE_CLASS),
            ("2402", "Bhavna Rao", POETRY_CLASS),
        ],
    )


def test_dual_subject_student_has_a_session_per_section() -> None:
    metrics = student_metrics(dual_subject_store(), "2405")
    assert [(s.section, s.presence_mode) for s in metrics.sessions] == [
        ("Economics.02", "audio"),
        ("English.04", "chat-only"),
    ]
    assert metrics.sessions_on_record == 2
    assert metrics.sessions_by_mode == {"audio": 1, "chat-only": 1}


def test_section_filter_returns_only_that_sections_sessions() -> None:
    results = section_metrics(dual_subject_store(), "Economics.02")
    assert [s.student_id for s in results] == ["2405"]
    assert [s.section for s in results[0].sessions] == ["Economics.02"]
    assert results[0].sessions_on_record == 1
    assert results[0].sessions_by_mode == {"audio": 1}


def test_section_filter_recomputes_rollups_for_the_section() -> None:
    results = section_metrics(dual_subject_store(), "English.04")
    assert results[0].spoken_chunk_total == 0
    assert results[0].chat_chunk_total == 1


def test_section_metrics_lists_every_student_in_the_section() -> None:
    results = section_metrics(dual_subject_store(), "English.03")
    assert [s.student_id for s in results] == ["2402"]


def test_section_metrics_for_unknown_section_is_empty() -> None:
    assert section_metrics(dual_subject_store(), "History.09") == []


# --- tier rates ---


def test_build_tier_rates_computes_counts_and_rates() -> None:
    rates = build_tier_rates([("high", "groq", 5), ("medium", "groq", 3), ("low", "fallback", 2)])
    assert rates.total == 10
    assert rates.counts_by_grade == {"high": 5, "medium": 3, "low": 2}
    assert rates.rates_by_grade == {"high": 0.5, "medium": 0.3, "low": 0.2}
    assert rates.counts_by_answer_source == {"fallback": 2, "groq": 8}
    assert rates.rates_by_answer_source == {"fallback": 0.2, "groq": 0.8}


def test_low_tier_rate_is_the_gate_number() -> None:
    rates = build_tier_rates([("high", "groq", 3), ("low", "fallback", 1)])
    assert rates.low_tier_rate == 0.25
    assert rates.low_tier_rate == rates.rates_by_grade["low"]


def test_build_tier_rates_reports_zero_grades_present() -> None:
    rates = build_tier_rates([("high", "groq", 4)])
    assert rates.counts_by_grade == {"high": 4, "medium": 0, "low": 0}
    assert rates.low_tier_rate == 0.0


def test_build_tier_rates_on_empty_log() -> None:
    rates = build_tier_rates([])
    assert rates == TierRates(
        total=0,
        counts_by_grade={"high": 0, "medium": 0, "low": 0},
        rates_by_grade={"high": 0.0, "medium": 0.0, "low": 0.0},
        counts_by_answer_source={},
        rates_by_answer_source={},
        low_tier_rate=0.0,
    )


def test_build_tier_rates_merges_rows_sharing_a_grade() -> None:
    rates = build_tier_rates([("low", "fallback", 2), ("low", "groq", 1)])
    assert rates.counts_by_grade["low"] == 3
    assert rates.counts_by_answer_source == {"fallback": 2, "groq": 1}


def test_tier_rates_student_filter_is_passed_through() -> None:
    store = FakeStore(stats=[("high", "groq", 1)])
    tier_rates(store, student_id="2409")
    assert store.stats_calls == [(["2409"], None)]


def test_tier_rates_section_filter_resolves_member_ids() -> None:
    store = dual_subject_store()
    store._stats = [("high", "groq", 1)]
    tier_rates(store, section="Economics.02")
    assert store.stats_calls == [(["2405"], None)]


def test_tier_rates_since_filter_is_passed_through() -> None:
    store = FakeStore(stats=[("high", "groq", 1)])
    since = datetime(2026, 7, 1, tzinfo=UTC)
    tier_rates(store, since=since)
    assert store.stats_calls == [(None, since)]


def test_tier_rates_unfiltered_passes_no_student_ids() -> None:
    store = FakeStore(stats=[("high", "groq", 1)])
    tier_rates(store)
    assert store.stats_calls == [(None, None)]


# --- rendering ---


def test_render_students_markdown_is_a_plain_table_without_verdicts() -> None:
    rendered = render_students_markdown(section_metrics(dual_subject_store(), "Economics.02"))
    assert rendered.splitlines()[0].startswith("| student_id | student_name | session |")
    assert "2405" in rendered
    assert "Economics.02" in rendered
    assert "audio" in rendered
    for banned in ("verdict", "score", "rank", "rating", "grade"):
        assert banned not in rendered.casefold()


def test_render_students_markdown_handles_a_student_with_no_sessions() -> None:
    rendered = render_students_markdown([student_metrics(FakeStore(), "9999")])
    assert "no sessions on record" in rendered


def test_render_tier_markdown_shows_grades_and_sources() -> None:
    rendered = render_tier_markdown(build_tier_rates([("high", "groq", 3), ("low", "fallback", 1)]))
    assert "Answered turns: 4" in rendered
    assert "| low | 1 | 0.2500 |" in rendered
    assert "| fallback | 1 | 0.2500 |" in rendered


# --- CLI ---


def run(argv: list[str], store: Any) -> tuple[int, str]:
    captured: list[str] = []
    code = run_cli(build_parser().parse_args(argv), store=store, output_fn=captured.append)
    return code, "\n".join(captured)


def test_cli_student_markdown() -> None:
    code, out = run(["--student", "2405", "--markdown"], dual_subject_store())
    assert code == 0
    assert "Economics.02" in out and "English.04" in out


def test_cli_student_json_round_trips() -> None:
    code, out = run(["--student", "2405", "--json"], dual_subject_store())
    assert code == 0
    payload = json.loads(out)
    assert payload[0]["student_id"] == "2405"
    assert len(payload[0]["sessions"]) == 2
    assert payload[0]["sessions"][0]["presence_mode"] == "audio"


def test_cli_section_markdown() -> None:
    code, out = run(["--section", "Economics.02", "--markdown"], dual_subject_store())
    assert code == 0
    assert "2405" in out and "Bhavna" not in out


def test_cli_defaults_to_markdown() -> None:
    code, out = run(["--student", "2405"], dual_subject_store())
    assert code == 0
    assert out.startswith("| student_id |")


def test_cli_tier_report_markdown() -> None:
    store = FakeStore(stats=[("high", "groq", 3), ("low", "fallback", 1)])
    code, out = run(["--tier-report"], store)
    assert code == 0
    assert "Answered turns: 4" in out


def test_cli_tier_report_json_exposes_the_gate_number() -> None:
    store = FakeStore(stats=[("high", "groq", 3), ("low", "fallback", 1)])
    code, out = run(["--tier-report", "--json"], store)
    assert code == 0
    assert json.loads(out)["low_tier_rate"] == 0.25


def test_cli_tier_report_with_section_and_since() -> None:
    store = dual_subject_store()
    store._stats = [("high", "groq", 1)]
    code, _ = run(
        ["--tier-report", "--section", "Economics.02", "--since", "2026-07-01T00:00:00+00:00"],
        store,
    )
    assert code == 0
    assert store.stats_calls == [(["2405"], datetime(2026, 7, 1, tzinfo=UTC))]


def test_cli_rejects_a_bad_since() -> None:
    with pytest.raises(ValueError, match="ISO timestamp"):
        run(["--tier-report", "--since", "last tuesday"], FakeStore())


def test_cli_json_and_markdown_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--student", "2405", "--json", "--markdown"])


def test_cli_without_a_selector_errors() -> None:
    with pytest.raises(SystemExit):
        main([], store=FakeStore())


def test_cli_never_touches_llm_or_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.retrieval

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("metrics must not build an embedder or call the LLM")

    monkeypatch.setattr(scripts.retrieval.QueryEmbedder, "__init__", boom)
    code, out = run(["--student", "2405", "--json"], dual_subject_store())
    assert code == 0
    assert out


def test_main_uses_an_injected_store() -> None:
    assert main(["--student", "2405", "--json"], store=dual_subject_store()) == 0


# --- PII ---


def test_no_question_text_reaches_the_metrics_surface() -> None:
    store = FakeStore(stats=[("high", "groq", 2)])
    _, out = run(["--tier-report", "--json"], store)
    payload = json.loads(out)
    assert "question" not in json.dumps(payload).casefold()
    assert set(payload) == {
        "total",
        "counts_by_grade",
        "rates_by_grade",
        "counts_by_answer_source",
        "rates_by_answer_source",
        "low_tier_rate",
    }
