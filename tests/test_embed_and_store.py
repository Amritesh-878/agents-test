from __future__ import annotations

import pytest

from scripts.embed_and_store import (
    chunk_absent_summary,
    chunk_student_context,
    collect_all_records,
    stable_chunk_id,
)
from scripts.models.context import (
    AbsentStudentSummary,
    BuildContextMetadata,
    ContextSegment,
    StudentContext,
    StudentContextDocument,
)


def make_context(
    name: str = "Anshi",
    roll_no: str = "2301",
    spoken_text: str = "today we covered the supply function and demand curve analysis",
    missed_text: str = "teacher explained the concept of elastic and inelastic demand",
    present_text: str = "class covered time and work problems with scaffolding approach",
) -> StudentContext:
    spoken = [ContextSegment(start=0.0, end=5.0, text=spoken_text, speakers=[name])]
    missed = [ContextSegment(start=100.0, end=110.0, text=missed_text)]
    present = [ContextSegment(start=0.0, end=5.0, text=present_text, speakers=[name])]
    return StudentContext(
        name=name,
        roll_no=roll_no,
        status="present",
        spoken_segments=spoken,
        missed_segments=missed,
        present_segments=present,
        class_duration_seconds=3600.0,
    )


def make_absent(name: str = "Bob", roll_no: str = "9999") -> AbsentStudentSummary:
    return AbsentStudentSummary(
        name=name,
        roll_no=roll_no,
        class_duration_seconds=3600.0,
        teacher_name="Dr Smith",
        topics_discussed=["recursion", "loops"],
    )


def make_doc(
    present: dict[str, StudentContext] | None = None,
    absent: dict[str, AbsentStudentSummary] | None = None,
) -> StudentContextDocument:
    meta = BuildContextMetadata(
        total_enrolled=1, present_count=1, absent_count=0, unmatched_count=0
    )
    return StudentContextDocument(
        class_name="CS101",
        present_students=present or {},
        absent_students=absent or {},
        metadata=meta,
    )


def test_stable_chunk_id_deterministic() -> None:
    id1 = stable_chunk_id("2301", "spoken", "hello world")
    id2 = stable_chunk_id("2301", "spoken", "hello world")
    assert id1 == id2


def test_stable_chunk_id_unique() -> None:
    id1 = stable_chunk_id("2301", "spoken", "hello world")
    id2 = stable_chunk_id("2301", "missed", "hello world")
    id3 = stable_chunk_id("9999", "spoken", "hello world")
    assert id1 != id2
    assert id1 != id3


def test_stable_chunk_id_is_hex() -> None:
    chunk_id = stable_chunk_id("2301", "spoken", "text")
    assert all(c in "0123456789abcdef" for c in chunk_id)


def test_chunk_context_produces_records() -> None:
    ctx = make_context()
    records = chunk_student_context(ctx, "CS101")
    assert len(records) > 0


def test_chunk_context_spoken_type() -> None:
    text = "today we covered the supply function and demand curve analysis"
    ctx = make_context(spoken_text=text)
    records = chunk_student_context(ctx, "CS101")
    spoken = [r for r in records if r.chunk_type == "spoken"]
    assert len(spoken) == 1
    assert spoken[0].text == text


def test_chunk_context_chat_type_isolated_to_student() -> None:
    ctx = make_context(name="Anshi", roll_no="2301")
    ctx.chat_segments = [
        ContextSegment(
            start=3.0, end=3.0,
            text="is the supply curve always upward sloping in this case",
            speakers=["Anshi_2301"], source="chat",
        )
    ]
    records = chunk_student_context(ctx, "CS101")
    chat_records = [r for r in records if r.chunk_type == "chat"]
    assert len(chat_records) == 1
    assert chat_records[0].student_id == "2301"
    assert "supply curve always upward" in chat_records[0].text
    assert chat_records[0].speaker == "Anshi_2301"


def test_chunk_context_chat_quality_filter_drops_junk() -> None:
    ctx = make_context(name="Anshi", roll_no="2301")
    ctx.chat_segments = [
        ContextSegment(start=1.0, end=1.0, text="20/3", speakers=["Anshi_2301"], source="chat"),
        ContextSegment(
            start=2.0, end=2.0,
            text="https://docs.google.com/spreadsheets/d/1pnxFhabcdefg",
            speakers=["Anshi_2301"], source="chat",
        ),
    ]
    records = chunk_student_context(ctx, "CS101")
    assert [r for r in records if r.chunk_type == "chat"] == []


def test_chunk_context_missed_type() -> None:
    text = "teacher explained the concept of elastic and inelastic demand"
    ctx = make_context(missed_text=text)
    records = chunk_student_context(ctx, "CS101")
    missed = [r for r in records if r.chunk_type == "missed"]
    assert len(missed) == 1
    assert missed[0].text == text


def test_chunk_context_class_context_type() -> None:
    text = "class covered time and work problems with scaffolding approach used throughout"
    ctx = make_context(present_text=text)
    records = chunk_student_context(ctx, "CS101")
    cc = [r for r in records if r.chunk_type == "class_context"]
    assert len(cc) >= 1


def test_chunk_context_student_id_from_roll_no() -> None:
    ctx = make_context(roll_no="2301")
    records = chunk_student_context(ctx, "CS101")
    assert all(r.student_id == "2301" for r in records)


def test_chunk_context_class_name_set() -> None:
    ctx = make_context()
    records = chunk_student_context(ctx, "CS101")
    assert all(r.class_name == "CS101" for r in records)


def test_chunk_absent_summary_produces_record() -> None:
    summary = make_absent()
    records = chunk_absent_summary(summary, "CS101")
    assert len(records) == 1


def test_chunk_absent_summary_chunk_type() -> None:
    summary = make_absent()
    records = chunk_absent_summary(summary, "CS101")
    assert records[0].chunk_type == "class_context"


def test_chunk_absent_summary_contains_topics() -> None:
    summary = make_absent()
    records = chunk_absent_summary(summary, "CS101")
    assert "recursion" in records[0].text or "loops" in records[0].text


def test_collect_all_records_includes_present_and_absent() -> None:
    ctx = make_context()
    absent = make_absent()
    doc = make_doc(present={"2301": ctx}, absent={"9999": absent})
    records = collect_all_records(doc)
    types = {r.chunk_type for r in records}
    assert len(records) > 0
    assert "class_context" in types


def test_collect_all_records_empty_doc() -> None:
    doc = make_doc()
    records = collect_all_records(doc)
    assert records == []


def test_validate_inputs_missing_file(tmp_path: object) -> None:
    import pytest
    from pathlib import Path

    from scripts.embed_and_store import EmbedArgs, validate_inputs

    args = EmbedArgs(
        contexts_path=Path(str(tmp_path)) / "missing.json",
        db_url="postgresql://localhost/test",
    )
    with pytest.raises(ValueError, match="not found"):
        validate_inputs(args)


def test_validate_inputs_no_db_url(tmp_path: object) -> None:
    import pytest
    from pathlib import Path

    from scripts.embed_and_store import EmbedArgs, validate_inputs

    f = Path(str(tmp_path)) / "ctx.json"
    f.write_text("{}")
    args = EmbedArgs(contexts_path=f, db_url="")
    with pytest.raises(ValueError, match="Database URL"):
        validate_inputs(args)


def test_quality_text_good() -> None:
    from scripts.embed_and_store import is_quality_text
    text = "today we will build our foundation of time and work scaffolding"
    assert is_quality_text(text) is True


def test_quality_text_garbled_repetition() -> None:
    from scripts.embed_and_store import is_quality_text
    text = ("अवाईवोद अवाईवोद अवाईवोद अवाईवोद अवाईवोद अवाईवोद "
            "अवाईवोद अवाईवोद अवाईवोद अवाईवोद अवाईवोद अवाईवोद")
    assert is_quality_text(text) is False


def test_quality_text_phrase_repetition() -> None:
    from scripts.embed_and_store import is_quality_text
    phrase = "tadi subh puchya " * 6
    assert is_quality_text(phrase) is False


def test_quality_text_short_loop_rejected() -> None:
    from scripts.embed_and_store import is_quality_text
    phrase = "माज्यारा पास्ता वेन्द सप्लाइट्रू "
    assert is_quality_text(phrase * 3) is False


def test_quality_text_no_space_loop_rejected() -> None:
    from scripts.embed_and_store import is_quality_text
    text = "अच्छा " + "पाइ" * 60 + " अच्छा"
    assert is_quality_text(text) is False


def test_quality_text_long_ascii_token_kept() -> None:
    from scripts.embed_and_store import is_quality_text
    text = "see the worksheet at https://adira.example.com/math/time-and-work/unit-four-problems today"
    assert is_quality_text(text) is True


def test_quality_text_nukta_dense_garble_rejected() -> None:
    from scripts.embed_and_store import is_quality_text
    text = "ःखा क्या वसुद आई भागास्ँन्से श्याईदुँग़ा रव्दिश्या कर्क्बित्त्प्या और देश्ड़ी कचादाशाना रव्दिश्याईगच़ोग़ा"
    assert is_quality_text(text) is False


def test_quality_text_genuine_hindi_kept() -> None:
    from scripts.embed_and_store import is_quality_text
    text = "market supply ka matlab hai total quantity jo sab producers बेचते हैं different price पर"
    assert is_quality_text(text) is True


def test_quality_text_too_short() -> None:
    from scripts.embed_and_store import is_quality_text
    assert is_quality_text("yes") is False
    assert is_quality_text("   ") is False


def test_quality_text_replacement_chars() -> None:
    from scripts.embed_and_store import is_quality_text
    text = "good content but has ��� garbled here and everywhere indeed"
    assert is_quality_text(text) is False


def _install_fake_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types as pytypes

    class _Vec:
        def __init__(self, data: list[float]) -> None:
            self._data = data

        def tolist(self) -> list[float]:
            return self._data

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def encode(self, texts: list[str], show_progress_bar: bool = False) -> list[_Vec]:
            return [_Vec([0.1, 0.2]) for _ in texts]

    module = pytypes.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_embed_records_stamps_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sentence_transformer(monkeypatch)
    from scripts.embed_and_store import embed_records
    from scripts.models.pipeline import EmbeddingRecord

    records = [
        EmbeddingRecord(
            id="a",
            student_id="2301",
            student_name="Anshi",
            class_name="CS101",
            chunk_type="spoken",
            text="hello world",
        )
    ]
    embed_records(records, "some/model-v2")

    assert records[0].embedding == [0.1, 0.2]
    assert records[0].metadata["embedding_model"] == "some/model-v2"


def test_embed_records_deduped_stamps_every_record(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sentence_transformer(monkeypatch)
    from scripts.embed_and_store import embed_records_deduped
    from scripts.models.pipeline import EmbeddingRecord

    records = [
        EmbeddingRecord(
            id="a", student_id="2301", student_name="Anshi", class_name="CS101",
            chunk_type="material", text="shared material text",
        ),
        EmbeddingRecord(
            id="b", student_id="2302", student_name="Bhagya", class_name="CS101",
            chunk_type="material", text="shared material text",
        ),
    ]
    embed_records_deduped(records, "paraphrase-multilingual-MiniLM-L12-v2")

    assert all(
        record.metadata["embedding_model"] == "paraphrase-multilingual-MiniLM-L12-v2"
        for record in records
    )
