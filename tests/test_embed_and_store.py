from __future__ import annotations

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
    spoken_text: str = "hello world",
    missed_text: str = "missed content",
    present_text: str = "class content",
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


# --- stable_chunk_id ---


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


# --- chunk_student_context ---


def test_chunk_context_produces_records() -> None:
    ctx = make_context()
    records = chunk_student_context(ctx, "CS101")
    assert len(records) > 0


def test_chunk_context_spoken_type() -> None:
    ctx = make_context(spoken_text="I said this")
    records = chunk_student_context(ctx, "CS101")
    spoken = [r for r in records if r.chunk_type == "spoken"]
    assert len(spoken) == 1
    assert spoken[0].text == "I said this"


def test_chunk_context_missed_type() -> None:
    ctx = make_context(missed_text="I missed this")
    records = chunk_student_context(ctx, "CS101")
    missed = [r for r in records if r.chunk_type == "missed"]
    assert len(missed) == 1
    assert missed[0].text == "I missed this"


def test_chunk_context_class_context_type() -> None:
    ctx = make_context(present_text="whole class content")
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


# --- chunk_absent_summary ---


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


# --- collect_all_records ---


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


# --- validate_inputs ---


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
