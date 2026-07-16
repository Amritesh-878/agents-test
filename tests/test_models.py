from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts.models.identity import (
    AttendanceRecord,
    AudioFileIdentity,
    MatchResult,
    RosterEntry,
    StudentIdentity,
)
from scripts.models.pipeline import ClassSession, PipelineConfig
from scripts.models.transcript import (
    DualLanguageWord,
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)


def test_transcript_word_construction() -> None:
    word = TranscriptWord(start=0.0, end=1.5, word="hello", score=0.98)
    assert word.word == "hello"
    assert word.score == 0.98


def test_transcript_word_score_optional() -> None:
    word = TranscriptWord(start=0.0, end=1.0, word="hi")
    assert word.score is None


def test_transcript_word_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        TranscriptWord(start=0.0, end=1.0)  # type: ignore[call-arg]


def test_transcript_segment_construction() -> None:
    seg = TranscriptSegment(start=0.0, end=5.0, text="Hello world")
    assert seg.text == "Hello world"
    assert seg.words == []
    assert seg.language is None


def test_transcript_segment_with_words() -> None:
    word = TranscriptWord(start=0.0, end=1.0, word="hello", score=0.9)
    seg = TranscriptSegment(start=0.0, end=1.0, text="hello", words=[word], language="en")
    assert len(seg.words) == 1
    assert seg.language == "en"


def test_transcript_document_construction() -> None:
    doc = TranscriptDocument(model="small")
    assert doc.model == "small"
    assert doc.segments == []
    assert doc.language is None


def test_transcript_document_round_trip() -> None:
    seg = TranscriptSegment(start=0.0, end=2.0, text="test")
    doc = TranscriptDocument(model="small", language="en", segments=[seg])
    data = doc.model_dump()
    restored = TranscriptDocument.model_validate(data)
    assert restored.language == "en"
    assert len(restored.segments) == 1


def test_dual_language_word_construction() -> None:
    word = DualLanguageWord(start=0.0, end=1.0, word="नमस्ते", score=0.95, source_language="hi")
    assert word.source_language == "hi"
    assert word.score == 0.95


def test_dual_language_word_english() -> None:
    word = DualLanguageWord(start=1.0, end=2.0, word="hello", score=0.88, source_language="en")
    assert word.source_language == "en"


def test_dual_language_word_score_comparison() -> None:
    hi = DualLanguageWord(start=0.0, end=1.0, word="x", score=0.9, source_language="hi")
    en = DualLanguageWord(start=0.0, end=1.0, word="x", score=0.7, source_language="en")
    assert hi.score > en.score


def test_roster_entry_construction() -> None:
    entry = RosterEntry(name="Ansh Jain", roll_no="2022", email="ajain@example.com")
    assert entry.roll_no == "2022"


def test_roster_entry_leading_zero_roll_no() -> None:
    entry = RosterEntry(name="Bob", roll_no="0042", email="bob@example.com")
    assert entry.roll_no == "0042"


def test_roster_entry_missing_email() -> None:
    with pytest.raises(ValidationError):
        RosterEntry(name="Alice", roll_no="1234")  # type: ignore[call-arg]


def test_attendance_record_construction() -> None:
    rec = AttendanceRecord(name="Ansh_2022", duration_minutes=45.0)
    assert rec.duration_minutes == 45.0
    assert rec.roll_no is None
    assert rec.tags == []


def test_attendance_record_with_roll_no() -> None:
    rec = AttendanceRecord(name="Ansh_2022", roll_no="2022", duration_minutes=45.0)
    assert rec.roll_no == "2022"


def test_attendance_record_guest_flag() -> None:
    rec = AttendanceRecord(name="Guest User", duration_minutes=10.0, guest=True)
    assert rec.guest is True


def test_attendance_record_tags() -> None:
    rec = AttendanceRecord(name="Bob_1111", duration_minutes=5.0, tags=["short_duration"])
    assert "short_duration" in rec.tags


def test_audio_file_identity_construction() -> None:
    af = AudioFileIdentity(filename="audio_Ansh_202212345.m4a", roll_no_4digit="2022")
    assert af.roll_no_4digit == "2022"
    assert af.extracted_number is None


def test_audio_file_identity_all_none() -> None:
    af = AudioFileIdentity(filename="unknown.m4a")
    assert af.extracted_number is None
    assert af.roll_no_4digit is None
    assert af.display_name is None


def test_student_identity_from_roster() -> None:
    si = StudentIdentity(name="Ansh", roll_no="2022", email="a@b.com", source="roster")
    assert si.source == "roster"


def test_student_identity_from_audio_file() -> None:
    si = StudentIdentity(name="Ansh", source="audio_file")
    assert si.roll_no is None
    assert si.email is None


def test_student_identity_invalid_source() -> None:
    with pytest.raises(ValidationError):
        StudentIdentity(name="X", source="unknown")  # type: ignore[arg-type]


def test_match_result_roll_no_match() -> None:
    student = StudentIdentity(name="Ansh", roll_no="2022", source="roster")
    result = MatchResult(
        audio_file="audio_Ansh_2022.m4a",
        matched_student=student,
        method="roll_no",
        confidence=1.0,
    )
    assert result.method == "roll_no"
    assert result.confidence == 1.0


def test_match_result_no_match() -> None:
    result = MatchResult(audio_file="unknown.m4a", method="none", confidence=0.0)
    assert result.matched_student is None


def test_match_result_invalid_method() -> None:
    with pytest.raises(ValidationError):
        MatchResult(audio_file="x.m4a", method="exact", confidence=1.0)  # type: ignore[arg-type]


def test_pipeline_config_construction(tmp_path: pytest.TempPathFactory) -> None:
    config = PipelineConfig(input_path=tmp_path, output_dir=tmp_path)  # type: ignore[arg-type]
    assert config.teacher_names == []
    assert config.db_url is None


def test_pipeline_config_full(tmp_path: pytest.TempPathFactory) -> None:
    config = PipelineConfig(  # type: ignore[arg-type]
        input_path=tmp_path,
        output_dir=tmp_path,
        teacher_names=["Dr. Smith"],
        db_url="postgresql://localhost/test",
        single_language="en",
    )
    assert config.single_language == "en"
    assert config.teacher_names == ["Dr. Smith"]


def test_class_session_construction(tmp_path: pytest.TempPathFactory) -> None:
    session = ClassSession(  # type: ignore[arg-type]
        class_name="CS101",
        raw_dir=tmp_path,
        output_dir=tmp_path,
        teacher_name="Dr. Smith",
    )
    assert session.zip_path is None
    assert session.class_name == "CS101"


def test_class_session_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    session = ClassSession(  # type: ignore[arg-type]
        class_name="CS101",
        raw_dir=tmp_path,
        output_dir=tmp_path,
        teacher_name="Dr. Smith",
    )
    data = session.model_dump()
    restored = ClassSession.model_validate(data)
    assert restored.class_name == session.class_name
    assert restored.teacher_name == session.teacher_name
