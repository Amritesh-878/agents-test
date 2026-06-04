from __future__ import annotations

from scripts.build_student_context import (
    build_absent_summary,
    build_context_document,
    build_present_context,
    class_context_text,
    full_transcript_text,
    merged_seg_to_context,
    teacher_segments_from_transcript,
)
from scripts.models.context import ContextSegment
from scripts.models.identity import (
    AttendanceRecord,
    IdentityMap,
    IdentityMapEntry,
    RosterEntry,
)
from scripts.models.transcript import (
    MergedSegment,
    MergedTranscriptDocument,
    MergeMetadata,
    PerStudentTranscript,
    TranscriptDocument,
    TranscriptSegment,
)


def make_teacher_doc(
    segments: list[tuple[float, float, str]],
    audio_file: str = "audioNisha_0000.m4a",
) -> PerStudentTranscript:
    segs = [TranscriptSegment(start=s, end=e, text=t) for s, e, t in segments]
    return PerStudentTranscript(
        audio_file=audio_file,
        is_teacher=True,
        transcript=TranscriptDocument(model="small", segments=segs),
        merged_words=[],
    )


def cseg(
    start: float,
    end: float,
    text: str,
    speakers: list[str] | None = None,
    source: str = "teacher",
) -> ContextSegment:
    return ContextSegment(
        start=start, end=end, text=text, speakers=speakers or ["Dr Smith"], source=source
    )


# --- Helpers ---


def make_transcript(
    segments: list[tuple[float, float, str, list[str]]] | None = None,
    duration: float = 3600.0,
    class_name: str = "CS101",
    teacher: str = "Dr Smith",
) -> MergedTranscriptDocument:
    from scripts.models.transcript import MergedSegment

    segs = []
    for start, end, text, speakers in (segments or []):
        segs.append(
            MergedSegment(start=start, end=end, text=text, speakers=speakers, source="per_student")
        )
    meta = MergeMetadata(
        total_segments=len(segs),
        per_student_segments=len(segs),
        session_fallback_segments=0,
        multi_speaker_segments=0,
        alignment_mode="per_student_canonical",
        merge_method="cluster_then_fill",
    )
    return MergedTranscriptDocument(
        class_name=class_name,
        duration_seconds=duration,
        segments=segs,
        speakers=[],
        teacher_name=teacher,
        metadata=meta,
    )


def make_entry(
    audio_file: str = "audio.m4a",
    matched_name: str = "Anshi",
    matched_roll_no: str = "2301",
) -> IdentityMapEntry:
    return IdentityMapEntry(
        audio_file=audio_file,
        matched_name=matched_name,
        matched_roll_no=matched_roll_no,
        match_method="roll_no",
        match_confidence=1.0,
    )


def make_roster(name: str = "Anshi Kumar", roll_no: str = "2301") -> RosterEntry:
    return RosterEntry(name=name, roll_no=roll_no, email=f"{roll_no}@example.com")


# --- merged_seg_to_context ---


def test_merged_seg_to_context() -> None:
    seg = MergedSegment(start=0.0, end=5.0, text="hello", speakers=["Alice"], source="per_student")
    ctx_seg = merged_seg_to_context(seg)
    assert ctx_seg.text == "hello"
    assert ctx_seg.speakers == ["Alice"]
    assert ctx_seg.start == 0.0


# --- full_transcript_text ---


def test_full_transcript_text() -> None:
    transcript = make_transcript([(0, 5, "hello world", ["Alice"]), (5, 10, "foo bar", ["Bob"])])
    text = full_transcript_text(transcript)
    assert "hello world" in text
    assert "foo bar" in text


def test_full_transcript_text_empty() -> None:
    transcript = make_transcript()
    assert full_transcript_text(transcript) == ""


# --- build_present_context ---


def test_build_present_full_attendance() -> None:
    transcript = make_transcript(
        [(0, 10, "hello", ["Anshi"]), (10, 20, "world", ["Anshi"])], duration=20.0
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    att = {"2301": AttendanceRecord(name="Anshi", roll_no="2301", duration_minutes=0.34)}  # ~20s
    ctx = build_present_context(student, entry, transcript, att, ["recursion"])
    assert ctx.status == "present"
    assert len(ctx.missed_segments) == 0


def test_build_present_early_leave() -> None:
    # Student attended first 15s of a 30s class (left early), so segment at t=20-30 is missed.
    # With duration-only attendance, window_end = duration_minutes * 60 = 15s.
    transcript = make_transcript(
        [(0, 10, "attended", ["Anshi"]), (20, 30, "missed_this", ["Other"])], duration=30.0
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    att = {"2301": AttendanceRecord(name="Anshi", roll_no="2301", duration_minutes=0.25)}
    ctx = build_present_context(student, entry, transcript, att, [])
    missed_texts = [s.text for s in ctx.missed_segments]
    assert "missed_this" in missed_texts


def test_build_present_spoken_segments() -> None:
    transcript = make_transcript(
        [(0, 5, "spoken by anshi", ["Anshi"]), (5, 10, "spoken by bob", ["Bob"])], duration=10.0
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    ctx = build_present_context(student, entry, transcript, {}, [])
    assert len(ctx.spoken_segments) == 1
    assert ctx.spoken_segments[0].text == "spoken by anshi"


def test_build_present_no_speech() -> None:
    transcript = make_transcript(
        [(0, 10, "bob speaks", ["Bob"]), (10, 20, "also bob", ["Bob"])], duration=20.0
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    ctx = build_present_context(student, entry, transcript, {}, [])
    assert len(ctx.spoken_segments) == 0
    assert len(ctx.present_segments) == 2


def test_build_present_no_attendance_flags_missed_unknown() -> None:
    from scripts.build_student_context import MISSED_UNKNOWN_TAG

    transcript = make_transcript([(0, 10, "hello", ["Anshi"])], duration=20.0)
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    ctx = build_present_context(student, entry, transcript, {}, [])
    assert MISSED_UNKNOWN_TAG in ctx.tags
    assert len(ctx.missed_segments) == 0  # still empty, but now explicitly flagged


def test_build_present_with_attendance_has_no_missed_unknown_flag() -> None:
    from scripts.build_student_context import MISSED_UNKNOWN_TAG

    transcript = make_transcript([(0, 10, "hello", ["Anshi"])], duration=20.0)
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    att = {"2301": AttendanceRecord(name="Anshi", roll_no="2301", duration_minutes=0.34)}
    ctx = build_present_context(student, entry, transcript, att, [])
    assert MISSED_UNKNOWN_TAG not in ctx.tags


# --- build_absent_summary ---


def test_build_absent_summary() -> None:
    transcript = make_transcript(duration=3600.0, teacher="Dr Smith")
    student = make_roster(name="Bob", roll_no="9999")
    topics = ["recursion", "loops"]
    summary = build_absent_summary(student, transcript, topics)
    assert summary.name == "Bob"
    assert summary.teacher_name == "Dr Smith"
    assert summary.class_duration_seconds == 3600.0
    assert "recursion" in summary.topics_discussed


def test_absent_summary_no_segments() -> None:
    transcript = make_transcript()
    student = make_roster(name="Alice", roll_no="1111")
    summary = build_absent_summary(student, transcript, [])
    assert not hasattr(summary, "spoken_segments")


# --- build_context_document ---


def test_unmatched_entries_are_not_embedded() -> None:
    # An unmatched M4A (no usable roll) must NOT become a present student, because its
    # student_id would be the raw filename — unusable for login and noise in the demo.
    transcript = make_transcript([(0, 5, "hello", ["UNKNOWN"])], duration=10.0)
    unmatched = IdentityMapEntry(
        audio_file="audioUnknown_99991234.m4a",
        roll_no_4digit=None,
        match_method="none",
        match_confidence=0.0,
        is_unmatched=True,
        tags=["unmatched"],
    )
    imap = IdentityMap(teacher_name="Dr Smith", unmatched_entries=[unmatched])
    doc = build_context_document(transcript, imap, [], [], [])
    assert len(doc.present_students) == 0
    assert doc.metadata.unmatched_count == 1


def test_roster_driven_all_get_context() -> None:
    transcript = make_transcript(
        [(0, 10, "hello Anshi speaks", ["Anshi"])], duration=10.0
    )
    roster = [make_roster("Anshi Kumar", "2301"), make_roster("Bob Absent", "9999")]
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    imap = IdentityMap(teacher_name="Dr Smith", entries=[entry])
    doc = build_context_document(transcript, imap, roster, [], ["recursion"])
    assert "2301" in doc.present_students
    assert "9999" in doc.absent_students


def test_all_students_absent() -> None:
    transcript = make_transcript(duration=3600.0)
    roster = [make_roster("Alice", "1111"), make_roster("Bob", "2222")]
    imap = IdentityMap(teacher_name="Dr Smith")
    doc = build_context_document(transcript, imap, roster, [], [])
    assert len(doc.absent_students) == 2
    assert len(doc.present_students) == 0


def test_build_context_metadata_counts() -> None:
    transcript = make_transcript([(0, 5, "hello", ["Anshi"])], duration=10.0)
    roster = [make_roster()]
    entry = make_entry()
    imap = IdentityMap(teacher_name="Dr Smith", entries=[entry])
    doc = build_context_document(transcript, imap, roster, [], [])
    assert doc.metadata.total_enrolled == 1
    assert doc.metadata.present_count == 1
    assert doc.metadata.absent_count == 0


def test_context_document_round_trip() -> None:
    import json

    transcript = make_transcript(duration=100.0)
    roster = [make_roster()]
    imap = IdentityMap(teacher_name="Dr Smith")
    doc = build_context_document(transcript, imap, roster, [], [])
    data = json.loads(doc.model_dump_json())
    from scripts.models.context import StudentContextDocument

    restored = StudentContextDocument.model_validate(data)
    assert restored.class_name == doc.class_name


# --- teacher_segments_from_transcript ---


def test_teacher_segments_from_transcript_maps_and_attributes() -> None:
    teacher_doc = make_teacher_doc([(0, 10, "today we cover supply"), (10, 20, "and demand")])
    segs = teacher_segments_from_transcript(teacher_doc, "Nisha")
    assert len(segs) == 2
    assert segs[0].text == "today we cover supply"
    assert segs[0].speakers == ["Nisha"]
    assert segs[0].source == "teacher"


def test_teacher_segments_from_transcript_skips_blank() -> None:
    teacher_doc = make_teacher_doc([(0, 5, "   "), (5, 10, "real content here")])
    segs = teacher_segments_from_transcript(teacher_doc, "Nisha")
    assert len(segs) == 1
    assert segs[0].text == "real content here"


# --- class_context_text ---


def test_class_context_text_prefers_teacher() -> None:
    transcript = make_transcript([(0, 10, "noisy mixed mp4 text", ["UNKNOWN"])], duration=10.0)
    teacher_doc = make_teacher_doc([(0, 10, "clean teacher words")])
    text = class_context_text(transcript, teacher_doc)
    assert "clean teacher words" in text
    assert "noisy mixed mp4 text" not in text


def test_class_context_text_falls_back_to_merged() -> None:
    transcript = make_transcript([(0, 10, "merged timeline text", ["Anshi"])], duration=10.0)
    text = class_context_text(transcript, None)
    assert "merged timeline text" in text


# --- build_present_context: teacher M4A as primary class-context source ---


def test_build_present_teacher_plus_own_only() -> None:
    # Merged timeline mixes the student's own speech, a peer, and noisy session fallback.
    transcript = make_transcript(
        [
            (0, 5, "anshi own answer", ["Anshi"]),
            (5, 10, "peer bob speaks", ["Bob"]),
            (10, 15, "noisy mp4 fallback", ["UNKNOWN"]),
        ],
        duration=30.0,
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    teacher_segs = [cseg(0, 30, "teacher explains the topic")]
    att = {"2301": AttendanceRecord(name="Anshi", roll_no="2301", duration_minutes=0.5)}  # 30s
    ctx = build_present_context(student, entry, transcript, att, [], teacher_segs)

    present_texts = [s.text for s in ctx.present_segments]
    assert "teacher explains the topic" in present_texts
    assert "anshi own answer" in present_texts  # student's own contribution kept
    assert "peer bob speaks" not in present_texts  # peer excluded
    assert "noisy mp4 fallback" not in present_texts  # session fallback excluded
    # spoken is unaffected — still derived from the merged timeline by speaker name
    assert [s.text for s in ctx.spoken_segments] == ["anshi own answer"]


def test_build_present_teacher_segments_past_window_land_in_missed() -> None:
    # Teacher track runs PAST class_duration/window_end (offset-0 shared-clock edge).
    # Late teacher segments must land in `missed` without crashing.
    transcript = make_transcript([(0, 5, "anshi own answer", ["Anshi"])], duration=20.0)
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    teacher_segs = [cseg(0, 10, "during class"), cseg(25, 35, "after the session ended")]
    # No attendance -> window_end = class_duration = 20s.
    ctx = build_present_context(student, entry, transcript, {}, [], teacher_segs)

    present_texts = [s.text for s in ctx.present_segments]
    missed_texts = [s.text for s in ctx.missed_segments]
    assert "during class" in present_texts
    assert "after the session ended" in missed_texts


def test_build_present_fallback_parity_without_teacher() -> None:
    # teacher_segments=None -> identical to pre-teacher behavior (full merged timeline).
    transcript = make_transcript(
        [(0, 5, "anshi own answer", ["Anshi"]), (5, 10, "peer bob speaks", ["Bob"])],
        duration=20.0,
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    ctx = build_present_context(student, entry, transcript, {}, [], None)
    present_texts = [s.text for s in ctx.present_segments]
    assert "anshi own answer" in present_texts
    assert "peer bob speaks" in present_texts  # peer present in fallback mode


# --- build_unmatched_context with teacher segments ---


# --- build_context_document: teacher-primary end-to-end ---


def test_build_context_document_teacher_primary() -> None:
    transcript = make_transcript(
        [
            (0, 5, "anshi own answer", ["Anshi"]),
            (5, 10, "peer bob speaks", ["Bob"]),
        ],
        duration=20.0,
        teacher="Dr Smith",
    )
    roster = [make_roster("Anshi Kumar", "2301")]
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    imap = IdentityMap(
        teacher_name="Dr Smith",
        teacher_audio_file="audioSmith_0000.m4a",
        entries=[entry],
    )
    teacher_doc = make_teacher_doc([(0, 20, "teacher explains supply and demand")])
    doc = build_context_document(transcript, imap, roster, [], ["supply"], teacher_doc)

    ctx = doc.present_students["2301"]
    present_texts = [s.text for s in ctx.present_segments]
    assert "teacher explains supply and demand" in present_texts
    assert "anshi own answer" in present_texts
    assert "peer bob speaks" not in present_texts


def test_build_context_document_no_teacher_doc_is_fallback() -> None:
    transcript = make_transcript(
        [(0, 5, "anshi own answer", ["Anshi"]), (5, 10, "peer bob speaks", ["Bob"])],
        duration=20.0,
    )
    roster = [make_roster("Anshi Kumar", "2301")]
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    imap = IdentityMap(teacher_name="Dr Smith", entries=[entry])
    doc = build_context_document(transcript, imap, roster, [], [], None)
    present_texts = [s.text for s in doc.present_students["2301"].present_segments]
    assert "peer bob speaks" in present_texts  # full merged timeline when no teacher M4A
