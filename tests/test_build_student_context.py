from __future__ import annotations

import pytest

from scripts.build_student_context import (
    ATTENDANCE_ONLY_PRESENCE_TAG,
    MISSED_UNKNOWN_TAG,
    _collapse_consecutive,
    _dedup_exact,
    attribute_chat_to_students,
    build_absent_summary,
    build_attendance_only_context,
    build_context_document,
    build_present_context,
    class_context_text,
    full_transcript_text,
    merged_seg_to_context,
    teacher_segments_from_transcript,
)
from scripts.parse_chat import ChatMessage
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


def test_merged_seg_to_context() -> None:
    seg = MergedSegment(start=0.0, end=5.0, text="hello", speakers=["Alice"], source="per_student")
    ctx_seg = merged_seg_to_context(seg)
    assert ctx_seg.text == "hello"
    assert ctx_seg.speakers == ["Alice"]
    assert ctx_seg.start == 0.0


def test_full_transcript_text() -> None:
    transcript = make_transcript([(0, 5, "hello world", ["Alice"]), (5, 10, "foo bar", ["Bob"])])
    text = full_transcript_text(transcript)
    assert "hello world" in text
    assert "foo bar" in text


def test_full_transcript_text_empty() -> None:
    transcript = make_transcript()
    assert full_transcript_text(transcript) == ""


def test_build_present_full_attendance() -> None:
    transcript = make_transcript(
        [(0, 10, "hello", ["Anshi"]), (10, 20, "world", ["Anshi"])], duration=20.0
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    att = {"2301": AttendanceRecord(name="Anshi", roll_no="2301", duration_minutes=0.34)}
    ctx = build_present_context(student, entry, transcript, att, ["recursion"])
    assert ctx.status == "present"
    assert len(ctx.missed_segments) == 0


def test_build_present_early_leave() -> None:
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


def test_build_present_spoken_excludes_non_primary_overlap() -> None:
    transcript = make_transcript(
        [
            (0, 5, "anshi is primary here", ["Anshi", "Bob"]),
            (5, 10, "bob is primary here", ["Bob", "Anshi"]),
        ],
        duration=10.0,
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    ctx = build_present_context(student, entry, transcript, {}, [])
    spoken_texts = [s.text for s in ctx.spoken_segments]
    assert spoken_texts == ["anshi is primary here"]
    assert "bob is primary here" not in spoken_texts


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
    assert len(ctx.missed_segments) == 0


def test_build_present_with_attendance_has_no_missed_unknown_flag() -> None:
    from scripts.build_student_context import MISSED_UNKNOWN_TAG

    transcript = make_transcript([(0, 10, "hello", ["Anshi"])], duration=20.0)
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    att = {"2301": AttendanceRecord(name="Anshi", roll_no="2301", duration_minutes=0.34)}
    ctx = build_present_context(student, entry, transcript, att, [])
    assert MISSED_UNKNOWN_TAG not in ctx.tags


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


def test_unmatched_entries_are_not_embedded() -> None:
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


def _chat_roster() -> list[RosterEntry]:
    return [
        RosterEntry(name="Anshi Kumar", roll_no="2301", email=""),
        RosterEntry(name="Bhagyashree", roll_no="2302", email=""),
    ]


def test_attribute_chat_routes_to_each_sender_only() -> None:
    roster = _chat_roster()
    by_roll = {r.roll_no: r for r in roster}
    messages = [
        ChatMessage(time_str="00:00:00", timestamp_seconds=0.0, sender="Anshi_2301", text="anshi typed this"),
        ChatMessage(time_str="00:00:05", timestamp_seconds=5.0, sender="Bhagyashree_2302", text="bhagya typed this"),
        ChatMessage(time_str="00:00:09", timestamp_seconds=9.0, sender="Nisha", text="teacher msg dropped"),
    ]
    chat_by_roll = attribute_chat_to_students(messages, roster, by_roll)

    assert [s.text for s in chat_by_roll["2301"]] == ["anshi typed this"]
    assert [s.text for s in chat_by_roll["2302"]] == ["bhagya typed this"]
    assert "teacher msg dropped" not in {s.text for segs in chat_by_roll.values() for s in segs}
    assert "anshi typed this" not in {s.text for s in chat_by_roll["2302"]}


def test_present_student_gets_own_chat_segment() -> None:
    transcript = make_transcript([(0, 5, "spoken by anshi", ["Anshi"])], duration=10.0)
    roster = [make_roster("Anshi Kumar", "2301")]
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    imap = IdentityMap(teacher_name="Dr Smith", entries=[entry])
    messages = [
        ChatMessage(time_str="00:00:03", timestamp_seconds=3.0, sender="Anshi_2301", text="a question I typed in chat")
    ]
    doc = build_context_document(transcript, imap, roster, [], ["recursion"], None, messages)

    ctx = doc.present_students["2301"]
    assert [s.text for s in ctx.chat_segments] == ["a question I typed in chat"]
    assert ctx.chat_segments[0].source == "chat"


def test_chat_only_student_becomes_present_not_absent() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=10.0)
    roster = [make_roster("Bhavna Rao", "2350")]
    imap = IdentityMap(teacher_name="Dr Smith")
    teacher_doc = make_teacher_doc([(0, 10, "today we cover supply and demand")])
    messages = [
        ChatMessage(time_str="00:00:04", timestamp_seconds=4.0, sender="Bhavna_2350", text="is supply upward sloping?")
    ]
    doc = build_context_document(transcript, imap, roster, [], ["supply"], teacher_doc, messages)

    assert "2350" in doc.present_students
    assert "2350" not in doc.absent_students
    ctx = doc.present_students["2350"]
    assert ctx.status == "present"
    assert "chat_only_no_audio" in ctx.tags
    assert [s.text for s in ctx.chat_segments] == ["is supply upward sloping?"]
    assert any("supply and demand" in s.text for s in ctx.present_segments)
    assert ctx.spoken_segments == []


def test_attendance_only_student_becomes_present() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Video Only", "2401")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "today we cover the water cycle")])
    attendance = [AttendanceRecord(name="Video Only", roll_no="2401", duration_minutes=45.0)]
    doc = build_context_document(transcript, imap, roster, attendance, ["water"], teacher_doc)

    assert "2401" in doc.present_students
    assert "2401" not in doc.absent_students
    ctx = doc.present_students["2401"]
    assert ctx.status == "present"
    assert ATTENDANCE_ONLY_PRESENCE_TAG in ctx.tags
    assert ctx.attendance_duration_minutes == 45.0
    assert ctx.spoken_segments == []
    assert ctx.chat_segments == []
    assert any("water cycle" in s.text for s in ctx.present_segments)
    assert ctx.missed_segments == []


def test_attendance_below_threshold_stays_absent() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Brief Visit", "2402")]
    imap = IdentityMap(teacher_name="Nisha")
    attendance = [AttendanceRecord(name="Brief Visit", roll_no="2402", duration_minutes=3.0)]
    doc = build_context_document(transcript, imap, roster, attendance, [])
    assert "2402" in doc.absent_students
    assert "2402" not in doc.present_students


def test_attendance_only_falls_back_to_merged_timeline_without_teacher() -> None:
    transcript = make_transcript([(0, 5, "merged timeline content", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Video Only", "2401")]
    imap = IdentityMap(teacher_name="Nisha")
    attendance = [AttendanceRecord(name="Video Only", roll_no="2401", duration_minutes=45.0)]
    doc = build_context_document(transcript, imap, roster, attendance, [], None)
    ctx = doc.present_students["2401"]
    assert any("merged timeline content" in s.text for s in ctx.present_segments)


def test_chat_takes_precedence_over_attendance() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Typed", "2403")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "photosynthesis basics")])
    attendance = [AttendanceRecord(name="Typed", roll_no="2403", duration_minutes=45.0)]
    messages = [
        ChatMessage(
            time_str="00:00:04", timestamp_seconds=4.0, sender="Typed_2403", text="a typed question"
        )
    ]
    doc = build_context_document(transcript, imap, roster, attendance, [], teacher_doc, messages)
    ctx = doc.present_students["2403"]
    assert "chat_only_no_audio" in ctx.tags
    assert ATTENDANCE_ONLY_PRESENCE_TAG not in ctx.tags
    assert [s.text for s in ctx.chat_segments] == ["a typed question"]


def test_skip_absent_summaries_keeps_attendance_present() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Video Only", "2401"), make_roster("Truly Absent", "2999")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "content")])
    attendance = [AttendanceRecord(name="Video Only", roll_no="2401", duration_minutes=45.0)]
    doc = build_context_document(
        transcript, imap, roster, attendance, [], teacher_doc, None, skip_absent_summaries=True
    )
    assert "2401" in doc.present_students
    assert doc.absent_students == {}
    assert "2999" not in doc.present_students


def test_attendance_roll_absent_from_roster_warns(caplog: pytest.LogCaptureFixture) -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("On Roster", "2401")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "content")])
    attendance = [
        AttendanceRecord(name="On Roster", roll_no="2401", duration_minutes=45.0),
        AttendanceRecord(name="Ghost", roll_no="2407", duration_minutes=30.0),
    ]
    with caplog.at_level("WARNING"):
        doc = build_context_document(transcript, imap, roster, attendance, [], teacher_doc)
    assert "2407" not in doc.present_students
    assert "2407" not in doc.absent_students
    assert "2407" in caplog.text
    assert "roster" in caplog.text


def test_attendance_name_agreement_grants_presence() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Rani Sharma", "2620")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "class content")])
    attendance = [AttendanceRecord(name="Rani Sharma", roll_no="2620", duration_minutes=45.0)]
    doc = build_context_document(transcript, imap, roster, attendance, [], teacher_doc)
    assert "2620" in doc.present_students
    assert ATTENDANCE_ONLY_PRESENCE_TAG in doc.present_students["2620"].tags


def test_attendance_roll_name_mismatch_denies_presence(caplog: pytest.LogCaptureFixture) -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Priya Nair", "2407")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "class content")])
    attendance = [AttendanceRecord(name="Rani Sharma", roll_no="2407", duration_minutes=150.0)]
    with caplog.at_level("WARNING"):
        doc = build_context_document(transcript, imap, roster, attendance, [], teacher_doc)
    assert "2407" not in doc.present_students
    assert "2407" in doc.absent_students
    assert "Rani Sharma" in caplog.text
    assert "Priya Nair" in caplog.text


def test_empty_attendance_name_trusts_roll() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Anon Student", "2500")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "class content")])
    attendance = [AttendanceRecord(name="", roll_no="2500", duration_minutes=45.0)]
    doc = build_context_document(transcript, imap, roster, attendance, [], teacher_doc)
    assert "2500" in doc.present_students


def test_build_attendance_only_context_flags_missed_unknown() -> None:
    transcript = make_transcript([(0, 5, "content", ["Dr Smith"])], duration=600.0)
    student = make_roster("Video Only", "2401")
    teacher_segs = [cseg(0, 600, "teacher content")]
    att = AttendanceRecord(name="Video Only", roll_no="2401", duration_minutes=45.0)
    ctx = build_attendance_only_context(student, transcript, ["topic"], teacher_segs, att)
    assert MISSED_UNKNOWN_TAG in ctx.tags
    assert ATTENDANCE_ONLY_PRESENCE_TAG in ctx.tags
    assert ctx.missed_segments == []
    assert ctx.attendance_duration_minutes == 45.0
    assert [s.text for s in ctx.present_segments] == ["teacher content"]


def test_student_without_chat_stays_absent() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=10.0)
    roster = [make_roster("Silent Sam", "2360")]
    imap = IdentityMap(teacher_name="Dr Smith")
    doc = build_context_document(transcript, imap, roster, [], [], None, [])
    assert "2360" in doc.absent_students


def test_skip_absent_summaries_drops_only_pure_absent_students() -> None:
    transcript = make_transcript([(0, 5, "spoken by anshi", ["Anshi"])], duration=10.0)
    roster = [
        make_roster("Anshi Kumar", "2301"),
        make_roster("Bhavna Rao", "2350"),
        make_roster("Silent Sam", "2360"),
    ]
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    imap = IdentityMap(teacher_name="Dr Smith", entries=[entry])
    messages = [
        ChatMessage(time_str="00:00:04", timestamp_seconds=4.0, sender="Bhavna_2350", text="a question I typed here")
    ]
    doc = build_context_document(
        transcript, imap, roster, [], ["topic"], None, messages, skip_absent_summaries=True
    )

    assert doc.absent_students == {}
    assert "2301" in doc.present_students
    assert "2350" in doc.present_students
    assert "chat_only_no_audio" in doc.present_students["2350"].tags
    assert "2360" not in doc.present_students


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


def test_build_present_teacher_plus_own_only() -> None:
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
    att = {"2301": AttendanceRecord(name="Anshi", roll_no="2301", duration_minutes=0.5)}
    ctx = build_present_context(student, entry, transcript, att, [], teacher_segs)

    present_texts = [s.text for s in ctx.present_segments]
    assert "teacher explains the topic" in present_texts
    assert "anshi own answer" in present_texts
    assert "peer bob speaks" not in present_texts
    assert "noisy mp4 fallback" not in present_texts
    assert [s.text for s in ctx.spoken_segments] == ["anshi own answer"]


def test_build_present_teacher_segments_past_window_land_in_missed() -> None:
    transcript = make_transcript([(0, 5, "anshi own answer", ["Anshi"])], duration=20.0)
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    teacher_segs = [cseg(0, 10, "during class"), cseg(25, 35, "after the session ended")]
    ctx = build_present_context(student, entry, transcript, {}, [], teacher_segs)

    present_texts = [s.text for s in ctx.present_segments]
    missed_texts = [s.text for s in ctx.missed_segments]
    assert "during class" in present_texts
    assert "after the session ended" in missed_texts


def test_build_present_fallback_parity_without_teacher() -> None:
    transcript = make_transcript(
        [(0, 5, "anshi own answer", ["Anshi"]), (5, 10, "peer bob speaks", ["Bob"])],
        duration=20.0,
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    ctx = build_present_context(student, entry, transcript, {}, [], None)
    present_texts = [s.text for s in ctx.present_segments]
    assert "anshi own answer" in present_texts
    assert "peer bob speaks" in present_texts


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
    assert "peer bob speaks" in present_texts


def test_dedup_exact_keeps_first_of_interleaved_repeats() -> None:
    segs = [
        cseg(0, 1, "A"),
        cseg(1, 2, "B"),
        cseg(2, 3, "A"),
        cseg(3, 4, "C"),
        cseg(4, 5, "A"),
    ]
    assert [s.text for s in _dedup_exact(segs)] == ["A", "B", "C"]


def test_dedup_exact_normalizes_case_and_whitespace() -> None:
    segs = [cseg(0, 1, "Thank you."), cseg(1, 2, "  thank   YOU.  "), cseg(2, 3, "THANK YOU.")]
    assert [s.text for s in _dedup_exact(segs)] == ["Thank you."]


def test_collapse_consecutive_drops_adjacent_keeps_nonadjacent() -> None:
    segs = [cseg(0, 1, "T"), cseg(1, 2, "T"), cseg(2, 3, "U"), cseg(3, 4, "T")]
    assert [s.text for s in _collapse_consecutive(segs)] == ["T", "U", "T"]


def test_collapse_consecutive_normalizes_adjacent_variants() -> None:
    segs = [cseg(0, 1, "Hello world"), cseg(1, 2, "hello   world"), cseg(2, 3, "next")]
    assert [s.text for s in _collapse_consecutive(segs)] == ["Hello world", "next"]


def test_build_present_dedups_looped_spoken_segments() -> None:
    transcript = make_transcript(
        [
            (0, 1, "thank you.", ["Anshi"]),
            (1, 2, "thank you.", ["Anshi"]),
            (2, 3, "real content", ["Anshi"]),
            (3, 4, "thank you.", ["Anshi"]),
        ],
        duration=10.0,
    )
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    student = make_roster()
    ctx = build_present_context(student, entry, transcript, {}, [])
    assert [s.text for s in ctx.spoken_segments] == ["thank you.", "real content"]


def test_teacher_timeline_collapses_consecutive_repeats() -> None:
    transcript = make_transcript([(0, 5, "student speech", ["Anshi"])], duration=100.0)
    roster = [make_roster("Anshi Kumar", "2301")]
    entry = make_entry(matched_name="Anshi", matched_roll_no="2301")
    imap = IdentityMap(teacher_name="Dr Smith", entries=[entry])
    teacher_doc = make_teacher_doc(
        [
            (0, 5, "open the book"),
            (5, 10, "open the book"),
            (10, 15, "now solve"),
            (15, 20, "open the book"),
        ]
    )
    doc = build_context_document(transcript, imap, roster, [], [], teacher_doc)
    teacher_texts = [
        s.text for s in doc.present_students["2301"].present_segments if s.source == "teacher"
    ]
    assert teacher_texts == ["open the book", "now solve", "open the book"]


def test_attendance_only_merged_fallback_collapses_consecutive() -> None:
    transcript = make_transcript(
        [
            (0, 1, "loop", ["X"]),
            (1, 2, "loop", ["X"]),
            (2, 3, "distinct", ["X"]),
            (3, 4, "loop", ["X"]),
        ],
        duration=600.0,
    )
    roster = [make_roster("Video Only", "2401")]
    imap = IdentityMap(teacher_name="Nisha")
    attendance = [AttendanceRecord(name="Video Only", roll_no="2401", duration_minutes=45.0)]
    doc = build_context_document(transcript, imap, roster, attendance, [], None)
    ctx = doc.present_students["2401"]
    assert [s.text for s in ctx.present_segments] == ["loop", "distinct", "loop"]


def test_roll_less_attendance_name_resolves_to_presence() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Ramsha Khan", "2408")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "today's lesson")])
    attendance = [AttendanceRecord(name="Ramsha Khan", roll_no=None, duration_minutes=45.0)]
    doc = build_context_document(transcript, imap, roster, attendance, [], teacher_doc)
    assert "2408" in doc.present_students
    ctx = doc.present_students["2408"]
    assert ATTENDANCE_ONLY_PRESENCE_TAG in ctx.tags
    assert ctx.attendance_duration_minutes == 45.0


def test_roll_less_attendance_below_threshold_stays_absent() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Ramsha Khan", "2408")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "lesson")])
    attendance = [AttendanceRecord(name="Ramsha Khan", roll_no=None, duration_minutes=3.0)]
    doc = build_context_document(transcript, imap, roster, attendance, [], teacher_doc)
    assert "2408" in doc.absent_students
    assert "2408" not in doc.present_students


def test_roll_less_ambiguous_name_denies_presence(caplog: pytest.LogCaptureFixture) -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [
        RosterEntry(name="Disha", roll_no="2504", email=""),
        RosterEntry(name="Disha Rajesh", roll_no="2505", email=""),
    ]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "lesson")])
    attendance = [AttendanceRecord(name="Disha", roll_no=None, duration_minutes=45.0)]
    with caplog.at_level("WARNING"):
        doc = build_context_document(transcript, imap, roster, attendance, [], teacher_doc)
    assert "2504" not in doc.present_students
    assert "2505" not in doc.present_students
    assert "2504" in doc.absent_students
    assert "2505" in doc.absent_students


def test_originally_rolled_beats_name_resolved_for_same_roll() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    roster = [make_roster("Ramsha Khan", "2408")]
    imap = IdentityMap(teacher_name="Nisha")
    teacher_doc = make_teacher_doc([(0, 600, "lesson")])
    attendance = [
        AttendanceRecord(name="Ramsha Khan", roll_no=None, duration_minutes=45.0),
        AttendanceRecord(name="Ramsha Khan", roll_no="2408", duration_minutes=99.0),
    ]
    doc = build_context_document(transcript, imap, roster, attendance, [], teacher_doc)
    ctx = doc.present_students["2408"]
    assert ctx.attendance_duration_minutes == 99.0


def test_no_roster_skips_name_resolution() -> None:
    transcript = make_transcript([(0, 5, "teacher talk", ["Dr Smith"])], duration=600.0)
    imap = IdentityMap(teacher_name="Nisha")
    attendance = [AttendanceRecord(name="Ramsha Khan", roll_no=None, duration_minutes=45.0)]
    doc = build_context_document(transcript, imap, [], attendance, [])
    assert doc.present_students == {}
    assert doc.absent_students == {}
