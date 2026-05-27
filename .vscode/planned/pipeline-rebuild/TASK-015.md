# TASK-015: Transcript Merge (Speaker Attribution + Text Replacement + Alignment Detection)

## Overview

Combine per-student transcripts with the full-session transcript to produce a single speaker-attributed, text-replaced merged transcript. Detect whether per-student M4A timestamps are session-aligned or join-offset. This is the most algorithm-heavy task in the rebuild.

## Execution Snapshot

- Depends on: TASK-013, TASK-014
- Produces: `scripts/merge_transcripts.py`, `output/<class_name>/transcript_merged.json`, `output/<class_name>/transcript_review.md`
- Primary validation: `python scripts/merge_transcripts.py --session-transcript output/<class>/transcripts/session.json --student-transcripts output/<class>/transcripts/ --identity-map output/<class>/identity_map.json --output output/<class>/transcript_merged.json`
- Complexity: High

## Goals

1. **Alignment Detection**: Detect whether per-student M4As are session-aligned (silence preserved) or join-offset (audio starts at student's join time)
2. **Speaker Attribution**: Tag each segment with the student(s) who spoke, using per-student M4A timestamps as ground truth
3. **Text Replacement**: Use per-student M4A transcription text as canonical; fall back to session transcript for gaps
4. **Multi-Speaker**: Handle simultaneous speech by tagging segments with all overlapping speakers

---

## Reasoning

### Why alignment detection?

**Current Problems:**

- Zoom cloud recording per-student M4As may start from session beginning (silence preserved) OR from when the student joined
- This is format-dependent and hasn't been tested with real exports yet
- If we assume wrong, every timestamp will be off

**Solution:**

- Auto-detect by comparing first speech in per-student M4A against session transcript
- If timestamps match directly → session-aligned
- If there's a consistent offset → join-offset mode with calculated offset

### Why per-student text as canonical?

**Current Problems:**

- Mixed session audio has crosstalk, background noise, and multiple speakers overlapping
- WhisperX on mixed audio may attribute words to the wrong speaker or garble overlapping speech

**Solution:**

- Per-student M4A captures ONLY that student's microphone audio — clean, isolated
- Transcription of isolated audio is inherently more accurate
- Use session transcript only for gaps where no per-student M4A covers the time range

---

## Files to Change

### Files to CREATE

1. `scripts/merge_transcripts.py` — **MAJOR** — Alignment detection, speaker attribution, text replacement, merged output
2. `tests/test_merge_transcripts.py` — **MAJOR** — Algorithm tests for all merge paths

### Files to MODIFY

1. `scripts/models/transcript.py` — **MINOR** — Add `MergedSegment`, `MergedTranscriptDocument`, `MergeMetadata` models

---

## Implementation Approach

### Step 1: Alignment Detection

For each per-student transcript, detect alignment mode:

```
def detect_alignment(student_transcript, session_transcript) -> AlignmentResult:
    # Find first non-trivial speech segment in student transcript
    first_student_words = student_transcript.segments[0].words[:5]

    # Search session transcript for matching words near the student's timestamps
    for session_seg in session_transcript.segments:
        similarity = word_sequence_similarity(first_student_words, session_seg.words)
        if similarity > 0.8:
            time_delta = session_seg.start - student_transcript.segments[0].start
            if abs(time_delta) < 2.0:
                return AlignmentResult(mode="session_aligned", offset=0.0)
            else:
                return AlignmentResult(mode="join_offset", offset=time_delta)

    # Fallback: assume session-aligned, flag for review
    return AlignmentResult(mode="session_aligned", offset=0.0, uncertain=True)
```

### Step 2: Build Time-Indexed Student Speech Map

After alignment correction, build a unified timeline of who spoke when:

```
speech_events: list[SpeechEvent] = []
for student_name, transcript in student_transcripts.items():
    offset = alignment_results[student_name].offset
    for segment in transcript.segments:
        speech_events.append(SpeechEvent(
            start=segment.start + offset,
            end=segment.end + offset,
            speaker=student_name,
            text=segment.text,
            words=segment.words,
            source="per_student",
        ))
```

### Step 3: Merge with Session Timeline

```
merged_segments = []
for time_window in iterate_time_windows(session_transcript, speech_events):
    overlapping_students = find_overlapping_speech(speech_events, time_window)

    if overlapping_students:
        # Per-student text is canonical
        merged_segments.append(MergedSegment(
            start=time_window.start,
            end=time_window.end,
            speakers=[s.speaker for s in overlapping_students],
            text=overlapping_students[0].text,  # primary speaker's text
            words=overlapping_students[0].words,
            source="per_student",
        ))
    else:
        # No per-student coverage — use session transcript
        session_seg = find_session_segment(session_transcript, time_window)
        merged_segments.append(MergedSegment(
            start=time_window.start,
            end=time_window.end,
            speakers=["UNKNOWN"],
            text=session_seg.text,
            words=session_seg.words,
            source="session_fallback",
        ))
```

### Step 4: Multi-Speaker Handling

When multiple per-student M4As show speech at the same timestamp:

```
overlapping = find_overlapping_speech(speech_events, window)
if len(overlapping) > 1:
    # Sort by overlap duration (longest first = primary speaker)
    overlapping.sort(key=lambda s: overlap_duration(s, window), reverse=True)
    segment.speakers = [s.speaker for s in overlapping]
    segment.text = overlapping[0].text  # primary speaker's text
```

### Review Artifact

`transcript_review.md` contains:
- Alignment mode detected per student (and offset if join-offset)
- Segments flagged as multi-speaker
- Segments using session_fallback (gaps)
- Segment count breakdown: per_student vs session_fallback

### Pydantic Models

```
AlignmentResult
  - mode: Literal["session_aligned", "join_offset"]
  - offset: float
  - uncertain: bool

MergedSegment
  - start: float
  - end: float
  - text: str
  - speakers: list[str]
  - source: Literal["per_student", "session_fallback"]
  - words: list[DualLanguageWord]
  - confidence: float

MergedTranscriptDocument
  - class_name: str
  - duration_seconds: float
  - segments: list[MergedSegment]
  - speakers: list[str]
  - teacher_name: str
  - alignment_results: dict[str, AlignmentResult]
  - metadata: MergeMetadata

MergeMetadata
  - total_segments: int
  - per_student_segments: int
  - session_fallback_segments: int
  - multi_speaker_segments: int
  - alignment_mode: str
  - merge_method: str
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] Detects session-aligned per-student M4As (timestamps match session)
- [ ] Detects join-offset per-student M4As (applies offset correction)
- [ ] Per-student text replaces session text where available
- [ ] Session text used as fallback for uncovered time ranges
- [ ] Multi-speaker segments tagged with all overlapping speakers
- [ ] Writes `transcript_merged.json` with valid `MergedTranscriptDocument` schema
- [ ] Writes `transcript_review.md` with alignment and merge statistics

### Code Quality

- [ ] argparse CLI with `--session-transcript`, `--student-transcripts`, `--identity-map`, `--output`
- [ ] Complete type hints
- [ ] No bare except
- [ ] `ruff` and `mypy` pass clean

---

## Testing Requirements

### Unit Tests

1. **Alignment Detection**
   - Session-aligned: student and session timestamps match within 2s → offset=0
   - Join-offset: student timestamps are 300s ahead of matching session content → offset=300
   - No match found → fallback to session-aligned with uncertain=True

2. **Speaker Attribution**
   - Single student speaking → speakers=[student_name]
   - Two students overlapping → speakers=[primary, secondary] (sorted by overlap duration)
   - No student M4A at this time → speakers=["UNKNOWN"], source="session_fallback"

3. **Text Replacement**
   - Per-student text used when available
   - Session text used for gaps
   - Empty per-student segment → skip, use session

4. **Segment Splitting**
   - Session segment spans two student speech events → split into two merged segments
   - Two student segments within one session segment → merge into one with multi-speaker

5. **Edge Cases**
   - Student with no speech (muted entire time) → no segments attributed, no crash
   - Session transcript has gaps (silence) → merged transcript has same gaps
   - Single student in class → all attributed to them, no multi-speaker

Target: ~25 tests.

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Alignment detection picks wrong offset | Log detected offset per student in review artifact; visual inspection |
| Word sequence matching across languages fails | Use timestamp-only matching as fallback; fuzzy text matching is secondary |
| Per-student M4A completely empty (student never spoke) | Skip gracefully; log in review |
| Session transcript has different segmentation than per-student | Match by timestamp overlap, not by segment boundaries |

---

## Related Tasks

- **TASK-013**: Provides `identity_map.json` (who each M4A belongs to)
- **TASK-014**: Provides session and per-student transcripts
- **TASK-016**: Consumes `transcript_merged.json` for context building

---

## Notes

- **Complexity:** High (highest in the pipeline)
- **Files Affected:** ~3 files (2 created, 1 modified)
- **No GPU needed** — pure algorithm, works on transcript JSON files
- **Critical path**: This task's output quality directly determines chatbot answer quality
