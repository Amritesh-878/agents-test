# TASK-006: Per-Student Context Builder

## Overview

Map pyannote's anonymous speaker labels to real student names using Zoom attendance data, then compute each student's spoken segments, missed segments, and a structured context object ready for RAG ingestion.

## Execution Snapshot

- Depends on: TASK-005
- Produces: `scripts/build_context.py`, `data/sample_attendance.csv`, `output/student_contexts.json`
- Primary validation: `python scripts/build_context.py --transcript output/transcript_diarized.json --attendance data/sample_attendance.csv`
- Complexity: Medium

## Goals

1. **Speaker-to-Student Mapping**: Link `SPEAKER_00` etc. to real names using Zoom attendance CSV join timestamps
2. **Per-Student Context Object**: For each student, produce: attendance window, what they said, what they missed
3. **RAG-Ready Output**: Output schema designed for direct ingestion into the future vector store pipeline

---

## Reasoning

### Why Zoom attendance CSV for speaker mapping?

**Current Problems:**

- pyannote doesn't know student names — only produces anonymous speaker IDs
- Voice enrollment (storing speaker embeddings at enrolment) is more robust but requires upfront work
- Attendance CSV is already available from Zoom and has accurate join/leave timestamps

**Solution:**

- "First speaker segment after student joins" heuristic — when student X joins at time T, the first active speaker segment after T is likely student X
- This is a soft heuristic, good enough for Phase 1 testing
- Note mapping in output so it can be manually corrected

### Why compute "missed segments"?

**Current Problems:**

- The core use case of the chatbot is "what did I miss?"
- Students who join late or leave early need exactly this
- Without this, the chatbot can't personalize to what the student actually needs

**Solution:**

- Missed segments = transcript segments that fall entirely outside the student's attendance window
- Simple timestamp comparison, no ML needed

---

## Files to Change

### Files to CREATE

1. `scripts/build_context.py` — speaker mapping + context builder
2. `data/sample_attendance.csv` — sample Zoom attendance format for testing

---

## Implementation Approach

### `data/sample_attendance.csv`

**Purpose:** Sample attendance file matching Zoom's export format for testing.

**Zoom attendance CSV format:**

```csv
Name (Original Name),User Email,Join Time,Leave Time,Duration (Minutes)
Amritesh Kumar,amritesh@isl.org,2024-01-15 10:02:34,2024-01-15 11:05:12,63
Priya Sharma,priya@isl.org,2024-01-15 10:00:01,2024-01-15 11:05:12,65
Rahul Gupta,rahul@isl.org,2024-01-15 10:15:22,2024-01-15 10:58:44,43
```

**Note:** Join/Leave times are in the meeting host's timezone — must be converted to seconds-from-meeting-start for timestamp comparison.

---

### `scripts/build_context.py`

**Purpose:** Map speakers to students and build per-student context objects.

**Key Responsibilities:**

- Parse Zoom attendance CSV
- Convert join/leave wall-clock times to seconds from meeting start
- Map speaker IDs to student names using first-speaker heuristic
- For each student: extract their spoken segments, compute missed segments
- Save to `output/student_contexts.json`

**Output schema:**

```json
{
  "meeting_duration_seconds": 3780.0,
  "speaker_mapping": {
    "SPEAKER_00": "Priya Sharma",
    "SPEAKER_01": "Amritesh Kumar",
    "SPEAKER_02": "Rahul Gupta"
  },
  "students": {
    "Priya Sharma": {
      "email": "priya@isl.org",
      "attendance": {
        "joined_at": 1.0,
        "left_at": 3780.0,
        "duration_minutes": 63
      },
      "spoken_segments": [
        {"start": 45.2, "end": 52.1, "text": "Can you explain that again?"}
      ],
      "missed_segments": [],
      "was_present_full_class": true
    },
    "Rahul Gupta": {
      "email": "rahul@isl.org",
      "attendance": {
        "joined_at": 921.0,
        "left_at": 3524.0,
        "duration_minutes": 43
      },
      "spoken_segments": [...],
      "missed_segments": [
        {"start": 0.0, "end": 921.0, "text": "Introduction and context setting..."},
        {"start": 3524.0, "end": 3780.0, "text": "Summary and Q&A wrap-up..."}
      ],
      "was_present_full_class": false
    }
  }
}
```

**Speaker mapping heuristic:**

```
meeting_start = earliest join time in attendance CSV

for each student (sorted by join time):
  join_offset = (student.join_time - meeting_start).total_seconds()

  first_segment_after_join = first diarization segment where start >= join_offset
  if first_segment_after_join not yet claimed:
    speaker_mapping[first_segment_after_join.speaker] = student.name
```

**Missed segments computation:**

```
for each student:
  attendance_window = (join_offset, leave_offset)

  missed = []
  for seg in transcript["segments"]:
    if seg["end"] <= join_offset or seg["start"] >= leave_offset:
      missed.append(seg)

  student_context["missed_segments"] = missed
```

**Spoken segments:**

```
for each student:
  speaker_id = reverse_lookup(speaker_mapping, student.name)
  spoken = [seg for seg in transcript["segments"] if seg["speaker"] == speaker_id]
  student_context["spoken_segments"] = spoken
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] Each student in attendance CSV has a context object in output
- [ ] Speaker mapping covers all named students (UNKNOWN may remain for unmapped speakers)
- [ ] Missed segments correctly computed for late joiners / early leavers
- [ ] Student with full attendance has empty `missed_segments`
- [ ] Output JSON schema matches documented format

### Code Quality

- [ ] All functions have complete type hints
- [ ] `datetime` parsing uses `dateutil.parser.parse` not manual strptime (handles Zoom format variations)
- [ ] Speaker mapping stored in output JSON for manual review and correction

---

## Testing Requirements

### Manual Validation

1. Verify a student who joined late has missed segments from before their join time
2. Verify the teacher (earliest join, longest duration) maps to the most frequent speaker
3. Check that `spoken_segments` for a known quiet student has few entries

### Unit Tests

```python
def test_compute_missed_segments():
  segments = [
    {"start": 0.0, "end": 5.0, "text": "Intro"},
    {"start": 5.0, "end": 10.0, "text": "Topic 1"},
    {"start": 900.0, "end": 910.0, "text": "Wrap up"}
  ]
  # Student joined at 6.0, left at 895.0
  missed = compute_missed_segments(segments, join_at=6.0, leave_at=895.0)
  assert len(missed) == 2  # "Intro" and "Wrap up" both missed
  assert missed[0]["text"] == "Intro"
  assert missed[1]["text"] == "Wrap up"
```

---

## Risks and Mitigation

| Risk                                            | Mitigation                                                       |
| ----------------------------------------------- | ---------------------------------------------------------------- |
| Speaker mapping heuristic wrong                 | Mapping stored in output — can be manually corrected             |
| Zoom attendance timezone issues                 | Parse with `dateutil`, use UTC offset from CSV header if present |
| More speakers than students (background noise)  | UNKNOWN speakers kept in transcript, not mapped                  |
| Student rejoins (multiple rows for same person) | Group by email, use earliest join / latest leave                 |

---

## Related Tasks

- **TASK-005**: Must complete first — provides `output/transcript_diarized.json`
- **Phase 2 (future)**: Output feeds into RAG chunking and embedding pipeline

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~2 files
- **No GPU needed** — pure Python data processing
- **Phase 2 note:** The `student_contexts.json` output is designed to plug directly into the RAG ingestion layer. Each student's `missed_segments` and `spoken_segments` become the primary context chunks for their personalized chatbot.
