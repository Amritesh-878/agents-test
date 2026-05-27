# TASK-016: Student Context Builder (Roster, Attendance, Present/Missed/Absent, Topics)

## Overview

Build per-student context objects for every enrolled student from the master roster. Present students get full context (spoken/present/missed segments). Absent students get a summary with auto-extracted topics. Implement TF-IDF keyword extraction for topic discovery.

## Execution Snapshot

- Depends on: TASK-013, TASK-015
- Produces: `scripts/build_student_context.py`, `scripts/utils/topics.py`, `output/<class_name>/student_contexts.json`, review artifacts
- Primary validation: `python scripts/build_student_context.py --transcript output/<class>/transcript_merged.json --identity-map output/<class>/identity_map.json --roster roster.csv --attendance attendance.csv --output output/<class>/student_contexts.json`
- Complexity: Medium

## Goals

1. **Roster-Driven**: Every enrolled student (from master CSV) gets a context object, even if absent
2. **Present Students**: Attendance window, spoken segments, present segments, missed segments
3. **Absent Students**: Summary only — class date, duration, teacher, auto-extracted topics
4. **Topic Extraction**: TF-IDF keyword-based, local only, no LLM/API calls
5. **Unmatched Students**: Include M4A-only students (not in roster) with 'unmatched' tag

---

## Reasoning

### Why roster-driven instead of attendance-driven?

**Current Problems:**

- The old pipeline only created context for students in the attendance CSV
- Students who were absent got nothing — no chatbot at all
- The core use case is "what did I miss?" — which is MOST relevant for absent students

**Solution:**

- Master roster is the source of truth for all enrolled students
- Every student gets a chatbot, even if they missed the entire class
- Absent students get a summary with topics so their chatbot can answer "what was covered?"

### Why TF-IDF instead of LLM for topics?

**Current Problems:**

- LLM-based topic extraction requires API calls (Groq), adding latency and cost
- API failures would block the entire pipeline
- Overkill for a keyword list

**Solution:**

- `sklearn.feature_extraction.text.TfidfVectorizer` is already available transitively via sentence-transformers
- Combined English + Hindi stopword list for Hinglish content
- Extract top 10 keywords/bigrams from the full transcript
- No API calls, no failure modes beyond bad input text

---

## Files to Change

### Files to CREATE

1. `scripts/build_student_context.py` — **MAJOR** — Context builder CLI script
2. `scripts/utils/topics.py` — **MINOR** — TF-IDF topic extraction utility
3. `tests/test_build_student_context.py` — **MAJOR** — Context building tests
4. `tests/test_topics.py` — **MINOR** — Topic extraction tests

### Files to MODIFY

1. `scripts/models/identity.py` — **MINOR** — Add context-related models if needed

---

## Implementation Approach

### Roster Processing

```
roster_entries = load_roster_csv(roster_path)  # Name, RollNo, Email
identity_map = load_identity_map(identity_map_path)
merged_transcript = load_merged_transcript(transcript_path)
attendance = load_attendance_csv(attendance_path)

# Build lookup: roll_no -> identity_map_entry
identity_lookup = {e.matched_roll_no: e for e in identity_map.entries if e.matched_roll_no}

for student in roster_entries:
    identity = identity_lookup.get(student.roll_no)
    if identity and not identity.is_unmatched:
        # Student was present — build full context
        context = build_present_context(student, identity, merged_transcript, attendance)
    else:
        # Student was absent — build summary
        context = build_absent_summary(student, merged_transcript, topics)
```

### Present Student Context

```
def build_present_context(student, identity, transcript, attendance):
    att_record = find_attendance(attendance, student.roll_no)
    att_window = (0.0, transcript.duration_seconds)  # default: full class
    if att_record:
        att_window = (0.0, att_record.duration_minutes * 60)  # estimated

    spoken_segments = [seg for seg in transcript.segments
                       if student.name in seg.speakers
                       or identity.matched_name in seg.speakers]

    present_segments = [seg for seg in transcript.segments
                        if seg.end > att_window[0] and seg.start < att_window[1]]

    missed_segments = [seg for seg in transcript.segments
                       if seg.end <= att_window[0] or seg.start >= att_window[1]]

    return StudentContext(
        name=student.name,
        roll_no=student.roll_no,
        email=student.email,
        status="present",
        attendance_duration_minutes=att_record.duration_minutes if att_record else None,
        spoken_segments=spoken_segments,
        present_segments=present_segments,
        missed_segments=missed_segments,
        topics_discussed=topics,
        ...
    )
```

### Absent Student Summary

```
def build_absent_summary(student, transcript, topics):
    return AbsentStudentSummary(
        name=student.name,
        roll_no=student.roll_no,
        email=student.email,
        class_date=extract_class_date(transcript),
        class_duration_seconds=transcript.duration_seconds,
        teacher_name=transcript.teacher_name,
        topics_discussed=topics,
    )
```

### Topic Extraction (`scripts/utils/topics.py`)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

HINDI_STOPWORDS = {"ka", "ki", "ke", "hai", "hain", "mein", "se", "ko", "ne",
                   "yeh", "woh", "aur", "par", "toh", "kya", "kaise", "kab",
                   "jab", "tab", "lekin", "agar", "ya", "bhi", ...}

def extract_topics(transcript_text: str, top_n: int = 10) -> list[str]:
    stopwords = list(HINDI_STOPWORDS | set(sklearn_english_stopwords))
    vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),  # unigrams + bigrams
        stop_words=stopwords,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform([transcript_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    top_indices = scores.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices if scores[i] > 0]
```

### Pydantic Models

```
ContextSegment
  - start: float
  - end: float
  - text: str
  - speakers: list[str]
  - source: str

StudentContext
  - name: str
  - roll_no: str | None
  - email: str | None
  - status: Literal["present", "absent"]
  - attendance_duration_minutes: float | None
  - spoken_segments: list[ContextSegment]
  - present_segments: list[ContextSegment]
  - missed_segments: list[ContextSegment]
  - topics_discussed: list[str]
  - class_date: str | None
  - class_duration_seconds: float
  - teacher_name: str
  - is_teacher: bool
  - tags: list[str]

AbsentStudentSummary
  - name: str
  - roll_no: str | None
  - email: str | None
  - class_date: str | None
  - class_duration_seconds: float
  - teacher_name: str
  - topics_discussed: list[str]

StudentContextDocument
  - class_name: str
  - present_students: dict[str, StudentContext]
  - absent_students: dict[str, AbsentStudentSummary]
  - all_enrolled: list[str]
  - topics: list[str]
  - metadata: BuildContextMetadata
```

### Review Artifacts

- `student_context_review.md`: Per-student summary table (name, status, spoken count, missed count, tags)
- `student_context_segments.csv`: Flat CSV with one row per segment per student

---

## Acceptance Criteria

### Functional Requirements

- [ ] Every roster student gets a context object (present OR absent)
- [ ] Present students have spoken, present, and missed segment lists
- [ ] Absent students have summary with topics (no full transcript segments)
- [ ] TF-IDF extracts reasonable topics from Hinglish text
- [ ] Unmatched M4A students included with 'unmatched' tag
- [ ] Review artifacts generated (.md and .csv)
- [ ] Writes valid `student_contexts.json`

### Code Quality

- [ ] argparse CLI with `--transcript`, `--identity-map`, `--roster`, `--attendance`, `--output`
- [ ] Complete type hints
- [ ] No bare except
- [ ] `ruff` and `mypy` pass clean

---

## Testing Requirements

### Unit Tests

1. **Present Student Context**
   - Student with full attendance → empty missed_segments
   - Student who joined late → missed_segments before join
   - Student who left early → missed_segments after leave
   - Student with no speech → empty spoken_segments, non-empty present_segments

2. **Absent Student Summary**
   - Absent student gets topics, date, duration, teacher
   - No transcript segments in absent summary

3. **Topic Extraction**
   - English text → English keywords
   - Hindi text → Hindi keywords (stopwords filtered)
   - Mixed Hinglish → keywords from both languages
   - Empty text → empty topic list

4. **Edge Cases**
   - Roster with 1 student → works
   - All students absent → all get summaries
   - Teacher appears in roster → handled (is_teacher=True)
   - Unmatched M4A student → included with tag

Target: ~20 tests (16 context + 4 topics).

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| TF-IDF topics are low quality for Hinglish | Combined stopword list; bigram support; flag as "auto-extracted" |
| Attendance duration doesn't match session duration | Use min(attendance_duration, session_duration) as window |
| scikit-learn not available | Transitively installed via sentence-transformers; add explicit dep if needed |

---

## Related Tasks

- **TASK-013**: Provides `identity_map.json`
- **TASK-015**: Provides `transcript_merged.json`
- **TASK-017**: Consumes `student_contexts.json` for chunking + embedding

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~5 files (4 created, 1 modified)
- **No GPU needed** — data processing + TF-IDF (CPU only)
