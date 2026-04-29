# TASK-005: Transcript + Diarization Merge

Archived on 2026-04-30 after implementation in worktree `task-005-merge`.
See `MASTER_PLAN.md` for the completion handoff and validation notes.

## Overview

Merge the word-level whisperX transcript with pyannote's speaker segments using timestamp overlap, producing a single diarized transcript where each text segment has a speaker label.

## Execution Snapshot

- Depends on: TASK-003, TASK-004
- Produces: `scripts/merge.py`, `output/transcript_diarized.json`
- Primary validation: `python scripts/merge.py --transcript output/transcript_raw.json --diarization output/diarization.json`
- Complexity: Medium

## Goals

1. **Speaker-Labeled Transcript**: Assign each transcript segment a speaker label by matching timestamps
2. **Clean Output Format**: Produce a human-readable and machine-parseable JSON with `[timestamp] SPEAKER: text` structure
3. **Overlap Handling**: Handle edge cases where transcript segments span speaker transitions

---

## Reasoning

### Why timestamp-based merge (not whisperX's built-in diarization)?

**Current Problems:**

- whisperX has a built-in diarization integration but it requires an additional HF token scope and sometimes errors on Windows
- Running them separately gives us cleaner outputs to debug independently
- Manual merge gives full control over the matching algorithm

**Solution:**

- For each transcript segment, find which pyannote speaker segment has the most overlap with its time range
- "Majority speaker" assignment — the speaker whose time overlaps most with the segment wins

### Why segment-level (not word-level) speaker assignment?

**Current Problems:**

- Word-level speaker assignment is very granular and rarely changes mid-sentence in practice
- Segment-level is sufficient for the RAG use case — we want "what did this speaker say" not "which word came from which speaker"

**Solution:**

- Assign one speaker per whisperX segment using the majority overlap method
- Keep word-level timestamps in output for future use (they're already computed)

---

## Files to Change

### Files to CREATE

1. `scripts/merge.py` — timestamp-based merge logic

---

## Implementation Approach

### `scripts/merge.py`

**Purpose:** Load both JSON outputs, assign speaker labels to transcript segments, save diarized transcript.

**Key Responsibilities:**

- Load `output/transcript_raw.json` and `output/diarization.json`
- For each transcript segment, find majority speaker from diarization segments
- Produce merged output with speaker label on each segment
- Save to `output/transcript_diarized.json`

**Merge algorithm — majority overlap:**

```
for each transcript_segment (start_t, end_t, text):
  overlaps = {}
  for each diarization_segment (start_d, end_d, speaker):
    overlap = min(end_t, end_d) - max(start_t, start_d)
    if overlap > 0:
      overlaps[speaker] = overlaps.get(speaker, 0) + overlap

  if overlaps is empty:
    assigned_speaker = "UNKNOWN"
  else:
    assigned_speaker = speaker with max overlap
```

**Output schema:**

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 4.2,
      "speaker": "SPEAKER_00",
      "text": "So today we are going to cover gradient descent.",
      "words": [...]
    }
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "metadata": {
    "total_segments": 312,
    "unknown_segments": 3,
    "merge_method": "majority_overlap"
  }
}
```

**Integration Points:**

- Input: `output/transcript_raw.json` (TASK-003) + `output/diarization.json` (TASK-004)
- Output: `output/transcript_diarized.json` (consumed by TASK-006)

---

## Acceptance Criteria

### Functional Requirements

- [ ] Every transcript segment has a speaker label (or "UNKNOWN")
- [ ] No transcript segments are lost in the merge
- [ ] `UNKNOWN` segments are < 5% of total (indicates good diarization quality)
- [ ] Output is readable as `[start-end] SPEAKER: text` for manual review

### Code Quality

- [ ] All functions have complete type hints
- [ ] `get_majority_speaker` is a pure function with no side effects (easy to test)
- [ ] `UNKNOWN` segments logged as warnings, not errors

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~1 file
- **No GPU needed** — pure Python data processing, runs on CPU instantly
