# TASK-005: Transcript + Diarization Merge

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

### Pseudocode

```
function get_majority_speaker(
  seg_start: float,
  seg_end: float,
  diar_segments: list[dict]
) -> str:
  overlaps: dict[str, float] = {}

  for diar_seg in diar_segments:
    overlap = min(seg_end, diar_seg["end"]) - max(seg_start, diar_seg["start"])
    if overlap > 0:
      overlaps[diar_seg["speaker"]] = overlaps.get(diar_seg["speaker"], 0.0) + overlap

  if not overlaps:
    return "UNKNOWN"
  return max(overlaps, key=overlaps.get)

function merge(transcript_path: str, diarization_path: str, output_path: str) -> None:
  transcript = load_json(transcript_path)
  diarization = load_json(diarization_path)

  merged_segments = []
  unknown_count = 0

  for seg in transcript["segments"]:
    speaker = get_majority_speaker(
      seg["start"], seg["end"], diarization["segments"]
    )
    if speaker == "UNKNOWN":
      unknown_count += 1

    merged_segments.append({
      "start": seg["start"],
      "end": seg["end"],
      "speaker": speaker,
      "text": seg["text"].strip(),
      "words": seg.get("words", [])
    })

  output = {
    "segments": merged_segments,
    "speakers": diarization["speakers"],
    "metadata": {
      "total_segments": len(merged_segments),
      "unknown_segments": unknown_count,
      "merge_method": "majority_overlap"
    }
  }

  save_json(output, output_path)
  print f"✅ Merged: {len(merged_segments)} segments, {unknown_count} UNKNOWN → {output_path}"

function main() -> None:
  parse args: --transcript, --diarization, --output
  merge(args.transcript, args.diarization, args.output)
```

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

## Testing Requirements

### Unit Test

```python
# Test majority overlap logic directly
def test_get_majority_speaker():
  diar_segments = [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
    {"speaker": "SPEAKER_01", "start": 3.0, "end": 6.0}
  ]
  # Segment mostly in SPEAKER_00's range
  assert get_majority_speaker(0.5, 3.5, diar_segments) == "SPEAKER_00"

  # Segment fully in SPEAKER_01's range
  assert get_majority_speaker(3.5, 5.5, diar_segments) == "SPEAKER_01"

  # Segment outside all ranges
  assert get_majority_speaker(10.0, 12.0, diar_segments) == "UNKNOWN"
```

### Manual Quality Check

```python
import json
with open("output/transcript_diarized.json") as f:
    data = json.load(f)

for seg in data["segments"][:20]:
    print(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['speaker']}: {seg['text']}")
```

Verify speaker changes happen at natural conversation boundaries.

---

## Risks and Mitigation

| Risk                                | Mitigation                                                                            |
| ----------------------------------- | ------------------------------------------------------------------------------------- |
| High UNKNOWN count (>10%)           | Indicates diarization gaps — check if silence detection is too aggressive in pyannote |
| Speaker label flipping mid-sentence | Segment-level assignment prevents this by design                                      |
| Timestamp precision mismatch        | Both systems use float seconds — no conversion needed                                 |

---

## Related Tasks

- **TASK-003**: Must complete first — provides transcript
- **TASK-004**: Must complete first — provides diarization
- **TASK-006**: Consumes `output/transcript_diarized.json`

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~1 file
- **No GPU needed** — pure Python data processing, runs on CPU instantly
