# TASK-014: Dual-Language WhisperX Transcription

## Overview

Transcribe the full-session audio (extracted from MP4) and each per-student M4A using dual-language WhisperX (Hindi + English), merging results by word-level probability to handle Hinglish code-switching.

## Execution Snapshot

- Depends on: TASK-011, TASK-012
- Produces: `scripts/transcribe_dual.py`, `output/<class_name>/transcripts/session.json`, `output/<class_name>/transcripts/<audio_file>.json`
- Primary validation: `python scripts/transcribe_dual.py --manifest output/<class>/manifest.json --output-dir output/<class>/transcripts/`
- Complexity: High

## Goals

1. **Dual-Language Transcription**: Run WhisperX twice per audio file (language='hi' and language='en')
2. **Word-Level Merge**: For each word position, pick the language run with higher alignment probability
3. **M4A Support**: Extend audio extraction to handle M4A → WAV conversion
4. **VRAM Safety**: Process one audio file at a time with GPU cleanup between runs

---

## Reasoning

### Why dual-language?

**Current Problems:**

- Classes are in Hinglish (Hindi + English code-switching)
- Single-language WhisperX set to Hindi mangles English words and vice versa
- Auto-detect picks one language per segment, missing the other

**Solution:**

- Run WhisperX twice: once with `language='hi'`, once with `language='en'`
- Both runs produce word-level aligned output with probability scores
- For each word, pick the transcription from the run with higher confidence
- This captures Hindi words from the Hindi run and English words from the English run

### Why word-level merge (not segment-level)?

**Current Problems:**

- Hinglish code-switching happens mid-sentence: "Yeh function ka return type **string** hai"
- Segment-level merge would pick one language for the whole sentence, losing the other

**Solution:**

- Word-level comparison: each word independently evaluated
- Hindi run might give high-confidence "yeh" and low-confidence "string" (romanized to Hindi script)
- English run might give low-confidence "yeh" (mangled) and high-confidence "string"
- Pick the best of each → correct Hinglish transcription

---

## Files to Change

### Files to CREATE

1. `scripts/transcribe_dual.py` — **MAJOR** — Dual-language transcription + merge CLI script
2. `tests/test_transcribe_dual.py` — **MAJOR** — Tests for merge algorithm

### Files to MODIFY

1. `scripts/extract_audio.py` — **MINOR** — Accept M4A input in addition to MP4 (loosen input validation)

---

## Implementation Approach

### Audio Extraction

For MP4: existing ffmpeg extraction to 16kHz mono WAV (unchanged).
For M4A: same ffmpeg command, just different input extension. Loosen validation in `extract_audio.py` to accept `.m4a` alongside `.mp4`.

### Dual-Language Transcription

```
for each audio_file in [session_wav] + per_student_m4as:
  1. Convert to 16kHz mono WAV if M4A
  2. Run WhisperX(language='hi', model='small', compute_type='float16')
     → hi_result: TranscriptDocument with word-level scores
  3. Release GPU memory
  4. Run WhisperX(language='en', model='small', compute_type='float16')
     → en_result: TranscriptDocument with word-level scores
  5. Release GPU memory
  6. merged = merge_by_word_probability(hi_result, en_result)
  7. Write merged transcript JSON
```

### Word-Level Merge Algorithm

This is the core algorithm. Both runs produce word-aligned output where each word has `(start, end, word, score)`.

```
def merge_by_word_probability(hi_words, en_words) -> list[DualLanguageWord]:
    # Build time-indexed lookup for both runs
    # For each time window where words exist:
    #   - Find overlapping words from both runs
    #   - Pick the word with higher score
    #   - Tag with source_language

    result = []
    hi_idx, en_idx = 0, 0

    while hi_idx < len(hi_words) or en_idx < len(en_words):
        hi_word = hi_words[hi_idx] if hi_idx < len(hi_words) else None
        en_word = en_words[en_idx] if en_idx < len(en_words) else None

        if both overlap in time:
            # Pick higher score
            if hi_word.score >= en_word.score:
                result.append(DualLanguageWord(..., source_language="hi"))
            else:
                result.append(DualLanguageWord(..., source_language="en"))
            advance both
        elif only hi_word exists at this time:
            result.append(DualLanguageWord(..., source_language="hi"))
            advance hi_idx
        elif only en_word exists at this time:
            result.append(DualLanguageWord(..., source_language="en"))
            advance en_idx

    return result
```

Overlap detection: two words overlap if `max(start_a, start_b) < min(end_a, end_b)`.

### Re-segmentation

After word-level merge, re-group words into segments:
- Group consecutive words with gaps < 1.5 seconds
- Each segment gets `text = " ".join(word.word for word in segment_words)`

### Pydantic Models

```
DualLanguageWord (from TASK-011)
  - start, end, word, score, source_language

PerStudentTranscript
  - audio_file: str
  - student_name: str | None
  - roll_no: str | None
  - is_teacher: bool
  - transcript: TranscriptDocument  # with DualLanguageWord entries
  - hi_avg_score: float
  - en_avg_score: float
  - dominant_language: str
```

### VRAM Management

Reuse existing pattern from `scripts/transcribe.py`:
```python
def release_gpu_resources() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
```

Call between every WhisperX run. On RTX 3050 4GB, WhisperX small model uses ~1.5GB VRAM. Two sequential runs fit comfortably.

### `--single-language` Fallback

Add a `--single-language hi` or `--single-language en` flag that skips dual transcription and runs only one language. For debugging or when dual isn't needed.

---

## Acceptance Criteria

### Functional Requirements

- [ ] Transcribes session WAV with dual-language merge
- [ ] Transcribes each per-student M4A with dual-language merge
- [ ] Word-level merge picks higher-probability word correctly
- [ ] Handles words only present in one language run (no crash)
- [ ] `--single-language` flag works as fallback
- [ ] GPU memory released between runs (no OOM on RTX 3050 4GB)
- [ ] Writes transcript JSON per audio file to `output/<class>/transcripts/`

### Code Quality

- [ ] argparse CLI with `--manifest`, `--output-dir`, `--single-language`
- [ ] Complete type hints
- [ ] No bare except
- [ ] `ruff` and `mypy` pass clean

---

## Testing Requirements

### Unit Tests (mock WhisperX — no GPU needed for tests)

1. **Word-Level Merge**
   - Both runs have words at same timestamps → higher score wins
   - Hindi run has word, English doesn't → Hindi word kept
   - English run has word, Hindi doesn't → English word kept
   - No overlap at all → concatenate in time order

2. **Re-segmentation**
   - Words within 1.5s gap → same segment
   - Words with >1.5s gap → split into new segment
   - Single word → single segment

3. **Score Comparison**
   - Equal scores → Hindi preferred (primary language)
   - Score of None → treated as 0.0
   - Negative scores → handled gracefully

4. **M4A Extraction**
   - M4A input accepted by extract_audio
   - Invalid M4A → clear error message

5. **Single-Language Mode**
   - `--single-language hi` → only Hindi run, no merge

Target: ~20 tests.

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Dual-language merge worse than single | `--single-language` flag; compare on 5-min clip before full run |
| WhisperX word alignment missing scores | Treat None score as 0.0; log warning |
| OOM on long audio files (>1hr) | Process one file at a time; GPU cleanup; WhisperX small model |
| Word alignment timing differs between hi/en runs | Use overlap detection with tolerance (±0.1s); log misaligned words |

---

## Related Tasks

- **TASK-012**: Provides `manifest.json` with audio file paths
- **TASK-015**: Consumes per-student and session transcripts for merge
- **TASK-011**: Provides `DualLanguageWord` Pydantic model

---

## Notes

- **Complexity:** High
- **Files Affected:** ~3 files (2 created, 1 modified)
- **GPU required** for runtime but NOT for tests (mock WhisperX)
- **Runtime estimate**: ~2x current transcription time due to dual runs. A 1-hour class with 15 students = ~30 audio files × 2 runs = ~60 WhisperX runs. On RTX 3050, expect ~2-4 hours total per class.
