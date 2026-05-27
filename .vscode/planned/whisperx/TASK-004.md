# TASK-004: Speaker Diarization (pyannote)

## Overview

Run pyannote speaker diarization on the extracted WAV to produce speaker-labeled time segments, mapping who spoke when without yet knowing their names.

## Execution Snapshot

- Depends on: TASK-002
- Produces: `scripts/diarize.py`, `output/diarization.json`
- Primary validation: `python scripts/diarize.py --input output/audio.wav`
- Complexity: Medium

## Goals

1. **Speaker Segmentation**: Produce time-stamped segments labeled `SPEAKER_00`, `SPEAKER_01` etc.
2. **Gated Model Access**: Handle HuggingFace authentication correctly for pyannote's gated models
3. **Local Execution Safety**: Keep diarization independent in plan structure, but run it sequentially on the local 4GB GPU to avoid VRAM contention

---

## Reasoning

### Why run this separately from TASK-003?

**Current Problems:**

- whisperX and pyannote both use GPU — running simultaneously would likely OOM on 4GB VRAM
- They produce independent outputs that are merged in TASK-005
- Keeping them separate makes debugging easier (isolate transcription bugs from diarization bugs)

**Solution:**

- Keep TASK-003 and TASK-004 as independent phase outputs, but execute them sequentially on the local 4GB machine
- Each script explicitly frees VRAM on exit

### Why pyannote 3.1?

**Current Problems:**

- Older pyannote versions have worse speaker separation
- pyannote 3.1 has significantly improved overlapping speech handling
- It's the version whisperX's diarization integration expects

**Solution:**

- Pin `pyannote.audio==3.1.1` in requirements.txt
- Use the recommended `Pipeline.from_pretrained()` loading pattern

---

## Files to Change

### Files to CREATE

1. `scripts/diarize.py` — pyannote diarization pipeline

---

## Implementation Approach

### `scripts/diarize.py`

**Purpose:** Load pyannote speaker diarization model, run on WAV, save speaker-timestamped segments to JSON.

**Key Responsibilities:**

- Load `.env` and read `HF_TOKEN`
- Load `pyannote/speaker-diarization-3.1` pipeline
- Run diarization on `output/audio.wav`
- Convert output to JSON-serializable format
- Save to `output/diarization.json`

**Output schema:**

```json
{
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "segments": [
    { "speaker": "SPEAKER_00", "start": 0.5, "end": 4.2 },
    { "speaker": "SPEAKER_01", "start": 4.8, "end": 9.1 }
  ]
}
```

**Integration Points:**

- Input: `output/audio.wav` (from TASK-002)
- HF token: `.env` → `HF_TOKEN`
- Output: `output/diarization.json` (consumed by TASK-005)

**Considerations:**

- `Pipeline.from_pretrained()` downloads model on first run (~500MB) — note in README
- Use `pipeline.to(torch.device("cuda"))` to run on GPU
- pyannote outputs an `Annotation` object — convert with `.itertracks(yield_label=True)`
- `min_speakers` / `max_speakers` hint improves accuracy — default to `max_speakers=6` for a class

---

### Pseudocode

```
function diarize(audio_path: str, output_path: str, hf_token: str) -> None:
  # Step 1: Load pipeline
  pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token
  )
  pipeline.to(torch.device("cuda"))

  # Step 2: Run diarization
  print "Running diarization (this takes a few minutes)..."
  diarization = pipeline(audio_path, max_speakers=6)

  # Step 3: Convert Annotation to list of dicts
  segments = []
  speakers = set()
  for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({
      "speaker": speaker,
      "start": round(turn.start, 3),
      "end": round(turn.end, 3)
    })
    speakers.add(speaker)

  # Step 4: Save output
  output = {
    "speakers": sorted(list(speakers)),
    "segments": segments
  }
  save_json(output, output_path)
  print f"✅ Diarization complete: {len(speakers)} speakers, {len(segments)} segments → {output_path}"

function main() -> None:
  load_dotenv()
  hf_token = os.getenv("HF_TOKEN")
  if not hf_token:
    raise EnvironmentError("HF_TOKEN not set in .env")

  parse args: --input (default: output/audio.wav), --output (default: output/diarization.json)
  diarize(args.input, args.output, hf_token)
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] Diarization runs without 403 HuggingFace error
- [ ] Output JSON has `speakers` list and `segments` array with `speaker`, `start`, `end`
- [ ] Correctly identifies 2-5 distinct speakers in test recording
- [ ] Script handles case where HF_TOKEN is missing with clear error message

### Code Quality

- [ ] All functions have complete type hints
- [ ] No bare `except:` — catch `huggingface_hub.utils.RepositoryNotFoundError` (403) separately
- [ ] HF token read from `.env` via `python-dotenv`, never hardcoded

---

## Testing Requirements

### Manual Quality Check

After running, verify speaker count makes sense:

```python
import json
with open("output/diarization.json") as f:
    data = json.load(f)

print(f"Speakers found: {data['speakers']}")
print(f"Total segments: {len(data['segments'])}")

# Print first 10 segments
for seg in data["segments"][:10]:
    duration = seg["end"] - seg["start"]
    print(f"[{seg['start']:.1f}s-{seg['end']:.1f}s] ({duration:.1f}s) {seg['speaker']}")
```

Expected: 2-5 speakers for a class recording, reasonable segment durations (2-30s typical).

---

## Risks and Mitigation

| Risk                            | Mitigation                                        |
| ------------------------------- | ------------------------------------------------- |
| 403 on model load               | validate_env.py catches this — run TASK-001 first |
| First run downloads 500MB model | Note in README, takes ~2min on good connection    |
| Too many speakers detected      | Add `max_speakers=6` hint to pipeline call        |
| VRAM conflict with TASK-003     | Run sequentially, not simultaneously              |

---

## Related Tasks

- **TASK-002**: Must complete first — provides `output/audio.wav`
- **TASK-003**: Independent sibling task; on the local 4GB GPU run this after or before transcription, but not at the same time
- **TASK-005**: Consumes `output/diarization.json`

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~1 file
- **Expected runtime:** ~5-8 min for 1hr recording on RTX 3050
- **First run:** Slower due to model download (~500MB from HuggingFace)
