# TASK-003: WhisperX Transcription

## Overview

Run whisperX on the extracted WAV audio to produce a word-level timestamped transcript, using the `small` model to fit within the RTX 3050's 4GB VRAM constraint.

## Execution Snapshot

- Depends on: TASK-002
- Produces: `scripts/transcribe.py`, `output/transcript_raw.json`
- Primary validation: `python scripts/transcribe.py --input output/audio.wav`
- Complexity: Medium

## Goals

1. **Word-Level Timestamps**: Get per-word start/end times via whisperX's forced alignment — needed for accurate speaker assignment in TASK-005
2. **VRAM Safety**: Use `small` model with explicit VRAM management to avoid OOM on RTX 3050
3. **Quality Verification**: Output structured JSON that can be manually sampled for Indian English accuracy

---

## Reasoning

### Why whisperX over raw Whisper?

**Current Problems:**

- Raw Whisper only gives segment-level timestamps, not word-level
- Word-level timestamps are required for accurate merge with pyannote speaker segments
- Raw Whisper is slower — whisperX uses faster-whisper under the hood

**Solution:**

- whisperX runs faster-whisper for transcription, then runs forced alignment to get word-level timestamps
- Output schema has `word`, `start`, `end` per token — exactly what TASK-005 needs

### Why `small` model?

**Current Problems:**

- `large-v3` requires ~10GB VRAM — immediate OOM on RTX 3050 (4GB)
- `medium` is borderline (~5GB) and risky
- `small` fits comfortably in 4GB and still handles Indian English well

**Solution:**

- Hardcode `model_size = "small"` with a comment explaining the VRAM constraint
- If cloud GPU is available later, change to `large-v3` for better accuracy

---

## Files to Change

### Files to CREATE

1. `scripts/transcribe.py` — whisperX transcription pipeline

### Files to MODIFY

- None

---

## Implementation Approach

### `scripts/transcribe.py`

**Purpose:** Load whisperX, transcribe the WAV, run forced alignment, save word-level transcript JSON.

**Key Responsibilities:**

- Load whisperX `small` model on CUDA
- Transcribe `output/audio.wav`
- Run word-level alignment (whisperX `align` step)
- Save result to `output/transcript_raw.json`
- Free GPU memory explicitly after each step

**Output schema:**

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 4.5,
      "text": "So today we are going to cover...",
      "words": [
        { "word": "So", "start": 0.0, "end": 0.3 },
        { "word": "today", "start": 0.3, "end": 0.6 }
      ]
    }
  ],
  "language": "en",
  "model": "small"
}
```

**Integration Points:**

- Input: `output/audio.wav` (from TASK-002)
- Output: `output/transcript_raw.json` (consumed by TASK-005)

**Considerations:**

- whisperX alignment model is language-specific — detect language first or hardcode `"en"`
- Call `torch.cuda.empty_cache()` after transcription before alignment to avoid OOM
- `batch_size=8` is safe for 4GB VRAM with `small` model — reduce to 4 if OOM occurs
- Use `compute_type="float16"` for CUDA (half precision, fits VRAM better)

---

### Pseudocode

```
function transcribe(audio_path: str, output_path: str) -> None:
  device = "cuda"
  compute_type = "float16"
  model_size = "small"   # RTX 3050 4GB VRAM constraint

  # Step 1: Load model
  model = whisperx.load_model(model_size, device, compute_type=compute_type)

  # Step 2: Load audio
  audio = whisperx.load_audio(audio_path)

  # Step 3: Transcribe
  result = model.transcribe(audio, batch_size=8)
  print f"Detected language: {result['language']}"

  # Step 4: Free transcription model VRAM
  del model
  torch.cuda.empty_cache()

  # Step 5: Load alignment model
  model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
  )

  # Step 6: Align for word-level timestamps
  result = whisperx.align(
    result["segments"], model_a, metadata, audio, device,
    return_char_alignments=False
  )

  # Step 7: Free alignment model VRAM
  del model_a
  torch.cuda.empty_cache()

  # Step 8: Save output
  output = {
    "segments": result["segments"],
    "language": result["language"],
    "model": model_size
  }
  save_json(output, output_path)
  print f"✅ Transcript saved: {len(result['segments'])} segments → {output_path}"

function main() -> None:
  parse args: --input (default: output/audio.wav), --output (default: output/transcript_raw.json)
  transcribe(args.input, args.output)
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] Transcribes 1hr WAV without OOM error on RTX 3050
- [ ] Output JSON has `segments` with `words` array containing `start`, `end` per word
- [ ] Script prints detected language (should be `en`)
- [ ] VRAM is freed between transcription and alignment steps
- [ ] Manually verify 5-minute sample for Indian English accuracy

### Code Quality

- [ ] All functions have complete type hints
- [ ] No bare `except:` — catch `torch.cuda.OutOfMemoryError` with clear message to reduce `batch_size`
- [ ] `model_size = "small"` has inline comment explaining VRAM constraint

---

## Testing Requirements

### Manual Quality Check

After running, spot-check the output:

```python
import json
with open("output/transcript_raw.json") as f:
    data = json.load(f)

# Print first 5 segments
for seg in data["segments"][:5]:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")
```

Look for:

- Correct technical vocabulary (domain-specific terms)
- Indian accent phrases transcribed correctly
- No garbled text in silence/transition periods

---

## Risks and Mitigation

| Risk                                  | Mitigation                                     |
| ------------------------------------- | ---------------------------------------------- |
| OOM with batch_size=8                 | Catch OOM, retry with batch_size=4             |
| Alignment model fails for "en-IN"     | Force language_code="en"                       |
| Long silence segments bloating output | Normal — pyannote handles non-speech detection |
| Alignment takes too long              | Alignment is fast (~2-3min for 1hr on GPU)     |

---

## Related Tasks

- **TASK-002**: Must complete first — provides `output/audio.wav`
- **TASK-004**: Independent sibling task; on the local 4GB GPU execute it sequentially, not simultaneously
- **TASK-005**: Consumes `output/transcript_raw.json`

---

## Notes

- **Complexity:** Medium
- **Files Affected:** ~1 file
- **Expected runtime:** ~8-12 min for 1hr recording on RTX 3050 with `small` model
