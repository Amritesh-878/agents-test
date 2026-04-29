# TASK-002: Audio Extraction via ffmpeg

## Overview

Extract clean 16kHz mono WAV audio from the Zoom MP4 recording using ffmpeg, producing the standard input format required by both whisperX and pyannote.

## Execution Snapshot

- Depends on: TASK-001
- Produces: `scripts/extract_audio.py`, `output/audio.wav`
- Primary validation: `python scripts/extract_audio.py --input <zoom-recording.mp4>`
- Complexity: Low

## Goals

1. **Format Compliance**: Output 16kHz mono WAV — the exact format both ML models expect
2. **Robustness**: Handle long recordings (1hr+) without memory issues via streaming
3. **Reusability**: Script accepts any MP4 path as CLI argument so it works for future recordings

---

## Reasoning

### Why 16kHz mono?

**Current Problems:**

- Zoom records at 44.1kHz stereo — unnecessary data for speech models
- Both whisperX and pyannote expect 16kHz — mismatches cause silent quality degradation
- Large audio files slow down processing on 4GB VRAM machine

**Solution:**

- ffmpeg resamples to 16kHz mono in a single pass
- Output size is ~10x smaller than original, no quality loss for speech recognition

---

## Files to Change

### Files to CREATE

1. `scripts/extract_audio.py` — ffmpeg subprocess wrapper with CLI args
2. `output/` — directory (gitignored, README documents it)

### Files to MODIFY

- `README.md` — add usage section for extract_audio.py

---

## Implementation Approach

### `scripts/extract_audio.py`

**Purpose:** Wrap ffmpeg to extract and resample audio from Zoom MP4 recordings.

**Key Responsibilities:**

- Accept `--input` (MP4 path) and `--output` (WAV path) as CLI arguments
- Run ffmpeg as subprocess with correct flags for 16kHz mono WAV
- Validate output file exists and has non-zero size
- Print duration of extracted audio using ffprobe

**ffmpeg command:**

```
ffmpeg -i input.mp4 -ar 16000 -ac 1 -vn output/audio.wav
```

- `-ar 16000` — resample to 16kHz
- `-ac 1` — mix to mono
- `-vn` — strip video track

**Integration Points:**

- Input: any Zoom `.mp4` recording
- Output: `output/audio.wav` consumed by TASK-003 and TASK-004

**Considerations:**

- Use `subprocess.run` with `check=True` to raise on ffmpeg errors
- Capture stderr for error messages (ffmpeg writes to stderr not stdout)
- ffmpeg must be in system PATH (validated during TASK-001 setup)

---

### Pseudocode

```
function extract_audio(input_path: str, output_path: str) -> None:
  validate input_path exists and is .mp4
  create output directory if not exists

  cmd = [
    "ffmpeg", "-y",           # overwrite without prompt
    "-i", input_path,
    "-ar", "16000",           # 16kHz
    "-ac", "1",               # mono
    "-vn",                    # no video
    output_path
  ]

  result = subprocess.run(cmd, capture_output=True, check=True)

  validate output_path exists
  validate file size > 0

  duration = get_duration_ffprobe(output_path)
  print f"✅ Extracted audio: {duration:.1f}s → {output_path}"

function get_duration_ffprobe(wav_path: str) -> float:
  cmd = ["ffprobe", "-v", "quiet", "-show_entries",
         "format=duration", "-of", "csv=p=0", wav_path]
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  return float(result.stdout.strip())

function main() -> None:
  parse args: --input, --output (default: output/audio.wav)
  extract_audio(args.input, args.output)
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] Extracts WAV from Zoom MP4 without errors
- [ ] Output is confirmed 16kHz mono (via ffprobe check in script)
- [ ] Handles 1hr recording without memory issues (streaming via ffmpeg subprocess)
- [ ] Script exits with code 1 and clear message if input file not found
- [ ] Script exits with code 1 if ffmpeg not in PATH

### Code Quality

- [ ] All functions have complete type hints
- [ ] No bare `except:` — catch `subprocess.CalledProcessError`, `FileNotFoundError`
- [ ] Output directory auto-created with `Path.mkdir(parents=True, exist_ok=True)`

---

## Testing Requirements

### Manual Validation

1. Run on the actual 1hr Zoom recording
2. Verify output with: `ffprobe -v quiet -show_streams output/audio.wav`
   - Should show: `sample_rate=16000`, `channels=1`
3. Verify file size is reasonable (~115MB for 1hr at 16kHz mono PCM)

---

## Risks and Mitigation

| Risk                        | Mitigation                                                          |
| --------------------------- | ------------------------------------------------------------------- |
| ffmpeg not in PATH          | Script catches FileNotFoundError and prints PATH setup instructions |
| Zoom MP4 has no audio track | ffmpeg stderr captured and printed clearly                          |
| Output directory missing    | Script creates it automatically                                     |

---

## Related Tasks

- **TASK-001**: Must complete first — Python env must exist
- **TASK-003**: Consumes `output/audio.wav`
- **TASK-004**: Consumes `output/audio.wav`

---

## Notes

- **Complexity:** Low
- **Files Affected:** ~2 files
