# TASK-001: Environment Setup & CUDA Validation

## Overview

Set up the Python environment with all required dependencies for the whisperX + pyannote pipeline, and validate that CUDA, the HuggingFace token, and gated model access all work correctly on the local RTX 3050 machine before any ML work begins.

## Execution Snapshot

- Depends on: None
- Produces: `requirements.txt`, `scripts/validate_env.py`, `.env.example`, `README.md`
- Primary validation: `python scripts/validate_env.py`
- Complexity: Low

## Goals

1. **Reproducible Environment**: Pin all dependencies in `requirements.txt` so the pipeline runs consistently
2. **CUDA Validation**: Confirm PyTorch detects the RTX 3050 GPU so whisperX uses CUDA not CPU
3. **HuggingFace Access**: Validate the HF token can download the gated pyannote models without 403 errors

---

## Reasoning

### Why validate CUDA first?

**Current Problems:**

- whisperX on CPU is extremely slow for a 1hr recording (could take hours)
- A silent fallback to CPU is easy to miss — validation makes it explicit
- CUDA toolkit version must match the PyTorch build

**Solution:**

- Run `torch.cuda.is_available()` and assert True
- Print device name and VRAM to confirm RTX 3050 detected

### Why validate HuggingFace access separately?

**Current Problems:**

- pyannote models are gated — HF token alone isn't enough, model agreements must be accepted on the website
- A 403 error mid-pipeline wastes time after a long transcription run
- Token needs to be in `.env` not hardcoded

**Solution:**

- Pre-flight check: attempt to load the pyannote pipeline config from HF
- Fail fast with a clear error message if access is denied

---

## Files to Change

### Files to CREATE

1. `requirements.txt` — pinned dependencies
2. `scripts/validate_env.py` — validation script
3. `.env.example` — template for HF token
4. `README.md` — setup instructions for Windows + RTX 3050

### Files to MODIFY

- None (greenfield)

---

## Implementation Approach

### `requirements.txt`

**Purpose:** Pin all dependencies to avoid version conflicts between whisperX, pyannote, and PyTorch.

**Key dependencies:**

```
torch==2.1.0+cu118          # CUDA 11.8 build — matches most RTX 30xx drivers
torchaudio==2.1.0+cu118
whisperx==3.1.1
pyannote.audio==3.1.1
python-dotenv==1.0.0
```

**Considerations:**

- PyTorch CUDA build must match the installed CUDA toolkit version
- whisperX pins its own faster-whisper version internally — do not override
- Install PyTorch first separately before `pip install -r requirements.txt` to avoid index conflicts

**Install order (in README):**

```
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

### `scripts/validate_env.py`

**Purpose:** Pre-flight check that all components are accessible before running the pipeline.

**Key Responsibilities:**

- Check `torch.cuda.is_available()` — print device name + VRAM
- Load `.env` and check `HF_TOKEN` is set
- Attempt to load pyannote speaker diarization config to validate model access
- Print a clear ✅ / ❌ per check with actionable error messages

**Pseudocode:**

```
function validate():
  check_cuda():
    - assert torch.cuda.is_available()
    - print device name, VRAM

  check_hf_token():
    - load .env
    - assert HF_TOKEN is set and non-empty

  check_pyannote_access():
    - attempt: Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    - on 403: print "Accept model agreement at huggingface.co/pyannote/speaker-diarization-3.1"
    - on success: print "✅ pyannote access confirmed"

  check_whisperx():
    - import whisperx
    - whisperx.load_model("small", device="cuda")
    - print "✅ whisperX small model loaded"
    - unload model (free VRAM)
```

**Integration Points:**

- Reads from `.env` via `python-dotenv`
- No file I/O other than `.env` read

---

## Acceptance Criteria

### Functional Requirements

- [ ] `python scripts/validate_env.py` runs without unhandled exceptions
- [ ] CUDA check prints RTX 3050 device name and available VRAM
- [ ] HF token check fails clearly if `.env` is missing or token is empty
- [ ] pyannote access check correctly distinguishes 403 (agreement not accepted) from other errors
- [ ] whisperX `small` model loads on CUDA without OOM

### Code Quality

- [ ] All functions have complete type hints (parameters + return type)
- [ ] No bare `except:` — catch specific exceptions (`OSError`, `requests.HTTPError`)
- [ ] `.env.example` committed, `.env` in `.gitignore`

---

## Testing Requirements

### Manual Validation

1. **Happy path** — Run with valid `.env` and accepted HF agreements → all checks pass
2. **Missing token** — Remove `HF_TOKEN` from `.env` → clear error printed, exits with code 1
3. **Wrong CUDA** — If CUDA unavailable, clear message about CPU fallback risk

### No automated tests for this task

Environment validation scripts don't require unit tests — they are themselves the test.

---

## Pseudocode Examples

### validate_env.py structure

```
import sys, torch, os
from dotenv import load_dotenv

function main() -> None:
  load_dotenv()
  results = []

  results.append(check_cuda())
  results.append(check_hf_token())
  results.append(check_pyannote_access(token))
  results.append(check_whisperx())

  if any failed:
    print summary of failures
    sys.exit(1)
  else:
    print "✅ All checks passed. Ready to run pipeline."

function check_cuda() -> bool:
  if not torch.cuda.is_available():
    print "❌ CUDA not available. whisperX will run on CPU (very slow)."
    return False
  device = torch.cuda.get_device_name(0)
  vram = torch.cuda.get_device_properties(0).total_memory / 1e9
  print f"✅ CUDA: {device} — {vram:.1f}GB VRAM"
  return True
```

---

## Risks and Mitigation

| Risk                                  | Mitigation                                                |
| ------------------------------------- | --------------------------------------------------------- |
| PyTorch CUDA version mismatch         | README specifies exact install command with `--index-url` |
| pyannote model agreement not accepted | validate_env.py prints exact HF URL to accept             |
| 4GB VRAM insufficient for small model | Fallback note in README to use `base` model               |
| Windows PATH issues with pip          | README notes to use `python -m pip` not bare `pip`        |

---

## Related Tasks

- **TASK-002**: Blocked until this passes — needs Python env and ffmpeg
- **TASK-003**: Needs CUDA working and whisperX confirmed
- **TASK-004**: Needs HF token and pyannote access confirmed

---

## Notes

- **Complexity:** Low
- **Files Affected:** ~4 files
- **VRAM note:** RTX 3050 4GB — always use `device="cuda"` with `small` model. Never `large-v3` locally.
