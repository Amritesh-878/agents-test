from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PYANNOTE_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-3.1"


class DiarizationArgs(BaseModel):
    allow_cpu: bool = False
    input_path: Path = Path("output/audio.wav")
    max_speakers: int | None = 6
    min_speakers: int | None = None
    pipeline_name: str = "pyannote/speaker-diarization-3.1"
    output_path: Path = Path("output/diarization.json")


class RuntimeOptions(BaseModel):
    device: str


class DiarizationSegment(BaseModel):
    end: float
    speaker: str
    start: float


class DiarizationDocument(BaseModel):
    speakers: list[str] = Field(default_factory=list)
    segments: list[DiarizationSegment] = Field(default_factory=list)


class DiarizationError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> DiarizationArgs:
    parser = argparse.ArgumentParser(
        description="Run pyannote speaker diarization on an extracted WAV recording."
    )
    parser.add_argument(
        "--input",
        default="output/audio.wav",
        help="Path to the 16kHz mono WAV file from TASK-002.",
    )
    parser.add_argument(
        "--output",
        default="output/diarization.json",
        help="Path to the output JSON diarization file.",
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="pyannote diarization pipeline repo to load from Hugging Face.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Optional lower bound for speaker count.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=6,
        help="Optional upper bound for speaker count. Keep the default for class recordings.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a slow CPU fallback when CUDA is unavailable.",
    )
    namespace = parser.parse_args(argv)
    return DiarizationArgs(
        allow_cpu=namespace.allow_cpu,
        input_path=Path(namespace.input),
        max_speakers=namespace.max_speakers,
        min_speakers=namespace.min_speakers,
        pipeline_name=namespace.model,
        output_path=Path(namespace.output),
    )


def validate_inputs(args: DiarizationArgs) -> None:
    if not args.input_path.exists():
        raise ValueError(f"Input file does not exist: {args.input_path}")
    if not args.input_path.is_file():
        raise ValueError(f"Input path is not a file: {args.input_path}")
    if args.input_path.suffix.lower() != ".wav":
        raise ValueError("Input file must use the .wav extension.")
    if args.output_path.suffix.lower() != ".json":
        raise ValueError("Output file must use the .json extension.")
    if not args.pipeline_name.strip():
        raise ValueError("--model must not be empty.")
    if args.min_speakers is not None and args.min_speakers <= 0:
        raise ValueError("--min-speakers must be a positive integer when provided.")
    if args.max_speakers is not None and args.max_speakers <= 0:
        raise ValueError("--max-speakers must be a positive integer when provided.")
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        raise ValueError("--min-speakers cannot be greater than --max-speakers.")


def resolve_runtime_options(cuda_available: bool, allow_cpu: bool) -> RuntimeOptions:
    if cuda_available:
        return RuntimeOptions(device="cuda")
    if allow_cpu:
        return RuntimeOptions(device="cpu")
    raise DiarizationError(
        "CUDA is not available. Fix the GPU environment or rerun with --allow-cpu for a slow fallback."
    )


def get_runtime_options(allow_cpu: bool) -> RuntimeOptions:
    try:
        import torch
    except ImportError as error:
        raise DiarizationError(
            f"PyTorch is not installed: {error}. Install the project dependencies before diarization."
        ) from error

    return resolve_runtime_options(torch.cuda.is_available(), allow_cpu)


def load_hf_token() -> str | None:
    load_dotenv()
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    fallback_token = os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
    return fallback_token or None


def release_gpu_resources(*resources: object | None) -> None:
    for resource in resources:
        if resource is not None:
            del resource
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_torch_checkpoint_loading() -> None:
    try:
        import torch
    except ImportError:
        return

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    torch_version_module = getattr(torch, "torch_version", None)
    torch_version_class = getattr(torch_version_module, "TorchVersion", None)
    if callable(add_safe_globals) and torch_version_class is not None:
        add_safe_globals([torch_version_class])

    original_load = getattr(torch, "load", None)
    if not callable(original_load) or getattr(original_load, "_task004_compat", False):
        return

    def compatible_load(*args: object, **kwargs: object) -> object:
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    setattr(compatible_load, "_task004_compat", True)
    setattr(torch, "load", compatible_load)


def build_pipeline_kwargs(min_speakers: int | None, max_speakers: int | None) -> dict[str, int]:
    pipeline_kwargs: dict[str, int] = {}
    if min_speakers is not None:
        pipeline_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        pipeline_kwargs["max_speakers"] = max_speakers
    return pipeline_kwargs


def to_float(value: object, field_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as error:
            raise DiarizationError(f"Expected a numeric {field_name}, got {value!r}.") from error
    raise DiarizationError(f"Expected a numeric {field_name}, got {value!r}.")


def annotation_to_raw_segments(annotation: object) -> list[tuple[object, object, object]]:
    itertracks = getattr(annotation, "itertracks", None)
    if not callable(itertracks):
        raise DiarizationError(f"pyannote returned an invalid annotation payload: {annotation!r}")

    raw_segments: list[tuple[object, object, object]] = []
    for turn, _, speaker in itertracks(yield_label=True):
        raw_segments.append((getattr(turn, "start", None), getattr(turn, "end", None), speaker))
    return raw_segments


def build_diarization_segment(start: object, end: object, speaker: object) -> DiarizationSegment:
    if not isinstance(speaker, str) or not speaker.strip():
        raise DiarizationError(f"pyannote returned an invalid speaker label: {speaker!r}")

    start_value = to_float(start, "segment start")
    end_value = to_float(end, "segment end")
    if end_value <= start_value:
        raise DiarizationError(
            f"pyannote returned an invalid segment span for {speaker}: start={start_value}, end={end_value}."
        )

    return DiarizationSegment(
        speaker=speaker.strip(),
        start=round(start_value, 3),
        end=round(end_value, 3),
    )


def build_diarization_document(
    raw_segments: Sequence[tuple[object, object, object]],
) -> DiarizationDocument:
    segments = [
        build_diarization_segment(start, end, speaker)
        for start, end, speaker in raw_segments
    ]
    if not segments:
        raise DiarizationError("pyannote returned no diarization segments.")

    ordered_segments = sorted(
        segments,
        key=lambda segment: (segment.start, segment.end, segment.speaker),
    )
    return DiarizationDocument(
        speakers=sorted({segment.speaker for segment in ordered_segments}),
        segments=ordered_segments,
    )


def save_output(document: DiarizationDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


class DiarizationService:
    def __init__(self, args: DiarizationArgs) -> None:
        self.args = args

    def run(self) -> DiarizationDocument:
        token = load_hf_token()
        if token is None:
            raise DiarizationError(
                "No Hugging Face token was found in .env. Set HF_TOKEN and accept the pyannote model agreement before retrying."
            )

        runtime = get_runtime_options(self.args.allow_cpu)
        if runtime.device == "cpu":
            print("CUDA is unavailable; running diarization on CPU because --allow-cpu was set.")

        try:
            import torch
        except ImportError as error:
            raise DiarizationError(
                f"PyTorch is not installed: {error}. Install the project dependencies before diarization."
            ) from error

        try:
            from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
            from pyannote.audio import Pipeline
        except ImportError as error:
            raise DiarizationError(
                f"pyannote is not installed: {error}. Install requirements.txt before running diarization."
            ) from error

        pipeline: Any | None = None
        diarization: Any | None = None
        try:
            configure_torch_checkpoint_loading()
            print(f"Loading pyannote pipeline '{self.args.pipeline_name}' on {runtime.device}...")
            pipeline = Pipeline.from_pretrained(self.args.pipeline_name, use_auth_token=token)
            if pipeline is None:
                raise DiarizationError("pyannote did not return a diarization pipeline instance.")
            pipeline.to(torch.device(runtime.device))
            print(f"Running speaker diarization on {self.args.input_path}...")
            diarization = pipeline(
                str(self.args.input_path),
                **build_pipeline_kwargs(self.args.min_speakers, self.args.max_speakers),
            )
        except GatedRepoError as error:
            raise DiarizationError(
                f"Access to {self.args.pipeline_name} is gated. Accept the model agreement at {PYANNOTE_MODEL_URL} and retry."
            ) from error
        except RepositoryNotFoundError as error:
            raise DiarizationError(
                f"The pyannote repository was not found: {error}. Confirm the model name and retry."
            ) from error
        except HfHubHTTPError as error:
            status_code = error.response.status_code if error.response is not None else "unknown"
            raise DiarizationError(
                f"Hugging Face HTTP error while loading {self.args.pipeline_name}: status {status_code}. Verify the token scope and model agreement at {PYANNOTE_MODEL_URL}."
            ) from error
        except torch.cuda.OutOfMemoryError as error:
            raise DiarizationError(
                "CUDA ran out of memory during diarization. Close other GPU workloads and rerun this script on its own."
            ) from error
        except OSError as error:
            raise DiarizationError(
                f"pyannote failed to initialize or run diarization: {error}"
            ) from error
        finally:
            release_gpu_resources(pipeline)

        document = build_diarization_document(annotation_to_raw_segments(diarization))
        save_output(document, self.args.output_path)
        print(
            f"Diarization saved: {len(document.speakers)} speakers, {len(document.segments)} segments -> {self.args.output_path}"
        )
        release_gpu_resources()
        return document


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        validate_inputs(args)
        DiarizationService(args).run()
    except (DiarizationError, ValueError) as error:
        print(f"Speaker diarization failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()