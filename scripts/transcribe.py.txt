from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError

from pydantic import BaseModel, Field


class TranscriptionArgs(BaseModel):
    allow_cpu: bool = False
    batch_size: int = 8
    input_path: Path = Path("output/audio.wav")
    language: str | None = None
    whisper_model: str = "small"
    output_path: Path = Path("output/transcript_raw.json")


class RuntimeOptions(BaseModel):
    compute_type: str
    device: str


class TranscriptWord(BaseModel):
    end: float
    start: float
    word: str


class TranscriptSegment(BaseModel):
    end: float
    start: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)


class TranscriptDocument(BaseModel):
    language: str
    model: str
    segments: list[TranscriptSegment] = Field(default_factory=list)


class TranscriptionError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> TranscriptionArgs:
    parser = argparse.ArgumentParser(
        description="Run WhisperX transcription plus word-level alignment on a WAV recording."
    )
    parser.add_argument(
        "--input",
        default="output/audio.wav",
        help="Path to the 16kHz mono WAV file from TASK-002.",
    )
    parser.add_argument(
        "--output",
        default="output/transcript_raw.json",
        help="Path to the output JSON transcript.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Transcription batch size. The default 8 fits the local 4GB GPU, with a built-in retry at 4 on OOM.",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        help="WhisperX model size. Keep the default 'small' on the local RTX 3050 4GB GPU.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language override for the alignment model, for example 'en'.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a slow CPU fallback when CUDA is unavailable.",
    )
    namespace = parser.parse_args(argv)
    return TranscriptionArgs(
        allow_cpu=namespace.allow_cpu,
        batch_size=namespace.batch_size,
        input_path=Path(namespace.input),
        language=namespace.language,
        whisper_model=namespace.model_size,
        output_path=Path(namespace.output),
    )


def validate_inputs(args: TranscriptionArgs) -> None:
    if not args.input_path.exists():
        raise ValueError(f"Input file does not exist: {args.input_path}")
    if not args.input_path.is_file():
        raise ValueError(f"Input path is not a file: {args.input_path}")
    if args.input_path.suffix.lower() != ".wav":
        raise ValueError("Input file must use the .wav extension.")
    if args.output_path.suffix.lower() != ".json":
        raise ValueError("Output file must use the .json extension.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if not args.whisper_model.strip():
        raise ValueError("--model-size must not be empty.")
    if args.language is not None and not args.language.strip():
        raise ValueError("--language must not be empty when provided.")


def resolve_runtime_options(cuda_available: bool, allow_cpu: bool) -> RuntimeOptions:
    if cuda_available:
        return RuntimeOptions(device="cuda", compute_type="float16")
    if allow_cpu:
        return RuntimeOptions(device="cpu", compute_type="int8")
    raise TranscriptionError(
        "CUDA is not available. Fix the GPU environment or rerun with --allow-cpu for a slow fallback."
    )


def get_runtime_options(allow_cpu: bool) -> RuntimeOptions:
    try:
        import torch
    except ImportError as error:
        raise TranscriptionError(
            f"PyTorch is not installed: {error}. Install the project dependencies before transcription."
        ) from error

    return resolve_runtime_options(torch.cuda.is_available(), allow_cpu)


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


def get_batch_size_attempts(batch_size: int) -> list[int]:
    attempts = [batch_size]
    if batch_size > 4:
        attempts.append(4)
    return attempts


def build_asr_options(model_name: str) -> dict[str, object]:
    candidate_options: dict[str, object] = {
        "multilingual": not model_name.endswith(".en"),
        "max_new_tokens": None,
        "clip_timestamps": "0",
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }
    try:
        from faster_whisper.transcribe import TranscriptionOptions
    except ImportError:
        return candidate_options

    supported_options = set(inspect.signature(TranscriptionOptions).parameters)
    return {
        option_name: option_value
        for option_name, option_value in candidate_options.items()
        if option_name in supported_options
    }


def is_vad_bootstrap_redirect(error: HTTPError) -> bool:
    return error.code == 301


def normalize_language_code(language_code: str) -> str:
    normalized = language_code.strip().lower()
    if not normalized:
        raise TranscriptionError("WhisperX did not return a language code.")
    return normalized.split("-")[0]


def resolve_alignment_language(detected_language: str, override_language: str | None) -> str:
    if override_language is not None:
        return normalize_language_code(override_language)
    return normalize_language_code(detected_language)


def to_float(value: object, field_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as error:
            raise TranscriptionError(f"Expected a numeric {field_name}, got {value!r}.") from error
    raise TranscriptionError(f"Expected a numeric {field_name}, got {value!r}.")


def build_transcript_word(raw_word: object) -> TranscriptWord:
    if not isinstance(raw_word, dict):
        raise TranscriptionError(f"WhisperX returned an invalid word payload: {raw_word!r}")

    word_value = raw_word.get("word")
    if not isinstance(word_value, str) or not word_value.strip():
        raise TranscriptionError(f"WhisperX returned a word without text: {raw_word!r}")

    return TranscriptWord(
        word=word_value,
        start=to_float(raw_word.get("start"), "word start"),
        end=to_float(raw_word.get("end"), "word end"),
    )


def build_segment_words(
    raw_words: Sequence[object] | None,
    segment_start: float,
    segment_end: float,
    segment_text: str,
) -> list[TranscriptWord]:
    if raw_words is None:
        return [
            TranscriptWord(
                word=segment_text.strip(),
                start=segment_start,
                end=segment_end,
            )
        ]

    words: list[TranscriptWord] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            raise TranscriptionError(f"WhisperX returned an invalid word payload: {raw_word!r}")

        if raw_word.get("start") is None or raw_word.get("end") is None:
            continue
        words.append(build_transcript_word(raw_word))

    if words:
        return words

    return [
        TranscriptWord(
            word=segment_text.strip(),
            start=segment_start,
            end=segment_end,
        )
    ]


def build_transcript_segment(raw_segment: object) -> TranscriptSegment:
    if not isinstance(raw_segment, dict):
        raise TranscriptionError(f"WhisperX returned an invalid segment payload: {raw_segment!r}")

    text_value = raw_segment.get("text")
    if not isinstance(text_value, str) or not text_value.strip():
        raise TranscriptionError(f"WhisperX returned a segment without text: {raw_segment!r}")

    raw_words = raw_segment.get("words")
    if raw_words is not None and not isinstance(raw_words, list):
        raise TranscriptionError(
            f"WhisperX returned invalid aligned word metadata: {raw_words!r}"
        )

    segment_start = to_float(raw_segment.get("start"), "segment start")
    segment_end = to_float(raw_segment.get("end"), "segment end")

    return TranscriptSegment(
        start=segment_start,
        end=segment_end,
        text=text_value,
        words=build_segment_words(raw_words, segment_start, segment_end, text_value),
    )


def build_transcript_document(result: object, model_size: str) -> TranscriptDocument:
    if not isinstance(result, dict):
        raise TranscriptionError(f"WhisperX returned an invalid transcription payload: {result!r}")

    language_value = result.get("language")
    if not isinstance(language_value, str):
        raise TranscriptionError("WhisperX did not return the detected language.")

    raw_segments = result.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise TranscriptionError("WhisperX returned no aligned transcript segments.")

    return TranscriptDocument(
        language=normalize_language_code(language_value),
        model=model_size,
        segments=[build_transcript_segment(raw_segment) for raw_segment in raw_segments],
    )


def save_output(document: TranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def build_backend_transcription_result(
    segments: Sequence[object],
    language: str,
) -> dict[str, object]:
    normalized_segments: list[dict[str, object]] = []
    for segment in segments:
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        text = getattr(segment, "text", None)
        if not isinstance(text, str):
            raise TranscriptionError(f"Backend ASR returned a segment without text: {segment!r}")
        stripped_text = text.strip()
        if not stripped_text:
            continue
        normalized_segments.append(
            {
                "start": round(to_float(start, "segment start"), 3),
                "end": round(to_float(end, "segment end"), 3),
                "text": stripped_text,
            }
        )

    if not normalized_segments:
        raise TranscriptionError("Backend ASR returned no transcript segments.")

    return {
        "segments": normalized_segments,
        "language": normalize_language_code(language),
    }


class TranscriptionService:
    def __init__(self, args: TranscriptionArgs) -> None:
        self.args = args

    def run(self) -> TranscriptDocument:
        runtime = get_runtime_options(self.args.allow_cpu)

        try:
            import whisperx
        except ImportError as error:
            raise TranscriptionError(
                f"WhisperX is not installed: {error}. Install requirements.txt before running transcription."
            ) from error

        print(f"Loading audio from {self.args.input_path}...")
        audio = whisperx.load_audio(str(self.args.input_path))

        transcription_result = self._transcribe_audio(whisperx, audio, runtime)
        detected_language = transcription_result.get("language")
        if not isinstance(detected_language, str):
            raise TranscriptionError("WhisperX did not return a detected language.")

        print(f"Detected language: {normalize_language_code(detected_language)}")
        alignment_language = resolve_alignment_language(detected_language, self.args.language)
        aligned_result = self._align_words(whisperx, audio, transcription_result, runtime, alignment_language)
        aligned_result.setdefault("language", normalize_language_code(detected_language))
        document = build_transcript_document(aligned_result, self.args.whisper_model)
        save_output(document, self.args.output_path)

        print(
            f"Transcript saved: {len(document.segments)} segments -> {self.args.output_path}"
        )
        return document

    def _transcribe_audio(
        self,
        whisperx_module: Any,
        audio: Any,
        runtime: RuntimeOptions,
    ) -> dict[str, object]:
        try:
            import torch
        except ImportError as error:
            raise TranscriptionError(
                f"PyTorch is not installed: {error}. Install the project dependencies before transcription."
            ) from error

        last_error: BaseException | None = None
        for batch_size in get_batch_size_attempts(self.args.batch_size):
            model: Any | None = None
            try:
                print(
                    f"Loading WhisperX model '{self.args.whisper_model}' on {runtime.device} for batch size {batch_size}..."
                )
                # The local RTX 3050 has 4GB VRAM, so keep the default model at 'small'.
                model = whisperx_module.load_model(
                    self.args.whisper_model,
                    runtime.device,
                    asr_options=build_asr_options(self.args.whisper_model),
                    compute_type=runtime.compute_type,
                    language=self.args.language,
                )
                if model is None:
                    raise TranscriptionError("WhisperX did not return a transcription model instance.")
                print(f"Running transcription with batch size {batch_size}...")
                result = model.transcribe(audio, batch_size=batch_size)
                if not isinstance(result, dict):
                    raise TranscriptionError(
                        f"WhisperX returned an unexpected transcription payload: {result!r}"
                    )
                return result
            except HTTPError as error:
                if is_vad_bootstrap_redirect(error):
                    print(
                        "WhisperX VAD bootstrap returned HTTP 301; falling back to the direct ASR backend without VAD chunking."
                    )
                    return self._transcribe_with_backend(whisperx_module, audio, runtime)
                raise TranscriptionError(
                    f"WhisperX failed while downloading the VAD model: HTTP {error.code}."
                ) from error
            except torch.cuda.OutOfMemoryError as error:
                last_error = error
                print(
                    f"CUDA out of memory at batch size {batch_size}; releasing GPU memory before retry."
                )
            finally:
                release_gpu_resources(model)

        raise TranscriptionError(
            "CUDA ran out of memory during transcription even after retrying with batch size 4. "
            "Close other GPU workloads or rerun with a smaller --batch-size."
        ) from last_error

    def _transcribe_with_backend(
        self,
        whisperx_module: Any,
        audio: Any,
        runtime: RuntimeOptions,
    ) -> dict[str, object]:
        try:
            import torch
        except ImportError as error:
            raise TranscriptionError(
                f"PyTorch is not installed: {error}. Install the project dependencies before transcription."
            ) from error

        backend_model: Any | None = None
        try:
            print(
                f"Loading direct WhisperX ASR backend '{self.args.whisper_model}' on {runtime.device}..."
            )
            backend_model = whisperx_module.asr.WhisperModel(
                self.args.whisper_model,
                device=runtime.device,
                compute_type=runtime.compute_type,
            )
            print(
                "Running transcription through the direct ASR backend; the --batch-size setting is not used in this fallback path."
            )
            segments, info = backend_model.transcribe(
                audio,
                condition_on_previous_text=False,
                language=self.args.language,
                vad_filter=False,
                word_timestamps=False,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise TranscriptionError(
                "CUDA ran out of memory while loading or running the direct ASR backend. "
                "Close other GPU workloads and retry."
            ) from error
        finally:
            release_gpu_resources(backend_model)

        detected_language = getattr(info, "language", None)
        if not isinstance(detected_language, str):
            raise TranscriptionError("Backend ASR did not return a detected language.")

        return build_backend_transcription_result(list(segments), detected_language)

    def _align_words(
        self,
        whisperx_module: Any,
        audio: Any,
        transcription_result: dict[str, object],
        runtime: RuntimeOptions,
        alignment_language: str,
    ) -> dict[str, object]:
        try:
            import torch
        except ImportError as error:
            raise TranscriptionError(
                f"PyTorch is not installed: {error}. Install the project dependencies before transcription."
            ) from error

        alignment_model: object | None = None
        try:
            print(f"Loading alignment model for language '{alignment_language}' on {runtime.device}...")
            alignment_model, metadata = whisperx_module.load_align_model(
                language_code=alignment_language,
                device=runtime.device,
            )
            print("Running word-level alignment...")
            aligned = whisperx_module.align(
                transcription_result["segments"],
                alignment_model,
                metadata,
                audio,
                runtime.device,
                return_char_alignments=False,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise TranscriptionError(
                "CUDA ran out of memory during alignment. Close other GPU workloads and retry the transcription step."
            ) from error
        finally:
            release_gpu_resources(alignment_model)

        if not isinstance(aligned, dict):
            raise TranscriptionError(f"WhisperX returned an unexpected alignment payload: {aligned!r}")
        return aligned


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        validate_inputs(args)
        TranscriptionService(args).run()
    except (TranscriptionError, ValueError) as error:
        print(f"Transcription failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()