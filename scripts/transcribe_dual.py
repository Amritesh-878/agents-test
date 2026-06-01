from __future__ import annotations

import argparse
import gc
import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from scripts.extract_audio import AudioExtractionError, build_ffmpeg_command, run_command
from scripts.models.identity import ZoomFileManifest
from scripts.models.transcript import (
    DualLanguageWord,
    PerStudentTranscript,
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)

logger = logging.getLogger(__name__)

_VALID_LANGUAGES = {"hi", "en"}
_SEGMENT_GAP_THRESHOLD = 1.5  # seconds


class TranscribeArgs(BaseModel):
    manifest_path: Path
    output_dir: Path
    model: str = "small"
    single_language: str | None = None
    allow_cpu: bool = False


def parse_args(argv: Sequence[str] | None = None) -> TranscribeArgs:
    parser = argparse.ArgumentParser(
        description="Dual-language (Hindi+English) WhisperX transcription pipeline."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        dest="manifest_path",
        help="Path to manifest.json produced by ingest_zip.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        dest="output_dir",
        help="Directory to write transcript JSON files.",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="WhisperX model size (default: small).",
    )
    parser.add_argument(
        "--single-language",
        default=None,
        dest="single_language",
        metavar="LANG",
        help="Run only one language instead of dual (hi or en). Skips merge.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        dest="allow_cpu",
        help="Allow CPU fallback when CUDA is unavailable.",
    )
    namespace = parser.parse_args(argv)
    return TranscribeArgs.model_validate(vars(namespace))


def validate_inputs(args: TranscribeArgs) -> None:
    if not args.manifest_path.exists():
        raise ValueError(f"Manifest not found: {args.manifest_path}")
    if args.single_language is not None and args.single_language not in _VALID_LANGUAGES:
        raise ValueError(
            f"--single-language must be one of {sorted(_VALID_LANGUAGES)}, "
            f"got: {args.single_language!r}"
        )


# ---------------------------------------------------------------------------
# Pure helper functions — no GPU, fully testable
# ---------------------------------------------------------------------------


def _safe_score(word: TranscriptWord) -> float:
    return max(word.score or 0.0, 0.0)


def _words_overlap(a: TranscriptWord, b: TranscriptWord) -> bool:
    return max(a.start, b.start) < min(a.end, b.end)


def merge_by_word_probability(
    hi_words: list[TranscriptWord],
    en_words: list[TranscriptWord],
) -> list[DualLanguageWord]:
    """Pick the higher-probability word at each time position from two language runs.

    Hindi is preferred on equal scores (it is the primary class language).
    Words present in only one run are included unchanged.
    """
    result: list[DualLanguageWord] = []
    hi_idx = 0
    en_idx = 0

    while hi_idx < len(hi_words) or en_idx < len(en_words):
        hi_word = hi_words[hi_idx] if hi_idx < len(hi_words) else None
        en_word = en_words[en_idx] if en_idx < len(en_words) else None

        if hi_word is None:
            assert en_word is not None
            result.append(
                DualLanguageWord(
                    start=en_word.start,
                    end=en_word.end,
                    word=en_word.word,
                    score=_safe_score(en_word),
                    source_language="en",
                )
            )
            en_idx += 1
        elif en_word is None:
            result.append(
                DualLanguageWord(
                    start=hi_word.start,
                    end=hi_word.end,
                    word=hi_word.word,
                    score=_safe_score(hi_word),
                    source_language="hi",
                )
            )
            hi_idx += 1
        elif _words_overlap(hi_word, en_word):
            hi_score = _safe_score(hi_word)
            en_score = _safe_score(en_word)
            if hi_score >= en_score:
                result.append(
                    DualLanguageWord(
                        start=hi_word.start,
                        end=hi_word.end,
                        word=hi_word.word,
                        score=hi_score,
                        source_language="hi",
                    )
                )
            else:
                result.append(
                    DualLanguageWord(
                        start=en_word.start,
                        end=en_word.end,
                        word=en_word.word,
                        score=en_score,
                        source_language="en",
                    )
                )
            hi_idx += 1
            en_idx += 1
        elif hi_word.start <= en_word.start:
            result.append(
                DualLanguageWord(
                    start=hi_word.start,
                    end=hi_word.end,
                    word=hi_word.word,
                    score=_safe_score(hi_word),
                    source_language="hi",
                )
            )
            hi_idx += 1
        else:
            result.append(
                DualLanguageWord(
                    start=en_word.start,
                    end=en_word.end,
                    word=en_word.word,
                    score=_safe_score(en_word),
                    source_language="en",
                )
            )
            en_idx += 1

    return result


def resegment(
    words: list[DualLanguageWord],
    gap_threshold: float = _SEGMENT_GAP_THRESHOLD,
) -> list[TranscriptSegment]:
    """Group merged words into segments separated by gaps > gap_threshold seconds."""
    if not words:
        return []

    segments: list[TranscriptSegment] = []
    current: list[DualLanguageWord] = [words[0]]

    for word in words[1:]:
        if word.start - current[-1].end > gap_threshold:
            segments.append(_words_to_segment(current))
            current = [word]
        else:
            current.append(word)

    segments.append(_words_to_segment(current))
    return segments


def _words_to_segment(words: list[DualLanguageWord]) -> TranscriptSegment:
    return TranscriptSegment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.word for w in words),
        words=[
            TranscriptWord(start=w.start, end=w.end, word=w.word, score=w.score)
            for w in words
        ],
    )


def compute_language_stats(words: list[DualLanguageWord]) -> tuple[float, float, str]:
    """Return (hi_avg_score, en_avg_score, dominant_language)."""
    hi_words = [w for w in words if w.source_language == "hi"]
    en_words = [w for w in words if w.source_language == "en"]
    hi_avg = sum(w.score for w in hi_words) / len(hi_words) if hi_words else 0.0
    en_avg = sum(w.score for w in en_words) / len(en_words) if en_words else 0.0
    dominant = "hi" if len(hi_words) >= len(en_words) else "en"
    return hi_avg, en_avg, dominant


def build_transcript_document(
    merged_words: list[DualLanguageWord],
    model_name: str,
) -> TranscriptDocument:
    segments = resegment(merged_words)
    _, _, dominant = compute_language_stats(merged_words)
    for seg in segments:
        seg.language = dominant
    return TranscriptDocument(model=model_name, language=dominant, segments=segments)


# ---------------------------------------------------------------------------
# GPU-dependent functions — isolated for easy mocking in tests
# ---------------------------------------------------------------------------


def _get_device(allow_cpu: bool) -> tuple[str, str]:
    try:
        import torch
    except ImportError:
        if allow_cpu:
            return "cpu", "int8"
        raise RuntimeError("PyTorch is not installed.")
    if torch.cuda.is_available():
        return "cuda", "float16"
    if allow_cpu:
        return "cpu", "int8"
    raise RuntimeError("CUDA is not available. Use --allow-cpu for CPU fallback.")


def _release_gpu(resource: object | None = None) -> None:
    if resource is not None:
        del resource
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _extract_words_from_result(result: dict[str, Any]) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                logger.debug("Skipping word without timestamps: %s", w)
                continue
            words.append(
                TranscriptWord(
                    start=float(w["start"]),
                    end=float(w["end"]),
                    word=str(w.get("word", "")),
                    score=float(w["score"]) if w.get("score") is not None else None,
                )
            )
    return words


def is_hallucinated_segment(words_text: list[str], threshold: float = 0.7) -> bool:
    """Return True when one word dominates >threshold of a segment — Whisper silence artifact."""
    if len(words_text) < 8:
        return False
    from collections import Counter

    top_count = Counter(w.strip().lower() for w in words_text if w.strip()).most_common(1)
    if not top_count:
        return False
    return top_count[0][1] / len(words_text) > threshold


def _is_hallucinated(faster_whisper_words: list[Any]) -> bool:
    texts = [getattr(w, "word", "") for w in faster_whisper_words]
    return is_hallucinated_segment(texts)


def run_whisperx_language(
    audio: Any,
    language: str,
    model_name: str,
    device: str,
    compute_type: str,
) -> list[TranscriptWord]:
    # Use WhisperModel directly — bypasses VAD bootstrap URL (HTTP 301) and avoids
    # loading alignment models that have Wav2Vec2Processor API breaks in newer transformers.
    import whisperx.asr

    logger.info("Transcribing with language=%s model=%s device=%s", language, model_name, device)
    asr_model = whisperx.asr.WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_gen, _info = asr_model.transcribe(audio, language=language, word_timestamps=True)
    raw_segments = list(segments_gen)
    _release_gpu(asr_model)

    # Use raw ASR word timestamps, filtering hallucinated segments first.
    words: list[TranscriptWord] = []
    for seg in raw_segments:
        if _is_hallucinated(seg.words or []):
            logger.debug("Skipping hallucinated segment at %.1fs", seg.start)
            continue
        for w in seg.words or []:
            words.append(
                TranscriptWord(
                    start=float(w.start),
                    end=float(w.end),
                    word=str(w.word),
                    score=float(w.probability) if hasattr(w, "probability") else None,
                )
            )
    return words


def _load_wav(wav_path: Path) -> Any:
    """Load a 16kHz mono WAV as a float32 numpy array — no ffmpeg needed."""
    import soundfile as sf

    data, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def convert_to_wav(audio_path: Path, wav_dir: Path) -> Path:
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / (audio_path.stem + ".wav")
    if wav_path.exists():
        return wav_path
    cmd = build_ffmpeg_command(audio_path, wav_path)
    run_command(cmd, "ffmpeg")
    if not wav_path.exists():
        raise AudioExtractionError(f"ffmpeg did not create: {wav_path}")
    return wav_path


def transcribe_audio(
    audio_path: Path,
    wav_dir: Path,
    args: TranscribeArgs,
    student_name: str | None = None,
    roll_no: str | None = None,
    is_teacher: bool = False,
) -> PerStudentTranscript:
    wav_path = convert_to_wav(audio_path, wav_dir)
    device, compute_type = _get_device(args.allow_cpu)
    audio: Any = _load_wav(wav_path)

    if args.single_language:
        hi_words = run_whisperx_language(audio, args.single_language, args.model, device, compute_type)
        merged_words = [
            DualLanguageWord(
                start=w.start,
                end=w.end,
                word=w.word,
                score=_safe_score(w),
                source_language=args.single_language,
            )
            for w in hi_words
        ]
    else:
        hi_words = run_whisperx_language(audio, "hi", args.model, device, compute_type)
        en_words = run_whisperx_language(audio, "en", args.model, device, compute_type)
        merged_words = merge_by_word_probability(hi_words, en_words)

    doc = build_transcript_document(merged_words, args.model)
    hi_avg, en_avg, dominant = compute_language_stats(merged_words)

    return PerStudentTranscript(
        audio_file=audio_path.name,
        student_name=student_name,
        roll_no=roll_no,
        is_teacher=is_teacher,
        transcript=doc,
        merged_words=merged_words,
        hi_avg_score=hi_avg,
        en_avg_score=en_avg,
        dominant_language=dominant,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as exc:
        logger.error("Input validation failed: %s", exc)
        raise SystemExit(2) from exc

    manifest = ZoomFileManifest.model_validate_json(
        args.manifest_path.read_text(encoding="utf-8")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = args.output_dir / "_wav_tmp"

    processed = 0

    try:
        if manifest.session_mp4 is not None:
            logger.info("Transcribing session MP4: %s", manifest.session_mp4.name)
            result = transcribe_audio(manifest.session_mp4, wav_dir, args)
            out = args.output_dir / "session.json"
            out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            processed += 1

        for audio_file in manifest.per_student_m4as:
            logger.info("Transcribing per-student M4A: %s", audio_file.filename)
            result = transcribe_audio(
                audio_file.path,
                wav_dir,
                args,
                student_name=audio_file.display_name,
                roll_no=audio_file.roll_no_4digit,
            )
            out = args.output_dir / f"{audio_file.filename}.json"
            out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            processed += 1
    finally:
        # The WAV cache is purely intermediate; drop it so it doesn't grow unbounded.
        shutil.rmtree(wav_dir, ignore_errors=True)

    print(f"Transcribed {processed} audio file(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
