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
_SEGMENT_GAP_THRESHOLD = 1.5
_SELECTION_TIE_EPS = 0.05


class TranscribeArgs(BaseModel):
    manifest_path: Path
    output_dir: Path
    model: str = "small"
    single_language: str | None = None
    allow_cpu: bool = False
    gate_monolingual: bool = False
    vad_filter: bool = False
    beam_size: int = 5


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
        help="Run only one language instead of dual (hi or en). Skips selection.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        dest="allow_cpu",
        help="Allow CPU fallback when CUDA is unavailable.",
    )
    parser.add_argument(
        "--gate-monolingual",
        action="store_true",
        dest="gate_monolingual",
        help=(
            "Opt-in speed optimization: skip the second language pass when the whole "
            "track is confidently monolingual (multi-probe language ID). Default off; "
            "validate on >2 tracks before enabling for a corpus backfill."
        ),
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        dest="vad_filter",
        help="Opt-in: skip silence via faster-whisper VAD (default off).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        dest="beam_size",
        help="Decode beam size (default 5); pass 1 for ~2x faster decode.",
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


def _safe_score(word: TranscriptWord) -> float:
    return max(word.score or 0.0, 0.0)


def _group_by_gap(
    words: list[DualLanguageWord],
    gap_threshold: float,
) -> list[list[DualLanguageWord]]:
    if not words:
        return []

    groups: list[list[DualLanguageWord]] = []
    current: list[DualLanguageWord] = [words[0]]
    max_end = words[0].end

    for word in words[1:]:
        if word.start - max_end > gap_threshold:
            groups.append(current)
            current = [word]
            max_end = word.end
        else:
            current.append(word)
            max_end = max(max_end, word.end)

    groups.append(current)
    return groups


def _pick_language(
    hi_words: list[DualLanguageWord],
    en_words: list[DualLanguageWord],
    tie_eps: float,
) -> str:
    if not hi_words:
        return "en"
    if not en_words:
        return "hi"

    mean_hi = sum(w.score for w in hi_words) / len(hi_words)
    mean_en = sum(w.score for w in en_words) / len(en_words)
    if abs(mean_hi - mean_en) >= tie_eps:
        return "hi" if mean_hi > mean_en else "en"

    mass_hi = sum(w.score for w in hi_words)
    mass_en = sum(w.score for w in en_words)
    return "hi" if mass_hi >= mass_en else "en"


def select_language_per_segment(
    hi_segments: list[TranscriptSegment],
    en_segments: list[TranscriptSegment],
    gap_threshold: float = _SEGMENT_GAP_THRESHOLD,
    tie_eps: float = _SELECTION_TIE_EPS,
) -> list[DualLanguageWord]:
    tagged: list[DualLanguageWord] = []
    for seg in hi_segments:
        for w in seg.words:
            tagged.append(
                DualLanguageWord(
                    start=w.start, end=w.end, word=w.word, score=_safe_score(w), source_language="hi"
                )
            )
    for seg in en_segments:
        for w in seg.words:
            tagged.append(
                DualLanguageWord(
                    start=w.start, end=w.end, word=w.word, score=_safe_score(w), source_language="en"
                )
            )
    tagged.sort(key=lambda x: x.start)

    selected: list[DualLanguageWord] = []
    for group in _group_by_gap(tagged, gap_threshold):
        hi_w = [x for x in group if x.source_language == "hi"]
        en_w = [x for x in group if x.source_language == "en"]
        winner = _pick_language(hi_w, en_w, tie_eps)
        chosen = hi_w if winner == "hi" else en_w
        selected.extend(sorted(chosen, key=lambda x: x.start))
    return selected


def resegment(
    words: list[DualLanguageWord],
    gap_threshold: float = _SEGMENT_GAP_THRESHOLD,
) -> list[TranscriptSegment]:
    return [_words_to_segment(group) for group in _group_by_gap(words, gap_threshold)]


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


def is_hallucinated_segment(words_text: list[str], threshold: float = 0.7) -> bool:
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


def _segments_from_raw(raw_segments: list[Any]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for seg in raw_segments:
        seg_words = seg.words or []
        if _is_hallucinated(seg_words):
            logger.debug("Skipping hallucinated segment at %.1fs", getattr(seg, "start", -1.0))
            continue
        words = [
            TranscriptWord(
                start=float(w.start),
                end=float(w.end),
                word=str(w.word),
                score=float(w.probability) if hasattr(w, "probability") else None,
            )
            for w in seg_words
            if w.start is not None and w.end is not None
        ]
        if not words:
            continue
        segments.append(
            TranscriptSegment(
                start=words[0].start,
                end=words[-1].end,
                text=" ".join(w.word for w in words),
                words=words,
            )
        )
    return segments


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


def run_whisperx_segments(
    audio: Any,
    language: str,
    model_name: str,
    device: str,
    compute_type: str,
    vad_filter: bool = False,
    beam_size: int = 5,
) -> list[TranscriptSegment]:
    import whisperx.asr

    logger.info("Transcribing with language=%s model=%s device=%s", language, model_name, device)
    asr_model = whisperx.asr.WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_gen, _info = asr_model.transcribe(
        audio, language=language, word_timestamps=True, vad_filter=vad_filter, beam_size=beam_size
    )
    raw_segments = list(segments_gen)
    _release_gpu(asr_model)
    return _segments_from_raw(raw_segments)


def _decide_gate_language(probes: list[tuple[str, float]], prob_threshold: float) -> str | None:
    detected: set[str] = set()
    for lang, prob in probes:
        if lang not in _VALID_LANGUAGES or prob < prob_threshold:
            return None
        detected.add(lang)
    if len(detected) == 1:
        return next(iter(detected))
    return None


def detect_track_language(
    audio: Any,
    model_name: str,
    device: str,
    compute_type: str,
    n_probes: int = 3,
    prob_threshold: float = 0.85,
    window_seconds: float = 30.0,
    sample_rate: int = 16000,
    vad_filter: bool = False,
) -> str | None:
    import whisperx.asr

    asr_model = whisperx.asr.WhisperModel(model_name, device=device, compute_type=compute_type)
    try:
        window = int(window_seconds * sample_rate)
        total = len(audio)
        if total <= window or n_probes <= 1:
            offsets = [0]
        else:
            offsets = [int(i * (total - window) / (n_probes - 1)) for i in range(n_probes)]

        probes: list[tuple[str, float]] = []
        for off in offsets:
            clip = audio[off : off + window]
            _segs, info = asr_model.transcribe(
                clip, language=None, word_timestamps=False, vad_filter=vad_filter
            )
            lang = str(getattr(info, "language", "") or "")
            prob = float(getattr(info, "language_probability", 0.0) or 0.0)
            probes.append((lang, prob))
        decision = _decide_gate_language(probes, prob_threshold)
        if decision is not None:
            logger.info("Gate: track is confidently monolingual (%s); skipping other pass", decision)
        return decision
    finally:
        _release_gpu(asr_model)


def _load_wav(wav_path: Path) -> Any:
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


def _words_from_single_language(
    segments: list[TranscriptSegment], language: str
) -> list[DualLanguageWord]:
    return [
        DualLanguageWord(
            start=w.start, end=w.end, word=w.word, score=_safe_score(w), source_language=language
        )
        for seg in segments
        for w in seg.words
    ]


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
        segs = run_whisperx_segments(
            audio, args.single_language, args.model, device, compute_type,
            vad_filter=args.vad_filter, beam_size=args.beam_size,
        )
        merged_words = _words_from_single_language(segs, args.single_language)
    else:
        gated_language: str | None = None
        if args.gate_monolingual:
            gated_language = detect_track_language(
                audio, args.model, device, compute_type, vad_filter=args.vad_filter
            )
        if gated_language is not None:
            segs = run_whisperx_segments(
                audio, gated_language, args.model, device, compute_type,
                vad_filter=args.vad_filter, beam_size=args.beam_size,
            )
            merged_words = _words_from_single_language(segs, gated_language)
        else:
            hi_segs = run_whisperx_segments(
                audio, "hi", args.model, device, compute_type,
                vad_filter=args.vad_filter, beam_size=args.beam_size,
            )
            en_segs = run_whisperx_segments(
                audio, "en", args.model, device, compute_type,
                vad_filter=args.vad_filter, beam_size=args.beam_size,
            )
            merged_words = select_language_per_segment(hi_segs, en_segs)

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


def _transcribe_track(
    audio_path: Path,
    out_path: Path,
    wav_dir: Path,
    args: TranscribeArgs,
    *,
    student_name: str | None = None,
    roll_no: str | None = None,
    is_teacher: bool = False,
) -> bool:
    try:
        result = transcribe_audio(
            audio_path, wav_dir, args,
            student_name=student_name, roll_no=roll_no, is_teacher=is_teacher,
        )
    except Exception as exc:
        logger.warning("Skipping track %s: transcription failed: %s", audio_path.name, exc)
        return False
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return True


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
            if _transcribe_track(
                manifest.session_mp4, args.output_dir / "session.json", wav_dir, args
            ):
                processed += 1

        for audio_file in manifest.per_student_m4as:
            logger.info("Transcribing per-student M4A: %s", audio_file.filename)
            if _transcribe_track(
                audio_file.path,
                args.output_dir / f"{audio_file.filename}.json",
                wav_dir,
                args,
                student_name=audio_file.display_name,
                roll_no=audio_file.roll_no_4digit,
            ):
                processed += 1
    finally:
        shutil.rmtree(wav_dir, ignore_errors=True)

    print(f"Transcribed {processed} audio file(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
