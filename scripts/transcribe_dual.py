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
_SELECTION_TIE_EPS = 0.05  # |mean_hi - mean_en| below this is a near-tie -> mass tiebreak


class TranscribeArgs(BaseModel):
    manifest_path: Path
    output_dir: Path
    model: str = "small"
    single_language: str | None = None
    allow_cpu: bool = False
    gate_monolingual: bool = False


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


def _group_by_gap(
    words: list[DualLanguageWord],
    gap_threshold: float,
) -> list[list[DualLanguageWord]]:
    """Group time-ordered words into windows separated by gaps > gap_threshold seconds.

    Tracks the running max end of the current window so it stays correct when the input
    interleaves two overlapping language streams (the union built by per-segment
    selection). For a single non-overlapping stream this is identical to grouping on the
    previous word's end, so `resegment` behavior is preserved.
    """
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
    """Pick the language for one window by mean word confidence, mass-tiebroken.

    - One side empty -> the other language.
    - Clear mean winner (|mean_hi - mean_en| >= tie_eps) -> higher mean.
    - Near-tie -> higher total confidence mass (sum of scores = confidence x coverage);
      exact tie -> Hindi (primary class language convention).

    Low confidence on BOTH is not special-cased: we still pick the higher mean and keep
    the words. Dropping junk is the downstream is_quality_text filter's job, so this stays
    lossless and single-responsibility.
    """
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
    """Choose ONE language per gap-delimited window; never interleave languages mid-clause.

    Replaces per-word merging (which spliced confidently-wrong Devanagari into clean
    English). Builds language-neutral windows from the union of both passes' words — real
    inter-clause silences are gaps in both streams — then emits only the winning language's
    words for each window. See `_pick_language` for the selection metric.
    """
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
    """Group selected words into segments separated by gaps > gap_threshold seconds."""
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


def _segments_from_raw(raw_segments: list[Any]) -> list[TranscriptSegment]:
    """Shape raw faster-whisper segments into TranscriptSegments, dropping hallucinations.

    Pure (no GPU) so it is unit-testable with fake segment objects.
    """
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


def run_whisperx_segments(
    audio: Any,
    language: str,
    model_name: str,
    device: str,
    compute_type: str,
) -> list[TranscriptSegment]:
    # Use WhisperModel directly — bypasses VAD bootstrap URL (HTTP 301) and avoids
    # loading alignment models that have Wav2Vec2Processor API breaks in newer transformers.
    # Segment structure is preserved (not flattened) so per-segment language selection can
    # choose one language per clause.
    import whisperx.asr

    logger.info("Transcribing with language=%s model=%s device=%s", language, model_name, device)
    asr_model = whisperx.asr.WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_gen, _info = asr_model.transcribe(audio, language=language, word_timestamps=True)
    raw_segments = list(segments_gen)
    _release_gpu(asr_model)
    return _segments_from_raw(raw_segments)


def _decide_gate_language(probes: list[tuple[str, float]], prob_threshold: float) -> str | None:
    """Pure gate decision: return the agreed language iff every probe agrees on a valid
    language with probability >= threshold, else None (-> run both passes).
    """
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
) -> str | None:
    """Conservative multi-probe language ID over the whole track (opt-in gate).

    Samples n_probes evenly-spaced ~window_seconds windows (start/middle/end) and only
    returns a language when ALL probes agree with high confidence — so a track that opens
    in one language and switches mid-way is NOT mis-gated (a later probe forces dual).
    Returns None to mean "run both passes + per-segment selection".
    """
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
            _segs, info = asr_model.transcribe(clip, language=None, word_timestamps=False)
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
        segs = run_whisperx_segments(audio, args.single_language, args.model, device, compute_type)
        merged_words = _words_from_single_language(segs, args.single_language)
    else:
        gated_language: str | None = None
        if args.gate_monolingual:
            gated_language = detect_track_language(audio, args.model, device, compute_type)
        if gated_language is not None:
            segs = run_whisperx_segments(audio, gated_language, args.model, device, compute_type)
            merged_words = _words_from_single_language(segs, gated_language)
        else:
            hi_segs = run_whisperx_segments(audio, "hi", args.model, device, compute_type)
            en_segs = run_whisperx_segments(audio, "en", args.model, device, compute_type)
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
