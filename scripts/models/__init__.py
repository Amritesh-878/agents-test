from __future__ import annotations

from scripts.models.identity import (
    AttendanceRecord,
    AudioFileIdentity,
    IdentityMap,
    IdentityMapEntry,
    MatchResult,
    PerStudentAudioFile,
    RosterEntry,
    StudentIdentity,
    ZoomFileManifest,
)
from scripts.models.pipeline import ClassSession, PipelineConfig
from scripts.models.transcript import (
    AlignmentResult,
    DualLanguageWord,
    MergeMetadata,
    MergedSegment,
    MergedTranscriptDocument,
    PerStudentTranscript,
    TranscriptDocument,
    TranscriptSegment,
    TranscriptWord,
)

__all__ = [
    "AlignmentResult",
    "AttendanceRecord",
    "AudioFileIdentity",
    "ClassSession",
    "DualLanguageWord",
    "IdentityMap",
    "IdentityMapEntry",
    "MatchResult",
    "MergeMetadata",
    "MergedSegment",
    "MergedTranscriptDocument",
    "PerStudentAudioFile",
    "PerStudentTranscript",
    "PipelineConfig",
    "RosterEntry",
    "StudentIdentity",
    "TranscriptDocument",
    "TranscriptSegment",
    "TranscriptWord",
    "ZoomFileManifest",
]
