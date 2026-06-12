"""Collect developer pushback signals from existing Claude Code transcripts."""

from __future__ import annotations

from cc_pushback.detectors import Detector, detect
from cc_pushback.migrate import MigrationReport, migrate_corpus
from cc_pushback.models import DedupKey, FeedbackCandidate, SourceKind, dedup_key
from cc_pushback.scan import ScanReport, scan
from cc_pushback.spec import PUSHBACK_SPEC
from cc_pushback.store import FeedbackStore

# Not the retired export-control convention: this exists only so great-docs' API
# reference skips the SourceKind (Literal) and DedupKey (NewType) aliases, which its
# dynamic walker cannot render ("Cannot handle auto for object kind: TYPE_ALIAS").
# great-docs documents __all__ when present; keep it in sync with the re-exports above.
__all__ = [
    "PUSHBACK_SPEC",
    "Detector",
    "FeedbackCandidate",
    "FeedbackStore",
    "MigrationReport",
    "ScanReport",
    "dedup_key",
    "detect",
    "migrate_corpus",
    "scan",
]
