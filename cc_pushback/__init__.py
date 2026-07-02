"""Collect developer pushback signals from existing Claude Code transcripts."""

from __future__ import annotations

from cc_pushback.detectors import Detector, detect
from cc_pushback.models import FeedbackCandidate, dedup_key
from cc_pushback.scan import ScanReport, scan
from cc_pushback.spec import PUSHBACK_SPEC
from cc_pushback.store import FeedbackStore

# great-docs documents __all__ when present; keep it in sync with the re-exports above.
__all__ = [
    "PUSHBACK_SPEC",
    "Detector",
    "FeedbackCandidate",
    "FeedbackStore",
    "ScanReport",
    "dedup_key",
    "detect",
    "scan",
]
