"""Collect developer pushback signals from existing Claude Code transcripts."""

from __future__ import annotations

from cc_pushback.context import ContextSnapshot, ContextTurn, build_snapshot
from cc_pushback.detectors import Detector, detect
from cc_pushback.models import DedupKey, FeedbackCandidate, SourceKind, dedup_key
from cc_pushback.scan import ScanReport, scan
from cc_pushback.spec import PUSHBACK_SPEC
from cc_pushback.store import FeedbackStore
