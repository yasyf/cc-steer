"""cc-pushback's candidate-model policy constants over the platform's mining types."""

from __future__ import annotations

from cc_transcript.mining import DedupKey, FeedbackCandidate, SourceKind, dedup_key

PUSHBACK_SOURCE_KINDS = ("transcript_message", "plan_review", "interrupt_rejection", "review_comment")
"""The source kinds cc-pushback's detectors emit, for CLI choice lists."""

__all__ = ["PUSHBACK_SOURCE_KINDS", "DedupKey", "FeedbackCandidate", "SourceKind", "dedup_key"]
