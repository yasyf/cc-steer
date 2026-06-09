"""Re-exports the feedback candidate model from the mining domain.

Deprecated: import :class:`FeedbackCandidate`, :data:`DedupKey`, :func:`dedup_key`,
and :data:`SourceKind` from :mod:`cc_transcript.domains.mining`. This shim keeps
cc-pushback's historical import paths working for at least one release.
"""

from __future__ import annotations

from cc_transcript.domains.mining import DedupKey, FeedbackCandidate, SourceKind, dedup_key

PUSHBACK_SOURCE_KINDS = ("transcript_message", "plan_review", "interrupt_rejection", "review_comment")
"""The source kinds cc-pushback's detectors emit, for CLI choice lists."""

__all__ = ["PUSHBACK_SOURCE_KINDS", "DedupKey", "FeedbackCandidate", "SourceKind", "dedup_key"]
