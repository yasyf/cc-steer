"""cc-steer's candidate-model policy constants over the platform's mining types."""

from __future__ import annotations

from cc_transcript.mining import DedupKey, FeedbackCandidate, SourceKind, dedup_key

STEERING_SOURCE_KINDS = ("transcript_message", "plan_review", "interrupt_rejection", "review_comment", "question_answer")
"""The source kinds cc-steer's detectors emit, for CLI choice lists."""

__all__ = ["STEERING_SOURCE_KINDS", "DedupKey", "FeedbackCandidate", "SourceKind", "dedup_key"]
