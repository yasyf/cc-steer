"""Feedback-extraction sources over transcripts, GitHub, and issue files."""

from __future__ import annotations

from cc_pushback.sources.base import Source, dedup_key
from cc_pushback.sources.github import GitHubReviews
from cc_pushback.sources.interrupts import Interrupts
from cc_pushback.sources.issues import SupersetIssues, changed_issue_files
from cc_pushback.sources.plan_reviews import PlanReviews
from cc_pushback.sources.transcripts import TranscriptMessages, changed_files

__all__ = [
    "GitHubReviews",
    "Interrupts",
    "PlanReviews",
    "Source",
    "SupersetIssues",
    "TranscriptMessages",
    "changed_files",
    "changed_issue_files",
    "dedup_key",
]
