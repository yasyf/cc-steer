"""The SQLite feedback store: the mining-domain store with cc-pushback's default path.

The store mechanism lives in :mod:`cc_transcript.domains.mining`; this module adds
cc-pushback's default database location and re-exports the store building blocks for
back-compat.
"""

from __future__ import annotations

from pathlib import Path

from cc_transcript.domains.mining import FEEDBACK_DDL, Stats, event_row
from cc_transcript.domains.mining import FeedbackStore as BaseFeedbackStore

__all__ = ["FEEDBACK_DDL", "FeedbackStore", "Stats", "event_row"]


class FeedbackStore(BaseFeedbackStore):
    """Persistent store for collected feedback over a :class:`FileStateStore`.

    Layers the ``feedback_events`` table onto cc-transcript's file-mtime ledger.
    Recording a scanned file and inserting its candidates commit in one
    transaction, so a scan is atomic: it either records the file and all its
    candidates or neither.

    Example:
        >>> async with await FeedbackStore.open(FeedbackStore.default_path()) as store:
        ...     await store.record_file_scan(str(path), mtime, candidates)
    """

    @staticmethod
    def default_path() -> Path:
        """Returns the default database path, ``~/.cc-pushback/feedback.db``."""
        return Path.home() / ".cc-pushback" / "feedback.db"
