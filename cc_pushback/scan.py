"""The scan orchestrator: discover, parse, detect, and persist, incrementally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cc_transcript import TranscriptDiscovery, parse_events

from cc_pushback.detectors import detect

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cc_pushback.store import FeedbackStore


@dataclass(frozen=True, slots=True)
class ScanReport:
    """The outcome of one scan pass.

    Attributes:
        scanned: The number of transcripts parsed and recorded.
        inserted: The number of newly inserted feedback events.
        skipped: The transcripts that could not be parsed and were left unrecorded.
    """

    scanned: int
    inserted: int
    skipped: tuple[Path, ...]


def scan(store: FeedbackStore, roots: Sequence[Path], *, full: bool = False) -> ScanReport:
    """Scans transcripts under ``roots`` for feedback, incrementally.

    Each transcript is parsed only when new or modified since the last scan
    (unless ``full``), and every candidate is inserted idempotently. A transcript
    that fails to parse — for example one Claude Code is still appending to — is
    skipped and left unrecorded so the next scan retries it.

    Args:
        store: The store to read mtimes from and write candidates to.
        roots: The directories to search recursively for transcripts.
        full: When set, re-scan every transcript, ignoring recorded mtimes.

    Returns:
        The :class:`ScanReport` for this pass.
    """
    known = None if full else store.file_mtimes()
    scanned = 0
    inserted = 0
    skipped: list[Path] = []
    for root in roots:
        for path, mtime in TranscriptDiscovery.find_in(root, known_mtimes=known):
            try:
                events = parse_events(path)
            except (OSError, ValueError, KeyError):
                skipped.append(path)
                continue
            inserted += store.record_file_scan(str(path), mtime, detect(path, events))
            scanned += 1
    return ScanReport(scanned=scanned, inserted=inserted, skipped=tuple(skipped))
