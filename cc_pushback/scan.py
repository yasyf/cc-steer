"""The scan orchestrator: discover, parse, detect, and persist, incrementally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cc_transcript import TranscriptDiscovery, TranscriptParser

from cc_pushback.detectors import detect
from cc_pushback.sidecar import candidates_for, discover_sidecars

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cc_pushback.store import FeedbackStore


@dataclass(frozen=True, slots=True)
class ScanReport:
    """The outcome of one scan pass.

    Attributes:
        scanned: The number of transcripts and findings files parsed and recorded.
        inserted: The number of newly inserted feedback events.
    """

    scanned: int
    inserted: int


async def scan(
    store: FeedbackStore, roots: Sequence[Path], *, findings_dirs: Sequence[Path] = (), full: bool = False
) -> ScanReport:
    """Scans transcripts under ``roots`` for feedback, incrementally.

    Each transcript is parsed only when new or modified since the last scan
    (unless ``full``), parsing runs concurrently across files, and every candidate
    is inserted idempotently. A transcript that fails to parse — for example one
    Claude Code is still appending to — is silently skipped by the parser and left
    unrecorded, so the next scan retries it. After the transcript pass, every
    ``issues.jsonl`` findings file under ``findings_dirs`` is anchored to the
    closest session under ``roots`` and its findings are recorded through the same
    idempotent insert.

    Args:
        store: The store to read mtimes from and write candidates to.
        roots: The directories to search recursively for transcripts.
        findings_dirs: The directories to search recursively for ``issues.jsonl``
            superset findings files.
        full: When set, re-scan every transcript and findings file, ignoring
            recorded mtimes.

    Returns:
        The :class:`ScanReport` for this pass.
    """
    known = None if full else await store.file_mtimes()
    paths: list[tuple[Path, float]] = []
    for root in roots:
        paths.extend(await TranscriptDiscovery.find_in(root, known_mtimes=known))
    scanned = 0
    inserted = 0
    async for parsed in TranscriptParser.stream_transcripts(paths):
        inserted += await store.record_file_scan(str(parsed.path), parsed.mtime, detect(parsed.events))
        scanned += 1
    for sidecar in discover_sidecars(findings_dirs):
        mtime = sidecar.stat().st_mtime
        if known is not None and (prev := known.get(str(sidecar))) is not None and prev >= mtime:
            continue
        inserted += await store.record_file_scan(str(sidecar), mtime, await candidates_for(sidecar, roots))
        scanned += 1
    return ScanReport(scanned=scanned, inserted=inserted)
