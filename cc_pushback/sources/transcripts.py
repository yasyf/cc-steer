from __future__ import annotations

from typing import TYPE_CHECKING

from cc_transcript.discovery import TranscriptDiscovery
from cc_transcript.models import UserEvent
from cc_transcript.parser import parse_events

from cc_pushback.context import build_snapshot
from cc_pushback.models import FeedbackCandidate
from cc_pushback.sources.base import MESSAGE_JUNK_RE, dedup_key

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from cc_transcript.models import TranscriptEvent

    from cc_pushback.repo import Repository

__all__ = ["TranscriptMessages", "changed_files"]

SOURCE_KIND = "transcript_message"


def changed_files(
    repo: Repository, roots: Sequence[Path]
) -> Iterator[tuple[Path, float, list[TranscriptEvent]]]:
    """Yields parsed transcripts under ``roots`` that changed since last scan.

    Discovery is mtime-filtered against the repository's recorded file mtimes,
    so a file is parsed only when new or modified.

    Args:
        repo: The repository holding the file-mtime ledger.
        roots: The directories to search recursively for transcripts.

    Yields:
        ``(path, mtime, events)`` for each changed transcript.
    """
    known = repo.file_mtimes()
    return (
        (path, mtime, parse_events(path))
        for root in roots
        for path, mtime in TranscriptDiscovery.find_in(root, known_mtimes=known)
    )


class TranscriptMessages:
    """Extracts the user's typed messages from a transcript as candidates.

    Keeps non-junk, non-sidechain, non-meta user turns with text. Unlike the
    sentiment filter, interrupt and stop-hook markers are not treated as junk
    here: those carry pushback worth keeping.
    """

    def candidates_for_file(self, path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
        return (
            FeedbackCandidate(
                dedup_key=dedup_key(str(path), event.meta.uuid, SOURCE_KIND),
                source_kind=SOURCE_KIND,
                occurred_at=event.meta.timestamp,
                text=event.text,
                context=build_snapshot(events, index),
                session_id=event.meta.session_id,
                origin_path=path,
                origin_uuid=event.meta.uuid,
                cc_version=event.meta.cc_version,
            )
            for index, event in enumerate(events)
            if isinstance(event, UserEvent)
            if not event.meta.is_sidechain
            if not event.meta.is_meta
            if event.text.strip()
            if not MESSAGE_JUNK_RE.search(event.text)
        )
