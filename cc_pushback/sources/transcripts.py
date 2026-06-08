from __future__ import annotations

from typing import TYPE_CHECKING

from cc_transcript import PUSHBACK_SPEC, keep
from cc_transcript.discovery import TranscriptDiscovery
from cc_transcript.models import UserEvent
from cc_transcript.parser import parse_events

from cc_pushback.context import build_snapshot
from cc_pushback.formats import extract_all
from cc_pushback.models import FeedbackCandidate
from cc_pushback.sources.base import dedup_key

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from cc_transcript.models import TranscriptEvent

    from cc_pushback.repo import Repository

SOURCE_KIND = "transcript_message"
REVIEW_KIND = "review_comment"


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


def pushback_user_events(events: Sequence[TranscriptEvent]) -> Iterator[tuple[int, UserEvent]]:
    return (
        (index, event)
        for index, event in enumerate(events)
        if isinstance(event, UserEvent)
        if keep(event, PUSHBACK_SPEC)
    )


class TranscriptMessages:
    """Extracts the user's typed messages from a transcript as candidates.

    Applies cc-transcript's ``PUSHBACK_SPEC``: drops structural noise, trivial
    acknowledgements ("ok", "go ahead") and very short control messages, plus
    sidechain/meta/compacted/empty turns. Unlike the sentiment filter, interrupt
    and stop-hook markers are kept here: those carry pushback worth learning.
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
            for index, event in pushback_user_events(events)
        )


class ReviewComments:
    """Explodes review-formatted user messages into one candidate per comment.

    A message matching a declared :class:`~cc_pushback.formats.ReviewFormat`
    (superset inline cites, conductor findings or workstreams) yields one
    ``review_comment`` candidate per extracted comment, alongside the whole
    message's ``transcript_message`` row.
    """

    def candidates_for_file(self, path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
        return (
            FeedbackCandidate(
                dedup_key=dedup_key(str(path), event.meta.uuid, REVIEW_KIND, str(position)),
                source_kind=REVIEW_KIND,
                occurred_at=event.meta.timestamp,
                text=comment.comment,
                context=build_snapshot(events, index),
                session_id=event.meta.session_id,
                origin_path=path,
                origin_uuid=event.meta.uuid,
                cc_version=event.meta.cc_version,
                payload={
                    "format": fmt.name,
                    "file": comment.file,
                    "line_start": comment.line_start,
                    "line_end": comment.line_end,
                },
            )
            for index, event in pushback_user_events(events)
            for position, (fmt, comment) in enumerate(extract_all(event.text))
        )
