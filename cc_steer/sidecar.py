"""The issues.jsonl sidecar source: anchor superset findings to the closest session.

Superset writes review findings as ``.context/cleanup/*issues.jsonl`` next to a
worktree, with no per-finding timestamp — every finding shares the file's mtime.
This module reads those findings, anchors each findings file to the transcript
session whose event time-range sits closest to the file mtime (containment first,
nearest edge otherwise), and lifts each surviving finding into a
:class:`~cc_transcript.mining.FeedbackCandidate` that flows into the same store as
transcript steering. Review metadata that the candidate model has no field for
rides in the candidate's ``payload``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cc_transcript import TranscriptDiscovery
from cc_transcript.activity import SessionActivity
from cc_transcript.filterspec import event_meta
from cc_transcript.ids import EventRef, SessionId
from cc_transcript.mining import REVIEW_COMMENT, FeedbackCandidate, dedup_key, firm
from cc_transcript.parser import parse_events_async

from cc_steer.capture import capture_anchored_window

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cc_transcript.models import TranscriptEvent

SIDECAR_GLOB = "*issues.jsonl"


@dataclass(frozen=True, slots=True)
class Finding:
    """One superset review finding from an ``issues.jsonl`` file.

    Attributes:
        id: The finding's stable identifier within its file.
        file: The source file the finding faults.
        line: The faulted line number.
        rule: The rule the finding cites.
        severity: The finding's severity, e.g. ``HIGH``.
        track: The review track the finding belongs to, e.g. ``arch``.
        evidence: The reviewer's evidence for the finding.
        suggested_fix: The reviewer's suggested fix, or ``"none"`` when absent.
        dismissed: Whether the finding was dismissed at validation.
    """

    id: str
    file: str
    line: int
    rule: str
    severity: str
    track: str
    evidence: str
    suggested_fix: str
    dismissed: bool

    @classmethod
    def parse(cls, record: dict[str, object]) -> Finding:
        if not isinstance(line := record["line"], int):
            raise AssertionError(line)
        return cls(
            id=str(record["id"]),
            file=str(record["file"]),
            line=line,
            rule=str(record["rule"]),
            severity=str(record["severity"]),
            track=str(record["track"]),
            evidence=str(record["evidence"]),
            suggested_fix=str(record.get("suggested_fix", "none")),
            dismissed=bool(record.get("dismissed", False)),
        )


@dataclass(frozen=True, slots=True)
class Anchor:
    """The transcript event one findings file anchors to.

    Attributes:
        session_id: The closest session's id.
        ref: The resolvable reference to the anchor event.
        activity: The lifted session the anchor resolves within.
        occurred_at: The anchor event's timestamp.
    """

    session_id: SessionId
    ref: EventRef
    activity: SessionActivity
    occurred_at: datetime


def edge_distance(start: datetime, end: datetime, when: datetime) -> float:
    return 0.0 if start <= when <= end else min(abs((start - when).total_seconds()), abs((end - when).total_seconds()))


def time_range(events: Sequence[TranscriptEvent]) -> tuple[datetime, datetime] | None:
    stamps = [meta.timestamp for event in events if (meta := event_meta(event)) is not None]
    return (min(stamps), max(stamps)) if stamps else None


def closest_session(
    ranges: Sequence[tuple[Path, datetime, datetime]], when: datetime
) -> Path | None:
    """Picks the session whose event time-range sits closest to ``when``.

    Containment wins (distance ``0`` when ``when`` falls within a session's range),
    falling back to the nearest range edge. Stub sessions — those with no
    timestamped events, hence no range — are passed pre-filtered and never compete.

    Args:
        ranges: One ``(path, start, end)`` per non-stub candidate session.
        when: The instant to anchor to, normalized to UTC.

    Returns:
        The closest session's path, or None when ``ranges`` is empty.
    """
    return min(
        (path for path, _, _ in ranges),
        key=lambda path: next(edge_distance(s, e, when) for p, s, e in ranges if p == path),
        default=None,
    )


def candidate_session_dirs(roots: Sequence[Path], uuid: str) -> list[Path]:
    return [d for root in roots if root.is_dir() for d in root.iterdir() if d.is_dir() and uuid in d.name]


async def session_ranges(dirs: Sequence[Path]) -> list[tuple[Path, datetime, datetime]]:
    return [
        (path, start, end)
        for directory in dirs
        for path, _ in await TranscriptDiscovery.find_in(directory)
        if "subagents" not in path.parts
        if (rng := time_range(await parse_events_async(path))) is not None
        for start, end in (rng,)
    ]


async def anchor_for(roots: Sequence[Path], uuid: str, when: datetime) -> Anchor | None:
    if (winner := closest_session(await session_ranges(candidate_session_dirs(roots, uuid)), when)) is None:
        return None
    events = await parse_events_async(winner)
    activity = SessionActivity.from_events(SessionId(winner.stem), events)
    meta = min(
        (m for event in events if (m := event_meta(event)) is not None),
        key=lambda m: abs((m.timestamp - when).total_seconds()),
    )
    return Anchor(meta.session_id, EventRef(meta.session_id, meta.uuid), activity, meta.timestamp)


def worktree_uuid(sidecar: Path) -> str:
    return sidecar.parts[-6]


def read_findings(sidecar: Path) -> list[Finding]:
    return [
        finding
        for line in sidecar.read_text().splitlines()
        if line.strip()
        if not (finding := Finding.parse(json.loads(line))).dismissed
    ]


def candidate_text(finding: Finding) -> str:
    body = f"{finding.rule}\n{finding.evidence}"
    return body if finding.suggested_fix.lower() == "none" else f"{body}\nSuggested fix: {finding.suggested_fix}"


def to_candidate(sidecar: Path, finding: Finding, anchor: Anchor) -> FeedbackCandidate:
    return FeedbackCandidate(
        dedup_key=dedup_key("sidecar", str(sidecar), finding.id, finding.file, str(finding.line), finding.rule),
        source_kind=REVIEW_COMMENT,
        occurred_at=anchor.occurred_at,
        text=candidate_text(finding),
        window=capture_anchored_window(anchor.activity, anchor.ref),
        ref=anchor.ref,
        session_id=anchor.session_id,
        signal=firm("sidecar_finding", finding.severity.lower()),
        payload={
            "format": "issues_jsonl",
            "file": finding.file,
            "line_start": finding.line,
            "line_end": finding.line,
            "provenance": "surfaced",
            "severity": finding.severity,
            "track": finding.track,
            "finding_id": finding.id,
        },
    )


async def candidates_for(sidecar: Path, roots: Sequence[Path]) -> list[FeedbackCandidate]:
    """Lifts one ``issues.jsonl`` file into anchored feedback candidates.

    Reads the findings, skips dismissed ones, anchors the file to the closest
    session under ``roots``, and maps each surviving finding to a candidate.

    Args:
        sidecar: The ``issues.jsonl`` file to lift.
        roots: The transcript roots to resolve the anchor session under.

    Returns:
        One candidate per surviving finding, or ``[]`` when no candidate session
        matches the worktree uuid (an empty or stub-only project dir).
    """
    findings = read_findings(sidecar)
    when = datetime.fromtimestamp(sidecar.stat().st_mtime, tz=UTC)
    if not findings or (anchor := await anchor_for(roots, worktree_uuid(sidecar), when)) is None:
        return []
    return [to_candidate(sidecar, finding, anchor) for finding in findings]


def discover_sidecars(findings_dirs: Sequence[Path]) -> list[Path]:
    return sorted(path for directory in findings_dirs for path in directory.rglob(SIDECAR_GLOB))
