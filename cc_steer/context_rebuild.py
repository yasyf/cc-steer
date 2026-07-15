"""One-off repair of stored feedback contexts captured before anchor splitting."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING

from cc_transcript import TranscriptDiscovery
from cc_transcript.context import ContextWindow
from cc_transcript.ids import SessionId
from cc_transcript.mining import DedupKey
from cc_transcript.parser import parse_events_async

from cc_steer.detectors import detect
from cc_steer.negatives import W_MAX, truncated
from cc_steer.rendering import has_substantive_content, messages
from cc_steer.watcher.live import scrub_events

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from aiosqlite import Row
    from cc_transcript.mining import FeedbackCandidate

    from cc_steer.store import FeedbackStore

CONTEXTS_QUERY = """
SELECT dedup_key, session_id, event_uuid, context_json, quarantined_reason
FROM feedback_events
ORDER BY id
"""
GATE_REPAIR_QUERY = """
SELECT dedup_key, context_json FROM feedback_events WHERE context_json IS NOT NULL ORDER BY id
"""
SESSION_BATCH_SIZE = 20
PARSE_CONCURRENCY = 8
LOCK_FILENAME = "context_rebuild.lock"


@dataclass(frozen=True, slots=True)
class CopyParseFailure:
    """One transcript copy skipped because its bytes would not parse."""

    session_id: SessionId
    path: Path
    error: str


@dataclass(frozen=True, slots=True)
class ContextRebuildReport:
    """Counts from one idempotent stored-context rebuild pass."""

    found: int
    rebuilt: int
    quarantined: int
    gate_repaired: int = 0
    parse_failures: tuple[CopyParseFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class CopyCandidates:
    mtime: float
    path: Path
    by_dedup: Mapping[str, FeedbackCandidate]


@dataclass(frozen=True, slots=True)
class SessionCopies:
    paths_found: int
    copies: tuple[CopyCandidates, ...]
    failures: tuple[CopyParseFailure, ...]


@dataclass(frozen=True, slots=True)
class RebuiltContext:
    dedup_key: DedupKey
    context_json: str


@dataclass(frozen=True, slots=True)
class QuarantinedContext:
    dedup_key: DedupKey
    reason: str


@dataclass(frozen=True, slots=True)
class UnchangedContext:
    dedup_key: DedupKey


type RebuildOutcome = RebuiltContext | QuarantinedContext | UnchangedContext


async def database_path(store: FeedbackStore) -> Path:
    async with store.store.conn.execute("PRAGMA database_list") as cur:
        rows = [dict(row) async for row in cur]
    return next(Path(str(row["file"])) for row in rows if row["name"] == "main")


@asynccontextmanager
async def rebuild_lock(store: FeedbackStore) -> AsyncIterator[None]:
    lock = (await database_path(store)).with_name(LOCK_FILENAME)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise RuntimeError(f"context rebuild already running; lock held at {lock}") from None
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink()


async def parse_copy(
    session_id: SessionId, path: Path, semaphore: asyncio.Semaphore
) -> CopyCandidates | CopyParseFailure:
    async with semaphore:
        try:
            events = await parse_events_async(path)
        except (KeyError, ValueError, TypeError) as exc:
            return CopyParseFailure(session_id, path, f"{type(exc).__name__}: {exc}")
    return CopyCandidates(
        mtime=await TranscriptDiscovery.transcript_mtime(path),
        path=path,
        by_dedup={str(candidate.dedup_key): candidate for candidate in detect(scrub_events(events))},
    )


async def load_session_copies(
    session_id: SessionId, roots: Sequence[Path], semaphore: asyncio.Semaphore
) -> tuple[SessionId, SessionCopies]:
    paths = sorted(
        {path.resolve() for root in roots if root.is_dir() for path in root.rglob(f"{session_id}.jsonl")},
        key=str,
    )
    results = await asyncio.gather(*(parse_copy(session_id, path, semaphore) for path in paths))
    return session_id, SessionCopies(
        paths_found=len(paths),
        copies=tuple(result for result in results if isinstance(result, CopyCandidates)),
        failures=tuple(result for result in results if isinstance(result, CopyParseFailure)),
    )


def before_content_length(window: ContextWindow) -> int:
    return sum(len(message["content"].strip()) for message in messages(window.before))


def quarantine_outcome(row: Row, dedup_key: DedupKey, reason: str) -> RebuildOutcome:
    return (
        UnchangedContext(dedup_key)
        if row["quarantined_reason"] == reason
        else QuarantinedContext(dedup_key, reason)
    )


def rebuild_outcome(row: Row, session: SessionCopies) -> RebuildOutcome:
    dedup_key = DedupKey(str(row["dedup_key"]))
    if session.paths_found == 0:
        return quarantine_outcome(row, dedup_key, "transcript_not_found")
    if not session.copies:
        return quarantine_outcome(row, dedup_key, "transcript_parse_failed")
    matching = [
        (copy.mtime, candidate)
        for copy in session.copies
        if (candidate := copy.by_dedup.get(str(dedup_key))) is not None
    ]
    if not matching:
        return quarantine_outcome(row, dedup_key, "anchor_not_found")
    window = max(matching, key=lambda item: (before_content_length(item[1].window), item[0]))[1].window
    if not has_substantive_content(messages(window.before)):
        return quarantine_outcome(row, dedup_key, "rebuilt_context_empty")
    context_json = window.to_json()
    return (
        UnchangedContext(dedup_key)
        if context_json == str(row["context_json"]) and row["quarantined_reason"] is None
        else RebuiltContext(dedup_key, context_json)
    )


def persistence_rows(
    outcomes: Sequence[RebuildOutcome],
) -> tuple[list[tuple[DedupKey, str]], list[tuple[DedupKey, str]]]:
    rebuilt: list[tuple[DedupKey, str]] = []
    quarantined: list[tuple[DedupKey, str]] = []
    for outcome in outcomes:
        match outcome:
            case RebuiltContext(dedup_key=dedup_key, context_json=context_json):
                rebuilt.append((dedup_key, context_json))
            case QuarantinedContext(dedup_key=dedup_key, reason=reason):
                quarantined.append((dedup_key, reason))
            case UnchangedContext():
                pass
    return rebuilt, quarantined


def gate_sample_windows(dedup_key: DedupKey, context_json: str) -> list[tuple[str, str]]:
    window = ContextWindow.from_json(context_json)
    return [
        (f"hard:{dedup_key}", context_json),
        *[
            (f"pos:{dedup_key}:{offset}", rewound.to_json())
            for offset in range(W_MAX)
            if (rewound := truncated(window, offset)) is not None
        ],
    ]


async def repair_gate_samples(store: FeedbackStore) -> int:
    """Heals every gate sample whose window drifted from its feedback row's current context.

    Comparison-driven and unconditional: each row's post-sweep window regenerates
    the sample windows it seeds, and only stored samples that actually differ are
    updated, so pre-existing offset-zero mismatches converge whether or not the
    feedback row changed this pass.
    """
    cur = await store.store.conn.execute(GATE_REPAIR_QUERY)
    return await store.repair_gate_sample_windows(
        [
            repair
            async for row in cur
            for repair in gate_sample_windows(DedupKey(str(row["dedup_key"])), str(row["context_json"]))
        ]
    )


async def rebuild_contexts(store: FeedbackStore, roots: Sequence[Path]) -> ContextRebuildReport:
    """Rebuilds all stored contexts from exact session transcripts and quarantines failures."""
    async with rebuild_lock(store):
        rows = [row async for row in await store.store.conn.execute(CONTEXTS_QUERY)]
        rows_by_session = {
            SessionId(session_id): list(group)
            for session_id, group in groupby(
                sorted(rows, key=lambda row: str(row["session_id"])),
                key=lambda row: str(row["session_id"]),
            )
        }
        session_ids = tuple(rows_by_session)
        semaphore = asyncio.Semaphore(PARSE_CONCURRENCY)
        rebuilt = 0
        quarantined = 0
        failures: list[CopyParseFailure] = []
        for offset in range(0, len(session_ids), SESSION_BATCH_SIZE):
            batch = session_ids[offset : offset + SESSION_BATCH_SIZE]
            parsed = dict(
                await asyncio.gather(*(load_session_copies(session_id, roots, semaphore) for session_id in batch))
            )
            failures.extend(failure for copies in parsed.values() for failure in copies.failures)
            changes = await store.rebuild_context(
                *persistence_rows(
                    [
                        rebuild_outcome(row, parsed[session_id])
                        for session_id in batch
                        for row in rows_by_session[session_id]
                    ]
                )
            )
            rebuilt += changes.rebuilt
            quarantined += changes.quarantined
        return ContextRebuildReport(
            found=len(rows),
            rebuilt=rebuilt,
            quarantined=quarantined,
            gate_repaired=await repair_gate_samples(store),
            parse_failures=tuple(failures),
        )
