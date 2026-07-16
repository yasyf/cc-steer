"""One-off repair of stored feedback contexts captured before anchor splitting."""

from __future__ import annotations

import asyncio
import fcntl
import os
from collections import Counter
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
from cc_steer.negatives import W_MAX, GateSample, truncated
from cc_steer.rendering import has_substantive_content, messages
from cc_steer.store import REBUILD_QUARANTINE_REASONS
from cc_steer.watcher.live import scrub_events

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from aiosqlite import Row
    from cc_transcript.mining import FeedbackCandidate

    from cc_steer.store import FeedbackStore

CONTEXTS_QUERY = """
SELECT dedup_key, session_id, event_uuid, occurred_at, context_json, quarantined_reason
FROM feedback_events
ORDER BY id
"""
GATE_SAMPLES_QUERY = """
SELECT sample_key, kind, dedup_key, window_json, seed
FROM gate_sample
WHERE dedup_key IS NOT NULL AND kind IN ('positive_window', 'hard_negative')
ORDER BY dedup_key, id
"""
ROWS_COUNT_QUERY = "SELECT COUNT(*) AS n FROM feedback_events"
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
class DetectorDrift:
    """One stored event whose anchor now detects under a different dedup key."""

    old_dedup_key: DedupKey
    new_dedup_key: DedupKey
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class ContextRebuildReport:
    """Counts from one idempotent stored-context rebuild pass."""

    found: int
    rebuilt: int
    quarantined: int
    rows_at_start: int
    rows_at_end: int
    gate_repaired: int = 0
    family_mismatches: int = 0
    drifted: tuple[DetectorDrift, ...] = ()
    parse_failures: tuple[CopyParseFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class CopyCandidates:
    mtime: float
    path: Path
    candidates: tuple[FeedbackCandidate, ...]
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


@dataclass(frozen=True, slots=True)
class GateSampleParent:
    dedup_key: DedupKey
    session_id: str
    event_uuid: str
    occurred_at: str | None
    context_json: str


@dataclass(frozen=True, slots=True)
class GateSampleRepairs:
    updates: tuple[tuple[str, str], ...]
    deletes: tuple[str, ...]
    inserts: tuple[GateSample, ...]

    @property
    def total(self) -> int:
        return len(self.updates) + len(self.deletes) + len(self.inserts)


type RebuildOutcome = RebuiltContext | QuarantinedContext | UnchangedContext | DetectorDrift


async def database_path(store: FeedbackStore) -> Path | None:
    async with store.store.conn.execute("PRAGMA database_list") as cur:
        rows = [dict(row) async for row in cur]
    file = next(str(row["file"]) for row in rows if row["name"] == "main")
    return Path(file) if file and file != ":memory:" else None


def rebuild_lock_path(database: Path | None) -> Path | None:
    return None if database is None or str(database) == ":memory:" else database.with_name(LOCK_FILENAME)


def acquire_rebuild_lock(lock: Path) -> int:
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = os.read(fd, 64).decode().strip() or "unknown"
        os.close(fd)
        raise RuntimeError(f"context rebuild already running; lock held by pid {holder} at {lock}") from None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


@asynccontextmanager
async def rebuild_lock(database: Path | None) -> AsyncIterator[None]:
    """Serializes context rebuilds for a file-backed database."""
    if (lock := rebuild_lock_path(database)) is None:
        yield
        return
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_rebuild_lock(lock)
    try:
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


async def parse_copy(
    session_id: SessionId, path: Path, semaphore: asyncio.Semaphore
) -> CopyCandidates | CopyParseFailure:
    async with semaphore:
        try:
            mtime = await TranscriptDiscovery.transcript_mtime(path)
            events = await parse_events_async(path)
            candidates = tuple(detect(scrub_events(events)))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            return CopyParseFailure(session_id, path, f"{type(exc).__name__}: {exc}")
    return CopyCandidates(
        mtime=mtime,
        path=path,
        candidates=candidates,
        by_dedup=dict(reversed([(str(candidate.dedup_key), candidate) for candidate in candidates])),
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


def best_candidate(matching: Sequence[tuple[float, FeedbackCandidate]]) -> FeedbackCandidate:
    return max(matching, key=lambda item: (before_content_length(item[1].window), item[0]))[1]


def quarantine_outcome(row: Row, dedup_key: DedupKey, reason: str) -> RebuildOutcome:
    return (
        UnchangedContext(dedup_key)
        if row["quarantined_reason"] == reason
        else QuarantinedContext(dedup_key, reason)
    )


def rebuild_outcome(
    row: Row, session: SessionCopies, *, ambiguous_event_uuid: bool
) -> RebuildOutcome:
    dedup_key = DedupKey(str(row["dedup_key"]))
    if row["quarantined_reason"] not in (None, *REBUILD_QUARANTINE_REASONS):
        return UnchangedContext(dedup_key)
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
        drifted = [
            (copy.mtime, candidate)
            for copy in session.copies
            for candidate in copy.candidates
            if str(candidate.ref.event_uuid) == str(row["event_uuid"])
        ]
        if not drifted:
            return quarantine_outcome(row, dedup_key, "anchor_not_found")
        if not has_substantive_content(messages(ContextWindow.from_json(str(row["context_json"])).before)):
            return quarantine_outcome(row, dedup_key, "rebuilt_context_empty")
        best_score = max(
            (before_content_length(candidate.window), mtime)
            for mtime, candidate in drifted
        )
        return DetectorDrift(
            dedup_key,
            best_candidate(drifted).dedup_key,
            ambiguous=ambiguous_event_uuid
            or len(
                {
                    candidate.dedup_key
                    for mtime, candidate in drifted
                    if (before_content_length(candidate.window), mtime) == best_score
                }
            )
            > 1,
        )
    window = best_candidate(matching).window
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
            case UnchangedContext() | DetectorDrift():
                pass
    return rebuilt, quarantined


def gate_sample_parent(row: Row, outcome: RebuildOutcome) -> GateSampleParent | None:
    match outcome:
        case QuarantinedContext():
            return None
        case RebuiltContext(context_json=context_json):
            pass
        case UnchangedContext() | DetectorDrift():
            if row["quarantined_reason"] is not None:
                return None
            context_json = str(row["context_json"])
    return GateSampleParent(
        dedup_key=DedupKey(str(row["dedup_key"])),
        session_id=str(row["session_id"]),
        event_uuid=str(row["event_uuid"]),
        occurred_at=str(row["occurred_at"]) if row["occurred_at"] is not None else None,
        context_json=context_json,
    )


def positive_gate_samples(parent: GateSampleParent, seed: int) -> tuple[GateSample, ...]:
    window = ContextWindow.from_json(parent.context_json)
    return tuple(
        GateSample(
            sample_key=f"pos:{parent.dedup_key}:{offset}",
            kind="positive_window",
            dedup_key=str(parent.dedup_key),
            session_id=parent.session_id,
            anchor_uuid=parent.event_uuid,
            occurred_at=parent.occurred_at,
            offset_turns=-offset,
            window_json=rewound.to_json(),
            seed=seed,
        )
        for offset in range(W_MAX)
        if (rewound := truncated(window, offset)) is not None
    )


def gate_sample_repairs(
    parents: Mapping[str, GateSampleParent], rows: Sequence[Row]
) -> GateSampleRepairs:
    updates: list[tuple[str, str]] = []
    deletes: list[str] = []
    inserts: list[GateSample] = []
    for dedup_key, grouped in groupby(rows, key=lambda row: str(row["dedup_key"])):
        if (parent := parents.get(dedup_key)) is None:
            continue
        stored = tuple(grouped)
        positive = tuple(row for row in stored if row["kind"] == "positive_window")
        if positive:
            expected = {
                sample.sample_key: sample for sample in positive_gate_samples(parent, int(positive[0]["seed"]))
            }
            current = {str(row["sample_key"]): row for row in positive}
            updates.extend(
                (sample_key, sample.window_json)
                for sample_key, sample in expected.items()
                if sample_key in current and current[sample_key]["window_json"] != sample.window_json
            )
            deletes.extend(sample_key for sample_key in current if sample_key not in expected)
            inserts.extend(sample for sample_key, sample in expected.items() if sample_key not in current)
        updates.extend(
            (str(row["sample_key"]), parent.context_json)
            for row in stored
            if row["kind"] == "hard_negative" and row["window_json"] != parent.context_json
        )
    return GateSampleRepairs(tuple(updates), tuple(deletes), tuple(inserts))


async def repair_gate_samples(
    store: FeedbackStore, parents: Mapping[str, GateSampleParent], *, dry_run: bool
) -> int:
    """Reconciles stored gate families with their active parents' effective contexts."""
    if dry_run:
        rows = [row async for row in await store.store.conn.execute(GATE_SAMPLES_QUERY)]
        return gate_sample_repairs(parents, rows).total
    return await store.repair_gate_samples(
        GATE_SAMPLES_QUERY,
        lambda rows: gate_sample_repairs(parents, rows),
    )


async def execute_context_rebuild(
    store: FeedbackStore, roots: Sequence[Path], *, dry_run: bool
) -> ContextRebuildReport:
    rows = [row async for row in await store.store.conn.execute(CONTEXTS_QUERY)]
    rows_by_session = {
        SessionId(session_id): list(group)
        for session_id, group in groupby(
            sorted(rows, key=lambda row: str(row["session_id"])),
            key=lambda row: str(row["session_id"]),
        )
    }
    event_uuid_counts = Counter(str(row["event_uuid"]) for row in rows)
    session_ids = tuple(rows_by_session)
    semaphore = asyncio.Semaphore(PARSE_CONCURRENCY)
    rebuilt = 0
    quarantined = 0
    failures: list[CopyParseFailure] = []
    resolved: list[tuple[Row, RebuildOutcome]] = []
    for offset in range(0, len(session_ids), SESSION_BATCH_SIZE):
        batch = session_ids[offset : offset + SESSION_BATCH_SIZE]
        parsed = dict(
            await asyncio.gather(*(load_session_copies(session_id, roots, semaphore) for session_id in batch))
        )
        failures.extend(failure for copies in parsed.values() for failure in copies.failures)
        batch_resolved = [
            (
                row,
                rebuild_outcome(
                    row,
                    parsed[session_id],
                    ambiguous_event_uuid=event_uuid_counts[str(row["event_uuid"])] > 1,
                ),
            )
            for session_id in batch
            for row in rows_by_session[session_id]
        ]
        changes = await store.rebuild_context(
            *persistence_rows([outcome for _, outcome in batch_resolved]),
            dry_run=dry_run,
        )
        rebuilt += changes.rebuilt
        quarantined += changes.quarantined
        resolved.extend(batch_resolved)
    parents = {
        str(parent.dedup_key): parent
        for row, outcome in resolved
        if (parent := gate_sample_parent(row, outcome)) is not None
    }
    gate_repaired = await repair_gate_samples(store, parents, dry_run=dry_run)
    mismatch_keys = await store.gate_sample_family_mismatch_keys()
    [count_row] = await (await store.store.conn.execute(ROWS_COUNT_QUERY)).fetchall()
    return ContextRebuildReport(
        found=len(rows),
        rebuilt=rebuilt,
        quarantined=quarantined,
        rows_at_start=len(rows),
        rows_at_end=int(count_row["n"]),
        gate_repaired=gate_repaired,
        family_mismatches=len(mismatch_keys & parents.keys()),
        drifted=tuple(outcome for _, outcome in resolved if isinstance(outcome, DetectorDrift)),
        parse_failures=tuple(failures),
    )


async def rebuild_contexts(
    store: FeedbackStore,
    roots: Sequence[Path],
    *,
    dry_run: bool = False,
    acquire_lock: bool = True,
) -> ContextRebuildReport:
    """Rebuilds all stored contexts from exact session transcripts and quarantines failures.

    Args:
        store: The open feedback store to inspect and repair.
        roots: Transcript roots searched for every stored session.
        dry_run: Compute the complete report without modifying the database.
        acquire_lock: Acquire the database's rebuild lock. The CLI disables this
            only after acquiring the same lock before opening the store.

    Returns:
        Counts and diagnostics for the rebuild pass.
    """
    if not acquire_lock:
        return await execute_context_rebuild(store, roots, dry_run=dry_run)
    async with rebuild_lock(await database_path(store)):
        return await execute_context_rebuild(store, roots, dry_run=dry_run)
