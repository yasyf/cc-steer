"""Stage 4 of the pipeline: ground each refined pair in the code evidence behind it.

The refine stage distills accepted steering into atomic direction pairs; this stage
links each pair to concrete code evidence. It hands each pair's steering anchor and
direction text to cc-transcript's shared correction extractor
(:func:`cc_transcript.extract.extract_correction`), which harvests the candidate
incorrect edits — and the corrections that later overwrote them, from the same
session or from git history — picks the one edit the direction faults, and appends
it to the shared ``corrections`` ledger (``~/.cc-transcript/corrections.db``) for
every consumer to join. The extractor is idempotent per anchor, so pairs that share
one anchor produce a single row. The dashboard reads its evidence straight from that
ledger; this stage no longer keeps a local evidence table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from cc_transcript.activity import SessionActivity
from cc_transcript.corrections import CorrectionLog
from cc_transcript.discovery import TranscriptExpiredError
from cc_transcript.extract import extract_correction, usable_backend
from cc_transcript.filterspec import event_meta
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.parser import parse

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cc_transcript.activity import Turn
    from spawnllm import LlmBackend, TModel

    from cc_steer.store import FeedbackStore

SOURCE = "cc-steer"
MAX_CONSECUTIVE_FAILURES = 5
"""Consecutive per-pair failures that abort a pass.

Refine isolates every failed row and never caps; enrich adds this circuit breaker
because a streak this long means the backend is down, not that one transcript is
bad — grinding the backlog against a dead dependency only burns the retry budget.
Five tolerates a transient blip yet trips well inside one concurrency window.
"""


class EnrichError(RuntimeError):
    """A provider or transport failure resolving one refined pair's correction.

    The extraction boundary (:func:`resolve_pair`) converts the backend, timeout,
    and response-validation failures the shared extractor's LLM call surfaces into
    this one type, so the fan-out isolates them exactly as ``run_verdicts`` isolates
    :class:`~cc_transcript.judge.verdicts.JudgeError` for triage and refine.
    :func:`run_enrichments` catches exactly this: the pair is counted failed and left
    unpersisted so the next pass retries it, and :data:`MAX_CONSECUTIVE_FAILURES` in
    a row aborts the pass loudly. Anything else raised while resolving a pair is a
    programming error and propagates.
    """


@dataclass(frozen=True, slots=True)
class EnrichReport:
    """The outcome of one enrich pass.

    Attributes:
        corrections: How many corrections landed in the shared ``corrections`` ledger.
        skipped: How many pairs resolved without a correction (no anchor, expired
            transcript, compacted anchor, editless window, or no faulted edit).
        failed: How many pairs failed on a provider or transport error and stay
            pending, isolated so the pass finishes and the next run retries them.
        pending: How many refined pairs remain without a ledger correction.
        enriched: How many pairs resolved this pass, derived as ``corrections + skipped``.
    """

    corrections: int
    skipped: int
    failed: int
    pending: int

    @property
    def enriched(self) -> int:
        return self.corrections + self.skipped


def repo_of(turn: Turn, anchor: EventRef) -> Path | None:
    return next(
        (
            Path(meta.cwd)
            for event in turn.events
            if (meta := event_meta(event)) is not None and meta.uuid == anchor.event_uuid and meta.cwd
        ),
        None,
    )


async def load_activity(session_id: SessionId, origin_path: object) -> SessionActivity:
    """Builds the session activity, preferring the exact transcript the event was mined from.

    Re-discovery by session id only searches the default projects root, so a corpus
    mined from a mirror or any non-default ``--transcripts`` root would resolve as
    expired. The stored ``origin_path`` names the file directly; discovery is the
    fallback for events whose original file is gone.
    """
    if origin_path is not None and (path := Path(str(origin_path))).exists():
        return SessionActivity.from_events(session_id, parse(path).events)
    return SessionActivity.from_session(session_id)


async def resolve_pair(
    row: Mapping[str, object], log: CorrectionLog, *, tier: TModel, backend: LlmBackend | None
) -> bool:
    """Extracts and appends the correction one refined pair's direction faults.

    Returns True when a correction was appended, False when the pair resolves to no
    correction (no anchor, expired transcript, compacted anchor, editless window, or
    no faulted edit). The extractor is idempotent per anchor, so a pair sharing an
    anchor with one already in the ledger resolves to False without a write.

    Raises:
        EnrichError: When the shared extractor's LLM call fails on a provider or
            transport error, so :func:`run_enrichments` can isolate and retry it.
    """
    import json

    from pydantic import ValidationError
    from spawnllm import BackendCallError, BackendUnavailable

    match row["session_id"], row["event_uuid"]:
        case (None, _) | (_, None):
            return False
        case (session_id, event_uuid):
            anchor = EventRef(SessionId(str(session_id)), EventUuid(str(event_uuid)))
    try:
        activity = await load_activity(anchor.session_id, row["origin_path"])
    except TranscriptExpiredError:
        return False
    if (turn := activity.turn_of(anchor)) is None:
        return False
    try:
        correction = await extract_correction(
            log,
            activity,
            anchor,
            source=SOURCE,
            feedback=str(row["direction_verbatim"]),
            repo=repo_of(turn, anchor),
            tier=tier,
            backend=backend,
        )
    except (BackendCallError, BackendUnavailable, TimeoutError, ValidationError, json.JSONDecodeError) as error:
        raise EnrichError(str(error)) from error
    return correction is not None


async def run_enrichments(
    rows: Sequence[Mapping[str, object]],
    *,
    tier: TModel,
    concurrency: int,
    log: CorrectionLog,
    backend: LlmBackend | None,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> tuple[int, int, int]:
    counts = {"corrections": 0, "skipped": 0, "failed": 0}
    consecutive = 0
    limiter = anyio.CapacityLimiter(concurrency)

    async def worker(row: Mapping[str, object]) -> None:
        nonlocal consecutive
        async with limiter:
            try:
                resolved = await resolve_pair(row, log, tier=tier, backend=backend)
            except EnrichError:
                counts["failed"] += 1
                if (consecutive := consecutive + 1) >= max_consecutive_failures:
                    tg.cancel_scope.cancel()
                return
            counts["corrections" if resolved else "skipped"] += 1
            consecutive = 0

    async with anyio.create_task_group() as tg:
        for row in rows:
            tg.start_soon(worker, row)
    if tg.cancel_scope.cancel_called:
        raise EnrichError(f"enrich aborted after {max_consecutive_failures} consecutive pair failures")
    return counts["corrections"], counts["skipped"], counts["failed"]


async def enrich(
    store: FeedbackStore,
    *,
    tier: TModel = "medium",
    limit: int | None = None,
    concurrency: int = 8,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> EnrichReport:
    """Grounds every refined pair lacking a shared-ledger correction in the edit it faults.

    Hands each pair's anchor and direction to cc-transcript's shared extractor,
    which harvests the candidate edits, picks the one the direction faults — an LLM
    call when a backend is ready, the best-overlap candidate otherwise — and appends
    it to the shared ``corrections`` ledger. Incremental and idempotent: the ledger
    is the single source of truth for "done", so a pair settles once its anchor
    carries a row, the extractor never duplicates a shared anchor, and a refine
    re-run resurfaces its new pairs here automatically. Pairs that resolve to no
    correction (no anchor, expired transcript, editless window, or no faulted edit)
    cost no LLM call. A pair whose LLM call fails on a provider or transport error is
    isolated — counted failed and left unpersisted so the next pass retries it — so
    one bad pair never aborts the rest; only ``max_consecutive_failures`` failures in
    a row abort the pass loudly. Corrections already appended to the ledger persist,
    so a re-run resumes idempotently.

    Args:
        store: The open feedback store.
        tier: The linking model's abstract tier when a backend is ready.
        limit: When set, enrich at most this many pairs this pass.
        concurrency: The maximum number of concurrent extractions.
        max_consecutive_failures: How many pairs may fail in a row before the pass
            aborts, distinguishing one bad transcript from a downed backend.

    Returns:
        The pass's corrections/skipped/failed/pending counts.
    """
    log = await CorrectionLog.open()
    rows = await store.unenriched(log, limit=limit)
    corrections, skipped, failed = await run_enrichments(
        rows,
        tier=tier,
        concurrency=concurrency,
        log=log,
        backend=usable_backend(),
        max_consecutive_failures=max_consecutive_failures,
    )
    return EnrichReport(
        corrections=corrections, skipped=skipped, failed=failed, pending=len(await store.unenriched(log))
    )
