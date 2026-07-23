"""The cc-factory import contract: decision outcomes ingested as judged-pair candidates.

cc-factory writes its accept/reject/undo decisions back into cc-steer as the
flywheel's next round of raw material. Each decision lands as an ordinary feedback
candidate — same ``feedback_events`` shape :mod:`cc_steer.sidecar` produces — so it
flows through the identical triage judge → auditor → refiner pipeline as a natively
mined steer. The imported label is never trusted as a verdict: the LLM gate re-judges
every row, which is the poison filter that keeps a bad upstream decision out of the
training set. Two additive provenance columns, ``import_source`` and ``import_batch``,
keep imported rows distinguishable from mined ones forever, and a content-derived
dedup key keyed on ``(source, external_id)`` makes re-importing the same batch a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.mining import FeedbackCandidate, dedup_key, event_row, firm
from cc_transcript.mining.sourcekind import SourceKind
from cc_transcript.mining.store import now
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cc_transcript.mining import DedupKey

    from cc_steer.store import FeedbackStore

SCHEMA_ID = "cc-steer/import@1"
IMPORT_SOURCE_KIND = SourceKind("import")
SESSION_PREFIX = "cc-steer-import-"
IMPORT_PREVIEW_CHARS = 8_000
IN_CHUNK = 500

INSERT_IMPORT_EVENT = """
INSERT OR IGNORE INTO feedback_events (
  dedup_key, source_kind, session_id, event_uuid,
  occurred_at, text, payload_json, context_json, cc_version, ingested_at,
  import_source, import_batch
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

type ImportLabel = Literal["accepted", "rejected", "undone"]
type ImportStatus = Literal["new", "duplicate"]

__all__ = [
    "SCHEMA_ID",
    "ImportBatch",
    "ImportItem",
    "ImportOutcome",
    "ImportResult",
    "import_batch",
]


class ImportItem(BaseModel):
    """One cc-factory decision outcome offered as a candidate.

    Attributes:
        external_id: The decision's stable id in the source system; half of the
            dedup key that makes re-import idempotent.
        occurred_at: When the decision was made.
        repo: The repository the decision concerned, or ``None``.
        context: What the agent did — the model-visible turn the decision reacts to.
        verbatim: The steer text the judge weighs, exactly as written.
        label: The upstream outcome, kept as provenance and never trusted as a verdict.
    """

    model_config = ConfigDict(extra="forbid")

    external_id: str
    occurred_at: datetime
    repo: str | None = None
    context: str
    verbatim: str
    label: ImportLabel


class ImportBatch(BaseModel):
    """A versioned batch of cc-factory decision outcomes.

    Attributes:
        schema_: The literal schema tag ``cc-steer/import@1``, read from the
            ``schema`` JSON key.
        source: The upstream system, e.g. ``cc-factory``; half of every item's
            dedup key and the value stored in each row's ``import_source`` column.
        items: The decisions to ingest.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_: Literal["cc-steer/import@1"] = Field(alias="schema")
    source: str
    items: tuple[ImportItem, ...]


@dataclass(frozen=True, slots=True)
class ImportOutcome:
    """The disposition of one imported item.

    Attributes:
        external_id: The item's source id.
        dedup_key: The content-derived key the row was inserted under.
        status: ``'new'`` when the row would land, ``'duplicate'`` when its dedup
            key was already present.
    """

    external_id: str
    dedup_key: DedupKey
    status: ImportStatus


@dataclass(frozen=True, slots=True)
class ImportResult:
    """The outcome of an :func:`import_batch` pass.

    Attributes:
        source: The batch's upstream system.
        batch: The batch's content digest, also stored in each new row's
            ``import_batch`` column.
        dry_run: Whether the pass wrote anything.
        outcomes: One disposition per item, in batch order.

    Example:
        >>> result = await import_batch(path, db=store)
        >>> print(result.summary_line())
    """

    source: str
    batch: str
    dry_run: bool
    outcomes: tuple[ImportOutcome, ...]

    @property
    def new(self) -> tuple[ImportOutcome, ...]:
        """The items that would land (dry run) or did (live)."""
        return tuple(outcome for outcome in self.outcomes if outcome.status == "new")

    @property
    def duplicates(self) -> tuple[ImportOutcome, ...]:
        """The items skipped because their dedup key already existed."""
        return tuple(outcome for outcome in self.outcomes if outcome.status == "duplicate")

    def summary_line(self) -> str:
        """A one-line report of new versus skipped counts."""
        verb = "would import" if self.dry_run else "imported"
        return (
            f"{verb} {len(self.new)} new, skipped {len(self.duplicates)} duplicate "
            f"from {self.source} (batch {self.batch[:12]})"
        )


def load_batch(batch: ImportBatch | Path) -> ImportBatch:
    match batch:
        case ImportBatch():
            return batch
        case _:
            return ImportBatch.model_validate_json(batch.read_text())


def batch_digest(batch: ImportBatch) -> str:
    return dedup_key(SCHEMA_ID, batch.source, *sorted(item.external_id for item in batch.items))


def import_window(context: str, verbatim: str, ref: EventRef) -> ContextWindow:
    return ContextWindow(
        anchor=ref,
        before=(TurnRef(role="assistant", refs=(ref,), preview=context, tool_digests=()),),
        trigger=TurnRef(role="user", refs=(ref,), preview=verbatim, tool_digests=()),
        after=(),
        fidelity="summary",
        preview_chars=IMPORT_PREVIEW_CHARS,
    )


def to_candidate(source: str, item: ImportItem) -> FeedbackCandidate:
    key = dedup_key(source, item.external_id)
    ref = EventRef(SessionId(f"{SESSION_PREFIX}{key[:32]}"), EventUuid(key))
    return FeedbackCandidate(
        dedup_key=key,
        source_kind=IMPORT_SOURCE_KIND,
        occurred_at=item.occurred_at,
        text=item.verbatim,
        window=import_window(item.context, item.verbatim, ref),
        ref=ref,
        signal=firm("cc_factory_import", item.label),
        session_id=ref.session_id,
        payload={
            "format": "cc_steer_import",
            "import_source": source,
            "external_id": item.external_id,
            "label": item.label,
            "repo": item.repo,
            "provenance": "imported",
        },
    )


def classify(candidates: Sequence[FeedbackCandidate], existing: set[str]) -> list[ImportStatus]:
    seen = set(existing)
    statuses: list[ImportStatus] = []
    for candidate in candidates:
        statuses.append("duplicate" if candidate.dedup_key in seen else "new")
        seen.add(candidate.dedup_key)
    return statuses


async def existing_keys(db: FeedbackStore, keys: Sequence[DedupKey]) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(keys), IN_CHUNK):
        chunk = keys[start : start + IN_CHUNK]
        rows = await db.sql(
            f"SELECT dedup_key FROM feedback_events WHERE dedup_key IN ({','.join('?' * len(chunk))})",
            tuple(chunk),
        )
        found |= {str(row["dedup_key"]) for row in rows}
    return found


async def import_batch(batch: ImportBatch | Path, *, db: FeedbackStore, dry_run: bool = False) -> ImportResult:
    """Ingests a cc-factory decision batch as judged-pair candidates.

    Every item becomes a summary-fidelity feedback candidate carrying the agent's
    action as its context and the steer as its trigger, inserted with
    ``INSERT OR IGNORE`` keyed by ``dedup_key(source, external_id)`` — so a
    re-import of the same batch changes nothing. The imported label rides in the
    candidate payload as provenance; it is never written as a triage verdict or a
    refined pair, so the LLM gate re-judges every row. The batch's content digest
    and its ``source`` land in the additive ``import_batch`` and ``import_source``
    columns, keeping imported rows distinguishable from mined ones.

    Args:
        batch: A parsed :class:`ImportBatch` or a path to its JSON file.
        db: The open feedback store to ingest into.
        dry_run: When True, classify every item without writing anything.

    Returns:
        The per-item dispositions and new/duplicate counts.
    """
    parsed = load_batch(batch)
    digest = batch_digest(parsed)
    candidates = [to_candidate(parsed.source, item) for item in parsed.items]
    statuses = classify(candidates, await existing_keys(db, [candidate.dedup_key for candidate in candidates]))
    result = ImportResult(
        source=parsed.source,
        batch=digest,
        dry_run=dry_run,
        outcomes=tuple(
            ImportOutcome(item.external_id, candidate.dedup_key, status)
            for item, candidate, status in zip(parsed.items, candidates, statuses, strict=True)
        ),
    )
    if dry_run:
        return result
    ingested_at = now()
    async with db.db.transaction() as conn:
        await conn.executemany(
            INSERT_IMPORT_EVENT,
            [
                (*event_row(candidate, ingested_at), parsed.source, digest)
                for candidate, status in zip(candidates, statuses, strict=True)
                if status == "new"
            ],
        )
    return result
