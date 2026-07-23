"""The SQLite feedback store: the platform's mining store plus cc-steer's triage layer.

The store mechanism lives in :mod:`cc_transcript.mining`; this module composes it —
holding a :class:`~cc_transcript.mining.FeedbackStore` configured by cc-steer's
:data:`STEER_SCHEMA` — and adds cc-steer's default database location, the
``origin_path`` display hint and ``quarantined_reason`` columns, the ``triage``
verdict table, the ``refinement`` table, and two views: ``accepted_steering``
(judge-accepted candidates, the refine stage's input) and ``refined_pairs`` — the
pipeline's final deliverable. The enrich stage's code evidence no longer lives here:
it lands in cc-transcript's shared ``corrections`` ledger, keyed by the steering
anchor, and the dashboard reads it straight from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

from cc_transcript.context import ContextWindow
from cc_transcript.ids import EventUuid, SessionId
from cc_transcript.mining import FeedbackStore as BaseFeedbackStore
from cc_transcript.mining import Stats, StoreSchema, event_row
from cc_transcript.mining.store import now

from cc_steer.rendering import has_substantive_content, has_substantive_gate_content, messages

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import TracebackType

    from cc_transcript.corrections import CorrectionLog
    from cc_transcript.mining import DedupKey, FeedbackCandidate, Fidelity, SourceKind, VerdictLike

    from cc_steer.context_rebuild import GatePruneClassification, GateSampleRepairs
    from cc_steer.negatives import GateSample
    from cc_steer.refine import Refinement

__all__ = [
    "ACCRUED_EMPTY_REASON",
    "ContextRebuildChanges",
    "FeedbackStore",
    "STEER_SCHEMA",
    "STEER_SCHEMA_DDL",
    "Stats",
    "TriageStats",
    "event_row",
]

BUSY_TIMEOUT_MS = 2_000
ACCRUED_EMPTY_REASON = "accrued_context_empty"
REBUILD_QUARANTINE_REASONS = (
    ACCRUED_EMPTY_REASON,
    "transcript_not_found",
    "transcript_parse_failed",
    "anchor_not_found",
    "rebuilt_context_empty",
)

# The feedback_events columns cc-steer's candidate and lineage reads project.
EVENT_COLUMNS = (
    "e.id, e.dedup_key, e.source_kind, e.occurred_at, e.text, "
    "e.payload_json, e.context_json, e.session_id, e.event_uuid"
)

QUARANTINE_ELIGIBLE = (
    "(quarantined_reason IS NULL OR quarantined_reason IN ("
    + ", ".join(f"'{reason}'" for reason in REBUILD_QUARANTINE_REASONS)
    + "))"
)

UPDATE_CONTEXT = f"""
UPDATE feedback_events SET context_json = ?, quarantined_reason = NULL
WHERE dedup_key = ? AND {QUARANTINE_ELIGIBLE}
"""

QUARANTINE_CONTEXT = f"""
UPDATE feedback_events SET quarantined_reason = ?
WHERE dedup_key = ? AND {QUARANTINE_ELIGIBLE}
"""

STEER_SCHEMA_DDL = """CREATE TABLE files (
  path TEXT PRIMARY KEY,
  mtime REAL NOT NULL
);
CREATE TABLE feedback_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT NOT NULL UNIQUE,
  source_kind TEXT NOT NULL,
  session_id TEXT,
  event_uuid TEXT,
  occurred_at TEXT NOT NULL,
  text TEXT NOT NULL,
  payload_json TEXT,
  context_json TEXT NOT NULL,
  cc_version TEXT,
  ingested_at TEXT NOT NULL,
  origin_path TEXT,
  quarantined_reason TEXT,
  import_source TEXT,
  import_batch TEXT
);
CREATE INDEX idx_feedback_source ON feedback_events(source_kind);
CREATE INDEX idx_feedback_session ON feedback_events(session_id);
CREATE TABLE triage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT NOT NULL REFERENCES feedback_events(dedup_key),
  role TEXT NOT NULL,
  prompt_version INTEGER NOT NULL,
  model TEXT NOT NULL,
  category TEXT NOT NULL,
  is_steering INTEGER NOT NULL,
  what_claude_did TEXT NOT NULL,
  confidence REAL NOT NULL,
  rationale TEXT NOT NULL,
  canonical_key TEXT,
  fidelity TEXT NOT NULL CHECK(fidelity IN ('full','summary')),
  judged_at TEXT NOT NULL,
  UNIQUE(dedup_key, role, prompt_version)
);
CREATE INDEX idx_triage_dedup ON triage(dedup_key);
CREATE VIEW latest_judge AS
SELECT * FROM (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'judge'
) WHERE rn = 1;
CREATE VIEW latest_auditor AS
SELECT * FROM (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'auditor'
) WHERE rn = 1;
CREATE VIEW accepted_steering AS
SELECT
  e.id AS event_id,
  e.dedup_key,
  e.source_kind,
  e.text,
  e.context_json,
  e.payload_json,
  t.category,
  t.what_claude_did,
  e.origin_path
FROM feedback_events e
JOIN latest_judge t ON t.dedup_key = e.dedup_key
WHERE t.is_steering = 1 AND e.quarantined_reason IS NULL;
CREATE TABLE refinement (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT NOT NULL REFERENCES feedback_events(dedup_key),
  prompt_version INTEGER NOT NULL,
  model TEXT NOT NULL,
  pair_index INTEGER NOT NULL,
  action TEXT NOT NULL,
  direction_verbatim TEXT NOT NULL,
  direction TEXT NOT NULL,
  refined_at TEXT NOT NULL,
  UNIQUE(dedup_key, prompt_version, model, pair_index)
);
CREATE INDEX idx_refinement_dedup ON refinement(dedup_key);
CREATE VIEW latest_refinement AS
WITH gens AS (
  SELECT dedup_key, prompt_version, model, refined_at,
    ROW_NUMBER() OVER (
      PARTITION BY dedup_key ORDER BY prompt_version DESC, refined_at DESC
    ) AS g
  FROM (SELECT DISTINCT dedup_key, prompt_version, model, refined_at FROM refinement)
)
SELECT r.*
FROM refinement r
JOIN gens ON gens.dedup_key = r.dedup_key AND gens.prompt_version = r.prompt_version
         AND gens.model = r.model AND gens.refined_at = r.refined_at AND gens.g = 1;
CREATE VIEW refined_pairs AS
SELECT
  e.id AS event_id,
  r.dedup_key,
  r.pair_index,
  r.action,
  r.direction_verbatim,
  r.direction,
  e.text AS original_message,
  ap.category,
  e.source_kind,
  e.session_id,
  e.event_uuid,
  e.occurred_at,
  e.origin_path,
  r.prompt_version,
  r.model
FROM latest_refinement r
JOIN feedback_events e ON e.dedup_key = r.dedup_key
JOIN accepted_steering ap ON ap.dedup_key = r.dedup_key
ORDER BY e.id, r.pair_index;
CREATE TABLE gate_sample (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sample_key TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  dedup_key TEXT,
  session_id TEXT NOT NULL,
  anchor_uuid TEXT NOT NULL,
  occurred_at TEXT,
  offset_turns INTEGER NOT NULL DEFAULT 0,
  window_json TEXT NOT NULL,
  seed INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_gate_sample_kind ON gate_sample(kind);
CREATE INDEX idx_gate_sample_session ON gate_sample(session_id);
CREATE TABLE sampled_session (
  session_id TEXT PRIMARY KEY,
  sampled_at TEXT NOT NULL
);
CREATE TABLE exemplar_embedding (
  dedup_key TEXT NOT NULL REFERENCES feedback_events(dedup_key),
  model TEXT NOT NULL,
  text_digest TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(dedup_key, model)
);
CREATE VIRTUAL TABLE evidence_fts USING fts5(
  verbatim, direction, evidence,
  category UNINDEXED, repo UNINDEXED, source UNINDEXED
);
CREATE TABLE evidence_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

INSERT_REFINEMENT = """
INSERT OR IGNORE INTO refinement (
  dedup_key, prompt_version, model, pair_index, action, direction_verbatim, direction, refined_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_GATE_SAMPLE = """
INSERT OR IGNORE INTO gate_sample (
  sample_key, kind, dedup_key, session_id, anchor_uuid, occurred_at, offset_turns, window_json, seed, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_GATE_SAMPLE_WINDOW = """
UPDATE gate_sample SET window_json = ?
WHERE sample_key = ? AND window_json != ?
"""

DELETE_GATE_SAMPLE = "DELETE FROM gate_sample WHERE sample_key = ?"

INSERT_SAMPLED_SESSION = "INSERT OR IGNORE INTO sampled_session (session_id, sampled_at) VALUES (?, ?)"

GATE_FAMILY_MISMATCH_QUERY = """
SELECT DISTINCT g.dedup_key
FROM gate_sample g
JOIN latest_judge j ON j.dedup_key = g.dedup_key
WHERE (j.is_steering = 1 AND g.kind = 'hard_negative')
   OR (j.is_steering = 0 AND g.kind = 'positive_window')
"""

INSERT_EMBEDDING = """
INSERT INTO exemplar_embedding (dedup_key, model, text_digest, dim, vector, created_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(dedup_key, model) DO UPDATE SET
  text_digest = excluded.text_digest, dim = excluded.dim,
  vector = excluded.vector, created_at = excluded.created_at
"""

CANDIDATES_QUERY = f"""
WITH judge_flip AS (
  SELECT dedup_key, COUNT(DISTINCT is_steering) > 1 AS flipped
  FROM triage WHERE role = 'judge' GROUP BY dedup_key
),
refine_summary AS (
  SELECT dedup_key, COUNT(*) AS pair_count,
    MAX(prompt_version) AS refine_version, MAX(model) AS refine_model
  FROM latest_refinement GROUP BY dedup_key
)
SELECT {EVENT_COLUMNS}, e.origin_path,
  j.category, j.is_steering, j.confidence, j.prompt_version AS judge_version,
  j.model AS judge_model, j.what_claude_did,
  a.is_steering AS auditor_is_steering,
  COALESCE(f.flipped, 0) AS flipped,
  rs.pair_count, rs.refine_version, rs.refine_model
FROM feedback_events e
LEFT JOIN latest_judge j ON j.dedup_key = e.dedup_key
  LEFT JOIN latest_auditor a ON a.dedup_key = e.dedup_key
  LEFT JOIN judge_flip f ON f.dedup_key = e.dedup_key
  LEFT JOIN refine_summary rs ON rs.dedup_key = e.dedup_key
  WHERE e.quarantined_reason IS NULL
  ORDER BY e.id
  """

LINEAGE_VERDICTS_QUERY = (
    "SELECT role, prompt_version, model, category, is_steering, what_claude_did, "
    "confidence, rationale, judged_at FROM triage WHERE dedup_key = ? ORDER BY role, prompt_version, id"
)

LINEAGE_PAIRS_QUERY = """
SELECT r.pair_index, r.action, r.direction_verbatim, r.direction, r.prompt_version, r.model,
  e.session_id, e.event_uuid
FROM latest_refinement r
JOIN feedback_events e ON e.dedup_key = r.dedup_key
WHERE r.dedup_key = ?
ORDER BY r.pair_index
"""

REFINED_PAIRS_QUERY = """
SELECT dedup_key, prompt_version AS refine_version, model AS refine_model, pair_index,
  action, direction, direction_verbatim, source_kind, session_id, event_uuid, origin_path
FROM refined_pairs
ORDER BY event_id, pair_index"""

STEER_SCHEMA = StoreSchema(
    identity="cc-steer-feedback",
    ddl=STEER_SCHEMA_DDL,
    event_columns=("origin_path", "quarantined_reason"),
    verdict_table="triage",
    accepted_column="is_steering",
    summary_column="what_claude_did",
    event_filter="e.quarantined_reason IS NULL",
)


@dataclass(frozen=True, slots=True)
class TriageStats:
    """A snapshot of triage progress at one prompt version.

    Attributes:
        total: The total feedback events in the corpus.
        judged: How many carry a judge verdict at this prompt version.
        accepted: How many of those verdicts are steering.
        by_category: Verdict counts keyed by category.
    """

    total: int
    judged: int
    accepted: int
    by_category: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ContextRebuildChanges:
    """Counts of context repairs and quarantines persisted by one rebuild."""

    rebuilt: int
    quarantined: int


def gate_sample_row(sample: GateSample, created_at: str) -> tuple[object, ...]:
    return (
        sample.sample_key,
        sample.kind,
        sample.dedup_key,
        sample.session_id,
        sample.anchor_uuid,
        sample.occurred_at,
        sample.offset_turns,
        sample.window_json,
        sample.seed,
        created_at,
    )


class FeedbackStore:
    """Persistent store for collected feedback over the native mining engine.

    Composes cc-transcript's :class:`~cc_transcript.mining.FeedbackStore`, configured
    by :data:`STEER_SCHEMA`: the ``feedback_events`` table (extended with the
    ``origin_path`` provenance and ``quarantined_reason`` columns), the ``triage``
    verdict table (the engine's verdict tier pinned to cc-steer's column names), the
    ``refinement`` table, and the ``accepted_steering`` and ``refined_pairs`` views.
    Verdicts and refinements key on the content-derived dedup key, so they survive a
    database rebuild; the enrich stage's code evidence lives in cc-transcript's shared
    ``corrections`` ledger, keyed by the steering anchor.

    Example:
        >>> async with await FeedbackStore.open(FeedbackStore.default_path()) as store:
        ...     await store.record_file_scan(str(path), mtime, candidates)
    """

    def __init__(self, db: BaseFeedbackStore) -> None:
        self.db = db

    @staticmethod
    def default_path() -> Path:
        """Returns the default database path, ``~/.cc-steer/feedback.db``."""
        return Path.home() / ".cc-steer" / "feedback.db"

    @classmethod
    async def open(cls, path: Path) -> Self:
        """Opens (creating if needed) the feedback database at ``path``."""
        return cls(
            await BaseFeedbackStore.open(
                path,
                schema=STEER_SCHEMA,
                extensions=(),
                busy_timeout_ms=BUSY_TIMEOUT_MS,
            )
        )

    @classmethod
    async def open_readonly(cls, path: Path) -> Self:
        """Opens an existing feedback database without schema or data writes."""
        return cls(
            await BaseFeedbackStore.open(
                path,
                schema=STEER_SCHEMA,
                extensions=(),
                readonly=True,
                busy_timeout_ms=BUSY_TIMEOUT_MS,
            )
        )

    async def close(self) -> None:
        """Closes the underlying connection; a second close is a no-op."""
        await self.db.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    # --- primitive delegators ------------------------------------------------
    async def sql(self, statement: str, params: Sequence[object] = ()) -> list[dict[str, object]]:
        """Runs one parameterized statement, returning rows as dicts."""
        return await self.db.sql(statement, params)

    async def execute(self, statement: str, params: Sequence[object] = ()) -> int:
        """Runs one parameterized write statement, returning the modified-row count."""
        return await self.db.execute(statement, params)

    async def executemany(self, statement: str, seq: Sequence[Sequence[object]]) -> int:
        """Runs ``statement`` once per parameter set, returning the total modified-row count."""
        return await self.db.executemany(statement, seq)

    # --- corpus reads (base) -------------------------------------------------
    async def file_mtimes(self) -> dict[str, float]:
        """Returns the recorded ``path`` to ``mtime`` map for incremental scans."""
        return await self.db.file_mtimes()

    async def dedup_keys(self) -> set[str]:
        """Returns every stored event's dedup key."""
        return await self.db.dedup_keys()

    async def stats(self) -> Stats:
        """Returns ingestion counts by source kind and the scanned-file count."""
        return await self.db.stats()

    async def recent(self, *, source_kind: SourceKind | None = None, limit: int = 20) -> list[dict[str, object]]:
        """Returns the most recent feedback events, newest first."""
        return await self.db.recent(source_kind=source_kind, limit=limit)

    async def events(self, *, source_kind: SourceKind | None = None) -> list[dict[str, object]]:
        """Returns every feedback event, newest first, with the columns needed to render it."""
        return await self.db.events(source_kind=source_kind)

    # --- ingestion -----------------------------------------------------------
    async def record_file_scan(self, path: str, mtime: float, candidates: Sequence[FeedbackCandidate]) -> int:
        """Records a scanned file and its candidates in one transaction.

        The platform store's ingestion, plus the scanned path lands in each row's
        ``origin_path`` — a display hint only (the dashboard's project labels);
        transcript resolution always goes through discovery by session UUID. A
        candidate whose rendered context carries no substantive content is
        quarantined in the same transaction, so an empty capture never reaches
        judging, acceptance, or the frozen eval.

        Args:
            path: The scanned file's path.
            mtime: The file's modification time at scan.
            candidates: The candidates extracted from the file.

        Returns:
            The number of newly inserted feedback events.
        """
        ingested_at = now()
        by_key = {str(candidate.dedup_key): candidate for candidate in candidates}
        async with self.db.transaction() as db:
            inserted = await db.insert_candidates(
                [list(event_row(candidate, ingested_at)) for candidate in candidates],
                extras=[[path, None] for _ in candidates],
            )
            await db.executemany(
                QUARANTINE_CONTEXT,
                [
                    (ACCRUED_EMPTY_REASON, key)
                    for key in inserted
                    if not has_substantive_content(messages(by_key[key].window.before))
                ],
            )
            await db.record_file(path, mtime)
            return len(inserted)

    async def rebuild_context(
        self,
        rebuilt: Sequence[tuple[DedupKey, str]],
        quarantined: Sequence[tuple[DedupKey, str]],
        *,
        dry_run: bool = False,
    ) -> ContextRebuildChanges:
        """Persists rebuilt contexts and quarantine reasons in one transaction."""
        if dry_run:
            return ContextRebuildChanges(rebuilt=len(rebuilt), quarantined=len(quarantined))
        async with self.db.transaction() as db:
            rebuilt_count = await db.executemany(UPDATE_CONTEXT, [(context_json, key) for key, context_json in rebuilt])
            quarantined_count = await db.executemany(QUARANTINE_CONTEXT, [(reason, key) for key, reason in quarantined])
            return ContextRebuildChanges(rebuilt=rebuilt_count, quarantined=quarantined_count)

    async def quarantined_keys(self) -> set[str]:
        """Returns the dedup keys excluded from pipeline and dataset reads."""
        return {
            str(row["dedup_key"])
            for row in await self.sql("SELECT dedup_key FROM feedback_events WHERE quarantined_reason IS NOT NULL")
        }

    # --- verdict tier (thin delegators) --------------------------------------
    async def unjudged(
        self,
        *,
        role: str,
        prompt_version: int,
        limit: int | None = None,
        refresh_summary: bool = False,
        probe_hydration: bool = True,
    ) -> list[dict[str, object]]:
        """Returns non-quarantined events lacking a verdict for one role and prompt version."""
        return await self.db.unjudged(
            role=role,
            prompt_version=prompt_version,
            limit=limit,
            refresh_summary=refresh_summary,
            probe_hydration=probe_hydration,
        )

    async def judged(self, *, role: str, prompt_version: int) -> list[dict[str, object]]:
        """Returns non-quarantined events carrying a verdict for one role and prompt version."""
        return await self.db.judged(role=role, prompt_version=prompt_version)

    async def record_verdict(
        self, key: DedupKey, verdict: VerdictLike, *, role: str, prompt_version: int, model: str, fidelity: Fidelity
    ) -> None:
        """Records one verdict, idempotently, keyed by ``(dedup_key, role, prompt_version)``."""
        await self.db.record_verdict(
            key,
            verdict,
            role=role,
            prompt_version=prompt_version,
            model=model,
            fidelity=fidelity,
        )

    async def unrefined(self, *, prompt_version: int, model: str, limit: int | None = None) -> list[dict[str, object]]:
        """Returns accepted steering events lacking a refinement at ``(prompt_version, model)``.

        Surfaces the columns the refine prompt needs — ``dedup_key``, ``source_kind``,
        ``text``, ``context_json``, ``payload_json``, and the judge's ``what_claude_did``
        hint — oldest first.

        Args:
            prompt_version: The refine prompt version the refinement must carry.
            model: The resolved model name the refinement must carry.
            limit: When set, the maximum number of rows to return.

        Returns:
            One dict per accepted, unrefined event.
        """
        query = (
            "SELECT ap.dedup_key, ap.source_kind, ap.text, ap.context_json, ap.payload_json, ap.what_claude_did "
            "FROM accepted_steering ap "
            "LEFT JOIN refinement r ON r.dedup_key = ap.dedup_key "
            "AND r.prompt_version = ? AND r.model = ? "
            "WHERE r.id IS NULL ORDER BY ap.event_id"
        )
        params: tuple[object, ...] = (prompt_version, model)
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        return await self.sql(query, params)

    async def record_refinement(
        self, key: DedupKey, refinement: Refinement, *, prompt_version: int, model: str
    ) -> None:
        """Records one event's atomic refined pairs in a single transaction.

        Keyed by ``(dedup_key, prompt_version, model, pair_index)`` so re-running over
        a fully refined corpus is a no-op and every pair of one event commits together.

        Args:
            key: The refined event's dedup key.
            refinement: The atomic pairs to persist.
            prompt_version: The refine prompt version that produced them.
            model: The resolved model name that produced them.
        """
        refined_at = datetime.now(UTC).isoformat()
        async with self.db.transaction() as db:
            await db.executemany(
                INSERT_REFINEMENT,
                [
                    (
                        key,
                        prompt_version,
                        model,
                        index,
                        pair.action,
                        pair.direction_verbatim,
                        pair.direction,
                        refined_at,
                    )
                    for index, pair in enumerate(refinement.pairs)
                ],
            )

    async def record_gate_samples(self, samples: Sequence[GateSample]) -> int:
        """Records gate training samples idempotently, keyed by ``sample_key``.

        The single insert choke point drops any sample whose window renders no
        substantive gate content — a rewound-past-content positive or an
        empty-anchor negative is not valid model input and never reaches the
        table (mirrors accrual's empty-context quarantine at its own insert seam).

        Args:
            samples: The samples to persist; re-inserting an existing key is a no-op.

        Returns:
            The number of newly inserted samples.
        """
        created_at = now()
        async with self.db.transaction() as db:
            return await db.executemany(
                INSERT_GATE_SAMPLE,
                [
                    gate_sample_row(sample, created_at)
                    for sample in samples
                    if has_substantive_gate_content(ContextWindow.from_json(sample.window_json))
                ],
            )

    async def repair_gate_samples(
        self,
        query: str,
        planner: Callable[[Sequence[Mapping[str, object]]], GateSampleRepairs],
    ) -> int:
        """Reads, plans, and applies one gate-family reconciliation transactionally."""
        created_at = now()
        async with self.db.transaction() as db:
            repairs = planner(await db.sql(query))
            updated = await db.executemany(
                UPDATE_GATE_SAMPLE_WINDOW,
                [(window_json, sample_key, window_json) for sample_key, window_json in repairs.updates],
            )
            deleted = await db.executemany(DELETE_GATE_SAMPLE, [(sample_key,) for sample_key in repairs.deletes])
            inserted = await db.executemany(
                INSERT_GATE_SAMPLE,
                [gate_sample_row(sample, created_at) for sample in repairs.inserts],
            )
            return updated + deleted + inserted

    async def prune_gate_samples(
        self,
        query: str,
        classify: Callable[[Sequence[Mapping[str, object]]], GatePruneClassification],
        *,
        dry_run: bool,
    ) -> GatePruneClassification:
        """Scans and classifies over an unlocked read, then deletes in a short transaction.

        Classification renders every stored window, so it runs *outside* the write
        transaction — holding ``BEGIN IMMEDIATE`` across that O(table) work would starve
        the pipeline's 2s busy timeout. The reported counts therefore come from the scan
        snapshot; the delete transaction asserts its own changed-row count against the
        planned key count and raises on a mismatch, so a row added or removed between scan
        and delete is a loud retry rather than a silent miscount. ``dry_run`` deletes nothing.
        """
        classification = classify(await self.sql(query))
        if dry_run:
            return classification
        async with self.db.transaction() as db:
            deleted = await db.executemany(DELETE_GATE_SAMPLE, [(key,) for key in classification.empty_keys])
            if deleted != len(classification.empty_keys):
                raise RuntimeError(
                    f"prune deleted {deleted} rows but planned {len(classification.empty_keys)}; the gate table "
                    "changed between scan and delete — retry with the watch daemon unloaded"
                )
            return classification

    async def gate_sample_family_mismatch_keys(self) -> set[str]:
        """Returns parents whose stored gate family contradicts the latest judge verdict."""
        return {str(row["dedup_key"]) for row in await self.sql(GATE_FAMILY_MISMATCH_QUERY)}

    async def gate_samples(self, *, kind: str | None = None) -> list[dict[str, object]]:
        """Returns gate samples, oldest first, optionally restricted to one kind."""
        query = "SELECT * FROM gate_sample" + (" WHERE kind = ?" if kind else "") + " ORDER BY id"
        return await self.sql(query, (kind,) if kind else ())

    async def gate_sample_stats(self) -> Mapping[str, int]:
        """Returns gate sample counts keyed by kind."""
        return {
            str(row["kind"]): int(str(row["n"]))
            for row in await self.sql("SELECT kind, COUNT(*) AS n FROM gate_sample GROUP BY kind ORDER BY kind")
        }

    async def negative_sessions(self) -> set[str]:
        """Returns the sessions already parsed for random negatives.

        Doneness is completion, not survival: a session whose every empty-anchor
        sample was dropped at insert carries no ``random_negative`` row, so the
        ``sampled_session`` marker is unioned in to stop it re-parsing every pass.
        """
        return {
            str(row["session_id"])
            for row in await self.sql(
                "SELECT session_id FROM sampled_session "
                "UNION SELECT session_id FROM gate_sample WHERE kind = 'random_negative'"
            )
        }

    async def mark_sessions_sampled(self, session_ids: Sequence[str]) -> None:
        """Records that ``session_ids`` were parsed for random negatives, dropped-only included."""
        sampled_at = now()
        async with self.db.transaction() as db:
            await db.executemany(INSERT_SAMPLED_SESSION, [(session_id, sampled_at) for session_id in session_ids])

    async def record_embeddings(self, rows: Sequence[tuple[str, str, str, int, bytes]]) -> None:
        """Upserts exemplar embeddings as ``(dedup_key, model, text_digest, dim, vector)`` rows."""
        created_at = now()
        async with self.db.transaction() as db:
            await db.executemany(INSERT_EMBEDDING, [(*row, created_at) for row in rows])

    async def embeddings(self, *, model: str) -> list[dict[str, object]]:
        """Returns every stored exemplar embedding for ``model``, oldest first."""
        return await self.sql(
            "SELECT dedup_key, text_digest, dim, vector FROM exemplar_embedding WHERE model = ? ORDER BY rowid",
            (model,),
        )

    async def unenriched(self, log: CorrectionLog, *, limit: int | None = None) -> list[dict[str, object]]:
        """Returns refined pairs whose steering anchor carries no shared-ledger correction.

        A pair settles once its ``(session_id, event_uuid)`` anchor has a row in the
        shared ``corrections`` ledger — the single source of truth for "done". Since
        the extractor is idempotent per anchor, every pair sharing one anchor settles
        together the moment any of them writes its row. Pairs come from the latest
        refine generation only, so a refine re-run resurfaces its new pairs here
        automatically. Anchors that legitimately yield no correction (expired,
        editless, or no faulted edit) never settle, but resolving them costs no LLM
        call.

        Args:
            log: The shared correction ledger to check each anchor against.
            limit: When set, the maximum number of rows to return.

        Returns:
            One dict per unenriched pair with the columns the extractor and anchor
            resolution need, oldest event first.
        """
        unenriched = [
            row
            for row in await self.sql(REFINED_PAIRS_QUERY)
            if row["session_id"] and row["event_uuid"]
            if not await log.for_anchor(SessionId(str(row["session_id"])), EventUuid(str(row["event_uuid"])))
        ]
        return unenriched if limit is None else unenriched[:limit]

    async def pairs(self) -> list[dict[str, object]]:
        """Returns every row of the ``refined_pairs`` view, the pipeline's deliverable."""
        return await self.sql("SELECT * FROM refined_pairs ORDER BY event_id, pair_index")

    async def candidates(self) -> list[dict[str, object]]:
        """Returns one row per event with its latest judge verdict and refine summary.

        Powers the dashboard's candidate view across every pipeline status — refined,
        accepted-but-unrefined, judge-rejected noise, and unjudged. The verdict and
        refine columns are ``NULL`` for events that have not reached that stage.

        Returns:
            One dict per event: the event columns plus the latest judge verdict
            (``category``, ``is_steering``, ``confidence``, ``judge_version``,
            ``judge_model``, ``what_claude_did``), the latest auditor side
            (``auditor_is_steering``), the judge ``flipped`` flag, and the refine
            summary (``pair_count``, ``refine_version``, ``refine_model``).
        """
        return await self.sql(CANDIDATES_QUERY)

    async def lineage(self, dedup_key: str) -> dict[str, object]:
        """Returns one event with all its triage verdicts and latest refined pairs.

        Reads ``feedback_events``, ``triage``, and ``latest_refinement`` directly —
        the deliverable views drop the auditor, the older judge versions, and the
        payload the lineage needs.

        Args:
            dedup_key: The event's content-derived key.

        Returns:
            The event columns plus ``verdicts`` (every judge and auditor row, oldest
            first) and ``pairs`` (the latest refinement generation, by ``pair_index``),
            or ``{}`` when no event carries the key.
        """
        events = await self.sql(
            f"SELECT {EVENT_COLUMNS}, e.origin_path FROM feedback_events e WHERE e.dedup_key = ?", (dedup_key,)
        )
        if not events:
            return {}
        verdicts = await self.sql(LINEAGE_VERDICTS_QUERY, (dedup_key,))
        pairs = await self.sql(LINEAGE_PAIRS_QUERY, (dedup_key,))
        return {**events[0], "verdicts": verdicts, "pairs": pairs}

    async def triage_stats(self, *, prompt_version: int) -> TriageStats:
        """Returns triage coverage and acceptance at ``prompt_version``."""
        total_rows = await self.sql("SELECT COUNT(*) AS n FROM feedback_events WHERE quarantined_reason IS NULL")
        by_category_rows = await self.sql(
            "SELECT t.category, COUNT(*) AS n, SUM(t.is_steering) AS accepted FROM triage t "
            "JOIN feedback_events e ON e.dedup_key = t.dedup_key "
            "WHERE t.role = 'judge' AND t.prompt_version = ? AND e.quarantined_reason IS NULL "
            "GROUP BY t.category ORDER BY n DESC",
            (prompt_version,),
        )
        by_category = {
            str(row["category"]): (int(str(row["n"])), int(str(row["accepted"]))) for row in by_category_rows
        }
        return TriageStats(
            total=int(str(total_rows[0]["n"])),
            judged=sum(n for n, _ in by_category.values()),
            accepted=sum(accepted for _, accepted in by_category.values()),
            by_category={category: n for category, (n, _) in by_category.items()},
        )
