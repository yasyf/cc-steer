"""The SQLite feedback store: the platform's mining store plus cc-steer's triage layer.

The store mechanism lives in :mod:`cc_transcript.mining`; this module adds
cc-steer's default database location, the ``origin_path`` display hint and
``quarantined_reason`` columns,
the ``triage`` verdict table, the ``refinement`` table, and two views:
``accepted_steering`` (judge-accepted candidates, the refine stage's input) and
``refined_pairs`` — the pipeline's final deliverable. The enrich stage's code
evidence no longer lives here: it lands in cc-transcript's shared ``corrections``
ledger, keyed by the steering anchor, and the dashboard reads it straight from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

import aiosqlite
from cc_transcript.ids import EventUuid, SessionId
from cc_transcript.judge import VerdictStoreMixin
from cc_transcript.judge.verdicts import EVENT_COLUMNS, hydratable
from cc_transcript.mining import FEEDBACK_DDL as BASE_FEEDBACK_DDL
from cc_transcript.mining import FeedbackStore as BaseFeedbackStore
from cc_transcript.mining import Stats, event_row
from cc_transcript.mining.store import now
from cc_transcript.store import FileStateStore

from cc_steer.rendering import has_substantive_content, messages

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from aiosqlite import Row
    from cc_transcript.corrections import CorrectionLog
    from cc_transcript.mining import DedupKey, FeedbackCandidate

    from cc_steer.context_rebuild import GateSampleRepairs
    from cc_steer.negatives import GateSample
    from cc_steer.refine import Refinement

__all__ = [
    "ACCRUED_EMPTY_REASON",
    "FEEDBACK_DDL",
    "GATE_DDL",
    "REFINE_DDL",
    "TRIAGE_DDL",
    "ContextRebuildChanges",
    "FeedbackStore",
    "Stats",
    "TriageStats",
    "event_row",
]

FEEDBACK_DDL = BASE_FEEDBACK_DDL.replace(
    "  ingested_at TEXT NOT NULL\n",
    "  ingested_at TEXT NOT NULL,\n  origin_path TEXT,\n  quarantined_reason TEXT\n",
)

BUSY_TIMEOUT_MS = 2_000
ADD_QUARANTINE_COLUMN = "ALTER TABLE feedback_events ADD COLUMN quarantined_reason TEXT"
ACCRUED_EMPTY_REASON = "accrued_context_empty"
REBUILD_QUARANTINE_REASONS = (
    ACCRUED_EMPTY_REASON,
    "transcript_not_found",
    "transcript_parse_failed",
    "anchor_not_found",
    "rebuilt_context_empty",
)

INSERT_EVENT = """
INSERT OR IGNORE INTO feedback_events (
  dedup_key, source_kind, session_id, event_uuid,
  occurred_at, text, payload_json, context_json, cc_version, ingested_at, origin_path
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

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

TRIAGE_VIEWS_DDL = """DROP VIEW IF EXISTS training_pairs;
DROP VIEW IF EXISTS latest_judge;
CREATE VIEW latest_judge AS
SELECT * FROM (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'judge'
) WHERE rn = 1;
DROP VIEW IF EXISTS latest_auditor;
CREATE VIEW latest_auditor AS
SELECT * FROM (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'auditor'
) WHERE rn = 1;
DROP VIEW IF EXISTS accepted_steering;
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
"""

REFINE_DDL = """
CREATE TABLE IF NOT EXISTS refinement (
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
CREATE INDEX IF NOT EXISTS idx_refinement_dedup ON refinement(dedup_key);
DROP VIEW IF EXISTS latest_refinement;
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
DROP VIEW IF EXISTS refined_pairs;
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
"""

INSERT_REFINEMENT = """
INSERT OR IGNORE INTO refinement (
  dedup_key, prompt_version, model, pair_index, action, direction_verbatim, direction, refined_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

GATE_DDL = """
CREATE TABLE IF NOT EXISTS gate_sample (
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
CREATE INDEX IF NOT EXISTS idx_gate_sample_kind ON gate_sample(kind);
CREATE INDEX IF NOT EXISTS idx_gate_sample_session ON gate_sample(session_id);
CREATE TABLE IF NOT EXISTS exemplar_embedding (
  dedup_key TEXT NOT NULL REFERENCES feedback_events(dedup_key),
  model TEXT NOT NULL,
  text_digest TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(dedup_key, model)
);
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


class FeedbackStore(VerdictStoreMixin, BaseFeedbackStore):
    """Persistent store for collected feedback over a :class:`FileStateStore`.

    Layers the ``feedback_events`` table (extended with provenance and quarantine
    columns) onto cc-transcript's file-mtime ledger and adds the
    ``triage`` verdict table (the judge package's verdict mechanism pinned to
    cc-steer's column names), the ``refinement`` table, and the
    ``accepted_steering`` and ``refined_pairs`` views. Verdicts and refinements key
    on the content-derived dedup key, so they survive a database rebuild; the enrich
    stage's code evidence lives in cc-transcript's shared ``corrections`` ledger,
    keyed by the steering anchor.

    Example:
        >>> async with await FeedbackStore.open(FeedbackStore.default_path()) as store:
        ...     await store.record_file_scan(str(path), mtime, candidates)
    """

    VERDICT_TABLE = "triage"
    ACCEPTED_COLUMN = "is_steering"
    SUMMARY_COLUMN = "what_claude_did"

    @staticmethod
    def default_path() -> Path:
        """Returns the default database path, ``~/.cc-steer/feedback.db``."""
        return Path.home() / ".cc-steer" / "feedback.db"

    @classmethod
    async def open(cls, path: Path) -> Self:
        """Opens (creating if needed) the feedback database at ``path``."""
        store = await FileStateStore.open(path, extra_schema=FEEDBACK_DDL)
        await store.conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        columns = {str(row["name"]) async for row in await store.conn.execute("PRAGMA table_info(feedback_events)")}
        if "quarantined_reason" not in columns:
            await store.conn.execute(ADD_QUARANTINE_COLUMN)
        await store.conn.executescript(TRIAGE_DDL + REFINE_DDL + GATE_DDL)
        return cls(store)

    @classmethod
    async def open_readonly(cls, path: Path) -> Self:
        """Opens an existing feedback database without schema or data writes."""
        conn = await aiosqlite.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            isolation_level=None,
            uri=True,
        )
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA query_only = ON")
        await conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        return cls(FileStateStore(conn))

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
        async with self.store.transaction() as conn:
            inserted: list[FeedbackCandidate] = []
            for candidate in candidates:
                before = conn.total_changes
                await conn.execute(INSERT_EVENT, (*event_row(candidate, ingested_at), path))
                if conn.total_changes > before:
                    inserted.append(candidate)
            await conn.executemany(
                QUARANTINE_CONTEXT,
                [
                    (ACCRUED_EMPTY_REASON, candidate.dedup_key)
                    for candidate in inserted
                    if not has_substantive_content(messages(candidate.window.before))
                ],
            )
            await self.store.record_file(path, mtime)
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
        async with self.store.transaction() as conn:
            before = conn.total_changes
            await conn.executemany(UPDATE_CONTEXT, [(context_json, key) for key, context_json in rebuilt])
            rebuilt_count = conn.total_changes - before
            before = conn.total_changes
            await conn.executemany(QUARANTINE_CONTEXT, [(reason, key) for key, reason in quarantined])
            return ContextRebuildChanges(
                rebuilt=rebuilt_count,
                quarantined=conn.total_changes - before,
            )

    async def quarantined_keys(self) -> set[str]:
        """Returns the dedup keys excluded from pipeline and dataset reads."""
        cur = await self.store.conn.execute(
            "SELECT dedup_key FROM feedback_events WHERE quarantined_reason IS NOT NULL"
        )
        return {str(row["dedup_key"]) async for row in cur}

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
        await self.ensure_verdict_schema()
        if not refresh_summary:
            sql = (
                f"SELECT {EVENT_COLUMNS} FROM feedback_events e "
                f"LEFT JOIN {self.VERDICT_TABLE} t ON t.dedup_key = e.dedup_key "
                "AND t.role = ? AND t.prompt_version = ? "
                "WHERE t.id IS NULL AND e.quarantined_reason IS NULL ORDER BY e.id"
            )
            params: tuple[object, ...] = (role, prompt_version)
            if limit is not None:
                sql += " LIMIT ?"
                params = (*params, limit)
            async with self.store.conn.execute(sql, params) as cur:
                return [dict(row) async for row in cur]
        if limit == 0:
            return []
        sql = (
            f"SELECT {EVENT_COLUMNS}, t.id AS verdict_id FROM feedback_events e "
            f"LEFT JOIN {self.VERDICT_TABLE} t ON t.dedup_key = e.dedup_key "
            "AND t.role = ? AND t.prompt_version = ? "
            "WHERE e.quarantined_reason IS NULL AND (t.id IS NULL OR t.fidelity = 'summary') "
            "ORDER BY (t.id IS NOT NULL), e.id"
        )
        params = (role, prompt_version)
        kept: list[dict[str, object]] = []
        if limit is None:
            async with self.store.conn.execute(sql, params) as cur:
                async for raw in cur:
                    row = dict(raw)
                    fresh = row.pop("verdict_id") is None
                    if not probe_hydration or fresh or await hydratable(str(row["context_json"])):
                        kept.append(row)
            return kept
        offset = 0
        while len(kept) < limit:
            async with self.store.conn.execute(sql + " LIMIT ? OFFSET ?", (*params, limit, offset)) as cur:
                page = [dict(row) async for row in cur]
            if not page:
                break
            offset += len(page)
            for row in page:
                fresh = row.pop("verdict_id") is None
                if not probe_hydration or fresh or await hydratable(str(row["context_json"])):
                    kept.append(row)
                    if len(kept) >= limit:
                        break
        return kept

    async def judged(self, *, role: str, prompt_version: int) -> list[dict[str, object]]:
        """Returns non-quarantined events carrying a verdict for one role and prompt version."""
        await self.ensure_verdict_schema()
        cur = await self.store.conn.execute(
            f"SELECT {EVENT_COLUMNS}, t.category, t.{self.ACCEPTED_COLUMN} AS accepted, t.confidence, "
            f"t.{self.SUMMARY_COLUMN} AS summary, t.rationale, t.model "
            f"FROM feedback_events e JOIN {self.VERDICT_TABLE} t ON t.dedup_key = e.dedup_key "
            "WHERE t.role = ? AND t.prompt_version = ? AND e.quarantined_reason IS NULL ORDER BY e.id",
            (role, prompt_version),
        )
        return [dict(row) async for row in cur]

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
        cur = await self.store.conn.execute(query, params)
        return [dict(row) async for row in cur]

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
        async with self.store.transaction() as conn:
            await conn.executemany(
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

        Args:
            samples: The samples to persist; re-inserting an existing key is a no-op.

        Returns:
            The number of newly inserted samples.
        """
        created_at = now()
        async with self.store.transaction() as conn:
            before = conn.total_changes
            await conn.executemany(
                INSERT_GATE_SAMPLE,
                [gate_sample_row(sample, created_at) for sample in samples],
            )
            return conn.total_changes - before

    async def repair_gate_samples(
        self,
        query: str,
        planner: Callable[[Sequence[Row]], GateSampleRepairs],
    ) -> int:
        """Reads, plans, and applies one gate-family reconciliation transactionally."""
        created_at = now()
        async with self.store.transaction() as conn:
            rows = [row async for row in await conn.execute(query)]
            repairs = planner(rows)
            before = conn.total_changes
            await conn.executemany(
                UPDATE_GATE_SAMPLE_WINDOW,
                [
                    (window_json, sample_key, window_json)
                    for sample_key, window_json in repairs.updates
                ],
            )
            await conn.executemany(DELETE_GATE_SAMPLE, [(sample_key,) for sample_key in repairs.deletes])
            await conn.executemany(
                INSERT_GATE_SAMPLE,
                [gate_sample_row(sample, created_at) for sample in repairs.inserts],
            )
            return conn.total_changes - before

    async def gate_sample_family_mismatch_keys(self) -> set[str]:
        """Returns parents whose stored gate family contradicts the latest judge verdict."""
        cur = await self.store.conn.execute(GATE_FAMILY_MISMATCH_QUERY)
        return {str(row["dedup_key"]) async for row in cur}

    async def gate_samples(self, *, kind: str | None = None) -> list[dict[str, object]]:
        """Returns gate samples, oldest first, optionally restricted to one kind."""
        query = "SELECT * FROM gate_sample" + (" WHERE kind = ?" if kind else "") + " ORDER BY id"
        cur = await self.store.conn.execute(query, (kind,) if kind else ())
        return [dict(row) async for row in cur]

    async def gate_sample_stats(self) -> Mapping[str, int]:
        """Returns gate sample counts keyed by kind."""
        cur = await self.store.conn.execute("SELECT kind, COUNT(*) AS n FROM gate_sample GROUP BY kind ORDER BY kind")
        return {str(row["kind"]): int(row["n"]) async for row in cur}

    async def negative_sessions(self) -> set[str]:
        """Returns the sessions that already carry random-negative samples."""
        cur = await self.store.conn.execute(
            "SELECT DISTINCT session_id FROM gate_sample WHERE kind = 'random_negative'"
        )
        return {str(row["session_id"]) async for row in cur}

    async def record_embeddings(self, rows: Sequence[tuple[str, str, str, int, bytes]]) -> None:
        """Upserts exemplar embeddings as ``(dedup_key, model, text_digest, dim, vector)`` rows."""
        created_at = now()
        async with self.store.transaction() as conn:
            await conn.executemany(INSERT_EMBEDDING, [(*row, created_at) for row in rows])

    async def embeddings(self, *, model: str) -> list[dict[str, object]]:
        """Returns every stored exemplar embedding for ``model``, oldest first."""
        cur = await self.store.conn.execute(
            "SELECT dedup_key, text_digest, dim, vector FROM exemplar_embedding WHERE model = ? ORDER BY rowid",
            (model,),
        )
        return [dict(row) async for row in cur]

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
        cur = await self.store.conn.execute(REFINED_PAIRS_QUERY)
        unenriched = [
            row
            async for raw in cur
            if (row := dict(raw))["session_id"] and row["event_uuid"]
            if not log.for_anchor(SessionId(str(row["session_id"])), EventUuid(str(row["event_uuid"])))
        ]
        return unenriched if limit is None else unenriched[:limit]

    async def pairs(self) -> list[dict[str, object]]:
        """Returns every row of the ``refined_pairs`` view, the pipeline's deliverable."""
        cur = await self.store.conn.execute("SELECT * FROM refined_pairs ORDER BY event_id, pair_index")
        return [dict(row) async for row in cur]

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
        cur = await self.store.conn.execute(CANDIDATES_QUERY)
        return [dict(row) async for row in cur]

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
        conn = self.store.conn
        event_cur = await conn.execute(
            f"SELECT {EVENT_COLUMNS}, e.origin_path FROM feedback_events e WHERE e.dedup_key = ?", (dedup_key,)
        )
        events = [dict(row) async for row in event_cur]
        if not events:
            return {}
        verdict_cur = await conn.execute(LINEAGE_VERDICTS_QUERY, (dedup_key,))
        verdicts = [dict(row) async for row in verdict_cur]
        pair_cur = await conn.execute(LINEAGE_PAIRS_QUERY, (dedup_key,))
        pairs = [dict(row) async for row in pair_cur]
        return {**events[0], "verdicts": verdicts, "pairs": pairs}

    async def triage_stats(self, *, prompt_version: int) -> TriageStats:
        """Returns triage coverage and acceptance at ``prompt_version``."""
        conn = self.store.conn
        total_cur = await conn.execute("SELECT COUNT(*) AS n FROM feedback_events WHERE quarantined_reason IS NULL")
        by_category_cur = await conn.execute(
            "SELECT t.category, COUNT(*) AS n, SUM(t.is_steering) AS accepted FROM triage t "
            "JOIN feedback_events e ON e.dedup_key = t.dedup_key "
            "WHERE t.role = 'judge' AND t.prompt_version = ? AND e.quarantined_reason IS NULL "
            "GROUP BY t.category ORDER BY n DESC",
            (prompt_version,),
        )
        by_category = {row["category"]: (row["n"], row["accepted"]) async for row in by_category_cur}
        return TriageStats(
            total=[row["n"] async for row in total_cur][0],
            judged=sum(n for n, _ in by_category.values()),
            accepted=sum(accepted for _, accepted in by_category.values()),
            by_category={category: n for category, (n, _) in by_category.items()},
        )


TRIAGE_DDL = FeedbackStore.verdicts_ddl() + TRIAGE_VIEWS_DDL
