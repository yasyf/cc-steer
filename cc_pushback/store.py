"""The SQLite feedback store: the platform's mining store plus cc-pushback's triage layer.

The store mechanism lives in :mod:`cc_transcript.mining`; this module adds
cc-pushback's default database location, the ``origin_path`` display-hint column,
the ``triage`` verdict table, the ``refinement`` table, and two views:
``accepted_pushback`` (judge-accepted candidates, the refine stage's input) and
``refined_pairs`` — the pipeline's final deliverable. The enrich stage's code
evidence no longer lives here: it lands in cc-transcript's shared ``corrections``
ledger, keyed by the pushback anchor, and the dashboard reads it straight from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

from cc_transcript.ids import EventUuid, SessionId
from cc_transcript.judge import VerdictStoreMixin
from cc_transcript.judge.verdicts import EVENT_COLUMNS
from cc_transcript.mining import FEEDBACK_DDL as BASE_FEEDBACK_DDL
from cc_transcript.mining import FeedbackStore as BaseFeedbackStore
from cc_transcript.mining import Stats, event_row
from cc_transcript.mining.store import now
from cc_transcript.store import FileStateStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cc_transcript.corrections import CorrectionLog
    from cc_transcript.mining import DedupKey, FeedbackCandidate

    from cc_pushback.refine import Refinement

__all__ = [
    "FEEDBACK_DDL",
    "REFINE_DDL",
    "TRIAGE_DDL",
    "FeedbackStore",
    "Stats",
    "TriageStats",
    "event_row",
]

FEEDBACK_DDL = BASE_FEEDBACK_DDL.replace(
    "  ingested_at TEXT NOT NULL\n",
    "  ingested_at TEXT NOT NULL,\n  origin_path TEXT\n",
)

INSERT_EVENT = """
INSERT OR IGNORE INTO feedback_events (
  dedup_key, source_kind, session_id, event_uuid,
  occurred_at, text, payload_json, context_json, cc_version, ingested_at, origin_path
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
DROP VIEW IF EXISTS accepted_pushback;
CREATE VIEW accepted_pushback AS
SELECT
  e.id AS event_id,
  e.dedup_key,
  e.source_kind,
  e.text,
  e.context_json,
  t.category,
  t.what_claude_did,
  e.origin_path
FROM feedback_events e
JOIN latest_judge t ON t.dedup_key = e.dedup_key
WHERE t.is_pushback = 1;
"""

REFINE_DDL = """
CREATE TABLE IF NOT EXISTS refinement (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT NOT NULL REFERENCES feedback_events(dedup_key),
  prompt_version INTEGER NOT NULL,
  model TEXT NOT NULL,
  pair_index INTEGER NOT NULL,
  action TEXT NOT NULL,
  complaint_verbatim TEXT NOT NULL,
  complaint TEXT NOT NULL,
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
  r.complaint_verbatim,
  r.complaint,
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
JOIN accepted_pushback ap ON ap.dedup_key = r.dedup_key
ORDER BY e.id, r.pair_index;
"""

INSERT_REFINEMENT = """
INSERT OR IGNORE INTO refinement (
  dedup_key, prompt_version, model, pair_index, action, complaint_verbatim, complaint, refined_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

CANDIDATES_QUERY = f"""
WITH judge_flip AS (
  SELECT dedup_key, COUNT(DISTINCT is_pushback) > 1 AS flipped
  FROM triage WHERE role = 'judge' GROUP BY dedup_key
),
refine_summary AS (
  SELECT dedup_key, COUNT(*) AS pair_count,
    MAX(prompt_version) AS refine_version, MAX(model) AS refine_model
  FROM latest_refinement GROUP BY dedup_key
)
SELECT {EVENT_COLUMNS}, e.origin_path,
  j.category, j.is_pushback, j.confidence, j.prompt_version AS judge_version,
  j.model AS judge_model, j.what_claude_did,
  a.is_pushback AS auditor_is_pushback,
  COALESCE(f.flipped, 0) AS flipped,
  rs.pair_count, rs.refine_version, rs.refine_model
FROM feedback_events e
LEFT JOIN latest_judge j ON j.dedup_key = e.dedup_key
LEFT JOIN latest_auditor a ON a.dedup_key = e.dedup_key
LEFT JOIN judge_flip f ON f.dedup_key = e.dedup_key
LEFT JOIN refine_summary rs ON rs.dedup_key = e.dedup_key
ORDER BY e.id
"""

LINEAGE_VERDICTS_QUERY = (
    "SELECT role, prompt_version, model, category, is_pushback, what_claude_did, "
    "confidence, rationale, judged_at FROM triage WHERE dedup_key = ? ORDER BY role, prompt_version, id"
)

LINEAGE_PAIRS_QUERY = """
SELECT r.pair_index, r.action, r.complaint_verbatim, r.complaint, r.prompt_version, r.model,
  e.session_id, e.event_uuid
FROM latest_refinement r
JOIN feedback_events e ON e.dedup_key = r.dedup_key
WHERE r.dedup_key = ?
ORDER BY r.pair_index
"""

REFINED_PAIRS_QUERY = """
SELECT dedup_key, prompt_version AS refine_version, model AS refine_model, pair_index,
  action, complaint, complaint_verbatim, source_kind, session_id, event_uuid, origin_path
FROM refined_pairs
ORDER BY event_id, pair_index"""


@dataclass(frozen=True, slots=True)
class TriageStats:
    """A snapshot of triage progress at one prompt version.

    Attributes:
        total: The total feedback events in the corpus.
        judged: How many carry a judge verdict at this prompt version.
        accepted: How many of those verdicts are pushback.
        by_category: Verdict counts keyed by category.
    """

    total: int
    judged: int
    accepted: int
    by_category: Mapping[str, int]


class FeedbackStore(VerdictStoreMixin, BaseFeedbackStore):
    """Persistent store for collected feedback over a :class:`FileStateStore`.

    Layers the ``feedback_events`` table (extended with the ``origin_path``
    display-hint column) onto cc-transcript's file-mtime ledger and adds the
    ``triage`` verdict table (the judge package's verdict mechanism pinned to
    cc-pushback's column names), the ``refinement`` table, and the
    ``accepted_pushback`` and ``refined_pairs`` views. Verdicts and refinements key
    on the content-derived dedup key, so they survive a database rebuild; the enrich
    stage's code evidence lives in cc-transcript's shared ``corrections`` ledger,
    keyed by the pushback anchor.

    Example:
        >>> async with await FeedbackStore.open(FeedbackStore.default_path()) as store:
        ...     await store.record_file_scan(str(path), mtime, candidates)
    """

    VERDICT_TABLE = "triage"
    ACCEPTED_COLUMN = "is_pushback"
    SUMMARY_COLUMN = "what_claude_did"

    @staticmethod
    def default_path() -> Path:
        """Returns the default database path, ``~/.cc-pushback/feedback.db``."""
        return Path.home() / ".cc-pushback" / "feedback.db"

    @classmethod
    async def open(cls, path: Path) -> Self:
        """Opens (creating if needed) the feedback database at ``path``."""
        return cls(await FileStateStore.open(path, extra_schema=FEEDBACK_DDL + TRIAGE_DDL + REFINE_DDL))

    async def record_file_scan(self, path: str, mtime: float, candidates: Sequence[FeedbackCandidate]) -> int:
        """Records a scanned file and its candidates in one transaction.

        The platform store's ingestion, plus the scanned path lands in each row's
        ``origin_path`` — a display hint only (the dashboard's project labels);
        transcript resolution always goes through discovery by session UUID.

        Args:
            path: The scanned file's path.
            mtime: The file's modification time at scan.
            candidates: The candidates extracted from the file.

        Returns:
            The number of newly inserted feedback events.
        """
        ingested_at = now()
        async with self.store.transaction() as conn:
            before = conn.total_changes
            await conn.executemany(
                INSERT_EVENT, [(*event_row(candidate, ingested_at), path) for candidate in candidates]
            )
            inserted = conn.total_changes - before
            await self.store.record_file(path, mtime)
            return inserted

    async def unrefined(self, *, prompt_version: int, model: str, limit: int | None = None) -> list[dict[str, object]]:
        """Returns accepted pushback events lacking a refinement at ``(prompt_version, model)``.

        Surfaces the columns the refine prompt needs — ``dedup_key``, ``source_kind``,
        ``text``, ``context_json``, and the judge's ``what_claude_did`` hint — oldest first.

        Args:
            prompt_version: The refine prompt version the refinement must carry.
            model: The resolved model name the refinement must carry.
            limit: When set, the maximum number of rows to return.

        Returns:
            One dict per accepted, unrefined event.
        """
        query = (
            "SELECT ap.dedup_key, ap.source_kind, ap.text, ap.context_json, ap.what_claude_did "
            "FROM accepted_pushback ap "
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
                        pair.complaint_verbatim,
                        pair.complaint,
                        refined_at,
                    )
                    for index, pair in enumerate(refinement.pairs)
                ],
            )

    async def unenriched(self, log: CorrectionLog, *, limit: int | None = None) -> list[dict[str, object]]:
        """Returns refined pairs whose pushback anchor carries no shared-ledger correction.

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
            (``category``, ``is_pushback``, ``confidence``, ``judge_version``,
            ``judge_model``, ``what_claude_did``), the latest auditor side
            (``auditor_is_pushback``), the judge ``flipped`` flag, and the refine
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
        total_cur = await conn.execute("SELECT COUNT(*) AS n FROM feedback_events")
        by_category_cur = await conn.execute(
            "SELECT category, COUNT(*) AS n, SUM(is_pushback) AS accepted FROM triage "
            "WHERE role = 'judge' AND prompt_version = ? GROUP BY category ORDER BY n DESC",
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
