"""The SQLite feedback store: the mining-domain store plus cc-pushback's triage layer.

The store mechanism lives in :mod:`cc_transcript.domains.mining`; this module adds
cc-pushback's default database location, the ``triage`` verdict table, the
``refinement`` table, and two views: ``accepted_pushback`` (judge-accepted candidates,
the refine stage's input) and ``refined_pairs`` — the pipeline's final deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

from cc_transcript.domains.mining import FEEDBACK_DDL, Stats, event_row
from cc_transcript.domains.mining import FeedbackStore as BaseFeedbackStore
from cc_transcript.store import FileStateStore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cc_transcript.domains.mining import DedupKey

    from cc_pushback.refine import Refinement
    from cc_pushback.triage import Verdict

__all__ = ["FEEDBACK_DDL", "REFINE_DDL", "TRIAGE_DDL", "FeedbackStore", "Stats", "TriageStats", "event_row"]

TRIAGE_DDL = """
CREATE TABLE IF NOT EXISTS triage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT NOT NULL REFERENCES feedback_events(dedup_key),
  role TEXT NOT NULL,
  prompt_version INTEGER NOT NULL,
  model TEXT NOT NULL,
  category TEXT NOT NULL,
  is_pushback INTEGER NOT NULL,
  what_claude_did TEXT NOT NULL,
  confidence REAL NOT NULL,
  rationale TEXT NOT NULL,
  judged_at TEXT NOT NULL,
  UNIQUE(dedup_key, role, prompt_version, model)
);
CREATE INDEX IF NOT EXISTS idx_triage_dedup ON triage(dedup_key);
DROP VIEW IF EXISTS training_pairs;
DROP VIEW IF EXISTS accepted_pushback;
CREATE VIEW accepted_pushback AS
WITH latest AS (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'judge'
)
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
JOIN latest t ON t.dedup_key = e.dedup_key AND t.rn = 1
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
DROP VIEW IF EXISTS refined_pairs;
CREATE VIEW refined_pairs AS
WITH gens AS (
  SELECT dedup_key, prompt_version, model, refined_at,
    ROW_NUMBER() OVER (
      PARTITION BY dedup_key ORDER BY prompt_version DESC, refined_at DESC
    ) AS g
  FROM (SELECT DISTINCT dedup_key, prompt_version, model, refined_at FROM refinement)
)
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
  e.occurred_at,
  e.origin_path,
  r.prompt_version,
  r.model
FROM refinement r
JOIN gens ON gens.dedup_key = r.dedup_key AND gens.prompt_version = r.prompt_version
         AND gens.model = r.model AND gens.refined_at = r.refined_at AND gens.g = 1
JOIN feedback_events e ON e.dedup_key = r.dedup_key
LEFT JOIN accepted_pushback ap ON ap.dedup_key = r.dedup_key
ORDER BY e.id, r.pair_index;
"""

INSERT_VERDICT = """
INSERT OR IGNORE INTO triage (
  dedup_key, role, prompt_version, model, category, is_pushback,
  what_claude_did, confidence, rationale, judged_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_REFINEMENT = """
INSERT OR IGNORE INTO refinement (
  dedup_key, prompt_version, model, pair_index, action, complaint_verbatim, complaint, refined_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

EVENT_COLUMNS = (
    "e.id, e.dedup_key, e.source_kind, e.occurred_at, e.text, "
    "e.payload_json, e.context_json, e.session_id, e.origin_path"
)


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


class FeedbackStore(BaseFeedbackStore):
    """Persistent store for collected feedback over a :class:`FileStateStore`.

    Layers the ``feedback_events`` table onto cc-transcript's file-mtime ledger and
    adds the ``triage`` verdict table, the ``refinement`` table, and the
    ``accepted_pushback`` and ``refined_pairs`` views. Verdicts and refinements key
    on the content-derived dedup key, so they survive a database rebuild.

    Example:
        >>> async with await FeedbackStore.open(FeedbackStore.default_path()) as store:
        ...     await store.record_file_scan(str(path), mtime, candidates)
    """

    @staticmethod
    def default_path() -> Path:
        """Returns the default database path, ``~/.cc-pushback/feedback.db``."""
        return Path.home() / ".cc-pushback" / "feedback.db"

    @classmethod
    async def open(cls, path: Path) -> Self:
        """Opens (creating if needed) the feedback database at ``path``."""
        return cls(await FileStateStore.open(path, extra_schema=FEEDBACK_DDL + TRIAGE_DDL + REFINE_DDL))

    async def unjudged(
        self, *, role: str, prompt_version: int, model: str, limit: int | None = None
    ) -> list[dict[str, object]]:
        """Returns events lacking a verdict for ``(role, prompt_version, model)``, oldest first.

        Args:
            role: The verdict role to check, ``judge`` or ``auditor``.
            prompt_version: The prompt version the verdict must carry.
            model: The resolved model name the verdict must carry.
            limit: When set, the maximum number of rows to return.

        Returns:
            One dict per event with the columns needed to build its prompt.
        """
        query = (
            f"SELECT {EVENT_COLUMNS} FROM feedback_events e "
            "LEFT JOIN triage t ON t.dedup_key = e.dedup_key "
            "AND t.role = ? AND t.prompt_version = ? AND t.model = ? "
            "WHERE t.id IS NULL ORDER BY e.id"
        )
        params: tuple[object, ...] = (role, prompt_version, model)
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        cur = await self.store.conn.execute(query, params)
        return [dict(row) async for row in cur]

    async def record_verdict(
        self, key: DedupKey, verdict: Verdict, *, role: str, prompt_version: int, model: str
    ) -> None:
        """Records one verdict, idempotently, keyed by ``(dedup_key, role, prompt_version, model)``.

        Args:
            key: The judged event's dedup key.
            verdict: The structured verdict to persist.
            role: Who produced it, ``judge`` or ``auditor``.
            prompt_version: The prompt version that produced it.
            model: The resolved model name that produced it.
        """
        await self.store.conn.execute(
            INSERT_VERDICT,
            (
                key,
                role,
                prompt_version,
                model,
                verdict.category,
                verdict.is_pushback,
                verdict.what_claude_did,
                verdict.confidence,
                verdict.rationale,
                datetime.now(UTC).isoformat(),
            ),
        )

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

    async def judged(self, *, role: str, prompt_version: int) -> list[dict[str, object]]:
        """Returns events joined with their ``(role, prompt_version)`` verdicts.

        Args:
            role: The verdict role to join, ``judge`` or ``auditor``.
            prompt_version: The prompt version to join.

        Returns:
            One dict per verdict-bearing event: the event columns plus the
            verdict's ``category``, ``is_pushback``, ``confidence``,
            ``what_claude_did``, ``rationale``, and ``model``.
        """
        cur = await self.store.conn.execute(
            f"SELECT {EVENT_COLUMNS}, t.category, t.is_pushback, t.confidence, "
            "t.what_claude_did, t.rationale, t.model "
            "FROM feedback_events e JOIN triage t ON t.dedup_key = e.dedup_key "
            "WHERE t.role = ? AND t.prompt_version = ? ORDER BY e.id",
            (role, prompt_version),
        )
        return [dict(row) async for row in cur]

    async def dedup_keys(self) -> set[str]:
        """Returns every stored event's dedup key."""
        cur = await self.store.conn.execute("SELECT dedup_key FROM feedback_events")
        return {str(row["dedup_key"]) async for row in cur}

    async def pairs(self) -> list[dict[str, object]]:
        """Returns every row of the ``refined_pairs`` view, the pipeline's deliverable."""
        cur = await self.store.conn.execute("SELECT * FROM refined_pairs ORDER BY event_id, pair_index")
        return [dict(row) async for row in cur]

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
