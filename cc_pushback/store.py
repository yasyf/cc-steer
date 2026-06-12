"""The SQLite feedback store: the platform's mining store plus cc-pushback's triage layer.

The store mechanism lives in :mod:`cc_transcript.mining`; this module adds
cc-pushback's default database location, the ``origin_path`` display-hint column,
the ``triage`` verdict table, the ``refinement`` table, the ``pair_evidence``
table (the enrich stage's generation-keyed code evidence), and three views:
``accepted_pushback`` (judge-accepted candidates, the refine stage's input),
``pair_evidence_latest`` (the newest evidence generation per refined pair), and
``refined_pairs`` — the pipeline's final deliverable, evidence columns included.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

from cc_transcript.judge import VerdictStoreMixin
from cc_transcript.judge.verdicts import EVENT_COLUMNS
from cc_transcript.mining import FEEDBACK_DDL as BASE_FEEDBACK_DDL
from cc_transcript.mining import FeedbackStore as BaseFeedbackStore
from cc_transcript.mining import Stats, event_row
from cc_transcript.mining.store import now
from cc_transcript.store import FileStateStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cc_transcript.mining import DedupKey, FeedbackCandidate

    from cc_pushback.enrich import CodeEvidence, Source
    from cc_pushback.refine import Refinement

__all__ = [
    "ENRICH_DDL",
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

ENRICH_DDL = """
CREATE TABLE IF NOT EXISTS pair_evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT NOT NULL REFERENCES feedback_events(dedup_key),
  refine_version INTEGER NOT NULL,
  refine_model TEXT NOT NULL,
  pair_index INTEGER NOT NULL,
  enrich_version INTEGER NOT NULL,
  enrich_model TEXT NOT NULL,
  extractor_version INTEGER NOT NULL,
  evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('code','no_code')),
  file_path TEXT,
  incorrect_old TEXT,
  incorrect_new TEXT,
  correct_old TEXT,
  correct_new TEXT,
  note TEXT NOT NULL,
  source TEXT CHECK(source IN ('session','git')),
  enriched_at_ms INTEGER NOT NULL,
  UNIQUE(dedup_key, refine_version, refine_model, pair_index, enrich_version, enrich_model, extractor_version)
);
CREATE INDEX IF NOT EXISTS idx_pair_evidence_dedup ON pair_evidence(dedup_key);
DROP VIEW IF EXISTS pair_evidence_latest;
CREATE VIEW pair_evidence_latest AS
WITH gens AS (
  SELECT dedup_key, refine_version, refine_model, enrich_version, enrich_model, extractor_version,
    ROW_NUMBER() OVER (
      PARTITION BY dedup_key, refine_version, refine_model
      ORDER BY extractor_version DESC, enriched_at_ms DESC, id DESC
    ) AS g
  FROM pair_evidence
)
SELECT pe.*
FROM pair_evidence pe
JOIN gens ON gens.dedup_key = pe.dedup_key AND gens.refine_version = pe.refine_version
  AND gens.refine_model = pe.refine_model AND gens.enrich_version = pe.enrich_version
  AND gens.enrich_model = pe.enrich_model AND gens.extractor_version = pe.extractor_version
  AND gens.g = 1;
"""

INSERT_EVIDENCE = """
INSERT OR IGNORE INTO pair_evidence (
  dedup_key, refine_version, refine_model, pair_index, enrich_version, enrich_model, extractor_version,
  evidence_kind, file_path, incorrect_old, incorrect_new, correct_old, correct_new, note, source, enriched_at_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

EVIDENCE_COLUMNS = """\
  COALESCE(px.evidence_kind, ps.evidence_kind) AS evidence_kind,
  COALESCE(px.file_path, ps.file_path) AS evidence_file_path,
  COALESCE(px.incorrect_old, ps.incorrect_old) AS incorrect_old,
  COALESCE(px.incorrect_new, ps.incorrect_new) AS incorrect_new,
  COALESCE(px.correct_old, ps.correct_old) AS correct_old,
  COALESCE(px.correct_new, ps.correct_new) AS correct_new,
  COALESCE(px.note, ps.note) AS evidence_note,
  COALESCE(px.source, ps.source) AS evidence_source,
  COALESCE(px.enrich_version, ps.enrich_version) AS enrich_version,
  COALESCE(px.enrich_model, ps.enrich_model) AS enrich_model,
  COALESCE(px.extractor_version, ps.extractor_version) AS extractor_version"""

EVIDENCE_JOIN = """\
LEFT JOIN pair_evidence_latest px ON px.dedup_key = r.dedup_key AND px.refine_version = r.prompt_version
  AND px.refine_model = r.model AND px.pair_index = r.pair_index
LEFT JOIN pair_evidence_latest ps ON ps.dedup_key = r.dedup_key AND ps.refine_version = r.prompt_version
  AND ps.refine_model = r.model AND ps.pair_index = -1"""

REFINE_DDL = f"""
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
  r.model,
{EVIDENCE_COLUMNS}
FROM refinement r
JOIN gens ON gens.dedup_key = r.dedup_key AND gens.prompt_version = r.prompt_version
         AND gens.model = r.model AND gens.refined_at = r.refined_at AND gens.g = 1
JOIN feedback_events e ON e.dedup_key = r.dedup_key
LEFT JOIN accepted_pushback ap ON ap.dedup_key = r.dedup_key
{EVIDENCE_JOIN}
ORDER BY e.id, r.pair_index;
"""

INSERT_REFINEMENT = """
INSERT OR IGNORE INTO refinement (
  dedup_key, prompt_version, model, pair_index, action, complaint_verbatim, complaint, refined_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

CANDIDATES_QUERY = f"""
WITH latest_judge AS (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'judge'
),
latest_auditor AS (
  SELECT t.dedup_key, t.is_pushback, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.prompt_version DESC, t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'auditor'
),
judge_flip AS (
  SELECT dedup_key, COUNT(DISTINCT is_pushback) > 1 AS flipped
  FROM triage WHERE role = 'judge' GROUP BY dedup_key
),
refine_summary AS (
  SELECT r.dedup_key, COUNT(*) AS pair_count, g.prompt_version AS refine_version, g.model AS refine_model
  FROM refinement r
  JOIN (
    SELECT dedup_key, prompt_version, model, refined_at, ROW_NUMBER() OVER (
      PARTITION BY dedup_key ORDER BY prompt_version DESC, refined_at DESC
    ) AS gen
    FROM (SELECT DISTINCT dedup_key, prompt_version, model, refined_at FROM refinement)
  ) g ON g.dedup_key = r.dedup_key AND g.prompt_version = r.prompt_version
     AND g.model = r.model AND g.refined_at = r.refined_at AND g.gen = 1
  GROUP BY r.dedup_key
)
SELECT {EVENT_COLUMNS}, e.origin_path,
  j.category, j.is_pushback, j.confidence, j.prompt_version AS judge_version,
  j.model AS judge_model, j.what_claude_did,
  a.is_pushback AS auditor_is_pushback,
  COALESCE(f.flipped, 0) AS flipped,
  rs.pair_count, rs.refine_version, rs.refine_model
FROM feedback_events e
LEFT JOIN latest_judge j ON j.dedup_key = e.dedup_key AND j.rn = 1
LEFT JOIN latest_auditor a ON a.dedup_key = e.dedup_key AND a.rn = 1
LEFT JOIN judge_flip f ON f.dedup_key = e.dedup_key
LEFT JOIN refine_summary rs ON rs.dedup_key = e.dedup_key
ORDER BY e.id
"""

LINEAGE_VERDICTS_QUERY = (
    "SELECT role, prompt_version, model, category, is_pushback, what_claude_did, "
    "confidence, rationale, judged_at FROM triage WHERE dedup_key = ? ORDER BY role, prompt_version, id"
)

LINEAGE_PAIRS_QUERY = f"""
SELECT r.pair_index, r.action, r.complaint_verbatim, r.complaint, r.prompt_version, r.model,
{EVIDENCE_COLUMNS}
FROM refinement r
JOIN (
  SELECT prompt_version, model, refined_at, ROW_NUMBER() OVER (
    ORDER BY prompt_version DESC, refined_at DESC
  ) AS gen
  FROM (SELECT DISTINCT prompt_version, model, refined_at FROM refinement WHERE dedup_key = ?)
) g ON g.prompt_version = r.prompt_version AND g.model = r.model AND g.refined_at = r.refined_at AND g.gen = 1
{EVIDENCE_JOIN}
WHERE r.dedup_key = ?
ORDER BY r.pair_index
"""

UNENRICHED_QUERY = """
SELECT rp.dedup_key, rp.prompt_version AS refine_version, rp.model AS refine_model, rp.pair_index,
  rp.action, rp.complaint, rp.complaint_verbatim, rp.source_kind, rp.session_id, e.event_uuid, e.origin_path
FROM refined_pairs rp
JOIN feedback_events e ON e.dedup_key = rp.dedup_key
LEFT JOIN pair_evidence px ON px.dedup_key = rp.dedup_key AND px.refine_version = rp.prompt_version
  AND px.refine_model = rp.model AND px.pair_index = rp.pair_index
  AND px.enrich_version = ? AND px.enrich_model = ? AND px.extractor_version = ?
LEFT JOIN pair_evidence ps ON ps.dedup_key = rp.dedup_key AND ps.refine_version = rp.prompt_version
  AND ps.refine_model = rp.model AND ps.pair_index = -1
  AND ps.enrich_version = ? AND ps.enrich_model = ? AND ps.extractor_version = ?
WHERE px.id IS NULL AND ps.id IS NULL
ORDER BY rp.event_id, rp.pair_index"""


def now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


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
    cc-pushback's column names), the ``refinement`` table, the ``pair_evidence``
    table, and the ``accepted_pushback``, ``pair_evidence_latest``, and
    ``refined_pairs`` views. Verdicts, refinements, and evidence key on the
    content-derived dedup key, so they survive a database rebuild.

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
        return cls(await FileStateStore.open(path, extra_schema=FEEDBACK_DDL + TRIAGE_DDL + ENRICH_DDL + REFINE_DDL))

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

    async def unenriched(
        self, *, enrich_version: int, enrich_model: str, extractor_version: int, limit: int | None = None
    ) -> list[dict[str, object]]:
        """Returns refined pairs lacking code evidence at the given enrich generation.

        A pair counts as enriched when it carries its own ``pair_evidence`` row at
        exactly ``(enrich_version, enrich_model, extractor_version)`` or its refine
        generation carries the ``pair_index=-1`` no-code sentinel there. Pairs come
        from the latest refine generation only, so a refine re-run resurfaces its
        new pairs here automatically — and so does bumping any version in the key.

        Args:
            enrich_version: The enrich prompt version the evidence must carry.
            enrich_model: The resolved model name the evidence must carry.
            extractor_version: The platform's deterministic-extraction version.
            limit: When set, the maximum number of rows to return.

        Returns:
            One dict per unenriched pair with the columns the enrich prompt and
            anchor resolution need, oldest event first.
        """
        query = UNENRICHED_QUERY
        params: tuple[object, ...] = (enrich_version, enrich_model, extractor_version) * 2
        if limit is not None:
            query += " LIMIT ?"
            params = (*params, limit)
        cur = await self.store.conn.execute(query, params)
        return [dict(row) async for row in cur]

    async def record_evidence(
        self,
        key: DedupKey,
        evidence: CodeEvidence,
        *,
        refine_version: int,
        refine_model: str,
        pair_index: int,
        enrich_version: int,
        enrich_model: str,
        extractor_version: int,
        source: Source | None,
    ) -> None:
        """Records one pair's code evidence, idempotently, under the full generation key.

        Keyed by ``(dedup_key, refine_version, refine_model, pair_index,
        enrich_version, enrich_model, extractor_version)`` with ``INSERT OR
        IGNORE``, so re-running over an enriched corpus is a no-op and bumping any
        version in the key re-derives. ``pair_index=-1`` encodes the no-code
        sentinel covering every pair of the refine generation.

        Args:
            key: The enriched event's dedup key.
            evidence: The code evidence to persist.
            refine_version: The refine prompt version of the annotated pair.
            refine_model: The resolved refine model of the annotated pair.
            pair_index: The annotated pair's index, or ``-1`` for the sentinel.
            enrich_version: The enrich prompt version that produced the evidence.
            enrich_model: The resolved model name that produced it.
            extractor_version: The platform's deterministic-extraction version.
            source: Where the correction came from, or None when there is none.
        """
        await self.store.conn.execute(
            INSERT_EVIDENCE,
            (
                key,
                refine_version,
                refine_model,
                pair_index,
                enrich_version,
                enrich_model,
                extractor_version,
                evidence.kind,
                evidence.file_path,
                evidence.incorrect_edit.old if evidence.incorrect_edit else None,
                evidence.incorrect_edit.new if evidence.incorrect_edit else None,
                evidence.correct_edit.old if evidence.correct_edit else None,
                evidence.correct_edit.new if evidence.correct_edit else None,
                evidence.note,
                source,
                now_ms(),
            ),
        )

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

        Reads ``feedback_events``, ``triage``, and ``refinement`` directly — the
        views drop the auditor, the older judge versions, and the payload the
        lineage needs.

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
        pair_cur = await conn.execute(LINEAGE_PAIRS_QUERY, (dedup_key, dedup_key))
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
