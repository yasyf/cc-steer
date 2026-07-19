"""The evidence query surface: full-text search over refined pairs and their code corrections.

Indexes the pipeline's refined-pair deliverable — each atomic direction plus the
grounding correction the enrich stage wrote to cc-transcript's shared ledger — into
an FTS5 table inside the feedback database, and answers BM25 queries over it. The
join to the shared ``corrections`` ledger is the dashboard's anchor join
(:func:`cc_steer.dashboard.evidence_resolver`): each pair's code evidence folds into
the searchable text, so a query hits the file and edit a steer produced, not only
its words. The index rebuilds lazily whenever the refined corpus or the shared
corrections ledger changes, keyed on the refined-pair count, the max feedback event
id, and a revision of the ``SOURCE``-scoped ledger (its row count and newest
timestamp), so a query rebuilds before reading whenever any of those has moved.
With ``rerank`` the BM25 shortlist is re-ordered by the exemplar index's MMR
machinery, trading pure lexical rank for embedding relevance diversified across the
results.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from cc_transcript.corrections import CorrectionLog
from cc_transcript.ids import EventUuid, SessionId

from cc_steer.enrich import SOURCE
from cc_steer.report import EvidenceRow, project_label
from cc_steer.store import FeedbackStore

if TYPE_CHECKING:
    from pathlib import Path

    from cc_transcript.corrections import Correction

BM25_LIMIT = 50
BUSY_TIMEOUT_MS = 2_000
WORD = re.compile(r"\w+")

EVIDENCE_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
  verbatim, direction, evidence,
  category UNINDEXED, repo UNINDEXED, source UNINDEXED
);
CREATE TABLE IF NOT EXISTS evidence_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

STALENESS_QUERY = (
    "SELECT (SELECT COUNT(*) FROM refined_pairs) AS n, (SELECT COALESCE(MAX(id), 0) FROM feedback_events) AS m"
)

CORRECTIONS_REVISION_QUERY = (
    f"SELECT COUNT(*) AS c, COALESCE(MAX(ts_ms), 0) AS t FROM corrections WHERE source = '{SOURCE}'"
)

REFINED_PAIRS_QUERY = """
SELECT direction_verbatim, direction, category, source_kind, origin_path, session_id, event_uuid
FROM refined_pairs
ORDER BY event_id, pair_index
"""

INSERT_HIT = (
    "INSERT INTO evidence_fts (rowid, verbatim, direction, evidence, category, repo, source) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)

UPSERT_STALENESS = (
    "INSERT INTO evidence_meta (key, value) VALUES ('staleness', ?) "
    "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
)


@dataclass(frozen=True, slots=True)
class EvidenceHit:
    """One refined pair retrieved from the evidence index.

    Attributes:
        pair_id: The pair's ordinal within the current index generation, in
            ``(event_id, pair_index)`` order.
        repo: The repository the steer happened in, from the transcript's origin
            path, or None when the pair carries none.
        category: The judge's category for the parent steering event.
        verbatim: The user's verbatim steering excerpt for this pair.
        direction: The distilled one-sentence direction.
        score: Retrieval relevance, higher is better — negated BM25 by default, or
            the MMR query-cosine under ``rerank``.
        source: The feedback source kind the pair was mined from.
    """

    pair_id: int
    repo: str | None
    category: str
    verbatim: str
    direction: str
    score: float
    source: str


async def open_corrections() -> CorrectionLog:
    return await CorrectionLog.open()


def match_terms(query: str) -> str:
    return " ".join(f'"{token}"' for token in WORD.findall(query))


async def staleness_key(conn: sqlite3.Connection, log: CorrectionLog) -> str:
    pairs = conn.execute(STALENESS_QUERY).fetchone()
    revision = (await log.sql(CORRECTIONS_REVISION_QUERY))[0]
    return f"{pairs['n']}:{pairs['m']}:{revision['c']}:{revision['t']}"


async def evidence_text(log: CorrectionLog, session_id: object, event_uuid: object) -> str:
    if not (session_id and event_uuid):
        return ""
    corrections: tuple[Correction, ...] = await log.for_anchor(SessionId(str(session_id)), EventUuid(str(event_uuid)))
    row = next((EvidenceRow.from_correction(c) for c in corrections if c.source == SOURCE), None)
    if row is None:
        return ""
    return "\n".join(part for part in (row.file_path, *row.incorrect, *(row.correct or ())) if part)


async def rebuild_index(conn: sqlite3.Connection, log: CorrectionLog, key: str) -> None:
    rows = conn.execute(REFINED_PAIRS_QUERY).fetchall()
    conn.execute("DELETE FROM evidence_fts")
    conn.executemany(
        INSERT_HIT,
        [
            (
                index,
                str(row["direction_verbatim"]),
                str(row["direction"]),
                await evidence_text(log, row["session_id"], row["event_uuid"]),
                str(row["category"]),
                project_label(str(row["origin_path"])) if row["origin_path"] else None,
                str(row["source_kind"]),
            )
            for index, row in enumerate(rows, start=1)
        ],
    )
    conn.execute(UPSERT_STALENESS, (key,))
    conn.commit()


def bm25_shortlist(conn: sqlite3.Connection, match: str, repo: str | None) -> list[EvidenceHit]:
    clause, params = ("", (match, BM25_LIMIT)) if repo is None else (" AND repo = ?", (match, repo, BM25_LIMIT))
    return [
        EvidenceHit(
            pair_id=int(row["rowid"]),
            repo=row["repo"],
            category=str(row["category"]),
            verbatim=str(row["verbatim"]),
            direction=str(row["direction"]),
            score=-float(row["rank"]),
            source=str(row["source"]),
        )
        for row in conn.execute(
            "SELECT rowid, verbatim, direction, category, repo, source, bm25(evidence_fts) AS rank "
            f"FROM evidence_fts WHERE evidence_fts MATCH ?{clause} ORDER BY rank LIMIT ?",
            params,
        )
    ]


def rerank_shortlist(query: str, shortlist: list[EvidenceHit], limit: int) -> list[EvidenceHit]:
    import numpy as np

    from cc_steer.exemplars import EMBED_MODEL, mmr_select, query_encoder

    if not shortlist:
        return []
    encoder = query_encoder(EMBED_MODEL)
    vectors = encoder.encode([f"{hit.verbatim}\n{hit.direction}" for hit in shortlist])
    matrix = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return [
        replace(shortlist[index], score=similarity)
        for index, similarity in mmr_select(encoder.encode([query])[0], matrix, top_n=len(shortlist), k=limit)
    ]


async def search(
    query: str, *, repo: str | None = None, limit: int = 10, rerank: bool = False, db: Path | None = None
) -> list[EvidenceHit]:
    """Searches the refined-pair evidence index and returns the best hits.

    Opens (and, on the first call or after the refined corpus changes, rebuilds)
    the FTS5 index inside the feedback database, matches ``query`` by BM25 to a
    top-50 shortlist, then truncates to ``limit``. With ``rerank`` the shortlist is
    re-ordered by the exemplar index's MMR embedder instead — which requires an
    embedding backend (``VOYAGE_API_KEY`` for the default model).

    Args:
        query: The free-text query; its word tokens become AND-ed FTS terms.
        repo: When set, restricts hits to that repository label.
        limit: The maximum number of hits to return.
        rerank: When True, re-rank the BM25 shortlist with the MMR embedder.
        db: The feedback database path; defaults to ``~/.cc-steer/feedback.db``.

    Returns:
        The matching :class:`EvidenceHit` rows, most relevant first.
    """
    if not (match := match_terms(query)):
        return []
    conn = sqlite3.connect(db or FeedbackStore.default_path())
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    try:
        conn.executescript(EVIDENCE_DDL)
        log = await open_corrections()
        try:
            key = await staleness_key(conn, log)
            stored = conn.execute("SELECT value FROM evidence_meta WHERE key = 'staleness'").fetchone()
            if stored is None or stored["value"] != key:
                await rebuild_index(conn, log, key)
        finally:
            await log.close()
        shortlist = bm25_shortlist(conn, match, repo)
    finally:
        conn.close()
    return rerank_shortlist(query, shortlist, limit) if rerank else shortlist[:limit]
