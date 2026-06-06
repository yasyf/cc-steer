from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

from cc_transcript.store import FileStateStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from sqlite3 import Connection
    from types import TracebackType

    from cc_pushback.models import FeedbackCandidate, SourceKind

__all__ = ["MatchRow", "Repository", "Stats", "now"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_cursors (
  source_key TEXT PRIMARY KEY,
  cursor TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dedup_key TEXT NOT NULL UNIQUE,
  source_kind TEXT NOT NULL,
  session_id TEXT,
  pr_ref TEXT,
  origin_path TEXT,
  origin_uuid TEXT,
  occurred_at TEXT NOT NULL,
  text TEXT NOT NULL,
  payload_json TEXT,
  context_json TEXT NOT NULL,
  cc_version TEXT,
  ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_source ON feedback_events(source_kind);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback_events(session_id);
CREATE TABLE IF NOT EXISTS pattern_matches (
  feedback_id INTEGER NOT NULL REFERENCES feedback_events(id) ON DELETE CASCADE,
  pattern_name TEXT NOT NULL,
  backend TEXT NOT NULL,
  taxonomy_version TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  severity TEXT,
  what_claude_did TEXT,
  rule TEXT,
  novel INTEGER NOT NULL DEFAULT 0,
  model TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (feedback_id, pattern_name, taxonomy_version, prompt_version, backend)
);
"""

INSERT_EVENT = """
INSERT OR IGNORE INTO feedback_events (
  dedup_key, source_kind, session_id, pr_ref, origin_path, origin_uuid,
  occurred_at, text, payload_json, context_json, cc_version, ingested_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPSERT_CURSOR = """
INSERT INTO source_cursors (source_key, cursor, updated_at) VALUES (?, ?, ?)
ON CONFLICT(source_key) DO UPDATE SET cursor = excluded.cursor, updated_at = excluded.updated_at
"""

INSERT_MATCH = """
INSERT OR IGNORE INTO pattern_matches (
  feedback_id, pattern_name, backend, taxonomy_version, prompt_version,
  severity, what_claude_did, rule, novel, model, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def now() -> str:
    return datetime.now(UTC).isoformat()


def event_row(candidate: FeedbackCandidate, ingested_at: str) -> tuple[object, ...]:
    return (
        candidate.dedup_key,
        candidate.source_kind,
        candidate.session_id,
        candidate.pr_ref,
        str(candidate.origin_path) if candidate.origin_path else None,
        candidate.origin_uuid,
        candidate.occurred_at.isoformat(),
        candidate.text,
        json.dumps(dict(candidate.payload)) if candidate.payload is not None else None,
        candidate.context.to_json(),
        candidate.cc_version,
        ingested_at,
    )


@dataclass(frozen=True, slots=True)
class MatchRow:
    """One row of the ``pattern_matches`` table, ready to persist.

    Cheap matcher rows carry ``backend='matcher'`` and leave the model-only
    columns ``None``; language-model rows fill ``severity``, ``what_claude_did``,
    ``rule``, and ``model``.

    Attributes:
        feedback_id: The classified event's ``feedback_events`` id.
        pattern_name: The taxonomy or novel pattern name.
        backend: The classifier that produced the row, e.g. ``matcher`` or ``claude``.
        taxonomy_version: The taxonomy version this classification covers.
        prompt_version: The prompt version this classification covers.
        severity: The model-assessed severity, or ``None`` for matcher rows.
        what_claude_did: The behavior that drew the pushback, or ``None``.
        rule: The corrective rule, or ``None`` for matcher rows.
        novel: ``1`` when ``pattern_name`` is a model-proposed novel pattern.
        model: The provider model name, or ``None`` for matcher rows.
        created_at: When the row was produced.
    """

    feedback_id: int
    pattern_name: str
    backend: str
    taxonomy_version: str
    prompt_version: str
    severity: str | None
    what_claude_did: str | None
    rule: str | None
    novel: int
    model: str | None
    created_at: str


def match_insert_row(row: MatchRow) -> tuple[object, ...]:
    return (
        row.feedback_id,
        row.pattern_name,
        row.backend,
        row.taxonomy_version,
        row.prompt_version,
        row.severity,
        row.what_claude_did,
        row.rule,
        row.novel,
        row.model,
        row.created_at,
    )


@dataclass(frozen=True, slots=True)
class Stats:
    """A snapshot of ingestion progress.

    Attributes:
        total: The total feedback events ingested.
        files: The number of scanned files recorded.
        by_source: Event counts keyed by source kind.
        cursors: Stored cursor values keyed by source key.
    """

    total: int
    files: int
    by_source: Mapping[str, int]
    cursors: Mapping[str, str]


class Repository:
    """Persistent store for ingested feedback over a :class:`FileStateStore`.

    Wraps the transcript ingestion-state store with the ``cc-pushback`` schema:
    feedback events, source cursors, and the (deliverable-2) pattern-match
    table. Every write codepath composes with the file-mtime ledger inside a
    single transaction, so a scan that records a file and inserts its candidates
    commits atomically or not at all.

    Example:
        >>> with Repository.open(Repository.default_path()) as repo:
        ...     repo.record_file_scan(str(path), mtime, candidates)
    """

    def __init__(self, store: FileStateStore) -> None:
        self.store = store

    @staticmethod
    def default_path() -> Path:
        """Returns the default database path, ``~/.cc-pushback/feedback.db``."""
        return Path.home() / ".cc-pushback" / "feedback.db"

    @classmethod
    def open(cls, path: Path) -> Self:
        """Opens (creating if needed) the repository database at ``path``."""
        return cls(FileStateStore.open(path, extra_schema=SCHEMA))

    def close(self) -> None:
        """Closes the underlying store."""
        self.store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def file_mtimes(self) -> dict[str, float]:
        """Returns the recorded ``path`` to ``mtime`` map for incremental scans."""
        return self.store.file_mtimes()

    def cursor_for(self, source_key: str) -> str | None:
        """Returns the stored cursor for ``source_key``, or ``None`` if unseen."""
        row = self.store.conn.execute(
            "SELECT cursor FROM source_cursors WHERE source_key = ?", (source_key,)
        ).fetchone()
        return row["cursor"] if row else None

    def record_file_scan(self, path: str, mtime: float, candidates: Sequence[FeedbackCandidate]) -> int:
        """Records a scanned file and its candidates in one transaction.

        Upserts the file's mtime and inserts every candidate with
        ``INSERT OR IGNORE`` so re-ingesting the same file is a no-op.

        Args:
            path: The scanned file's path.
            mtime: The file's modification time at scan.
            candidates: The candidates extracted from the file.

        Returns:
            The number of newly inserted feedback events.
        """
        ingested_at = now()
        with self.store.transaction() as conn:
            before = self.total_changes(conn)
            conn.executemany(INSERT_EVENT, [event_row(candidate, ingested_at) for candidate in candidates])
            self.store.record_file(path, mtime)
            return self.total_changes(conn) - before - 1

    def advance_github_cursor(self, source_key: str, cursor: str, candidates: Sequence[FeedbackCandidate]) -> int:
        """Advances a source cursor and inserts its candidates in one transaction.

        Args:
            source_key: The cursor's source key, e.g. ``github:owner/repo``.
            cursor: The new cursor value to persist.
            candidates: The candidates discovered up to ``cursor``.

        Returns:
            The number of newly inserted feedback events.
        """
        ingested_at = now()
        with self.store.transaction() as conn:
            before = self.total_changes(conn)
            conn.executemany(INSERT_EVENT, [event_row(candidate, ingested_at) for candidate in candidates])
            conn.execute(UPSERT_CURSOR, (source_key, cursor, ingested_at))
            return self.total_changes(conn) - before - 1

    def save_matches(self, rows: Sequence[MatchRow]) -> int:
        """Persists classification rows with ``INSERT OR IGNORE``.

        The ``pattern_matches`` primary key makes re-running a classification a
        no-op, so this is safe to call repeatedly.

        Args:
            rows: The classification rows to persist.

        Returns:
            The number of newly inserted rows.
        """
        with self.store.transaction() as conn:
            before = self.total_changes(conn)
            conn.executemany(INSERT_MATCH, [match_insert_row(row) for row in rows])
            return self.total_changes(conn) - before

    def stats(self) -> Stats:
        """Returns ingestion counts by source kind, file count, and cursors."""
        conn = self.store.conn
        return Stats(
            total=conn.execute("SELECT COUNT(*) AS n FROM feedback_events").fetchone()["n"],
            files=conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"],
            by_source={
                row["source_kind"]: row["n"]
                for row in conn.execute(
                    "SELECT source_kind, COUNT(*) AS n FROM feedback_events GROUP BY source_kind ORDER BY source_kind"
                )
            },
            cursors={
                row["source_key"]: row["cursor"]
                for row in conn.execute("SELECT source_key, cursor FROM source_cursors ORDER BY source_key")
            },
        )

    def recent(self, *, source_kind: SourceKind | None = None, limit: int = 20) -> list[dict[str, object]]:
        """Returns the most recent feedback events, newest first.

        Args:
            source_kind: When set, restrict to this source kind.
            limit: The maximum number of rows to return.

        Returns:
            One dict per event with its stored columns.
        """
        query = "SELECT source_kind, occurred_at, text FROM feedback_events"
        params: tuple[object, ...] = ()
        if source_kind is not None:
            query += " WHERE source_kind = ?"
            params = (source_kind,)
        query += " ORDER BY occurred_at DESC, id DESC LIMIT ?"
        return [dict(row) for row in self.store.conn.execute(query, (*params, limit))]

    @staticmethod
    def total_changes(conn: Connection) -> int:
        return conn.total_changes
