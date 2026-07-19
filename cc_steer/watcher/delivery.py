"""Proposal delivery: where a finished cascade verdict goes.

Shadow mode is the only implemented delivery: :class:`ShadowDelivery` appends
every proposal to a local SQLite ledger and never touches the session. A
``CaptHookDelivery`` that injects the steer into the live session comes later,
gated on shadow metrics — hit rate against real interventions and nuisance
rate — proving the cascade is worth interrupting a session for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

import aiosqlite
from cc_transcript.mining.store import now

if TYPE_CHECKING:
    from types import TracebackType

    from cc_steer.watcher.types import ScoredMoment, SteerProposal

SHADOW_DDL = """
CREATE TABLE IF NOT EXISTS proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  anchor_uuid TEXT NOT NULL,
  turn_index INTEGER NOT NULL,
  ts TEXT NOT NULL,
  gate_score REAL,
  sentinel_prob REAL,
  draft TEXT,
  steer TEXT,
  window_render TEXT,
  project TEXT,
  exemplar_keys TEXT NOT NULL,
  stage_versions TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(session_id, anchor_uuid)
);
"""

SCORED_MOMENTS_DDL = """
CREATE TABLE IF NOT EXISTS scored_moments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  turn_index INTEGER NOT NULL,
  ts TEXT NOT NULL,
  project TEXT,
  gate_score REAL NOT NULL,
  gate_threshold REAL NOT NULL,
  gate_passed INTEGER NOT NULL,
  stage2_prob REAL,
  stage2_threshold REAL,
  created_at TEXT NOT NULL,
  UNIQUE(session_id, turn_index)
);
"""

INSERT_PROPOSAL = """
INSERT OR IGNORE INTO proposals (
  session_id, anchor_uuid, turn_index, ts, gate_score, sentinel_prob, draft, steer,
  window_render, project, exemplar_keys, stage_versions, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_SCORED = """
INSERT INTO scored_moments (
  session_id, turn_index, ts, project, gate_score, gate_threshold, gate_passed,
  stage2_prob, stage2_threshold, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(session_id, turn_index) DO UPDATE SET
  ts = excluded.ts,
  project = excluded.project,
  gate_score = excluded.gate_score,
  gate_threshold = excluded.gate_threshold,
  gate_passed = excluded.gate_passed,
  stage2_prob = excluded.stage2_prob,
  stage2_threshold = excluded.stage2_threshold
"""


async def ensure_columns(conn: aiosqlite.Connection) -> None:
    cur = await conn.execute("PRAGMA table_info(proposals)")
    columns = {row["name"] async for row in cur}
    for column in ("window_render", "project"):
        if column not in columns:
            await conn.execute(f"ALTER TABLE proposals ADD COLUMN {column} TEXT")


class SteerDelivery(Protocol):
    """Anything that takes a finished proposal off the watcher's hands."""

    async def deliver(self, proposal: SteerProposal) -> None: ...


class ScoredSink(Protocol):
    """Anything that durably records a scored moment for shadow analysis."""

    async def record_scored(self, moment: ScoredMoment) -> None: ...


class ShadowDelivery:
    """Records every proposal in the local shadow ledger; never touches a session.

    Idempotent by ``(session_id, anchor_uuid)`` — a replayed proposal for the
    same anchored moment is a no-op, so daemon restarts never duplicate rows.

    Example:
        >>> async with await ShadowDelivery.open() as delivery:
        ...     await delivery.deliver(proposal)
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    @staticmethod
    def default_path() -> Path:
        """Returns the default ledger path, ``~/.cc-steer/shadow.db``."""
        return Path.home() / ".cc-steer" / "shadow.db"

    @classmethod
    async def open(cls, path: Path | None = None) -> Self:
        """Opens (creating if needed) the shadow ledger at ``path``."""
        target = path or cls.default_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(target), isolation_level=None)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=2000")
        await conn.executescript(SHADOW_DDL + SCORED_MOMENTS_DDL)
        await ensure_columns(conn)
        return cls(conn)

    async def close(self) -> None:
        """Closes the underlying connection."""
        await self.conn.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def deliver(self, proposal: SteerProposal) -> None:
        """Appends one proposal row; a duplicate ``(session, anchor)`` is a no-op."""
        await self.conn.execute(
            INSERT_PROPOSAL,
            (
                proposal.session_id,
                proposal.anchor_uuid,
                proposal.turn_index,
                proposal.ts,
                proposal.gate_score,
                proposal.sentinel_prob,
                proposal.draft,
                proposal.steer,
                proposal.window_render,
                proposal.project,
                json.dumps(list(proposal.exemplar_keys)),
                proposal.stage_versions,
                now(),
            ),
        )

    async def proposals(self) -> list[dict[str, object]]:
        """Returns every recorded proposal, oldest first."""
        cur = await self.conn.execute("SELECT * FROM proposals ORDER BY id")
        return [dict(row) async for row in cur]

    async def record_scored(self, moment: ScoredMoment) -> None:
        """Records one scored-moment row; a duplicate ``(session, turn)`` refreshes it, last observation wins."""
        await self.conn.execute(
            INSERT_SCORED,
            (
                moment.session_id,
                moment.turn_index,
                moment.ts,
                moment.project,
                moment.gate_score,
                moment.gate_threshold,
                int(moment.gate_passed),
                moment.stage2_prob,
                moment.stage2_threshold,
                now(),
            ),
        )

    async def scored_moments(self, *, since: str | None = None) -> list[dict[str, object]]:
        """Returns scored moments at or after ISO-8601 ``since`` (every one when None), oldest first."""
        where = "" if since is None else " WHERE ts >= ?"
        params = () if since is None else (since,)
        cur = await self.conn.execute(f"SELECT * FROM scored_moments{where} ORDER BY id", params)
        return [dict(row) async for row in cur]

    async def scored_count(self) -> int:
        """Returns the grand total of scored moments, all time — cheap, no row materialization."""
        cur = await self.conn.execute("SELECT COUNT(*) FROM scored_moments")
        row = await cur.fetchone()
        return int(row[0]) if row is not None else 0
