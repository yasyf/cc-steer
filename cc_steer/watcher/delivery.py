"""Proposal delivery: where a finished cascade verdict goes.

Shadow mode is the only implemented delivery: :class:`ShadowDelivery` appends
every proposal to a local SQLite ledger and never touches the session. A
``CaptHookDelivery`` that injects the steer into the live session comes later,
gated on shadow metrics — hit rate against real interventions and nuisance
rate — proving the cascade is worth interrupting a session for.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

import aiosqlite
from cc_transcript.mining.store import now

if TYPE_CHECKING:
    from types import TracebackType

    from cc_steer.watcher.types import ScoredMoment, SteerProposal

SHADOW_SCHEMA_COMPONENT = "cc-steer-shadow-v1"
SHADOW_SCHEMA_VERSION = 1
EXPECTED_SHADOW_DDL_FINGERPRINT = "cdc1f1fc38a8a0bbe17a72a14ed7eb07bf38b78f5c612d28bee1b0f1c9e2d274"
EXPECTED_SHADOW_OBJECT_FINGERPRINT = "282ca47626f69efb53d6cd50bbf4040b7c29dc9f64a3159c100026824a4a39ae"

SHADOW_SCHEMA_DDL = """CREATE TABLE cc_steer_shadow_schema_v1 (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  component TEXT NOT NULL CHECK (component = 'cc-steer-shadow-v1'),
  schema_version INTEGER NOT NULL CHECK (schema_version = 1),
  ddl_fingerprint TEXT NOT NULL CHECK (length(ddl_fingerprint) = 64),
  object_fingerprint TEXT NOT NULL CHECK (length(object_fingerprint) = 64)
);
CREATE TABLE proposals (
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
CREATE TABLE scored_moments (
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
CREATE TABLE deliveries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  proposal_id INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  project TEXT,
  ts TEXT NOT NULL,
  mode TEXT NOT NULL,
  ttl TEXT NOT NULL,
  holdout INTEGER NOT NULL,
  state TEXT NOT NULL,
  decided_at TEXT,
  UNIQUE(proposal_id)
);
CREATE TABLE reactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  proposal_id INTEGER NOT NULL,
  delivery_id INTEGER,
  kind TEXT NOT NULL,
  source TEXT NOT NULL,
  feedback_dedup_key TEXT,
  similarity REAL,
  ts TEXT NOT NULL,
  UNIQUE(proposal_id)
);
CREATE TABLE scored_outcomes (
  session_id TEXT NOT NULL,
  turn_index INTEGER NOT NULL,
  ts TEXT NOT NULL,
  fired INTEGER NOT NULL,
  steered INTEGER NOT NULL,
  steer_turn INTEGER,
  steer_dedup_key TEXT,
  distance INTEGER,
  radius INTEGER NOT NULL,
  resolved_at TEXT NOT NULL,
  UNIQUE(session_id, turn_index)
);
"""

_SCHEMA_DDL_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
    }
)
_SCHEMA_DML_ACTIONS = frozenset({sqlite3.SQLITE_DELETE, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE})
_PROTECTED_SCHEMA_TABLES = frozenset({"cc_steer_shadow_schema_v1", "sqlite_master", "sqlite_schema"})
_PROTECTED_PRAGMAS = frozenset({"user_version", "writable_schema"})

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


def shadow_ddl_fingerprint() -> str:
    return hashlib.sha256(b"cc-steer-shadow-ddl-v1\0" + SHADOW_SCHEMA_DDL.encode()).hexdigest()


async def shadow_object_fingerprint(conn: aiosqlite.Connection) -> str:
    digest = hashlib.sha256(b"cc-steer-shadow-objects-v1\0")
    rows = await conn.execute_fetchall("SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY type, name")
    for object_type, name, table, statement in rows:
        for field in (object_type, name, table, statement or ""):
            digest.update(str(field).encode())
            digest.update(b"\0")
    return digest.hexdigest()


async def _required_row(conn: aiosqlite.Connection, statement: str) -> sqlite3.Row:
    row = await (await conn.execute(statement)).fetchone()
    if row is None:
        raise RuntimeError(f"shadow schema query returned no row: {statement}")
    return row


async def verify_shadow_schema(conn: aiosqlite.Connection) -> None:
    version = int((await _required_row(conn, "PRAGMA user_version"))[0])
    if version != SHADOW_SCHEMA_VERSION:
        raise RuntimeError(f"shadow schema version {version}, want exactly {SHADOW_SCHEMA_VERSION}")
    row = await (
        await conn.execute(
            "SELECT component, schema_version, ddl_fingerprint, object_fingerprint "
            "FROM cc_steer_shadow_schema_v1 WHERE id=1"
        )
    ).fetchone()
    if row is None:
        raise RuntimeError("shadow schema identity row is missing")
    component, marker_version, stored_ddl, stored_objects = row
    if component != SHADOW_SCHEMA_COMPONENT:
        raise RuntimeError(f"shadow schema component {component!r}, want exactly {SHADOW_SCHEMA_COMPONENT!r}")
    if marker_version != SHADOW_SCHEMA_VERSION:
        raise RuntimeError(f"shadow marker version {marker_version}, want exactly {SHADOW_SCHEMA_VERSION}")
    if stored_ddl != EXPECTED_SHADOW_DDL_FINGERPRINT:
        raise RuntimeError(f"shadow DDL fingerprint {stored_ddl!r}, want exactly {EXPECTED_SHADOW_DDL_FINGERPRINT!r}")
    if stored_objects != EXPECTED_SHADOW_OBJECT_FINGERPRINT:
        raise RuntimeError(
            f"shadow stored object fingerprint {stored_objects!r}, want exactly {EXPECTED_SHADOW_OBJECT_FINGERPRINT!r}"
        )
    if (actual := await shadow_object_fingerprint(conn)) != EXPECTED_SHADOW_OBJECT_FINGERPRINT:
        raise RuntimeError(f"shadow object fingerprint {actual!r}, want exactly {EXPECTED_SHADOW_OBJECT_FINGERPRINT!r}")


async def create_shadow_schema(conn: aiosqlite.Connection) -> None:
    for statement in SHADOW_SCHEMA_DDL.split(";"):
        if statement := statement.strip():
            await conn.execute(statement)
    await conn.execute(f"PRAGMA user_version = {SHADOW_SCHEMA_VERSION}")
    if (actual := await shadow_object_fingerprint(conn)) != EXPECTED_SHADOW_OBJECT_FINGERPRINT:
        raise RuntimeError(f"shadow object fingerprint {actual!r}, want exactly {EXPECTED_SHADOW_OBJECT_FINGERPRINT!r}")
    await conn.execute(
        "INSERT INTO cc_steer_shadow_schema_v1"
        "(id, component, schema_version, ddl_fingerprint, object_fingerprint) VALUES(1, ?, 1, ?, ?)",
        (SHADOW_SCHEMA_COMPONENT, EXPECTED_SHADOW_DDL_FINGERPRINT, EXPECTED_SHADOW_OBJECT_FINGERPRINT),
    )


def _authorize_exact_shadow_schema(
    action: int,
    argument1: str | None,
    argument2: str | None,
    database: str | None,
    _source: str | None,
) -> int:
    if action == sqlite3.SQLITE_ATTACH:
        return sqlite3.SQLITE_DENY
    if database == "main":
        if action in _SCHEMA_DDL_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action in _SCHEMA_DML_ACTIONS and (argument1 or "").casefold() in _PROTECTED_SCHEMA_TABLES:
            return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA and argument2 is not None and (argument1 or "").casefold() in _PROTECTED_PRAGMAS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


async def open_shadow_sqlite(path: Path) -> aiosqlite.Connection:
    if shadow_ddl_fingerprint() != EXPECTED_SHADOW_DDL_FINGERPRINT:
        raise RuntimeError(
            f"shadow compiled DDL fingerprint {shadow_ddl_fingerprint()!r}, "
            f"want exactly {EXPECTED_SHADOW_DDL_FINGERPRINT!r}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path), isolation_level=None)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA busy_timeout=2000")
    committed = False
    created = False
    try:
        await conn.execute("BEGIN IMMEDIATE")
        version = int((await _required_row(conn, "PRAGMA user_version"))[0])
        row = await _required_row(
            conn,
            "SELECT count(*) FROM sqlite_schema "
            "WHERE type IN ('table', 'index', 'trigger', 'view') "
            "AND lower(substr(name, 1, 7)) <> 'sqlite_'",
        )
        created = version == 0 and int(row[0]) == 0
        await (create_shadow_schema(conn) if created else verify_shadow_schema(conn))
        await conn.execute("COMMIT")
        committed = True
    finally:
        if not committed:
            if conn.in_transaction:
                await conn.execute("ROLLBACK")
            await conn.close()
    if created:
        path.chmod(0o600)
    mode = str((await _required_row(conn, "PRAGMA journal_mode=WAL"))[0])
    if mode != "wal":
        await conn.close()
        raise RuntimeError(f"enable shadow WAL: mode {mode!r}")
    await conn.set_authorizer(_authorize_exact_shadow_schema)
    return conn


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
        return cls(await open_shadow_sqlite(target))

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
