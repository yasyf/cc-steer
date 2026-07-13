"""The ``UserPromptSubmit`` hook: pop the freshest live steer and surface it, fail-open on anything.

Claude Code runs this synchronously in every session on the machine, so it is built to never harm a
session: a hard ~200ms budget, a busy-timed WAL read, the kill switch checked before any database is
touched, and one outer guard that swallows every error into a silent exit 0. The most it ever does is
emit a single ``hookSpecificOutput.additionalContext`` carrying the steer; the least — and the default
on any surprise — is nothing.

Mirror mode records the would-be delivery and emits nothing (the mirror-week deliverable data); the
live modes emit for allowed projects, holding out a deterministic fraction and stopping at the daily
budget. The daemon queues; this hook resolves and is the only writer of a delivery's terminal state.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cc_steer.watcher.live import (
    LiveConfig,
    State,
    format_additional_context,
    is_killed,
    shadow_db_path,
    today_prefix,
)

if TYPE_CHECKING:
    from types import FrameType

BUDGET_S = 0.2
HOOK_EVENT = "UserPromptSubmit"

SELECT_FRESHEST = """
SELECT d.id, d.proposal_id, d.holdout, p.steer
FROM deliveries d JOIN proposals p ON p.id = d.proposal_id
WHERE d.session_id = ? AND d.state = 'queued'
ORDER BY d.ts DESC, d.id DESC LIMIT 1
"""

EXPIRE_STALE = """
UPDATE deliveries SET state = 'expired', decided_at = ?
WHERE session_id = ? AND state = 'queued' AND ttl <= ?
"""


def delivered_today(conn: sqlite3.Connection, at: datetime) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM deliveries WHERE state = 'delivered' AND substr(decided_at, 1, 10) = ?",
        (today_prefix(at),),
    ).fetchone()[0]


def decide(config: LiveConfig, *, cwd: str, held_out: bool, delivered: int) -> tuple[State, bool]:
    """The delivery verdict for one popped steer: the terminal state to record and whether to emit."""
    match config.mode:
        case "mirror":
            return ("mirror", False)
        case "live_allow" if not config.allows(cwd):
            return ("mirror", False)
        case "live_allow" | "live_all":
            if held_out:
                return ("holdout", False)
            if delivered >= config.max_live_per_day:
                return ("suppressed_budget", False)
            return ("delivered", True)
        case "shadow":
            return ("mirror", False)


def resolve(conn: sqlite3.Connection, config: LiveConfig, *, session_id: str, cwd: str, at: datetime) -> str | None:
    """Expires stale steers, pops the session's freshest queued one, records its verdict, returns the emission."""
    stamp = at.isoformat()
    conn.execute(EXPIRE_STALE, (stamp, session_id, stamp))
    if (row := conn.execute(SELECT_FRESHEST, (session_id,)).fetchone()) is None:
        return None
    state, emit = decide(config, cwd=cwd, held_out=bool(row["holdout"]), delivered=delivered_today(conn, at))
    conn.execute("UPDATE deliveries SET state = ?, decided_at = ? WHERE id = ?", (state, stamp, row["id"]))
    return format_additional_context(int(row["proposal_id"]), row["steer"]) if emit else None


def additional_context() -> str | None:
    """The steer to surface for the current prompt, or None; every failure mode returns None."""
    if is_killed():
        return None
    payload = json.loads(sys.stdin.read() or "{}")
    if not (session_id := payload.get("session_id")) or not (cwd := payload.get("cwd")):
        return None
    if (config := LiveConfig.load()).mode == "shadow" or not (path := shadow_db_path()).exists():
        return None
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=BUDGET_S)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 100")
        return resolve(conn, config, session_id=session_id, cwd=os.path.realpath(cwd), at=datetime.now(UTC))
    finally:
        conn.close()


def _budget_exceeded(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError


def run() -> None:
    """The hook entrypoint: guard the budget, resolve fail-open, emit at most one context, exit 0."""
    signal.signal(signal.SIGALRM, _budget_exceeded)
    signal.setitimer(signal.ITIMER_REAL, BUDGET_S)
    try:
        context = additional_context()
    except BaseException:
        context = None
    signal.setitimer(signal.ITIMER_REAL, 0)
    if context is not None:
        json.dump({"hookSpecificOutput": {"hookEventName": HOOK_EVENT, "additionalContext": context}}, sys.stdout)
    raise SystemExit(0)
