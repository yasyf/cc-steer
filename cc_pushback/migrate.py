"""One-time conversion of a pre-2.0 corpus onto the cc-transcript 2.0 shapes.

Legacy databases persisted ``context_json`` as the pre-2.0 ``ContextSnapshot``
shape and carried the event uuid in an ``origin_uuid`` column. ``migrate-corpus``
rewrites every legacy row into a ``cc-transcript.context/1`` document — previews
only, labeled summary fidelity, ``origin='migrated'``, ``anchor`` null where the
row has no resolvable uuid — and adds the columns the platform shapes need
(``feedback_events.event_uuid``, ``triage.fidelity``). Idempotent: rows already
in the new schema are skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import zip_longest
from typing import TYPE_CHECKING

from cc_transcript.context import ContextWindow, SchemaError, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import aiosqlite

    from cc_pushback.store import FeedbackStore

MIGRATED_PREVIEW_CHARS = 2000


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """The outcome of one ``migrate-corpus`` pass.

    Attributes:
        migrated: How many rows were converted to the new context schema.
        skipped: How many rows already carried it.
    """

    migrated: int
    skipped: int


def turn_preview(turn: Mapping[str, Any]) -> str:
    tools = "".join(
        f"\n  {name}({input})" if input else f"\n  {name}()"
        for name, input in zip_longest(turn["tool_calls"], turn.get("tool_inputs", ()), fillvalue="")
    )
    return f"{turn['role']}: {turn['text']}{tools}"


def migrated_turn(turn: Mapping[str, Any]) -> TurnRef:
    return TurnRef(
        role="user" if turn["role"] == "user" else "assistant",
        refs=(),
        preview=turn_preview(turn),
        tool_digests=(),
    )


def window_from_snapshot(context_json: str, anchor: EventRef | None) -> ContextWindow:
    data = json.loads(context_json)
    return ContextWindow(
        anchor=anchor,
        before=tuple(migrated_turn(turn) for turn in data["before"]),
        trigger=migrated_turn(data["trigger"]) if data["trigger"] else None,
        after=tuple(migrated_turn(turn) for turn in data["after"]),
        fidelity="summary",
        preview_chars=MIGRATED_PREVIEW_CHARS,
        origin="migrated",
    )


def anchor_of(row: Mapping[str, Any]) -> EventRef | None:
    match row["session_id"], row["event_uuid"]:
        case str() as session, str() as uuid:
            return EventRef(SessionId(session), EventUuid(uuid))
        case _:
            return None


async def column_names(conn: aiosqlite.Connection, table: str) -> set[str]:
    cur = await conn.execute(f"PRAGMA table_info({table})")
    return {str(row["name"]) async for row in cur}


async def migrate_corpus(store: FeedbackStore) -> MigrationReport:
    """Converts a legacy corpus in place, idempotently.

    Adds the ``event_uuid`` column (backfilled from the legacy ``origin_uuid``)
    and the ``triage.fidelity`` column (existing verdicts were judged on baked
    summaries, so they default to ``'summary'`` and stay re-judgeable via
    ``triage --refresh-summary``), then rewrites every legacy ``context_json``
    snapshot into a ``cc-transcript.context/1`` document.

    Args:
        store: The open feedback store to migrate.

    Returns:
        The pass's migrated/skipped row counts.
    """
    conn = store.store.conn
    if "event_uuid" not in await column_names(conn, "feedback_events"):
        await conn.execute("ALTER TABLE feedback_events ADD COLUMN event_uuid TEXT")
        await conn.execute("UPDATE feedback_events SET event_uuid = origin_uuid")
    if "fidelity" not in await column_names(conn, "triage"):
        await conn.execute("ALTER TABLE triage ADD COLUMN fidelity TEXT NOT NULL DEFAULT 'summary'")
    cur = await conn.execute("SELECT id, session_id, event_uuid, context_json FROM feedback_events")
    rows = [dict(row) async for row in cur]
    migrated = skipped = 0
    for row in rows:
        try:
            ContextWindow.from_json(str(row["context_json"]))
        except SchemaError:
            window = window_from_snapshot(str(row["context_json"]), anchor_of(row))
            await conn.execute(
                "UPDATE feedback_events SET context_json = ? WHERE id = ?", (window.to_json(), row["id"])
            )
            migrated += 1
        else:
            skipped += 1
    return MigrationReport(migrated=migrated, skipped=skipped)
