"""The scored-moment outcome resolver: did the user actually steer near each gate decision.

:mod:`cc_steer.watcher.shadow` measures the watcher against real interventions with a
coarse join — a fired proposal counts as a hit when *any* mined intervention lands in
the same session within 30 minutes. That join only ever sees fired proposals, reads
its interventions from the mined ``feedback_events`` corpus, and can credit an
unrelated later steer to a proposal.

This resolver keys ground truth to every scored moment instead. For each
:class:`~cc_steer.watcher.delivery.ShadowDelivery` scored row — fire *and* no-fire — it
runs :func:`cc_steer.detectors.detect` fresh over the session transcript, maps each
detected steer to its turn, and records whether the user steered on or within
``radius`` turns of the scored turn. The result is a ``scored_outcomes`` table beside
the shadow ledger, one row per scored moment, carrying the gate's own fire bit next to
the resolved outcome — the exact ``(fired, steered)`` pair stream
:mod:`cc_steer.watcher.wsr` reads to bound the gate's live precision and recall.

Both sides come from one :class:`~cc_transcript.activity.SessionActivity` so the
scored ``turn_index`` and the steer turns share an index space; a session whose
transcript has aged out is skipped, never guessed. Run it nightly:
``python -m cc_steer.watcher.outcomes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import aiosqlite
from cc_transcript.activity import SessionActivity
from cc_transcript.discovery import TranscriptExpiredError, resolve
from cc_transcript.ids import SessionId
from cc_transcript.mining.store import now
from cc_transcript.parser import parse

from cc_steer.detectors import detect
from cc_steer.watcher.delivery import ShadowDelivery, open_shadow_sqlite
from cc_steer.watcher.wsr import FireOutcome

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

    from cc_transcript.models import TranscriptEvent

DEFAULT_RADIUS = 1

INSERT_OUTCOME = """
INSERT INTO scored_outcomes (
  session_id, turn_index, ts, fired, steered, steer_turn, steer_dedup_key, distance, radius, resolved_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(session_id, turn_index) DO UPDATE SET
  ts = excluded.ts,
  fired = excluded.fired,
  steered = excluded.steered,
  steer_turn = excluded.steer_turn,
  steer_dedup_key = excluded.steer_dedup_key,
  distance = excluded.distance,
  radius = excluded.radius,
  resolved_at = excluded.resolved_at
"""


@dataclass(frozen=True, slots=True)
class SteerTurn:
    """One real steer the user made, located in the session's turn index space.

    Attributes:
        turn_index: The turn the steering message anchors to.
        occurred_at: When the steer landed, ISO-8601.
        dedup_key: The detected candidate's key — the outcome's link to the steer.
    """

    turn_index: int
    occurred_at: str
    dedup_key: str


@dataclass(frozen=True, slots=True)
class Outcome:
    """One scored moment's resolved ground truth.

    Attributes:
        session_id: The session the scored moment came from.
        turn_index: The scored turn — the key into the shadow ledger's scored moments.
        ts: When the moment was scored, ISO-8601.
        fired: Whether the gate cleared its threshold on this moment.
        steered: Whether a real steer landed within ``radius`` turns of it.
        steer_turn: The matched steer's turn, or None when the user did not steer.
        steer_dedup_key: The matched steer's key, or None.
        distance: Turns between the scored moment and the matched steer, or None.
    """

    session_id: str
    turn_index: int
    ts: str
    fired: bool
    steered: bool
    steer_turn: int | None
    steer_dedup_key: str | None
    distance: int | None


@dataclass(frozen=True, slots=True)
class OutcomeReport:
    """One resolver pass's headline counts.

    Attributes:
        scored: Scored rows read from the ledger.
        resolved: Outcomes written — scored rows in sessions whose transcript survives.
        sessions: Sessions whose transcript resolved.
        expired_sessions: Sessions skipped because their transcript aged out.
        fired: Resolved outcomes whose gate fired.
        steered: Resolved outcomes that landed near a real steer.
    """

    scored: int
    resolved: int
    sessions: int
    expired_sessions: int
    fired: int
    steered: int

    def summary_line(self) -> str:
        return (
            f"resolved {self.resolved}/{self.scored} scored moments across {self.sessions} sessions "
            f"({self.expired_sessions} expired) — fired {self.fired}, steered {self.steered}"
        )


def steer_turns(session_id: str, events: Sequence[TranscriptEvent]) -> list[SteerTurn]:
    """Every real steer in the session, located by turn — :func:`detect` mapped through its activity."""
    activity = SessionActivity.from_events(SessionId(session_id), events)
    return [
        SteerTurn(turn.index, candidate.occurred_at.isoformat(), str(candidate.dedup_key))
        for candidate in detect(events)
        if (turn := activity.turn_of(candidate.ref)) is not None
    ]


def nearest(scored_turn: int, steers: Sequence[SteerTurn], *, radius: int) -> SteerTurn | None:
    """The closest steer within ``radius`` turns of ``scored_turn``, ties broken by earliest turn."""
    return min(
        (steer for steer in steers if abs(steer.turn_index - scored_turn) <= radius),
        key=lambda steer: (abs(steer.turn_index - scored_turn), steer.turn_index),
        default=None,
    )


def resolve_session(
    scored_rows: Sequence[Mapping[str, object]],
    session_id: str,
    events: Sequence[TranscriptEvent],
    *,
    radius: int = DEFAULT_RADIUS,
) -> list[Outcome]:
    """Resolves every scored row in one session against the steers its transcript actually holds."""
    steers = steer_turns(session_id, events)
    return [
        Outcome(
            session_id=session_id,
            turn_index=(turn_index := int(str(row["turn_index"]))),
            ts=str(row["ts"]),
            fired=bool(row["gate_passed"]),
            steered=(match := nearest(turn_index, steers, radius=radius)) is not None,
            steer_turn=match.turn_index if match else None,
            steer_dedup_key=match.dedup_key if match else None,
            distance=abs(match.turn_index - turn_index) if match else None,
        )
        for row in scored_rows
    ]


def default_load_events(session_id: str, *, root: Path | None = None) -> list[TranscriptEvent] | None:
    """The session's transcript events, or None when no transcript resolves; raises when it aged out."""
    if (path := resolve(SessionId(session_id), root=root)) is None:
        return None
    return list(parse(path).events)


class OutcomeStore:
    """The ``scored_outcomes`` table beside the shadow ledger — the one write path for resolved outcomes."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    @classmethod
    async def open(cls, path: Path | None = None) -> Self:
        """Opens (creating if needed) the outcomes table in the shared ledger at ``path``."""
        target = path or ShadowDelivery.default_path()
        return cls(await open_shadow_sqlite(target))

    async def close(self) -> None:
        await self.conn.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    async def record(self, outcome: Outcome, *, radius: int) -> None:
        """Writes one outcome; a duplicate ``(session, turn)`` refreshes it, last observation wins."""
        await self.conn.execute(
            INSERT_OUTCOME,
            (
                outcome.session_id,
                outcome.turn_index,
                outcome.ts,
                int(outcome.fired),
                int(outcome.steered),
                outcome.steer_turn,
                outcome.steer_dedup_key,
                outcome.distance,
                radius,
                now(),
            ),
        )

    async def pairs(self) -> list[FireOutcome]:
        """Every resolved outcome as a fire-outcome pair, oldest first — the WSR stream."""
        cur = await self.conn.execute("SELECT fired, steered FROM scored_outcomes ORDER BY ts, session_id, turn_index")
        return [FireOutcome(fired=bool(row["fired"]), steered=bool(row["steered"])) async for row in cur]


def group_by_session(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    """The scored rows bucketed by session id, each bucket in the order read."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["session_id"]), []).append(row)
    return grouped


async def resolve_outcomes(
    *,
    shadow_db: Path | None = None,
    root: Path | None = None,
    radius: int = DEFAULT_RADIUS,
    load_events: Callable[[str], list[TranscriptEvent] | None] | None = None,
) -> OutcomeReport:
    """Resolves every scored moment's outcome and writes the ``scored_outcomes`` table.

    Reads the shadow ledger's scored moments, groups them by session, and for each
    session whose transcript still resolves runs :func:`detect` over it and matches
    every scored turn against the real steers. Sessions whose transcript has aged out
    are skipped and counted, never guessed. Idempotent: a rerun refreshes each row.

    Args:
        shadow_db: The shared ledger path; None uses the default.
        root: The transcript discovery root; None uses ``~/.claude/projects``.
        radius: How many turns from a scored moment a steer may land and still count.
        load_events: The session-to-events loader, injectable for tests; None resolves
            each session's transcript from disk.

    Returns:
        The :class:`OutcomeReport` for this pass.
    """
    loader = load_events or (lambda session_id: default_load_events(session_id, root=root))
    async with await ShadowDelivery.open(shadow_db) as ledger:
        scored = await ledger.scored_moments()
    outcomes: list[Outcome] = []
    sessions = expired = 0
    for session_id, rows in group_by_session(scored).items():
        try:
            events = loader(session_id)
        except TranscriptExpiredError:
            expired += 1
            continue
        if events is None:
            expired += 1
            continue
        outcomes.extend(resolve_session(rows, session_id, events, radius=radius))
        sessions += 1
    async with await OutcomeStore.open(shadow_db) as store:
        for outcome in outcomes:
            await store.record(outcome, radius=radius)
    return OutcomeReport(
        scored=len(scored),
        resolved=len(outcomes),
        sessions=sessions,
        expired_sessions=expired,
        fired=sum(outcome.fired for outcome in outcomes),
        steered=sum(outcome.steered for outcome in outcomes),
    )


def main() -> None:
    import anyio

    print(anyio.run(resolve_outcomes).summary_line())


if __name__ == "__main__":
    main()
