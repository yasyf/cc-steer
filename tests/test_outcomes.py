from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from cc_transcript.discovery import TranscriptExpiredError

from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.outcomes import (
    OutcomeStore,
    SteerTurn,
    nearest,
    resolve_outcomes,
    resolve_session,
    steer_turns,
)
from cc_steer.watcher.wsr import FireOutcome
from tests.builders import assistant_text, parse, user_text
from tests.test_delivery import make_scored

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cc_transcript.models import TranscriptEvent

pytestmark = pytest.mark.anyio

SESSION = "sess-1"


def transcript() -> list[TranscriptEvent]:
    """turn 0 asks; turns 1 and 2 are real steers the detector recognizes."""
    return parse(
        [
            user_text("please implement the parser"),
            assistant_text("I'll use a regex-based approach for this"),
            user_text("no, don't use regex, write a proper recursive descent parser instead"),
            assistant_text("switching to recursive descent"),
            user_text("actually revert all of that and start over with a simpler design"),
            assistant_text("reverting"),
        ]
    )


# --- detection to turns -----------------------------------------------------


def test_steer_turns_locates_every_real_steer() -> None:
    turns = steer_turns(SESSION, transcript())
    assert [turn.turn_index for turn in turns] == [1, 2]
    assert all(turn.dedup_key for turn in turns)


def test_steer_turns_is_empty_without_steers() -> None:
    events = parse([user_text("hello"), assistant_text("hi, how can I help?")])
    assert steer_turns(SESSION, events) == []


# --- pure nearest-turn matching --------------------------------------------


def steer(turn: int, key: str = "k") -> SteerTurn:
    return SteerTurn(turn_index=turn, occurred_at="2026-07-07T10:00:00+00:00", dedup_key=key)


def test_nearest_picks_the_closest_within_radius() -> None:
    picked = nearest(5, [steer(3, "far"), steer(6, "close"), steer(8, "farther")], radius=2)
    assert picked is not None and picked.dedup_key == "close"


def test_nearest_breaks_ties_toward_the_earlier_turn() -> None:
    picked = nearest(5, [steer(6, "after"), steer(4, "before")], radius=1)
    assert picked is not None and picked.dedup_key == "before"


def test_nearest_returns_none_outside_the_radius() -> None:
    assert nearest(5, [steer(8), steer(1)], radius=1) is None
    assert nearest(5, [], radius=3) is None


# --- session resolution -----------------------------------------------------


def scored_row(turn_index: int, *, gate_passed: bool, ts: str = "2026-07-07T10:00:00+00:00") -> dict[str, object]:
    return {"session_id": SESSION, "turn_index": turn_index, "ts": ts, "gate_passed": int(gate_passed)}


def test_resolve_session_labels_fire_and_no_fire_rows() -> None:
    rows: Sequence[dict[str, object]] = [
        scored_row(0, gate_passed=True),  # steer at turn 1 is one turn away
        scored_row(1, gate_passed=False),  # steer sits on this very turn
        scored_row(5, gate_passed=True),  # no steer within radius
    ]
    by_turn = {out.turn_index: out for out in resolve_session(rows, SESSION, transcript(), radius=1)}
    assert (by_turn[0].fired, by_turn[0].steered, by_turn[0].distance) == (True, True, 1)
    assert (by_turn[1].fired, by_turn[1].steered, by_turn[1].distance) == (False, True, 0)
    assert (by_turn[5].fired, by_turn[5].steered) == (True, False)
    assert by_turn[5].steer_turn is None and by_turn[5].distance is None
    assert by_turn[1].steer_dedup_key is not None


def test_resolve_session_radius_zero_demands_an_exact_turn() -> None:
    [out] = resolve_session([scored_row(0, gate_passed=True)], SESSION, transcript(), radius=0)
    assert out.steered is False


# --- the end-to-end resolver over a real ledger -----------------------------


async def seed_scored(db: Path, turns: Sequence[tuple[int, bool]]) -> None:
    async with await ShadowDelivery.open(db) as ledger:
        for turn_index, gate_passed in turns:
            await ledger.record_scored(make_scored(session_id=SESSION, turn_index=turn_index, gate_passed=gate_passed))


async def test_resolve_outcomes_writes_the_table_and_pairs(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    await seed_scored(db, [(0, True), (1, False), (5, True)])
    report = await resolve_outcomes(shadow_db=db, radius=1, load_events=lambda _sid: transcript())
    assert (report.scored, report.resolved, report.sessions, report.expired_sessions) == (3, 3, 1, 0)
    assert (report.fired, report.steered) == (2, 2)
    async with await OutcomeStore.open(db) as store:
        pairs = await store.pairs()
        rows = {int(row["turn_index"]): row async for row in await store.conn.execute("SELECT * FROM scored_outcomes")}
    assert rows[0]["steered"] == 1 and rows[0]["distance"] == 1
    assert rows[5]["steered"] == 0 and rows[5]["distance"] is None
    assert sorted(pairs, key=lambda p: (p.fired, p.steered)) == sorted(
        [FireOutcome(True, True), FireOutcome(False, True), FireOutcome(True, False)],
        key=lambda p: (p.fired, p.steered),
    )


async def test_resolve_outcomes_skips_an_aged_out_transcript(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    await seed_scored(db, [(0, True)])

    def loader(_session_id: str) -> list[TranscriptEvent]:
        raise TranscriptExpiredError(SESSION)

    report = await resolve_outcomes(shadow_db=db, load_events=loader)
    assert (report.scored, report.resolved, report.sessions, report.expired_sessions) == (1, 0, 0, 1)
    async with await OutcomeStore.open(db) as store:
        assert await store.pairs() == []


async def test_resolve_outcomes_counts_a_missing_transcript_as_expired(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    await seed_scored(db, [(0, True)])
    report = await resolve_outcomes(shadow_db=db, load_events=lambda _sid: None)
    assert report.expired_sessions == 1 and report.resolved == 0


async def test_resolve_outcomes_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    await seed_scored(db, [(0, True), (1, False)])
    first = await resolve_outcomes(shadow_db=db, radius=1, load_events=lambda _sid: transcript())
    second = await resolve_outcomes(shadow_db=db, radius=1, load_events=lambda _sid: transcript())
    assert first == second
    async with await OutcomeStore.open(db) as store:
        [[count]] = [[row[0]] async for row in await store.conn.execute("SELECT COUNT(*) FROM scored_outcomes")]
    assert count == 2


def test_module_runs_as_main_over_an_empty_ledger(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cc_steer.watcher.outcomes"],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "resolved 0/0 scored moments" in result.stdout
