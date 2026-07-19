from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import aiosqlite
import anyio
import pytest

from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.types import ScoredMoment, SteerProposal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

LEGACY_DDL = """
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
  exemplar_keys TEXT NOT NULL,
  stage_versions TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(session_id, anchor_uuid)
);
"""


def make_proposal(**overrides: object) -> SteerProposal:
    return dataclasses.replace(
        SteerProposal(
            session_id="sess-live",
            anchor_uuid="a1",
            turn_index=3,
            ts="2026-07-07T10:00:00+00:00",
            gate_score=1.0,
            draft="draft steer",
            steer="final steer",
            exemplar_keys=("k-train",),
            stage_versions=json.dumps({"stage2_model": "medium"}),
            window_render="<user>\nplease do step\n\n<assistant>\ndid step",
        ),
        **overrides,
    )


def make_scored(**overrides: object) -> ScoredMoment:
    return dataclasses.replace(
        ScoredMoment(
            session_id="sess-live",
            turn_index=3,
            ts="2026-07-07T10:00:00+00:00",
            gate_score=0.42,
            gate_threshold=0.5,
            gate_passed=False,
        ),
        **overrides,
    )


async def test_shadow_delivery_round_trips_a_proposal(tmp_path: Path) -> None:
    async with await ShadowDelivery.open(tmp_path / "shadow.db") as delivery:
        await delivery.deliver(make_proposal())
        rows = await delivery.proposals()
    assert len(rows) == 1
    row = rows[0]
    assert (row["session_id"], row["anchor_uuid"], row["turn_index"]) == ("sess-live", "a1", 3)
    assert (row["ts"], row["gate_score"]) == ("2026-07-07T10:00:00+00:00", 1.0)
    assert (row["draft"], row["steer"]) == ("draft steer", "final steer")
    assert row["window_render"] == "<user>\nplease do step\n\n<assistant>\ndid step"
    assert json.loads(str(row["exemplar_keys"])) == ["k-train"]
    assert json.loads(str(row["stage_versions"])) == {"stage2_model": "medium"}
    assert row["created_at"]


async def test_shadow_delivery_survives_a_duplicate_anchor(tmp_path: Path) -> None:
    async with await ShadowDelivery.open(tmp_path / "shadow.db") as delivery:
        await delivery.deliver(make_proposal())
        await delivery.deliver(make_proposal(steer="replayed"))
        await delivery.deliver(make_proposal(anchor_uuid="a2", turn_index=9))
        rows = await delivery.proposals()
    assert [(row["anchor_uuid"], row["steer"]) for row in rows] == [("a1", "final steer"), ("a2", "final steer")]


async def test_shadow_delivery_stores_abstentions_as_nulls_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "shadow.db"
    async with await ShadowDelivery.open(path) as delivery:
        await delivery.deliver(make_proposal(draft=None, steer=None, exemplar_keys=()))
    async with await ShadowDelivery.open(path) as reopened:
        rows = await reopened.proposals()
    assert len(rows) == 1
    assert (rows[0]["draft"], rows[0]["steer"]) == (None, None)
    assert json.loads(str(rows[0]["exemplar_keys"])) == []


async def test_shadow_delivery_round_trips_the_sentinel_prob(tmp_path: Path) -> None:
    async with await ShadowDelivery.open(tmp_path / "shadow.db") as delivery:
        await delivery.deliver(make_proposal(sentinel_prob=0.2373))
        await delivery.deliver(make_proposal(anchor_uuid="a2", turn_index=9))
        rows = await delivery.proposals()
    assert [row["sentinel_prob"] for row in rows] == [0.2373, None]


async def test_scored_moments_round_trip_gate_and_stage2_fields(tmp_path: Path) -> None:
    path = tmp_path / "shadow.db"
    async with await ShadowDelivery.open(path) as delivery:
        await delivery.record_scored(make_scored(project="/repo"))
        await delivery.record_scored(
            make_scored(turn_index=7, gate_score=0.91, gate_passed=True, stage2_prob=0.2, stage2_threshold=0.6)
        )
    async with await ShadowDelivery.open(path) as reopened:
        rows = await reopened.scored_moments()
    assert len(rows) == 2
    suppressed, fired = rows
    assert (suppressed["turn_index"], suppressed["gate_passed"], suppressed["project"]) == (3, 0, "/repo")
    assert (suppressed["gate_score"], suppressed["gate_threshold"]) == (0.42, 0.5)
    assert (suppressed["stage2_prob"], suppressed["stage2_threshold"]) == (None, None)
    assert (fired["turn_index"], fired["gate_passed"]) == (7, 1)
    assert (fired["stage2_prob"], fired["stage2_threshold"]) == (0.2, 0.6)
    assert suppressed["created_at"] and fired["created_at"]


async def test_scored_moments_refresh_on_session_and_turn_conflict(tmp_path: Path) -> None:
    async with await ShadowDelivery.open(tmp_path / "shadow.db") as delivery:
        await delivery.record_scored(make_scored(gate_score=0.42))
        await delivery.record_scored(make_scored(gate_score=0.99, gate_passed=True, stage2_prob=0.1, stage2_threshold=0.6))
        await delivery.record_scored(make_scored(turn_index=9, gate_score=0.77))
        rows = await delivery.scored_moments()
    assert [
        (row["turn_index"], row["gate_score"], row["gate_passed"], row["stage2_prob"], row["stage2_threshold"])
        for row in rows
    ] == [(3, 0.99, 1, 0.1, 0.6), (9, 0.77, 0, None, None)]


async def test_open_sets_wal_and_busy_timeout(tmp_path: Path) -> None:
    async with await ShadowDelivery.open(tmp_path / "shadow.db") as delivery:
        mode = await (await delivery.conn.execute("PRAGMA journal_mode")).fetchone()
        timeout = await (await delivery.conn.execute("PRAGMA busy_timeout")).fetchone()
    assert mode is not None and mode[0] == "wal"
    assert timeout is not None and timeout[0] == 2000


async def test_record_scored_waits_out_a_concurrent_immediate_writer(tmp_path: Path) -> None:
    path = tmp_path / "shadow.db"
    async with await ShadowDelivery.open(path) as delivery:
        holder = await aiosqlite.connect(str(path), isolation_level=None)
        try:
            await holder.execute("BEGIN IMMEDIATE")
            await holder.execute(
                "INSERT INTO scored_moments "
                "(session_id, turn_index, ts, gate_score, gate_threshold, gate_passed, created_at) "
                "VALUES ('holder', 0, ?, 0.1, 0.5, 0, ?)",
                ("2026-07-07T10:00:00+00:00", "2026-07-07T10:00:00+00:00"),
            )
            async with anyio.create_task_group() as tg:
                tg.start_soon(delivery.record_scored, make_scored())
                await anyio.sleep(0.2)
                await holder.execute("COMMIT")
        finally:
            await holder.close()
        rows = await delivery.scored_moments()
    assert {row["session_id"] for row in rows} == {"holder", "sess-live"}


async def test_scored_moments_bounds_by_since_and_counts_all(tmp_path: Path) -> None:
    async with await ShadowDelivery.open(tmp_path / "shadow.db") as delivery:
        await delivery.record_scored(make_scored(turn_index=0, ts="2026-01-01T00:00:00+00:00"))
        await delivery.record_scored(make_scored(turn_index=1, ts="2026-07-01T00:00:00+00:00"))
        await delivery.record_scored(make_scored(turn_index=2, ts="2026-07-17T00:00:00+00:00"))
        recent = await delivery.scored_moments(since="2026-06-01T00:00:00+00:00")
        total = await delivery.scored_count()
        everything = await delivery.scored_moments()
    assert [row["turn_index"] for row in recent] == [1, 2]
    assert total == 3
    assert [row["turn_index"] for row in everything] == [0, 1, 2]


async def test_open_migrates_a_legacy_ledger_missing_window_render(tmp_path: Path) -> None:
    path = tmp_path / "shadow.db"
    conn = await aiosqlite.connect(str(path), isolation_level=None)
    await conn.executescript(LEGACY_DDL)
    await conn.execute(
        "INSERT INTO proposals "
        "(session_id, anchor_uuid, turn_index, ts, exemplar_keys, stage_versions, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("legacy", "a0", 1, "2026-07-01T00:00:00+00:00", "[]", "{}", "2026-07-01T00:00:00+00:00"),
    )
    await conn.close()
    async with await ShadowDelivery.open(path) as delivery:
        await delivery.deliver(make_proposal(anchor_uuid="a1", window_render="fresh render"))
        rows = await delivery.proposals()
    assert [(row["anchor_uuid"], row["window_render"]) for row in rows] == [("a0", None), ("a1", "fresh render")]
