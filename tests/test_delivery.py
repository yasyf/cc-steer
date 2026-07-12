from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import aiosqlite
import pytest

from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.types import SteerProposal

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
