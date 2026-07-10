from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.types import SteerProposal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


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
