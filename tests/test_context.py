from __future__ import annotations

import pytest

from cc_pushback.context import ContextSnapshot, ContextTurn, build_snapshot
from tests.builders import assistant_text, parse, user_text

# 0:A0  1:U1  2:A2(trigger)  3:U3(feedback)  4:A4  5:U5
EVENTS = parse(
    [
        assistant_text("A0"),
        user_text("U1"),
        assistant_text("A2"),
        user_text("U3 feedback"),
        assistant_text("A4"),
        user_text("U5"),
    ]
)


@pytest.mark.unit
def test_trigger_is_nearest_preceding_assistant_turn() -> None:
    snapshot = build_snapshot(EVENTS, 3)
    assert snapshot.trigger is not None
    assert snapshot.trigger.role == "assistant"
    assert snapshot.trigger.text == "A2"


@pytest.mark.unit
def test_window_runs_before_to_after_in_order() -> None:
    snapshot = build_snapshot(EVENTS, 3)
    assert [(t.role, t.text) for t in snapshot.before] == [
        ("assistant", "A0"),
        ("user", "U1"),
        ("assistant", "A2"),
    ]
    assert [(t.role, t.text) for t in snapshot.after] == [("assistant", "A4"), ("user", "U5")]


@pytest.mark.unit
def test_lower_bound_clamps_the_before_window() -> None:
    snapshot = build_snapshot(EVENTS, 3, lower_bound=2)
    assert [t.text for t in snapshot.before] == ["A2"]


@pytest.mark.unit
def test_assistant_turn_records_tool_calls() -> None:
    snapshot = build_snapshot(EVENTS, 3)
    assert snapshot.before[0].tool_calls == ()


@pytest.mark.unit
def test_snapshot_json_round_trips() -> None:
    snapshot = ContextSnapshot(
        before=(ContextTurn(role="user", text="hi"),),
        trigger=ContextTurn(role="assistant", text="did x", tool_calls=("Edit",)),
        after=(),
    )
    assert ContextSnapshot.from_json(snapshot.to_json()) == snapshot
