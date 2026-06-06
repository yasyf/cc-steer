from __future__ import annotations

from cc_pushback.context import build_snapshot
from tests.builders import assistant_text, assistant_tool_use, parse, user_text


def test_window_caps_before_and_after() -> None:
    events = parse(
        [user_text(f"u{i}") for i in range(10)]
        + [assistant_text("trigger")]
        + [user_text("feedback")]
        + [user_text(f"a{i}") for i in range(5)]
    )
    snapshot = build_snapshot(events, 11, before=6, after=2)

    assert len(snapshot.before) == 6
    assert len(snapshot.after) == 2
    assert snapshot.before[-1].text == "trigger"
    assert snapshot.before[0].text == "u5"


def test_trigger_is_nearest_preceding_assistant() -> None:
    events = parse(
        [user_text("u0"), assistant_text("earlier"), user_text("u1"), assistant_text("nearest"), user_text("fb")]
    )
    snapshot = build_snapshot(events, 4)

    assert snapshot.trigger is not None
    assert snapshot.trigger.text == "nearest"


def test_assistant_text_truncated_to_limit() -> None:
    events = parse([assistant_text("x" * 5000), user_text("fb")])
    snapshot = build_snapshot(events, 1)

    assert snapshot.trigger is not None
    assert len(snapshot.trigger.text) == 2000


def test_tool_calls_captured_in_turn() -> None:
    events = parse([assistant_tool_use("t1", "Edit", {"file_path": "/a"}), user_text("fb")])
    snapshot = build_snapshot(events, 1)

    assert snapshot.trigger is not None
    assert snapshot.trigger.tool_calls == ("Edit",)


def test_lower_bound_stops_reach_back() -> None:
    events = parse([assistant_text("hidden"), user_text("u1"), assistant_text("visible"), user_text("fb")])
    snapshot = build_snapshot(events, 3, lower_bound=2)

    assert [turn.text for turn in snapshot.before] == ["visible"]
    assert snapshot.trigger is not None
    assert snapshot.trigger.text == "visible"
