from __future__ import annotations

from cc_pushback.models import ContextSnapshot, ContextTurn


def test_snapshot_roundtrips_through_json() -> None:
    snapshot = ContextSnapshot(
        before=(ContextTurn(role="user", text="do X"),),
        trigger=ContextTurn(role="assistant", text="did Y", tool_calls=("Edit", "Write")),
        after=(ContextTurn(role="tool", text="ok"),),
    )

    assert ContextSnapshot.from_json(snapshot.to_json()) == snapshot


def test_snapshot_without_trigger_roundtrips() -> None:
    snapshot = ContextSnapshot(before=(), trigger=None, after=())

    restored = ContextSnapshot.from_json(snapshot.to_json())

    assert restored == snapshot
    assert restored.trigger is None


def test_turn_tool_calls_default_empty() -> None:
    assert ContextTurn(role="assistant", text="hi").tool_calls == ()
