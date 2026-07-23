from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from cc_transcript.activity import SessionActivity
from cc_transcript.filterspec import event_meta
from cc_transcript.ids import EventRef, SessionId
from cc_transcript.mining.sampling import fold_trigger
from cc_transcript.watch import WatchEvent

from cc_steer.capture import capture_window
from cc_steer.watcher.daemon import Watcher, session_cwd
from tests.builders import assistant_text, mode_entry, parse, user_text
from tests.test_delivery import make_proposal

if TYPE_CHECKING:
    from cc_transcript.context import ContextWindow

    from cc_steer.watcher.types import SteerProposal

pytestmark = pytest.mark.anyio

SESSION = "sess-live"


class RecordingCascade:
    def __init__(self, result: SteerProposal | None = None) -> None:
        self.result = result
        self.calls: list[tuple[str, int, str, ContextWindow, str | None]] = []

    async def evaluate(
        self, session_id: str, *, turn_index: int, anchor_uuid: str, window: ContextWindow, project: str | None = None
    ) -> SteerProposal | None:
        self.calls.append((session_id, turn_index, anchor_uuid, window, project))
        return self.result


class CollectingDelivery:
    def __init__(self) -> None:
        self.delivered: list[SteerProposal] = []

    async def deliver(self, proposal: SteerProposal) -> None:
        self.delivered.append(proposal)


def entries(pairs: int, *, start: int = 0, session: str = SESSION) -> list[dict[str, Any]]:
    return [
        entry
        for index in range(start, start + pairs)
        for entry in (
            user_text(f"please do step {index}", sessionId=session, uuid=f"u{index}"),
            assistant_text(f"did step {index}", sessionId=session, uuid=f"a{index}"),
        )
    ]


def watch_events(items: list[dict[str, Any]], *, session: str = SESSION, sidechain: bool = False) -> list[WatchEvent]:
    return [
        WatchEvent(path=Path(f"/{session}.jsonl"), session_id=SessionId(session), is_sidechain=sidechain, event=event)
        for event in parse(items)
    ]


def watcher_with(cascade: RecordingCascade, delivery: CollectingDelivery | None = None, **overrides: Any) -> Watcher:
    return Watcher(cascade, delivery or CollectingDelivery(), roots=(), **overrides)


async def test_evaluate_session_targets_the_last_completed_turn() -> None:
    cascade = RecordingCascade()
    watcher = watcher_with(cascade)
    watcher.ingest(watch_events(entries(8)), now=0.0)
    assert await watcher.evaluate_session(SESSION) is None
    [(session_id, turn_index, anchor_uuid, window, project)] = cascade.calls
    assert (session_id, turn_index, anchor_uuid, project) == (SESSION, 6, "a6", "/repo")
    assert len(window.before) == 6
    assert window.trigger is None
    assert window.after == ()
    assert "did step 6" in window.before[-1].preview
    assert "step 7" not in window.before[-1].preview


async def test_live_window_is_byte_compatible_with_training_capture() -> None:
    cascade = RecordingCascade()
    watcher = watcher_with(cascade)
    watcher.ingest(watch_events(entries(8)), now=0.0)
    await watcher.evaluate_session(SESSION)
    activity = SessionActivity.from_events(SessionId(SESSION), parse(entries(8)))
    meta = next(meta for event in activity.turns[6].events if (meta := event_meta(event)) is not None)
    anchor = EventRef(SessionId(SESSION), meta.uuid)
    expected = fold_trigger(capture_window(activity, anchor, before=6, after=0, preview_chars=200), keep=6)
    assert cascade.calls[0][3].to_json() == expected.to_json()


async def test_the_in_flight_turn_alone_is_never_evaluated() -> None:
    cascade = RecordingCascade()
    watcher = watcher_with(cascade)
    watcher.ingest(watch_events(entries(1)), now=0.0)
    assert await watcher.evaluate_session(SESSION) is None
    assert cascade.calls == []


async def test_unchanged_turn_count_skips_the_cascade() -> None:
    cascade = RecordingCascade()
    watcher = watcher_with(cascade)
    watcher.ingest(watch_events(entries(4)), now=0.0)
    await watcher.evaluate_session(SESSION)
    watcher.ingest(watch_events([assistant_text("still going", sessionId=SESSION, uuid="a3-extra")]), now=1.0)
    assert await watcher.evaluate_session(SESSION) is None
    assert [turn_index for _, turn_index, _, _, _ in cascade.calls] == [2]


async def test_a_newly_completed_turn_advances_the_target() -> None:
    cascade = RecordingCascade()
    watcher = watcher_with(cascade)
    watcher.ingest(watch_events(entries(8)), now=0.0)
    await watcher.evaluate_session(SESSION)
    watcher.ingest(watch_events(entries(1, start=8)), now=1.0)
    await watcher.evaluate_session(SESSION)
    assert [turn_index for _, turn_index, _, _, _ in cascade.calls] == [6, 7]


def test_sidechain_events_are_never_buffered() -> None:
    watcher = watcher_with(RecordingCascade())
    watcher.ingest(watch_events(entries(2), sidechain=True), now=0.0)
    assert watcher.buffers == {}


def test_buffers_are_capped_to_the_most_recent_events() -> None:
    watcher = watcher_with(RecordingCascade(), buffer_limit=8)
    watcher.ingest(watch_events(entries(8)), now=0.0)
    assert len(watcher.buffers[SESSION].events) == 8


async def test_step_debounces_and_delivers_every_proposal() -> None:
    delivery = CollectingDelivery()
    cascade = RecordingCascade(result=make_proposal(project="/repo"))
    watcher = watcher_with(cascade, delivery, debounce_s=2.0)
    assert await watcher.step(watch_events(entries(4)), now=10.0) == []
    assert delivery.delivered == []
    assert await watcher.step([], now=13.0) == [make_proposal(project="/repo")]
    assert delivery.delivered == [make_proposal(project="/repo")]
    assert cascade.calls[-1][4] == "/repo"
    assert await watcher.step([], now=20.0) == []


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        pytest.param([user_text("hi"), assistant_text("ok")], "/repo", id="realistic-transcript-cwd"),
        pytest.param([user_text("hi", cwd=None), assistant_text("ok", cwd=None)], None, id="cwd-absent-underivable"),
        pytest.param([mode_entry("default")], None, id="no-meta-bearing-events"),
    ],
)
def test_session_cwd_derives_the_project_from_event_meta(items: list[dict[str, Any]], expected: str | None) -> None:
    assert session_cwd(parse(items)) == expected
