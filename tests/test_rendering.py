from __future__ import annotations

from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId

from cc_steer.rendering import (
    DRAFT_CHAR_CAP,
    NO_STEER,
    context_turns,
    gate_text,
    strip_think,
    tail_messages,
    truncated,
    watcher_prompt,
)

STEER = "no, use the other approach"


def turn(role: str, preview: str, uuid: str) -> TurnRef:
    ref = EventRef(session_id=SessionId("sess-1"), event_uuid=EventUuid(uuid), tool_use_id=None)
    return TurnRef(role=role, refs=(ref,), preview=preview, tool_digests=())  # type: ignore[arg-type]


def window(*, trigger: TurnRef | None, before: tuple[TurnRef, ...]) -> ContextWindow:
    anchor = EventRef(session_id=SessionId("sess-1"), event_uuid=EventUuid("anchor"), tool_use_id=None)
    return ContextWindow(anchor=anchor, before=before, trigger=trigger, after=(), fidelity="full", preview_chars=200)


BEFORE = (turn("user", "please add a test", "u1"), turn("user", "I added the test to the suite", "u2"))
USER_ANCHORED = window(trigger=turn("user", STEER, "t1"), before=BEFORE)
NO_TRIGGER = window(trigger=None, before=BEFORE)
ASSISTANT_ANCHORED = window(trigger=turn("assistant", "I will refactor everything", "t2"), before=BEFORE)


def test_user_steer_trigger_is_excluded_from_context() -> None:
    assert context_turns(USER_ANCHORED) == BEFORE
    assert context_turns(NO_TRIGGER) == BEFORE


def test_assistant_trigger_joins_the_context() -> None:
    turns = context_turns(ASSISTANT_ANCHORED)
    assert turns[-1].preview == "I will refactor everything"
    assert turns[:-1] == BEFORE


def test_watcher_prompt_is_context_messages_only() -> None:
    prompt = watcher_prompt(USER_ANCHORED)
    assert prompt == [
        {"role": "user", "content": "please add a test"},
        {"role": "user", "content": "I added the test to the suite"},
    ]


def test_gate_text_never_leaks_the_user_steer() -> None:
    text = gate_text(USER_ANCHORED)
    assert STEER not in text
    assert "<user>\nI added the test to the suite" in text
    assert text == gate_text(USER_ANCHORED)


def test_truncated_rewinds_before_turns() -> None:
    assert truncated(USER_ANCHORED, 0) is USER_ANCHORED
    rewound = truncated(USER_ANCHORED, 1)
    assert rewound is not None
    assert [item.preview for item in rewound.before] == ["please add a test"]
    assert rewound.trigger == USER_ANCHORED.trigger


def test_truncated_stops_when_nothing_remains() -> None:
    assert truncated(USER_ANCHORED, 2) is None
    assert truncated(USER_ANCHORED, 3) is None


def test_no_steer_sentinel_is_stable() -> None:
    assert NO_STEER == "NO_STEER"


def test_no_steer_sentinel_and_draft_cap_are_stable() -> None:
    assert DRAFT_CHAR_CAP == 10_000


def test_tail_messages_keeps_the_most_recent_whole_messages() -> None:
    prompt = [
        {"role": "user", "content": "a" * 60},
        {"role": "assistant", "content": "b" * 30},
        {"role": "user", "content": "c" * 30},
    ]
    assert tail_messages(prompt, 60) == prompt[1:]
    assert tail_messages(prompt, 200) == prompt


def test_tail_messages_boundary_is_inclusive_on_exact_fit() -> None:
    prompt = [{"role": "user", "content": "aaaa"}, {"role": "assistant", "content": "bb"}]
    assert tail_messages(prompt, 6) == prompt
    assert tail_messages(prompt, 5) == prompt[1:]


def test_tail_messages_oversized_final_message_keeps_its_tail() -> None:
    prompt = [{"role": "user", "content": "early"}, {"role": "assistant", "content": "x" * 50}]
    assert tail_messages(prompt, 10) == [{"role": "assistant", "content": "x" * 10}]


def test_tail_messages_rejects_a_nonpositive_cap() -> None:
    import pytest

    with pytest.raises(ValueError, match="cap"):
        tail_messages([], 0)


def test_strip_think_removes_the_template_scaffold() -> None:
    assert strip_think("<think>\n\n</think>\n\nNO_STEER") == "NO_STEER"
    assert strip_think("<think>plan</think>do the thing") == "do the thing"
    assert strip_think("plain steer") == "plain steer"
