"""The single capture seam: an anchor-aware context window every consumer shares.

The window is built directly over a lifted
:class:`~cc_transcript.activity.SessionActivity`, so it captures the scrubbed
in-memory event stream rather than the raw transcript bytes — the native
:func:`~cc_transcript.context.capture_window` reparses raw and would bypass
cc-steer's scrubbing. :func:`turn_ref` renders one lifted turn into a persisted
:class:`~cc_transcript.context.TurnRef`; the daemon's live window shares it.

A plain window centers on the anchor's whole turn. That is wrong for a
tool-result-anchored correction — a rejected ``ExitPlanMode`` result, a review
comment — whose carrier is a tool-result-only user event that never opens a
turn: the assistant material the user reacted to (the submitted plan, the prose
under review) lives inside the anchor's own turn, *before* the carrier, and the
plain window folds the plan, the correction, and every post-anchor event into one
trigger turn that the rendering contract then drops from model input entirely.

:func:`capture_anchored_window` splits the anchor's turn at the carrier: the
renderable prefix strictly before the anchor becomes an extra ``before`` turn
(the agent action the correction reacts to), and the carrier onward becomes the
trigger, excluded from model input. The scan pipeline and the sidecar source both
capture through here, so the fix lands once for live accrual, mirror scans, and
the export that reads what they stored.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from cc_transcript.activity import ToolUse, Turn, event_stamps, position_in, result_index
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.filterspec import event_meta
from cc_transcript.ids import EventRef
from cc_transcript.models import AssistantEvent, ToolUseBlock, UserEvent
from cc_transcript.render import Budget, render_turn
from cc_transcript.tools import edits_of, parse_tool_call

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from cc_transcript.activity import SessionActivity
    from cc_transcript.ids import SessionId, ToolUseId
    from cc_transcript.models import ToolResultBlock, TranscriptEvent

DEFAULT_PREVIEW_CHARS = 200


def turn_ref(turn: Turn, budget: Budget) -> TurnRef:
    """Renders one lifted turn into a persisted :class:`~cc_transcript.context.TurnRef`.

    The refs re-hydrate the turn from the transcript; the preview renders it now
    at ``budget`` so a summary-fidelity window stays legible once the transcript
    is gone.
    """
    return TurnRef(
        role="user" if turn.prompt else "assistant",
        refs=tuple(
            EventRef(meta.session_id, meta.uuid) for event in turn.events if (meta := event_meta(event)) is not None
        ),
        preview=render_turn(turn, budget=budget),
        tool_digests=tuple(use.call.digest for use in turn.tool_uses),
    )


def lift_turn(
    session_id: SessionId,
    index: int,
    prompt: str,
    events: tuple[TranscriptEvent, ...],
    results: Mapping[ToolUseId, tuple[ToolResultBlock, datetime | None]],
) -> Turn:
    started_at, ended_at = event_stamps(events)
    return Turn(
        index=index,
        prompt=prompt,
        started_at=started_at,
        ended_at=ended_at,
        events=events,
        tool_uses=tuple(
            ToolUse(
                ref=EventRef(session_id, event.meta.uuid, block.id),
                call=(call := parse_tool_call(block.name, block.input, on_error="other")),
                result=pair[0] if (pair := results.get(block.id)) is not None else None,
                result_ts=pair[1] if pair is not None else None,
                edits=edits_of(call),
                turn_index=index,
                ts=event.meta.timestamp,
            )
            for event in events
            if isinstance(event, AssistantEvent)
            for block in event.blocks
            if isinstance(block, ToolUseBlock)
        ),
    )


def window_around(
    activity: SessionActivity,
    anchor: EventRef,
    trigger: Turn,
    *,
    before: int,
    after: int,
    preview_chars: int,
) -> ContextWindow:
    budget = Budget(turn_chars=preview_chars, tool_chars=preview_chars)
    return ContextWindow(
        anchor=anchor,
        before=tuple(turn_ref(turn, budget) for turn in activity.turns[max(0, trigger.index - before) : trigger.index]),
        trigger=turn_ref(trigger, budget),
        after=tuple(turn_ref(turn, budget) for turn in activity.turns[trigger.index + 1 : trigger.index + 1 + after]),
        fidelity="full",
        preview_chars=preview_chars,
    )


def capture_window(
    activity: SessionActivity,
    anchor: EventRef,
    *,
    before: int = 6,
    after: int = 2,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> ContextWindow:
    """Captures the turns around ``anchor`` as a full-fidelity window over ``activity``.

    Centers on the anchor's own turn as ``trigger``, with up to ``before`` turns
    preceding and ``after`` turns following. Rendered over the lifted (and, for the
    scan path, scrubbed) event stream, never the raw transcript.

    Raises:
        ValueError: When ``anchor`` does not resolve within ``activity``.
    """
    if (trigger := activity.turn_of(anchor)) is None:
        raise ValueError(f"anchor {anchor.event_uuid} not found in session {activity.session_id}")
    return window_around(activity, anchor, trigger, before=before, after=after, preview_chars=preview_chars)


def capture_anchored_window(
    activity: SessionActivity,
    anchor: EventRef,
    *,
    before: int = 6,
    after: int = 2,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> ContextWindow:
    """Captures ``anchor``'s window, splitting its turn at a mid-turn user carrier.

    Builds :func:`capture_window` — ``before`` turns, the anchor's own turn as
    ``trigger``, then ``after`` turns — and returns it unchanged unless the anchor
    is a user event sitting *inside* its turn: the tool-result-only carrier shape.
    For that shape the anchor's turn is split at the carrier: the events before it
    join ``before`` as the agent action, the carrier and everything after become
    the trigger, kept out of model input.

    Args:
        activity: The lifted session containing the anchor.
        anchor: The correction's carrier event.
        before: Turns before the anchor's turn to capture.
        after: Turns after the anchor's turn to capture.
        preview_chars: The per-chunk preview budget, persisted on the window.

    Raises:
        ValueError: When ``anchor`` does not resolve within ``activity``.
    """
    if (trigger := activity.turn_of(anchor)) is None:
        raise ValueError(f"anchor {anchor.event_uuid} not found in session {activity.session_id}")
    window = window_around(activity, anchor, trigger, before=before, after=after, preview_chars=preview_chars)
    budget = Budget(turn_chars=preview_chars, tool_chars=preview_chars)
    event_pos, _ = position_in(trigger, anchor)
    if event_pos == 0 or not isinstance(trigger.events[event_pos], UserEvent):
        return window
    results = result_index([event for turn in activity.turns for event in turn.events])
    prefix = lift_turn(activity.session_id, trigger.index, trigger.prompt, trigger.events[:event_pos], results)
    suffix = lift_turn(activity.session_id, trigger.index, "", trigger.events[event_pos:], results)
    return replace(
        window,
        before=(*window.before, turn_ref(prefix, budget)),
        trigger=replace(turn_ref(suffix, budget), role="user"),
    )
