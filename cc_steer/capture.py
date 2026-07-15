"""The single capture seam: an anchor-aware context window every consumer shares.

:func:`~cc_transcript.context.capture_window` centers a window on the anchor's
whole turn. That is wrong for a tool-result-anchored correction — a rejected
``ExitPlanMode`` result, a review comment — whose carrier is a tool-result-only
user event that never opens a turn: the assistant material the user reacted to
(the submitted plan, the prose under review) lives inside the anchor's own turn,
*before* the carrier, and the plain window folds the plan, the correction, and
every post-anchor event into one trigger turn that the rendering contract then
drops from model input entirely.

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

from cc_transcript.activity import lift_turn, position_in, result_index
from cc_transcript.context import capture_window, turn_ref
from cc_transcript.models import UserEvent
from cc_transcript.render import Budget

if TYPE_CHECKING:
    from cc_transcript.activity import SessionActivity
    from cc_transcript.context import ContextWindow
    from cc_transcript.ids import EventRef

DEFAULT_PREVIEW_CHARS = 200


def capture_anchored_window(
    activity: SessionActivity,
    anchor: EventRef,
    *,
    before: int = 6,
    after: int = 2,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> ContextWindow:
    """Captures ``anchor``'s window, splitting its turn at a mid-turn user carrier.

    Falls through to a plain :func:`~cc_transcript.context.capture_window` unless
    the anchor is a user event sitting *inside* its turn — the tool-result-only
    carrier shape. For that shape the anchor's turn is split at the carrier: the
    events before it join ``before`` as the agent action, the carrier and
    everything after become the trigger, kept out of model input.

    Args:
        activity: The lifted session containing the anchor.
        anchor: The correction's carrier event.
        before: Turns before the anchor's turn to capture.
        after: Turns after the anchor's turn to capture.
        preview_chars: The per-chunk preview budget, persisted on the window.
    """
    window = capture_window(activity, anchor, before=before, after=after, preview_chars=preview_chars)
    if (turn := activity.turn_of(anchor)) is None:
        return window
    event_pos, _ = position_in(turn, anchor)
    if event_pos == 0 or not isinstance(turn.events[event_pos], UserEvent):
        return window
    budget = Budget(turn_chars=preview_chars, tool_chars=preview_chars)
    results = result_index([event for other in activity.turns for event in other.events])
    prefix = lift_turn(activity.session_id, turn.index, turn.prompt, turn.events[:event_pos], results)
    suffix = lift_turn(activity.session_id, turn.index, "", turn.events[event_pos:], results)
    return replace(
        window,
        before=(*window.before, turn_ref(prefix, budget)),
        trigger=replace(turn_ref(suffix, budget), role="user"),
    )
