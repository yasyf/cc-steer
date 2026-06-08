from __future__ import annotations

from typing import TYPE_CHECKING

from cc_transcript.models import AssistantEvent, ToolUseBlock, UserEvent

from cc_pushback.models import ContextSnapshot, ContextTurn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cc_transcript.models import TranscriptEvent

ASSISTANT_TEXT_LIMIT = 2000


def turn_for(event: UserEvent | AssistantEvent) -> ContextTurn:
    match event:
        case UserEvent():
            return ContextTurn(role="user", text=event.text)
        case AssistantEvent():
            return ContextTurn(
                role="assistant",
                text=event.text[:ASSISTANT_TEXT_LIMIT],
                tool_calls=tuple(block.name for block in event.blocks if isinstance(block, ToolUseBlock)),
            )


def trigger_for(events: Sequence[TranscriptEvent], index: int, lower: int) -> ContextTurn | None:
    return next(
        (
            turn_for(event)
            for i in range(index - 1, lower - 1, -1)
            if isinstance(event := events[i], AssistantEvent)
        ),
        None,
    )


def build_snapshot(
    events: Sequence[TranscriptEvent],
    index: int,
    *,
    before: int = 6,
    after: int = 2,
    lower_bound: int | None = None,
) -> ContextSnapshot:
    """Builds the conversational window around the event at ``index``.

    A turn is a :class:`UserEvent` or :class:`AssistantEvent`; system, mode, and
    other events are skipped. The trigger is the nearest preceding assistant
    turn — the action the feedback responds to.

    Args:
        events: The full ordered event stream for one transcript.
        index: The index of the event the feedback was attached to.
        before: The maximum number of turns to capture before the trigger.
        after: The maximum number of turns to capture after the index.
        lower_bound: When set, an event index the ``before`` window and trigger
            search may not reach back past — used to anchor plan-review context
            to the triggering edit cycle.

    Returns:
        The assembled :class:`ContextSnapshot`.
    """
    lower = lower_bound if lower_bound is not None else 0
    return ContextSnapshot(
        before=tuple(
            turn_for(event)
            for i in range(index - 1, lower - 1, -1)
            if isinstance(event := events[i], UserEvent | AssistantEvent)
        )[:before][::-1],
        trigger=trigger_for(events, index, lower),
        after=tuple(
            turn_for(event)
            for i in range(index + 1, len(events))
            if isinstance(event := events[i], UserEvent | AssistantEvent)
        )[:after],
    )
