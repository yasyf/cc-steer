from __future__ import annotations

from typing import TYPE_CHECKING

from cc_transcript import STRUCTURAL_NOISE_RE
from cc_transcript.models import AssistantEvent, ModeEvent, ToolResultBlock, ToolUseBlock, UserEvent

from cc_pushback.context import build_snapshot
from cc_pushback.models import FeedbackCandidate
from cc_pushback.sources.base import (
    DENIAL_PREFIX,
    USER_SAID_MARKER,
    USER_SAID_TRAILER,
    dedup_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from cc_transcript.models import ToolUseId, TranscriptEvent

SOURCE_KIND = "plan_review"
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
REENTRY_LOOKBACK = 40


def embedded_user_text(content: str) -> str | None:
    if (start := content.find(USER_SAID_MARKER)) == -1:
        return None
    tail = content[start + len(USER_SAID_MARKER) :]
    return tail.split(USER_SAID_TRAILER, 1)[0].strip()


def tool_uses(events: Sequence[TranscriptEvent]) -> dict[ToolUseId, ToolUseBlock]:
    return {
        block.id: block
        for event in events
        if isinstance(event, AssistantEvent)
        for block in event.blocks
        if isinstance(block, ToolUseBlock)
    }


def denial_results(event: UserEvent) -> Iterator[ToolResultBlock]:
    return (
        block
        for block in event.blocks
        if isinstance(block, ToolResultBlock)
        if block.is_error
        if block.content.startswith(DENIAL_PREFIX)
    )


def last_edit_index(events: Sequence[TranscriptEvent], index: int) -> int | None:
    return next(
        (
            i
            for i in range(index - 1, max(index - REENTRY_LOOKBACK, 0) - 1, -1)
            if isinstance(event := events[i], AssistantEvent)
            if any(isinstance(b, ToolUseBlock) and b.name in EDIT_TOOLS for b in event.blocks)
        ),
        None,
    )


def next_user_message(events: Sequence[TranscriptEvent], index: int) -> tuple[int, UserEvent] | None:
    return next(
        (
            (i, event)
            for i in range(index, len(events))
            if isinstance(event := events[i], UserEvent)
            if event.text.strip()
        ),
        None,
    )


class PlanReviews:
    """Extracts plan-review feedback from a transcript's ordered events.

    Two detectors run over each file:

    - ``exit_plan_rejection``: an ``ExitPlanMode`` tool use the user rejected,
      whose embedded "the user said:" payload is review feedback.
    - ``plan_reentry``: a user message that re-enters plan mode right after an
      edit cycle, treated as review of the just-written code.

    A re-entry message may also surface as a ``transcript_message`` candidate.
    The two carry distinct dedup keys, so both rows persist intentionally.
    """

    def candidates_for_file(self, path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
        uses = tool_uses(events)
        yield from self.exit_plan_rejections(path, events, uses)
        yield from self.plan_reentries(path, events)

    def exit_plan_rejections(
        self, path: Path, events: Sequence[TranscriptEvent], uses: Mapping[ToolUseId, ToolUseBlock]
    ) -> Iterator[FeedbackCandidate]:
        return (
            FeedbackCandidate(
                dedup_key=dedup_key(str(path), event.meta.uuid, result.tool_use_id, SOURCE_KIND),
                source_kind=SOURCE_KIND,
                occurred_at=event.meta.timestamp,
                text=text,
                context=build_snapshot(events, index),
                session_id=event.meta.session_id,
                origin_path=path,
                origin_uuid=event.meta.uuid,
                cc_version=event.meta.cc_version,
                payload={"detector": "exit_plan_rejection"},
            )
            for index, event in enumerate(events)
            if isinstance(event, UserEvent)
            for result in denial_results(event)
            if (use := uses.get(result.tool_use_id)) is not None
            if use.name == "ExitPlanMode"
            if (text := embedded_user_text(result.content)) is not None
        )

    def plan_reentries(
        self, path: Path, events: Sequence[TranscriptEvent]
    ) -> Iterator[FeedbackCandidate]:
        seen: set[str] = set()
        for index, event in enumerate(events):
            if not (isinstance(event, ModeEvent) and event.value == "plan"):
                continue
            if (user := next_user_message(events, index)) is None:
                continue
            user_index, user_event = user
            if user_event.meta.uuid in seen or STRUCTURAL_NOISE_RE.search(user_event.text):
                continue
            if (edit := last_edit_index(events, user_index)) is None:
                continue
            seen.add(user_event.meta.uuid)
            yield FeedbackCandidate(
                dedup_key=dedup_key(str(path), user_event.meta.uuid, "plan_reentry", SOURCE_KIND),
                source_kind=SOURCE_KIND,
                occurred_at=user_event.meta.timestamp,
                text=user_event.text,
                context=build_snapshot(events, user_index, lower_bound=edit),
                session_id=user_event.meta.session_id,
                origin_path=path,
                origin_uuid=user_event.meta.uuid,
                cc_version=user_event.meta.cc_version,
                payload={"detector": "plan_reentry"},
            )
