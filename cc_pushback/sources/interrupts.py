from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cc_transcript.models import ToolResultBlock, UserEvent

from cc_pushback.context import build_snapshot
from cc_pushback.models import FeedbackCandidate
from cc_pushback.sources.base import DENIAL_PREFIX, INTERRUPT_RE, MESSAGE_JUNK_RE, dedup_key
from cc_pushback.sources.plan_reviews import embedded_user_text, next_user_message, tool_uses

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from cc_transcript.models import ToolUseBlock, ToolUseId, TranscriptEvent

__all__ = ["Interrupts"]

SOURCE_KIND = "interrupt_rejection"


def denied_tool_payload(use: ToolUseBlock) -> dict[str, Any]:
    return {"tool": use.name, "file_path": use.input.get("file_path")}


def marker_in(event: UserEvent) -> str | None:
    return next(
        (
            match.group(0)
            for block in event.blocks
            if isinstance(block, ToolResultBlock)
            if (match := INTERRUPT_RE.search(block.content))
        ),
        None,
    )


class Interrupts:
    """Extracts permission denials and user interrupts from a transcript.

    Permission denials carry the user's embedded correction when present, else
    the full denial content, plus the denied tool's name and ``file_path`` when
    recoverable. ``ExitPlanMode`` denials belong to the plan-review source and
    are not claimed here. Interrupt markers capture the user's following
    correction as the candidate text when one exists.
    """

    def candidates_for_file(self, path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
        uses = tool_uses(events)
        for index, event in enumerate(events):
            if not isinstance(event, UserEvent):
                continue
            yield from self.denials(path, events, index, event, uses)
            yield from self.markers(path, events, index, event)

    def denials(
        self,
        path: Path,
        events: Sequence[TranscriptEvent],
        index: int,
        event: UserEvent,
        uses: Mapping[ToolUseId, ToolUseBlock],
    ) -> Iterator[FeedbackCandidate]:
        return (
            FeedbackCandidate(
                dedup_key=dedup_key(str(path), event.meta.uuid, block.tool_use_id, SOURCE_KIND),
                source_kind=SOURCE_KIND,
                occurred_at=event.meta.timestamp,
                text=embedded_user_text(block.content) or block.content,
                context=build_snapshot(events, index),
                session_id=event.meta.session_id,
                origin_path=path,
                origin_uuid=event.meta.uuid,
                cc_version=event.meta.cc_version,
                payload=denied_tool_payload(paired) if paired else None,
            )
            for block in event.blocks
            if isinstance(block, ToolResultBlock)
            if block.is_error
            if block.content.startswith(DENIAL_PREFIX)
            if (paired := uses.get(block.tool_use_id)) is None or paired.name != "ExitPlanMode"
        )

    def markers(
        self, path: Path, events: Sequence[TranscriptEvent], index: int, event: UserEvent
    ) -> Iterator[FeedbackCandidate]:
        if (marker := marker_in(event)) is None:
            return
        following = next_user_message(events, index + 1)
        correction = following[1].text if following and not MESSAGE_JUNK_RE.search(following[1].text) else None
        yield FeedbackCandidate(
            dedup_key=dedup_key(str(path), event.meta.uuid, "interrupt", SOURCE_KIND),
            source_kind=SOURCE_KIND,
            occurred_at=event.meta.timestamp,
            text=correction or marker,
            context=build_snapshot(events, index),
            session_id=event.meta.session_id,
            origin_path=path,
            origin_uuid=event.meta.uuid,
            cc_version=event.meta.cc_version,
            payload={"detector": "interrupt"},
        )
