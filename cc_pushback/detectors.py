"""The pushback detectors and the entry point that runs them over a transcript."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cc_transcript import STRUCTURAL_NOISE_RE, keep
from cc_transcript.models import ModeEvent, UserEvent

from cc_pushback.context import build_snapshot, trigger_for
from cc_pushback.formats import extract_all
from cc_pushback.models import FeedbackCandidate, dedup_key
from cc_pushback.nav import (
    denial_results,
    denied_tool_payload,
    embedded_user_text,
    is_bare_interrupt_marker,
    last_edit_index,
    marker_in,
    next_user_message,
    tool_uses,
)
from cc_pushback.spec import PUSHBACK_SPEC

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path

    from cc_transcript.models import ToolUseBlock, ToolUseId, TranscriptEvent

type Detector = Callable[[Path, Sequence[TranscriptEvent]], Iterator[FeedbackCandidate]]


def pushback_user_events(events: Sequence[TranscriptEvent]) -> Iterator[tuple[int, UserEvent]]:
    return (
        (index, event)
        for index, event in enumerate(events)
        if isinstance(event, UserEvent)
        if keep(event, PUSHBACK_SPEC)
    )


def correction_text(events: Sequence[TranscriptEvent], index: int) -> str | None:
    while (found := next_user_message(events, index + 1)) is not None:
        i, event = found
        if not is_bare_interrupt_marker(event.text) and not STRUCTURAL_NOISE_RE.search(event.text):
            return event.text
        index = i
    return None


def transcript_messages(path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
    return (
        FeedbackCandidate(
            dedup_key=dedup_key(event.meta.session_id, "transcript_message", event.text),
            source_kind="transcript_message",
            occurred_at=event.meta.timestamp,
            text=event.text,
            context=build_snapshot(events, index),
            session_id=event.meta.session_id,
            origin_path=path,
            origin_uuid=event.meta.uuid,
            cc_version=event.meta.cc_version,
        )
        for index, event in pushback_user_events(events)
        if not is_bare_interrupt_marker(event.text)
        if trigger_for(events, index, 0) is not None
    )


def exit_plan_rejections(
    path: Path, events: Sequence[TranscriptEvent], uses: Mapping[ToolUseId, ToolUseBlock]
) -> Iterator[FeedbackCandidate]:
    return (
        FeedbackCandidate(
            dedup_key=dedup_key(event.meta.session_id, "plan_review", "exit_plan", text),
            source_kind="plan_review",
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


def plan_reentries(path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
    seen: set[str] = set()
    for index, event in enumerate(events):
        if not (isinstance(event, ModeEvent) and event.value == "plan"):
            continue
        if (user := next_user_message(events, index)) is None:
            continue
        user_index, user_event = user
        if (
            user_event.meta.uuid in seen
            or not keep(user_event, PUSHBACK_SPEC)
            or is_bare_interrupt_marker(user_event.text)
        ):
            continue
        if (edit := last_edit_index(events, user_index)) is None:
            continue
        seen.add(user_event.meta.uuid)
        yield FeedbackCandidate(
            dedup_key=dedup_key(user_event.meta.session_id, "plan_review", "plan_reentry", user_event.text),
            source_kind="plan_review",
            occurred_at=user_event.meta.timestamp,
            text=user_event.text,
            context=build_snapshot(events, user_index, lower_bound=edit),
            session_id=user_event.meta.session_id,
            origin_path=path,
            origin_uuid=user_event.meta.uuid,
            cc_version=user_event.meta.cc_version,
            payload={"detector": "plan_reentry"},
        )


def plan_reviews(path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
    uses = tool_uses(events)
    yield from exit_plan_rejections(path, events, uses)
    yield from plan_reentries(path, events)


def denials(
    path: Path,
    events: Sequence[TranscriptEvent],
    index: int,
    event: UserEvent,
    uses: Mapping[ToolUseId, ToolUseBlock],
) -> Iterator[FeedbackCandidate]:
    return (
        FeedbackCandidate(
            dedup_key=dedup_key(event.meta.session_id, "interrupt_rejection", text),
            source_kind="interrupt_rejection",
            occurred_at=event.meta.timestamp,
            text=text,
            context=build_snapshot(events, index),
            session_id=event.meta.session_id,
            origin_path=path,
            origin_uuid=event.meta.uuid,
            cc_version=event.meta.cc_version,
            payload=denied_tool_payload(paired) if paired else None,
        )
        for block in denial_results(event)
        if (paired := uses.get(block.tool_use_id)) is None or paired.name not in {"ExitPlanMode", "AskUserQuestion"}
        if (text := embedded_user_text(block.content) or correction_text(events, index))
    )


def interrupt_markers(
    path: Path, events: Sequence[TranscriptEvent], index: int, event: UserEvent
) -> Iterator[FeedbackCandidate]:
    if marker_in(event) is None or (correction := correction_text(events, index)) is None:
        return
    yield FeedbackCandidate(
        dedup_key=dedup_key(event.meta.session_id, "interrupt_rejection", correction),
        source_kind="interrupt_rejection",
        occurred_at=event.meta.timestamp,
        text=correction,
        context=build_snapshot(events, index),
        session_id=event.meta.session_id,
        origin_path=path,
        origin_uuid=event.meta.uuid,
        cc_version=event.meta.cc_version,
        payload={"detector": "interrupt"},
    )


def interrupt_rejections(path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
    uses = tool_uses(events)
    for index, event in enumerate(events):
        if not isinstance(event, UserEvent):
            continue
        yield from denials(path, events, index, event, uses)
        yield from interrupt_markers(path, events, index, event)


def review_comments(path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
    return (
        FeedbackCandidate(
            dedup_key=dedup_key(
                event.meta.session_id,
                "review_comment",
                comment.file or "",
                str(comment.line_start or ""),
                str(comment.line_end or ""),
                comment.comment,
            ),
            source_kind="review_comment",
            occurred_at=event.meta.timestamp,
            text=comment.comment,
            context=build_snapshot(events, index),
            session_id=event.meta.session_id,
            origin_path=path,
            origin_uuid=event.meta.uuid,
            cc_version=event.meta.cc_version,
            payload={
                "format": fmt.name,
                "file": comment.file,
                "line_start": comment.line_start,
                "line_end": comment.line_end,
            },
        )
        for index, event in pushback_user_events(events)
        for fmt, comment in extract_all(event.text)
    )


def detect(path: Path, events: Sequence[TranscriptEvent]) -> list[FeedbackCandidate]:
    """Runs every detector over one transcript's events.

    Args:
        path: The transcript file the events came from.
        events: The transcript's full ordered event stream.

    Returns:
        Every feedback candidate the detectors found, in detector order.
    """
    pipeline: tuple[Detector, ...] = (transcript_messages, plan_reviews, interrupt_rejections, review_comments)
    return [candidate for detector in pipeline for candidate in detector(path, events)]
