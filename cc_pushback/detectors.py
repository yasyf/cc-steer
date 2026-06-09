"""cc-pushback's detector policy: map neutral mining facts to feedback candidates.

The fact-recognition mechanism lives in :mod:`cc_transcript.domains.mining`; this
module injects cc-pushback's policy — its filter spec, its trigger-absence
disqualification, and its review formats — and maps each surviving
:class:`MiningSignal` to a :class:`FeedbackCandidate`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cc_transcript import keep
from cc_transcript.domains.mining import (
    FeedbackCandidate,
    build_snapshot,
    dedup_key,
    iter_interrupt_marker_signals,
    iter_plan_reentry_signals,
    iter_plan_rejection_signals,
    iter_review_comment_signals,
    iter_tool_denial_signals,
    iter_user_message_signals,
)

from cc_pushback.formats import formats
from cc_pushback.spec import PUSHBACK_SPEC

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path
    from typing import Any

    from cc_transcript.domains.mining import MiningSignal
    from cc_transcript.models import TranscriptEvent

type Detector = Callable[[Path, Sequence[TranscriptEvent]], Iterator[FeedbackCandidate]]

SPEC_DETECTORS = frozenset({"transcript_message", "plan_reentry", "review_comment"})


def survives(events: Sequence[TranscriptEvent], sig: MiningSignal) -> bool:
    if sig.detector in SPEC_DETECTORS and not keep(events[sig.event_index], PUSHBACK_SPEC):
        return False
    return not (sig.detector == "transcript_message" and sig.trigger_index is None)


def parts(sig: MiningSignal) -> tuple[str, ...]:
    match sig.detector:
        case "transcript_message":
            return (sig.session_id, "transcript_message", sig.text)
        case "exit_plan_rejection":
            return (sig.session_id, "plan_review", "exit_plan", sig.text)
        case "plan_reentry":
            return (sig.session_id, "plan_review", "plan_reentry", sig.text)
        case "denial" | "interrupt":
            return (sig.session_id, "interrupt_rejection", sig.text)
        case "review_comment":
            return (
                sig.session_id,
                "review_comment",
                sig.evidence["file"] or "",
                str(sig.evidence["line_start"] or ""),
                str(sig.evidence["line_end"] or ""),
                sig.text,
            )
    raise AssertionError(sig.detector)


def payload_of(sig: MiningSignal) -> Mapping[str, Any] | None:
    match sig.detector:
        case "transcript_message":
            return None
        case "exit_plan_rejection" | "plan_reentry" | "interrupt":
            return {"detector": sig.detector}
        case "denial":
            return dict(sig.evidence) or None
        case "review_comment":
            return {key: sig.evidence[key] for key in ("format", "file", "line_start", "line_end")}
    raise AssertionError(sig.detector)


def to_candidate(path: Path, events: Sequence[TranscriptEvent], sig: MiningSignal) -> FeedbackCandidate:
    return FeedbackCandidate(
        dedup_key=dedup_key(*parts(sig)),
        source_kind=sig.kind,
        occurred_at=sig.occurred_at,
        text=sig.text,
        context=build_snapshot(events, sig.event_index, lower_bound=sig.lower_bound),
        session_id=sig.session_id,
        origin_path=path,
        origin_uuid=sig.event_uuid,
        cc_version=sig.cc_version,
        payload=payload_of(sig),
        signal=sig.signal,
    )


def candidates_from(
    path: Path, events: Sequence[TranscriptEvent], *streams: Iterator[MiningSignal]
) -> Iterator[FeedbackCandidate]:
    return (to_candidate(path, events, sig) for stream in streams for sig in stream if survives(events, sig))


def transcript_messages(path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
    return candidates_from(path, events, iter_user_message_signals(events))


def plan_reviews(path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
    return candidates_from(path, events, iter_plan_rejection_signals(events), iter_plan_reentry_signals(events))


def interrupt_rejections(path: Path, events: Sequence[TranscriptEvent]) -> Iterator[FeedbackCandidate]:
    return candidates_from(path, events, iter_tool_denial_signals(events), iter_interrupt_marker_signals(events))


def detect(path: Path, events: Sequence[TranscriptEvent]) -> list[FeedbackCandidate]:
    """Runs every detector over one transcript's events.

    Args:
        path: The transcript file the events came from.
        events: The transcript's full ordered event stream.

    Returns:
        Every feedback candidate the detectors found, in detector order.
    """
    return list(
        candidates_from(
            path,
            events,
            iter_user_message_signals(events),
            iter_plan_rejection_signals(events),
            iter_plan_reentry_signals(events),
            iter_tool_denial_signals(events),
            iter_interrupt_marker_signals(events),
            iter_review_comment_signals(events, formats()),
        )
    )
