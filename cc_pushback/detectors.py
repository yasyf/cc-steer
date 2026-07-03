"""cc-pushback's detector policy: map neutral mining facts to feedback candidates.

The fact-recognition mechanism lives in :mod:`cc_transcript.mining`; this module
injects cc-pushback's policy — its filter spec, its trigger-absence
disqualification, and its review formats (carried by :data:`PUSHBACK_MINING_SPEC`'s
:class:`~cc_transcript.mining.ReviewSpec`) — and maps each surviving
:class:`MiningSignal` from a single :func:`~cc_transcript.mining.mine` pass to a
:class:`FeedbackCandidate` whose durable
:class:`~cc_transcript.context.ContextWindow` is captured over the lifted
:class:`~cc_transcript.activity.SessionActivity`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cc_transcript import keep
from cc_transcript.activity import SessionActivity
from cc_transcript.context import capture_window
from cc_transcript.filterspec import event_meta
from cc_transcript.ids import EventRef
from cc_transcript.mining import FeedbackCandidate, MiningSpec, dedup_key, mine

from cc_pushback.formats import review_spec
from cc_pushback.spec import PUSHBACK_SPEC

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from typing import Any

    from cc_transcript.mining import MiningSignal
    from cc_transcript.models import TranscriptEvent

type Detector = Callable[[Sequence[TranscriptEvent]], list[FeedbackCandidate]]

DEFAULT_BEFORE = 6
PUSHBACK_MINING_SPEC = MiningSpec(review=review_spec())


def human_authored(events: Sequence[TranscriptEvent], sig: MiningSignal) -> bool:
    match sig.evidence["provenance"]:
        case "typed":
            return keep(events[sig.event_index], PUSHBACK_SPEC)
        case "surfaced":
            return True
    raise AssertionError(sig.evidence["provenance"])


def survives(events: Sequence[TranscriptEvent], sig: MiningSignal) -> bool:
    match sig.detector:
        case "review_comment":
            return human_authored(events, sig)
        case "transcript_message":
            return keep(events[sig.event_index], PUSHBACK_SPEC) and sig.trigger_index is not None
        case "plan_reentry":
            return keep(events[sig.event_index], PUSHBACK_SPEC)
        case _:
            return True


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
            return {key: sig.evidence[key] for key in ("format", "file", "line_start", "line_end", "provenance")}
    raise AssertionError(sig.detector)


def clamped_before(activity: SessionActivity, events: Sequence[TranscriptEvent], sig: MiningSignal) -> int:
    if sig.lower_bound is None or (meta := event_meta(events[sig.lower_bound])) is None:
        return DEFAULT_BEFORE
    match (
        activity.turn_of(EventRef(meta.session_id, meta.uuid)),
        activity.turn_of(EventRef(sig.session_id, sig.event_uuid)),
    ):
        case (None, _) | (_, None):
            return DEFAULT_BEFORE
        case lower, anchor:
            return min(DEFAULT_BEFORE, anchor.index - lower.index)


def to_candidate(activity: SessionActivity, events: Sequence[TranscriptEvent], sig: MiningSignal) -> FeedbackCandidate:
    anchor = EventRef(sig.session_id, sig.event_uuid)
    return FeedbackCandidate(
        dedup_key=dedup_key(*parts(sig)),
        source_kind=sig.kind,
        occurred_at=sig.occurred_at,
        text=sig.text,
        window=capture_window(activity, anchor, before=clamped_before(activity, events, sig)),
        ref=anchor,
        session_id=sig.session_id,
        cc_version=sig.cc_version,
        signal=sig.signal,
        payload=payload_of(sig),
    )


def candidates_from(events: Sequence[TranscriptEvent], signals: Iterable[MiningSignal]) -> list[FeedbackCandidate]:
    surviving = [sig for sig in signals if survives(events, sig)]
    if not surviving:
        return []
    activity = SessionActivity.from_events(surviving[0].session_id, events)
    return [to_candidate(activity, events, sig) for sig in surviving]


def for_detectors(events: Sequence[TranscriptEvent], detectors: frozenset[str]) -> list[FeedbackCandidate]:
    return candidates_from(events, (sig for sig in mine(events, PUSHBACK_MINING_SPEC) if sig.detector in detectors))


def transcript_messages(events: Sequence[TranscriptEvent]) -> list[FeedbackCandidate]:
    return for_detectors(events, frozenset({"transcript_message"}))


def plan_reviews(events: Sequence[TranscriptEvent]) -> list[FeedbackCandidate]:
    return for_detectors(events, frozenset({"exit_plan_rejection", "plan_reentry"}))


def interrupt_rejections(events: Sequence[TranscriptEvent]) -> list[FeedbackCandidate]:
    return for_detectors(events, frozenset({"denial", "interrupt"}))


def detect(events: Sequence[TranscriptEvent]) -> list[FeedbackCandidate]:
    """Runs every detector over one transcript's events.

    Args:
        events: The transcript's full ordered event stream.

    Returns:
        Every feedback candidate the detectors found, in detector order.
    """
    return candidates_from(events, mine(events, PUSHBACK_MINING_SPEC))
