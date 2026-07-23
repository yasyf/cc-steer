"""The live watcher daemon: tail transcripts, evaluate quiet sessions, deliver proposals.

Mirrors :class:`cc_transcript.watch.Watcher`'s tick/stream split: :meth:`Watcher.step`
is one deterministic step over an already-tailed event batch and an explicit
clock — directly drivable by tests — and :meth:`Watcher.run` is the thin
poll-forever loop that drives a held :class:`cc_transcript.watch.Watcher`'s
:meth:`~cc_transcript.watch.Watcher.tick` and the monotonic clock through it.
(:meth:`~cc_transcript.watch.Watcher.stream` yields nothing while sessions are
silent, which is exactly when the debounce must fire, so the loop drives ``tick`` —
the same event source — directly.)

A session is evaluated only when it goes quiet — no new event for
``debounce_s`` — and a turn has completed since the last look. The final turn
is possibly still in flight and never judged; the window ends at the last
completed turn, built as the negatives' triggerless shape: the anchored turn
is the last element of ``before``, previews render through
:func:`~cc_steer.capture.turn_ref` at the same ``Budget(200, 200)`` the
training pipeline's ``capture_window`` used, so live model input is
byte-compatible with training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import anyio
from cc_transcript.activity import SessionActivity
from cc_transcript.context import ContextWindow
from cc_transcript.filterspec import event_meta
from cc_transcript.ids import SessionId
from cc_transcript.render import Budget
from cc_transcript.watch import Watcher as TranscriptWatcher

from cc_steer.capture import turn_ref
from cc_steer.watcher.live import scrub_text

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cc_transcript.activity import Turn
    from cc_transcript.models import TranscriptEvent
    from cc_transcript.watch import WatchEvent

    from cc_steer.watcher.cascade import Cascade
    from cc_steer.watcher.delivery import SteerDelivery
    from cc_steer.watcher.types import SteerProposal

BUFFER_LIMIT = 4000
DEBOUNCE_S = 2.0
WINDOW_TURNS = 6
PREVIEW_CHARS = 200
WINDOW_BUDGET = Budget(turn_chars=PREVIEW_CHARS, tool_chars=PREVIEW_CHARS)


@dataclass(slots=True)
class SessionBuffer:
    """One tailed session's recent events and evaluation bookkeeping.

    Attributes:
        events: The buffered events, oldest first, capped at the buffer limit.
        last_event_at: The monotonic time the latest event arrived.
        dirty: Whether events arrived since the last evaluation.
        turn_count: The session's turn count at the last evaluation.
    """

    events: list[TranscriptEvent] = field(default_factory=list)
    last_event_at: float = 0.0
    dirty: bool = False
    turn_count: int = 0


class Watcher:
    """The daemon: per-session event buffers, debounce, cascade, delivery.

    Args:
        cascade: The three-stage cascade evaluating each quiet moment.
        delivery: Where every proposal goes; shadow mode records a ledger row.
        roots: The transcript directories to tail.
        debounce_s: How long a session must stay quiet before evaluation.
        poll: Seconds between tail polls in :meth:`run`.
        buffer_limit: The per-session event cap.
    """

    def __init__(
        self,
        cascade: Cascade,
        delivery: SteerDelivery,
        *,
        roots: Sequence[Path],
        debounce_s: float = DEBOUNCE_S,
        poll: float = 5.0,
        buffer_limit: int = BUFFER_LIMIT,
    ) -> None:
        self.cascade = cascade
        self.delivery = delivery
        self.roots = tuple(roots)
        self.debounce_s = debounce_s
        self.poll = poll
        self.buffer_limit = buffer_limit
        self.buffers: dict[str, SessionBuffer] = {}
        self._tailer = TranscriptWatcher(self.roots)

    def ingest(self, events: Sequence[WatchEvent], *, now: float) -> None:
        """Buffers freshly tailed main-session events; sidechains never steer."""
        for event in events:
            if event.is_sidechain:
                continue
            buffer = self.buffers.setdefault(str(event.session_id), SessionBuffer())
            buffer.events.append(event.event)
            del buffer.events[: -self.buffer_limit]
            buffer.last_event_at = now
            buffer.dirty = True

    def quiet_sessions(self, now: float) -> list[str]:
        """Sessions with unevaluated events that have been silent for the debounce."""
        return [
            session_id
            for session_id, buffer in self.buffers.items()
            if buffer.dirty and now - buffer.last_event_at >= self.debounce_s
        ]

    async def evaluate_session(self, session_id: str) -> SteerProposal | None:
        """Runs one quiet session's last completed turn through the cascade.

        Rebuilds the session's activity from its buffer, skips it unless a new
        turn completed since the last look, and evaluates the window ending at
        the last completed turn — the final turn is possibly in flight and
        never judged. The cascade itself guarantees a turn is evaluated at
        most once.

        Returns:
            The cascade's proposal, or None when no new turn completed, the
            window cannot anchor, or the cascade suppressed the moment.
        """
        buffer = self.buffers[session_id]
        buffer.dirty = False
        activity = SessionActivity.from_events(SessionId(session_id), buffer.events)
        if len(activity.turns) == buffer.turn_count:
            return None
        buffer.turn_count = len(activity.turns)
        if len(activity.turns) < 2:
            return None
        target = activity.turns[-2]
        if (window := live_window(activity.turns[: target.index + 1])) is None:
            return None
        anchor_uuid = str(window.before[-1].refs[-1].event_uuid)
        return await self.cascade.evaluate(
            session_id,
            turn_index=target.index,
            anchor_uuid=anchor_uuid,
            window=window,
            project=session_cwd(buffer.events),
        )

    async def step(self, events: Sequence[WatchEvent], *, now: float) -> list[SteerProposal]:
        """One deterministic step: ingest a batch, evaluate quiet sessions, deliver.

        Returns:
            The proposals delivered this step.
        """
        self.ingest(events, now=now)
        delivered: list[SteerProposal] = []
        for session_id in self.quiet_sessions(now):
            if (proposal := await self.evaluate_session(session_id)) is not None:
                await self.delivery.deliver(proposal)
                delivered.append(proposal)
        return delivered

    async def run(self) -> None:
        """Tails the roots forever, one :meth:`step` per poll; never returns."""
        while True:
            await self.step(await self._tailer.tick(), now=time.monotonic())
            await anyio.sleep(self.poll)


def session_cwd(events: Sequence[TranscriptEvent]) -> str | None:
    """The session's working directory from its latest cwd-bearing event, or None when none carries one."""
    return next(
        (meta.cwd for event in reversed(events) if (meta := event_meta(event)) is not None and meta.cwd is not None),
        None,
    )


def live_window(turns: Sequence[Turn]) -> ContextWindow | None:
    """The live moment as the negatives' window shape, anchored on the last turn.

    The last ``WINDOW_TURNS`` turns fold into ``before`` with no trigger and no
    ``after`` — byte-compatible with ``fold_trigger`` over a training
    ``capture_window`` — and the window anchors on the last turn's first
    meta-bearing event, matching the native negative-sampling anchor rule.

    Returns:
        The window, or None when the last turn carries no resolvable event.
    """
    refs = tuple(
        replace(ref := turn_ref(turn, WINDOW_BUDGET), preview=scrub_text(ref.preview))
        for turn in turns[-WINDOW_TURNS:]
    )
    if not refs or not refs[-1].refs:
        return None
    return ContextWindow(
        anchor=refs[-1].refs[0],
        before=refs,
        trigger=None,
        after=(),
        fidelity="full",
        preview_chars=PREVIEW_CHARS,
    )
