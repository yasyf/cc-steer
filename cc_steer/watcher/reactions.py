"""The true-label engine: turn delivered steers into accept/edit/diverge/dismiss reactions.

Shadow mode measures proposals against interventions after the fact
(:mod:`cc_steer.watcher.shadow`); once a steer is actually surfaced, the user's next
move IS the label. For every emitted delivery this pass finds the first authored
spec-surviving user turn within the window and classifies the reaction by how close
that turn is to the steer: a near-match is ``accepted``, a partial overlap ``edited``,
a real steer in another direction ``diverged``, and no reply at all ``ignored`` (a weak
negative — silence leans against the fire but proves nothing). A delivery that expired
before it was ever surfaced is ``expired``, never a dismissal. Explicit
``cc-steer live accept|dismiss|edit`` verbs are ``cli_verb`` reactions that always win
over this scan.

The attribution is proposal-id-scoped: the id travels with the delivery row, so a
delivered steer's reply joins by identity, not by the session+window heuristic the
shadow report still uses for holdout and undelivered proposals. Reply turns come from
the feedback store's ``feedback_events`` — authored turns that already survived the
mining spec — so each reaction links to the ``feedback_dedup_key`` of the turn it read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from cc_steer.watcher.live import LiveConfig, MailboxDelivery
from cc_steer.watcher.shadow import WINDOW_MINUTES, parse_ts

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from cc_steer.exemplars import Encoder
    from cc_steer.store import FeedbackStore
    from cc_steer.watcher.live import ReactionKind, ReactionSource

HIGH_BAND = 0.7
LOW_BAND = 0.3

REPLY_QUERY = """
SELECT session_id, occurred_at, text, dedup_key FROM feedback_events
WHERE session_id IN ({marks}) ORDER BY occurred_at, id
"""


@dataclass(frozen=True, slots=True)
class ReplyTurn:
    """One authored spec-surviving user turn, the raw material of an inferred reaction.

    Attributes:
        occurred_at: When the turn landed, ISO-8601.
        text: The turn's verbatim text.
        dedup_key: The feedback event's key — the reaction's ``feedback_dedup_key`` link.
    """

    occurred_at: str
    text: str
    dedup_key: str


@dataclass(frozen=True, slots=True)
class Reaction:
    """One attributed reaction, ready for :meth:`MailboxDelivery.record_reaction`.

    Attributes:
        proposal_id: The proposal the delivery carried — the reaction's identity.
        delivery_id: The delivery row the reaction was read off, when one exists.
        kind: The reaction verdict.
        source: ``scan_inferred`` for this pass; ``cli_verb`` for the explicit verbs.
        feedback_dedup_key: The reply turn's key, or None (``ignored``/``expired``).
        similarity: The steer-vs-reply similarity the kind was banded from, or None.
    """

    proposal_id: int
    delivery_id: int | None
    kind: ReactionKind
    source: ReactionSource
    feedback_dedup_key: str | None
    similarity: float | None


@dataclass(frozen=True, slots=True)
class ReactionReport:
    """The outcome of one attribution pass: counts per reaction kind."""

    counts: Mapping[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary_line(self) -> str:
        return f"attributed {self.total} " + "  ".join(f"{kind} {n}" for kind, n in sorted(self.counts.items()))


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard overlap, the encoder-free similarity fallback."""
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb) if (ta and tb) else 0.0


def similarity(steer: str, reply: str, encoder: Encoder | None) -> float:
    """Cosine similarity of the two texts under ``encoder``, or Jaccard when none is given."""
    if encoder is None:
        return jaccard(steer, reply)
    va, vb = encoder.encode([steer, reply])
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    return float(va @ vb / (na * nb)) if na and nb else 0.0


def classify(steer: str, reply: ReplyTurn | None, encoder: Encoder | None) -> tuple[ReactionKind, float | None]:
    """The reaction kind and its similarity for one delivered steer's first reply.

    No reply is ``ignored``; otherwise the steer-vs-reply similarity bands into
    ``accepted`` (near-identical), ``edited`` (partial overlap), or ``diverged``
    (a real steer in another direction).
    """
    if reply is None:
        return ("ignored", None)
    sim = similarity(steer, reply.text, encoder)
    if sim >= HIGH_BAND:
        return ("accepted", sim)
    return ("edited", sim) if sim >= LOW_BAND else ("diverged", sim)


def first_reply(replies: Sequence[ReplyTurn], after: datetime, *, window: timedelta) -> ReplyTurn | None:
    """The chronologically earliest reply in ``(after, after + window]``; None when the window is silent.

    Selects on the parsed instant, not row or string order, so mixed timezone
    offsets in ``occurred_at`` can never pick a later reply over an earlier one.
    """
    in_window = [
        (occurred, reply)
        for reply in replies
        if (occurred := parse_ts(reply.occurred_at)) is not None and after < occurred <= after + window
    ]
    return min(in_window, key=lambda pair: pair[0])[1] if in_window else None


def attribute(
    deliveries: Sequence[Mapping[str, object]],
    replies_by_session: Mapping[str, Sequence[ReplyTurn]],
    cli_proposals: frozenset[int],
    *,
    encoder: Encoder | None = None,
    window_minutes: int = WINDOW_MINUTES,
) -> list[Reaction]:
    """Attributes a reaction to every delivered or expired delivery, purely over rows.

    Args:
        deliveries: Delivery rows joined to their proposal's steer.
        replies_by_session: Authored spec-surviving turns per session, oldest first.
        cli_proposals: Proposals already carrying an explicit ``cli_verb`` reaction —
            skipped so the scan never overwrites an operator's own verdict.
        encoder: The similarity encoder, or None for the Jaccard fallback.
        window_minutes: Minutes after a delivery within which a reply is attributed to it.

    Returns:
        One :class:`Reaction` per delivered/expired delivery not already decided by a verb.
    """
    window = timedelta(minutes=window_minutes)
    reactions: list[Reaction] = []
    for row in deliveries:
        proposal_id = int(str(row["proposal_id"]))
        if proposal_id in cli_proposals:
            continue
        delivery_id = int(str(row["id"]))
        match row["state"]:
            case "expired":
                reactions.append(Reaction(proposal_id, delivery_id, "expired", "scan_inferred", None, None))
            case "delivered" if (at := parse_ts(row["decided_at"])) is not None:
                reply = first_reply(replies_by_session.get(str(row["session_id"]), ()), at, window=window)
                kind, sim = classify(str(row["steer"] or ""), reply, encoder)
                key = reply.dedup_key if reply is not None else None
                reactions.append(Reaction(proposal_id, delivery_id, kind, "scan_inferred", key, sim))
    return reactions


async def reply_turns(store: FeedbackStore, sessions: Sequence[str]) -> dict[str, list[ReplyTurn]]:
    """The authored spec-surviving turns for ``sessions``, grouped by session, oldest first."""
    if not sessions:
        return {}
    cur = await store.store.conn.execute(
        REPLY_QUERY.format(marks=",".join("?" for _ in sessions)), tuple(sessions)
    )
    grouped: dict[str, list[ReplyTurn]] = {}
    async for row in cur:
        grouped.setdefault(str(row["session_id"]), []).append(
            ReplyTurn(str(row["occurred_at"]), str(row["text"]), str(row["dedup_key"]))
        )
    return grouped


async def attribute_reactions(
    store: FeedbackStore,
    *,
    shadow_db: Path | None = None,
    encoder: Encoder | None = None,
    window_minutes: int = WINDOW_MINUTES,
) -> ReactionReport:
    """Runs one attribution pass over the shadow ledger — the SessionEnd-scan and nightly-pipeline step.

    Reads every delivery and its existing reactions, scans the feedback store for each
    delivered session's authored turns, attributes a reaction to each delivered or expired
    delivery (leaving explicit ``cli_verb`` verdicts untouched), and records them through
    the one persistence path. A ledger with no emitted deliveries — the mirror-week world —
    is a no-op.

    Args:
        store: The feedback store the reply turns are read from.
        shadow_db: The shadow ledger path; None uses the default.
        encoder: The similarity encoder, or None for the Jaccard fallback (the default,
            spend-free path).
        window_minutes: Minutes after a delivery within which a reply is attributed to it.

    Returns:
        The :class:`ReactionReport` for this pass.
    """
    async with await MailboxDelivery.open(shadow_db, config=LiveConfig.shadow()) as mailbox:
        deliveries = await mailbox.deliveries()
        cli_proposals = frozenset(
            int(str(row["proposal_id"])) for row in await mailbox.reactions() if row["source"] == "cli_verb"
        )
        sessions = sorted({str(row["session_id"]) for row in deliveries if row["state"] == "delivered"})
        replies = await reply_turns(store, sessions)
        reactions = attribute(deliveries, replies, cli_proposals, encoder=encoder, window_minutes=window_minutes)
        counts: dict[str, int] = {}
        for reaction in reactions:
            await mailbox.record_reaction(
                proposal_id=reaction.proposal_id,
                delivery_id=reaction.delivery_id,
                kind=reaction.kind,
                source=reaction.source,
                feedback_dedup_key=reaction.feedback_dedup_key,
                similarity=reaction.similarity,
            )
            counts[reaction.kind] = counts.get(reaction.kind, 0) + 1
    return ReactionReport(counts=counts)
