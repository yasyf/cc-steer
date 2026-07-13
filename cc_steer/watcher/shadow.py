"""Shadow-mode analysis: proposals joined against the interventions users actually made.

Nothing the watcher proposes reaches a session in shadow mode; this module
measures the proposals after the fact. Feedback events carry no turn index, so
the join key is time within a session: a steer counts as a HIT when the same
session shows a real intervention within ``window_minutes`` after the proposal
fired — the user did step in near the moment the watcher flagged — and as a
nuisance candidate otherwise.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from cc_steer.journal import Journal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from cc_steer.store import FeedbackStore

WINDOW_MINUTES = 30
REPORT_LOG_TITLE = "cc-steer shadow reports"
REPORT_LOG_LABEL = "shadow"


@dataclass(frozen=True, slots=True)
class ShadowSummary:
    """The shadow pass's headline numbers.

    Attributes:
        sessions: Distinct sessions that produced at least one proposal.
        proposals: Total proposal rows — one per stage-2 invocation.
        stage2_abstained: Proposals where the drafter answered ``NO_STEER``.
        stage3_abstained: Drafted proposals the refiner declined to send.
        steers: Proposals that produced a final steering message.
        hits: Steers followed by a real intervention within the window.
        nuisance: Steers with no nearby intervention — the would-be noise.
        window_minutes: The join window the numbers were computed at.
    """

    sessions: int
    proposals: int
    stage2_abstained: int
    stage3_abstained: int
    steers: int
    hits: int
    nuisance: int
    window_minutes: int
    hit_categories: Mapping[str, int] = dataclasses.field(default_factory=dict)
    sentinel_probs: SentinelStats | None = None

    @property
    def proposals_per_session(self) -> float:
        """Mean proposals per watched session; 0.0 with nothing watched."""
        return self.proposals / self.sessions if self.sessions else 0.0


@dataclass(frozen=True, slots=True)
class SentinelStats:
    """The drafter's abstain-score distribution over scored proposals.

    Attributes:
        n: Proposals carrying a sentinel probability.
        mean: The mean P(NO_STEER) across them.
        deciles: P(NO_STEER) at the 0.1..0.9 quantiles, for threshold reading.
    """

    n: int
    mean: float
    deciles: tuple[float, ...]

    @classmethod
    def from_probs(cls, probs: Sequence[float]) -> SentinelStats | None:
        if not probs:
            return None
        ordered = sorted(probs)
        deciles = tuple(ordered[min(len(ordered) - 1, int(len(ordered) * q / 10))] for q in range(1, 10))
        return cls(n=len(ordered), mean=sum(ordered) / len(ordered), deciles=deciles)


def parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def summarize(
    proposals: Sequence[Mapping[str, object]],
    interventions: Sequence[Mapping[str, object]],
    *,
    window_minutes: int = WINDOW_MINUTES,
    reactions: Mapping[int, str] | None = None,
) -> ShadowSummary:
    """Joins shadow proposals against real interventions, purely over row dicts.

    A proposal that was actually delivered (its id appears in ``reactions``) is
    scored by its id-scoped reaction — a positive kind is a hit — not the
    session+window heuristic, which stays the join for holdout and undelivered
    proposals whose reply can only be guessed at by time.

    Args:
        proposals: Rows from the shadow ledger's ``proposals`` table.
        interventions: Real feedback events as ``(session_id, occurred_at)`` rows.
        window_minutes: Minutes after a proposal within which an intervention
            in the same session counts as a hit.
        reactions: Attributed reaction kinds keyed by proposal id — the delivered
            steers' ground truth, overriding the window join for those rows.

    Returns:
        The :class:`ShadowSummary` at ``window_minutes``.
    """
    from cc_steer.watcher.live import POSITIVE_REACTIONS

    reacted = reactions or {}
    by_session: dict[str, list[tuple[datetime, str]]] = {}
    for row in interventions:
        if (occurred := parse_ts(row["occurred_at"])) is not None:
            category = str(row.get("category") or "")
            by_session.setdefault(str(row["session_id"]), []).append((occurred, category))
    window = timedelta(minutes=window_minutes)
    stage2_abstained = sum(row["draft"] is None for row in proposals)
    stage3_abstained = sum(row["draft"] is not None and row["steer"] is None for row in proposals)
    hits = 0
    steers = 0
    hit_categories: dict[str, int] = {}
    for row in proposals:
        if row["steer"] is None or (fired := parse_ts(row["ts"])) is None:
            continue
        steers += 1
        if (pid := row.get("id")) is not None and (kind := reacted.get(int(str(pid)))) is not None:
            if kind in POSITIVE_REACTIONS:
                hits += 1
                hit_categories[kind] = hit_categories.get(kind, 0) + 1
            continue
        nearby = [
            category
            for occurred, category in by_session.get(str(row["session_id"]), [])
            if fired <= occurred <= fired + window
        ]
        if nearby:
            hits += 1
            hit_categories[nearby[0] or "(unjudged)"] = hit_categories.get(nearby[0] or "(unjudged)", 0) + 1
    probs = [float(p) for row in proposals if isinstance(p := row.get("sentinel_prob"), int | float)]
    return ShadowSummary(
        sessions=len({str(row["session_id"]) for row in proposals}),
        proposals=len(proposals),
        stage2_abstained=stage2_abstained,
        stage3_abstained=stage3_abstained,
        steers=steers,
        hits=hits,
        nuisance=steers - hits,
        window_minutes=window_minutes,
        hit_categories=hit_categories,
        sentinel_probs=SentinelStats.from_probs(probs),
    )


async def report_summary(
    db: Path | None, shadow_db: Path | None, *, window_minutes: int = WINDOW_MINUTES
) -> ShadowSummary:
    """The shadow report over the on-disk ledgers — the one codepath behind the CLI and the nightly pipeline step.

    Args:
        db: Feedback store path; ``None`` uses the default.
        shadow_db: Shadow ledger path; ``None`` uses the default.
        window_minutes: The hit-join window.

    Returns:
        The :class:`ShadowSummary` at ``window_minutes``.
    """
    from cc_steer.store import FeedbackStore
    from cc_steer.watcher.delivery import ShadowDelivery
    from cc_steer.watcher.live import LiveConfig, MailboxDelivery

    async with await ShadowDelivery.open(shadow_db) as ledger:
        proposals = await ledger.proposals()
    async with await MailboxDelivery.open(shadow_db, config=LiveConfig.shadow()) as mailbox:
        delivered = {int(str(row["proposal_id"])) for row in await mailbox.deliveries() if row["state"] == "delivered"}
        reactions = {
            pid: str(row["kind"])
            for row in await mailbox.reactions()
            if (pid := int(str(row["proposal_id"]))) in delivered
        }
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        interventions = await intervention_rows(store)
    return summarize(proposals, interventions, window_minutes=window_minutes, reactions=reactions)


def payload_of(summary: ShadowSummary) -> dict[str, object]:
    """The summary as one JSON-able payload, including the derived per-session rate."""
    return dataclasses.asdict(summary) | {"proposals_per_session": summary.proposals_per_session}


def journal_shadow_report(journal_repo: Path, summary: ShadowSummary) -> bool:
    """Appends one report line to the shadow journal log; True when recorded."""
    line = f"shadow report | {json.dumps(payload_of(summary), sort_keys=True)}"
    return Journal(journal_repo, title=REPORT_LOG_TITLE, label=REPORT_LOG_LABEL).append(line)


async def intervention_rows(store: FeedbackStore) -> list[dict[str, object]]:
    """Every real intervention's ``(session_id, occurred_at, category)`` from the feedback store.

    Category comes from the latest triage verdict when one exists; unjudged
    events carry an empty string.
    """
    cur = await store.store.conn.execute(
        "SELECT f.session_id, f.occurred_at, COALESCE(j.category, '') AS category "
        "FROM feedback_events f LEFT JOIN latest_judge j ON j.dedup_key = f.dedup_key"
    )
    return [dict(row) async for row in cur]
