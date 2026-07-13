from __future__ import annotations

import json
from typing import TYPE_CHECKING

import anyio
import pytest
from click.testing import CliRunner

from cc_steer.cli import main
from cc_steer.store import FeedbackStore
from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.shadow import journal_shadow_report, payload_of, summarize
from tests.test_delivery import make_proposal
from tests.test_exemplars import TRAIN_SESSION, seed_steering

if TYPE_CHECKING:
    from pathlib import Path


def proposal_row(
    session: str = "s1",
    ts: str = "2026-07-07T10:00:00+00:00",
    draft: str | None = "d",
    steer: str | None = "s",
    sentinel_prob: float | None = None,
) -> dict[str, object]:
    return {"session_id": session, "ts": ts, "draft": draft, "steer": steer, "sentinel_prob": sentinel_prob}


def intervention(
    session: str = "s1", occurred: str = "2026-07-07T10:10:00+00:00", category: str = ""
) -> dict[str, object]:
    return {"session_id": session, "occurred_at": occurred, "category": category}


def test_a_nearby_intervention_is_a_hit() -> None:
    summary = summarize([proposal_row()], [intervention()])
    assert (summary.steers, summary.hits, summary.nuisance) == (1, 1, 0)
    assert (summary.sessions, summary.proposals) == (1, 1)
    assert summary.proposals_per_session == 1.0


def test_a_far_or_earlier_intervention_is_a_nuisance_candidate() -> None:
    late = summarize([proposal_row()], [intervention(occurred="2026-07-07T10:40:00+00:00")])
    assert (late.hits, late.nuisance) == (0, 1)
    earlier = summarize([proposal_row()], [intervention(occurred="2026-07-07T09:59:00+00:00")])
    assert (earlier.hits, earlier.nuisance) == (0, 1)
    other_session = summarize([proposal_row()], [intervention(session="s2")])
    assert (other_session.hits, other_session.nuisance) == (0, 1)


def test_the_window_is_tunable() -> None:
    summary = summarize(
        [proposal_row()], [intervention(occurred="2026-07-07T10:40:00+00:00")], window_minutes=60
    )
    assert (summary.hits, summary.nuisance) == (1, 0)


def test_abstentions_count_per_stage_and_never_join() -> None:
    rows = [
        proposal_row(draft=None, steer=None),
        proposal_row(session="s2", draft="d", steer=None),
        proposal_row(session="s2", ts="2026-07-07T11:00:00+00:00"),
    ]
    summary = summarize(rows, [intervention()])
    assert (summary.stage2_abstained, summary.stage3_abstained) == (1, 1)
    assert (summary.proposals, summary.sessions, summary.steers) == (3, 2, 1)
    assert (summary.hits, summary.nuisance) == (0, 1)


def test_naive_timestamps_are_treated_as_utc() -> None:
    summary = summarize([proposal_row()], [intervention(occurred="2026-07-07T10:10:00")])
    assert summary.hits == 1


def test_empty_ledger_summarizes_to_zeroes() -> None:
    summary = summarize([], [intervention()])
    assert (summary.sessions, summary.proposals, summary.steers) == (0, 0, 0)
    assert summary.proposals_per_session == 0.0
    assert summary.sentinel_probs is None
    assert summary.hit_categories == {}


def test_hits_count_per_intervention_category() -> None:
    rows = [proposal_row(), proposal_row(session="s2")]
    hits = [intervention(category="wrong_approach"), intervention(session="s2", category="direction")]
    summary = summarize(rows, hits)
    assert summary.hit_categories == {"wrong_approach": 1, "direction": 1}
    unjudged = summarize([proposal_row()], [intervention()])
    assert unjudged.hit_categories == {"(unjudged)": 1}


def test_sentinel_probs_summarize_over_scored_proposals() -> None:
    rows = [proposal_row(sentinel_prob=p / 10) for p in range(1, 11)] + [proposal_row(session="s2")]
    summary = summarize(rows, [])
    stats = summary.sentinel_probs
    assert stats is not None
    assert stats.n == 10
    assert stats.mean == pytest.approx(0.55)
    assert len(stats.deciles) == 9
    assert stats.deciles[0] <= stats.deciles[-1]


def test_payload_of_carries_the_derived_per_session_rate() -> None:
    summary = summarize([proposal_row(), proposal_row(session="s2")], [])
    payload = payload_of(summary)
    assert payload["proposals"] == 2
    assert payload["proposals_per_session"] == 1.0
    assert payload["window_minutes"] == summary.window_minutes


def test_journal_shadow_report_appends_one_sorted_json_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: list[tuple[Path, str, str, str]] = []

    class FakeJournal:
        def __init__(self, repo: Path, *, title: str, label: str) -> None:
            self.repo, self.title, self.label = repo, title, label

        def append(self, text: str) -> bool:
            recorded.append((self.repo, self.title, self.label, text))
            return True

    monkeypatch.setattr("cc_steer.watcher.shadow.Journal", FakeJournal)
    summary = summarize([proposal_row()], [intervention()])
    assert journal_shadow_report(tmp_path, summary)
    ((repo, title, label, text),) = recorded
    assert (repo, title, label) == (tmp_path, "cc-steer shadow reports", "shadow")
    prefix, _, body = text.partition(" | ")
    assert prefix == "shadow report"
    assert json.loads(body) == payload_of(summary)


def test_a_delivered_proposal_scores_by_its_reaction_not_the_window() -> None:
    proposals = [proposal_row() | {"id": 1}, proposal_row(session="s2") | {"id": 2}]
    summary = summarize(proposals, [intervention(session="s2")], reactions={1: "accepted", 2: "ignored"})
    assert (summary.steers, summary.hits, summary.nuisance) == (2, 1, 1)
    assert summary.hit_categories == {"accepted": 1}


def test_an_undelivered_proposal_keeps_the_window_join() -> None:
    summary = summarize([proposal_row() | {"id": 1}], [intervention()], reactions={})
    assert (summary.hits, summary.nuisance) == (1, 0)


@pytest.mark.integration
def test_watch_delivery_is_config_driven_not_a_live_flag() -> None:
    result = CliRunner().invoke(main, ["watch", "--live"])
    assert result.exit_code != 0
    assert "No such option" in result.output and "--live" in result.output
    help_result = CliRunner().invoke(main, ["watch", "--help"])
    assert "--shadow" in help_result.output
    assert "live.toml" in help_result.output


@pytest.mark.integration
def test_shadow_report_joins_the_two_databases(tmp_path: Path) -> None:
    async def seed() -> None:
        async with await FeedbackStore.open(tmp_path / "feedback.db") as store:
            await seed_steering(store, "k-train", TRAIN_SESSION, "u1")
        async with await ShadowDelivery.open(tmp_path / "shadow.db") as delivery:
            await delivery.deliver(
                make_proposal(session_id=TRAIN_SESSION, ts="2026-01-01T00:00:00+00:00", steer="final steer")
            )
            await delivery.deliver(make_proposal(session_id="sess-quiet", anchor_uuid="a9", draft=None, steer=None))

    anyio.run(seed)
    result = CliRunner().invoke(
        main,
        [
            "shadow",
            "report",
            "--db",
            str(tmp_path / "feedback.db"),
            "--shadow-db",
            str(tmp_path / "shadow.db"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sessions"] == 2
    assert payload["proposals"] == 2
    assert payload["stage2_abstained"] == 1
    assert (payload["steers"], payload["hits"], payload["nuisance"]) == (1, 1, 0)
    assert payload["proposals_per_session"] == 1.0
