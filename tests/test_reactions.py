from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import numpy as np
import pytest
from click.testing import CliRunner

from cc_steer.cli import main
from cc_steer.export import LIVE_LABEL, live_gate_row, live_watcher_row
from cc_steer.store import INSERT_EVENT, FeedbackStore
from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.live import LiveConfig, MailboxDelivery
from cc_steer.watcher.reactions import (
    ReplyTurn,
    attribute,
    attribute_reactions,
    classify,
    first_reply,
    jaccard,
    similarity,
)
from cc_steer.watcher.shadow import parse_ts
from tests.test_delivery import make_proposal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.anyio

STEER = "run the linter before you push the code"
ACCEPT_REPLY = "run the linter before you push the code"
EDIT_REPLY = "run the linter before pushing"
DIVERGE_REPLY = "actually revert that change entirely"


def delivery_row(
    proposal_id: int = 1,
    *,
    state: str = "delivered",
    decided_at: str | None = "2026-07-07T10:00:00+00:00",
    session: str = "s1",
    steer: str | None = STEER,
) -> dict[str, object]:
    return {
        "id": proposal_id,
        "proposal_id": proposal_id,
        "session_id": session,
        "state": state,
        "decided_at": decided_at,
        "steer": steer,
    }


def reply(text: str, *, occurred: str = "2026-07-07T10:05:00+00:00", key: str = "reply-k") -> ReplyTurn:
    return ReplyTurn(occurred_at=occurred, text=text, dedup_key=key)


# --- pure similarity and classification -------------------------------------


def test_jaccard_bands_the_three_reply_shapes() -> None:
    assert jaccard(STEER, ACCEPT_REPLY) == 1.0
    assert jaccard(STEER, EDIT_REPLY) == pytest.approx(0.5)
    assert jaccard(STEER, DIVERGE_REPLY) == 0.0


def test_jaccard_is_empty_safe() -> None:
    assert jaccard("", "") == 1.0
    assert jaccard("a b", "") == 0.0


def test_similarity_uses_the_encoder_when_given() -> None:
    class FakeEncoder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            table = {STEER: [1.0, 0.0], ACCEPT_REPLY: [1.0, 0.0], DIVERGE_REPLY: [0.0, 1.0]}
            return np.asarray([table[text] for text in texts], dtype=np.float32)

    assert similarity(STEER, ACCEPT_REPLY, FakeEncoder()) == pytest.approx(1.0)
    assert similarity(STEER, DIVERGE_REPLY, FakeEncoder()) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("text", "kind"),
    [(ACCEPT_REPLY, "accepted"), (EDIT_REPLY, "edited"), (DIVERGE_REPLY, "diverged")],
    ids=["high_is_accepted", "mid_is_edited", "low_is_diverged"],
)
def test_classify_bands_similarity(text: str, kind: str) -> None:
    result, sim = classify(STEER, reply(text), None)
    assert result == kind
    assert sim is not None


def test_classify_no_reply_is_ignored() -> None:
    assert classify(STEER, None, None) == ("ignored", None)


def test_first_reply_respects_the_window() -> None:
    at = parse_ts("2026-07-07T10:00:00+00:00")
    assert at is not None
    replies = [
        reply("too early", occurred="2026-07-07T09:59:00+00:00", key="early"),
        reply("in window", occurred="2026-07-07T10:10:00+00:00", key="hit"),
        reply("too late", occurred="2026-07-07T10:40:00+00:00", key="late"),
    ]
    from datetime import timedelta

    picked = first_reply(replies, at, window=timedelta(minutes=30))
    assert picked is not None and picked.dedup_key == "hit"
    assert first_reply(replies[:1], at, window=timedelta(minutes=30)) is None


def test_first_reply_picks_earliest_instant_across_timezones() -> None:
    at = parse_ts("2026-07-07T10:00:00+00:00")
    assert at is not None
    from datetime import timedelta

    replies = [
        reply("later", occurred="2026-07-07T05:20:00-05:00", key="later"),
        reply("earlier", occurred="2026-07-07T12:10:00+02:00", key="earlier"),
    ]
    picked = first_reply(replies, at, window=timedelta(minutes=30))
    assert picked is not None and picked.dedup_key == "earlier"


# --- pure attribution over rows ---------------------------------------------


def test_attribute_delivered_reply_bands_into_kind() -> None:
    replies = {"s1": [reply(ACCEPT_REPLY, key="k-accept")]}
    [reaction] = attribute([delivery_row()], replies, frozenset())
    assert (reaction.kind, reaction.source) == ("accepted", "scan_inferred")
    assert reaction.feedback_dedup_key == "k-accept"
    assert reaction.similarity == pytest.approx(1.0)


def test_attribute_silent_window_is_ignored_not_negative() -> None:
    [reaction] = attribute([delivery_row()], {"s1": []}, frozenset())
    assert reaction.kind == "ignored"
    assert reaction.feedback_dedup_key is None and reaction.similarity is None


def test_attribute_expired_delivery_is_its_own_kind() -> None:
    [reaction] = attribute([delivery_row(state="expired", decided_at=None)], {}, frozenset())
    assert reaction.kind == "expired"
    assert reaction.feedback_dedup_key is None


@pytest.mark.parametrize("state", ["mirror", "holdout", "queued", "suppressed_budget", "suppressed_invalid"])
def test_attribute_skips_undelivered_states(state: str) -> None:
    assert attribute([delivery_row(state=state)], {"s1": [reply(ACCEPT_REPLY)]}, frozenset()) == []


def test_attribute_respects_explicit_verb_precedence() -> None:
    assert attribute([delivery_row(proposal_id=7)], {"s1": [reply(ACCEPT_REPLY)]}, frozenset({7})) == []


def test_attribute_ignores_a_reply_before_the_delivery() -> None:
    replies = {"s1": [reply(ACCEPT_REPLY, occurred="2026-07-07T09:55:00+00:00")]}
    [reaction] = attribute([delivery_row()], replies, frozenset())
    assert reaction.kind == "ignored"


# --- persistence, precedence, and the end-to-end pass ------------------------


async def seed_proposals(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.watcher.live.is_killed", lambda: False)
    async with await ShadowDelivery.open(db) as shadow:
        await shadow.deliver(make_proposal(session_id="s1", anchor_uuid="a1", steer=STEER))
        await shadow.deliver(make_proposal(session_id="s2", anchor_uuid="a2", steer="rename the helper"))
    async with await MailboxDelivery.open(db, config=LiveConfig(mode="live_all")) as mailbox:
        await mailbox.deliver(make_proposal(session_id="s1", anchor_uuid="a1", steer=STEER))
        await mailbox.deliver(make_proposal(session_id="s2", anchor_uuid="a2", steer="rename the helper"))
        await mailbox.conn.execute(
            "UPDATE deliveries SET state='delivered', decided_at=? WHERE proposal_id=1",
            ("2026-07-07T10:00:00+00:00",),
        )
        await mailbox.conn.execute(
            "UPDATE deliveries SET state='expired', decided_at=? WHERE proposal_id=2",
            ("2026-07-07T10:00:00+00:00",),
        )


async def plant_reply(store: FeedbackStore, *, session: str, occurred: str, text: str, key: str) -> None:
    async with store.store.transaction() as conn:
        await conn.execute(
            INSERT_EVENT,
            (key, "transcript_message", session, f"u-{key}", occurred, text, "{}", "{}", "2.0.0", occurred, None),
        )


async def test_attribute_reactions_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "shadow.db"
    await seed_proposals(db, monkeypatch)
    async with await FeedbackStore.open(tmp_path / "feedback.db") as store:
        await plant_reply(store, session="s1", occurred="2026-07-07T10:05:00+00:00", text=EDIT_REPLY, key="reply-1")
        report = await attribute_reactions(store, shadow_db=db)
        assert report.counts == {"edited": 1, "expired": 1}
        async with await MailboxDelivery.open(db, config=LiveConfig.shadow()) as mailbox:
            by_proposal = {int(row["proposal_id"]): row for row in await mailbox.reactions()}
    assert by_proposal[1]["kind"] == "edited"
    assert by_proposal[1]["feedback_dedup_key"] == "reply-1"
    assert by_proposal[1]["source"] == "scan_inferred"
    assert by_proposal[2]["kind"] == "expired"


async def test_shadow_report_ignores_reactions_on_undelivered_proposals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cc_steer.watcher.shadow import report_summary

    db = tmp_path / "shadow.db"
    await seed_proposals(db, monkeypatch)
    async with await MailboxDelivery.open(db, config=LiveConfig.shadow()) as mailbox:
        await mailbox.conn.execute("UPDATE deliveries SET state='holdout', decided_at=NULL WHERE proposal_id=2")
        await mailbox.record_reaction(proposal_id=1, delivery_id=1, kind="accepted", source="cli_verb")
        await mailbox.record_reaction(proposal_id=2, delivery_id=2, kind="accepted", source="cli_verb")
    summary = await report_summary(tmp_path / "feedback.db", db)
    assert (summary.steers, summary.hits, summary.nuisance) == (2, 1, 1)
    assert summary.hit_categories == {"accepted": 1}


def test_live_verb_rejects_an_unknown_proposal(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["live", "accept", "999"])
    assert result.exit_code != 0
    assert "no proposal 999" in result.output


async def test_attribute_reactions_is_a_noop_without_deliveries(tmp_path: Path) -> None:
    db = tmp_path / "shadow.db"
    async with await FeedbackStore.open(tmp_path / "feedback.db") as store:
        report = await attribute_reactions(store, shadow_db=db)
    assert report.total == 0


async def test_cli_verb_wins_over_a_later_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "shadow.db"
    await seed_proposals(db, monkeypatch)
    async with await MailboxDelivery.open(db, config=LiveConfig.shadow()) as mailbox:
        await mailbox.record_reaction(proposal_id=1, delivery_id=1, kind="dismissed", source="cli_verb")
    async with await FeedbackStore.open(tmp_path / "feedback.db") as store:
        await plant_reply(store, session="s1", occurred="2026-07-07T10:05:00+00:00", text=ACCEPT_REPLY, key="reply-1")
        await attribute_reactions(store, shadow_db=db)
        async with await MailboxDelivery.open(db, config=LiveConfig.shadow()) as mailbox:
            by_proposal = {int(row["proposal_id"]): row for row in await mailbox.reactions()}
    assert by_proposal[1]["kind"] == "dismissed"
    assert by_proposal[1]["source"] == "cli_verb"


async def test_scan_reaction_does_not_clobber_a_cli_verb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "shadow.db"
    await seed_proposals(db, monkeypatch)
    async with await MailboxDelivery.open(db, config=LiveConfig.shadow()) as mailbox:
        await mailbox.record_reaction(proposal_id=1, delivery_id=1, kind="diverged", source="scan_inferred")
        await mailbox.record_reaction(proposal_id=1, delivery_id=1, kind="accepted", source="cli_verb")
        await mailbox.record_reaction(proposal_id=1, delivery_id=1, kind="ignored", source="scan_inferred")
        [row] = await mailbox.reactions()
    assert (row["kind"], row["source"]) == ("accepted", "cli_verb")


# --- label mapping (export) --------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "bucket", "label", "confidence"),
    [
        ("accepted", "pos", True, 0.9),
        ("edited", "pos-corrected", True, 0.7),
        ("diverged", "pos-corrected", True, 0.5),
        ("dismissed", "neg", False, 0.8),
        ("ignored", "weak-neg", False, 0.3),
    ],
)
def test_live_label_maps_every_kind_to_a_bucket(kind: str, bucket: str, label: bool, confidence: float) -> None:
    assert LIVE_LABEL[kind] == (bucket, label, confidence)


def test_expired_never_becomes_a_label() -> None:
    assert "expired" not in LIVE_LABEL


def reaction_dict(kind: str, *, key: str | None = None) -> dict[str, object]:
    return {
        "kind": kind,
        "proposal_id": 3,
        "session_id": "sess-0",
        "feedback_dedup_key": key,
        "steer": STEER,
        "window_render": "<user>\nplease do step\n\n<assistant>\ndid step",
    }


def test_live_watcher_row_uses_the_steer_when_accepted() -> None:
    row = live_watcher_row(reaction_dict("accepted"), {})
    assert row is not None
    assert row["label"] is True and row["label_confidence"] == 0.9
    assert row["source_kind"] == "live_reaction"
    assert row["completion"] == [{"role": "assistant", "content": STEER}]
    assert row["prompt"] == [
        {"role": "user", "content": "please do step"},
        {"role": "assistant", "content": "did step"},
    ]


def test_live_watcher_row_uses_the_reply_when_corrected() -> None:
    row = live_watcher_row(reaction_dict("edited", key="reply-1"), {"reply-1": EDIT_REPLY})
    assert row is not None
    assert row["completion"] == [{"role": "assistant", "content": EDIT_REPLY}]
    assert row["verbatim"] == EDIT_REPLY


def test_live_watcher_row_sentinels_a_negative() -> None:
    row = live_watcher_row(reaction_dict("dismissed"), {})
    assert row is not None
    assert row["label"] is False
    assert row["completion"] == [{"role": "assistant", "content": "NO_STEER"}]


def test_live_rows_skip_expired() -> None:
    assert live_watcher_row(reaction_dict("expired"), {}) is None
    assert live_gate_row(reaction_dict("expired")) is None


def test_live_gate_row_carries_the_flattened_window() -> None:
    row = live_gate_row(reaction_dict("ignored"))
    assert row is not None
    assert row["text"] == "<user>\nplease do step\n\n<assistant>\ndid step"
    assert (row["label"], row["label_confidence"], row["source_kind"]) == (False, 0.3, "live_reaction")


# --- the explicit CLI verbs --------------------------------------------------


@pytest.mark.parametrize(
    ("verb", "kind"), [("accept", "accepted"), ("dismiss", "dismissed"), ("edit", "edited")]
)
def test_live_verb_records_a_cli_reaction(tmp_path: Path, verb: str, kind: str) -> None:
    db = tmp_path / "shadow.db"

    async def seed() -> None:
        async with await ShadowDelivery.open(db) as shadow:
            await shadow.deliver(make_proposal(session_id="s1", anchor_uuid="a1", steer=STEER))

    async def read() -> list[dict[str, object]]:
        async with await MailboxDelivery.open(db, config=LiveConfig.shadow()) as mailbox:
            return await mailbox.reactions()

    anyio.run(seed)
    result = CliRunner().invoke(main, ["live", verb, "1"])
    assert result.exit_code == 0, result.output
    [row] = anyio.run(read)
    assert (row["kind"], row["source"]) == (kind, "cli_verb")
