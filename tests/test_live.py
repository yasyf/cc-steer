from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from cc_transcript import keep

from cc_steer.spec import STEERING_SPEC
from cc_steer.watcher.delivery import ShadowDelivery
from cc_steer.watcher.live import (
    LiveConfig,
    MailboxDelivery,
    TeeDelivery,
    format_additional_context,
    holdout,
    is_killed,
    scrub_events,
    scrub_text,
)
from tests.builders import parse, user_text
from tests.test_delivery import make_proposal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio


def write_config(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_missing_config_is_shadow(tmp_path: Path) -> None:
    assert LiveConfig.load(tmp_path / "absent.toml") == LiveConfig(mode="shadow")


def test_valid_config_parses_every_knob(tmp_path: Path) -> None:
    config = LiveConfig.load(
        write_config(
            tmp_path / "live.toml",
            'mode = "live_allow"\n'
            f'allow_projects = ["{tmp_path}"]\n'
            "cooldown_turns = 7\nmax_per_session = 9\nmax_live_per_day = 3\n"
            "steer_ttl_minutes = 45\nholdout_frac = 0.25\n",
        )
    )
    assert config.mode == "live_allow"
    assert config.allow_projects == (str(tmp_path),)
    assert (config.cooldown_turns, config.max_per_session, config.max_live_per_day) == (7, 9, 3)
    assert (config.steer_ttl_minutes, config.holdout_frac) == (45, 0.25)


def test_unknown_mode_crashes_loud(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown live mode"):
        LiveConfig.load(write_config(tmp_path / "live.toml", 'mode = "yolo"\n'))


def test_malformed_toml_crashes_loud(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="."):
        LiveConfig.load(write_config(tmp_path / "live.toml", "mode = = =\n"))


def test_config_round_trips_through_toml(tmp_path: Path) -> None:
    original = LiveConfig(mode="mirror", allow_projects=("/a", "/b"), cooldown_turns=4, holdout_frac=0.3)
    path = original.write(tmp_path / "live.toml")
    assert LiveConfig.load(path) == original


def test_allows_matches_exact_dir_and_descendants() -> None:
    config = LiveConfig(mode="live_allow", allow_projects=("/work/proj",))
    assert config.allows("/work/proj")
    assert config.allows("/work/proj/sub")
    assert not config.allows("/work/project-other")
    assert not config.allows("/work")


def test_holdout_is_deterministic_and_frac_shaped() -> None:
    assert all(holdout(i, 0.5) == holdout(i, 0.5) for i in range(50))
    assert all(holdout(i, 1.0) for i in range(50))
    assert not any(holdout(i, 0.0) for i in range(50))
    held = sum(holdout(i, 0.5) for i in range(2000))
    assert 850 <= held <= 1150


def test_kill_switch_reads_env_and_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cc_steer.watcher.live.live_off_path", lambda: tmp_path / "live.off")
    monkeypatch.delenv("CC_STEER_LIVE_OFF", raising=False)
    assert not is_killed()
    (tmp_path / "live.off").touch()
    assert is_killed()
    (tmp_path / "live.off").unlink()
    assert not is_killed()
    monkeypatch.setenv("CC_STEER_LIVE_OFF", "1")
    assert is_killed()


def test_scrub_text_strips_span_keeps_authored() -> None:
    text = "<cc-steer-proposal id=7>redo it with a real fixture</cc-steer-proposal>\n\nactually use pytest not unittest"
    assert scrub_text(text) == "actually use pytest not unittest"


def test_scrub_events_leaves_the_authored_reply_minable(tmp_path: Path) -> None:
    fused = "<cc-steer-proposal id=3>abstract the retry loop</cc-steer-proposal>\n\nno, keep it inline and add a test"
    [event] = scrub_events(parse([user_text(fused)]))
    assert event.text == "no, keep it inline and add a test"
    assert keep(event, STEERING_SPEC)


def test_scrub_events_drops_a_pure_steer_turn_to_short() -> None:
    [event] = scrub_events(parse([user_text("<cc-steer-proposal id=1>do the thing</cc-steer-proposal>")]))
    assert event.text.strip() == ""
    assert not keep(event, STEERING_SPEC)


def test_scrub_events_leaves_untagged_events_untouched() -> None:
    original = parse([user_text("just a normal correction")])
    assert scrub_events(original) == original


def test_format_additional_context_carries_the_tag_and_instruction() -> None:
    context = format_additional_context(42, "run the linter first")
    assert "<cc-steer-proposal id=42>" in context
    assert "run the linter first" in context
    assert "watcher suggests" in context
    assert "(proposal 42)" in context


async def test_mailbox_queues_a_would_be_delivery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.watcher.live.is_killed", lambda: False)
    db = tmp_path / "shadow.db"
    proposal = make_proposal(project="/work/p")
    config = LiveConfig(mode="mirror", steer_ttl_minutes=30, holdout_frac=0.5)
    async with await ShadowDelivery.open(db) as shadow:
        await shadow.deliver(proposal)
    async with await MailboxDelivery.open(db, config=config) as mailbox:
        await mailbox.deliver(proposal)
        rows = await mailbox.recent()
    [row] = rows
    assert (row["state"], row["mode"], row["project"]) == ("queued", "mirror", "/work/p")
    assert row["holdout"] in (0, 1)
    assert row["ttl"] > row["ts"]


async def test_mailbox_is_idempotent_by_proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.watcher.live.is_killed", lambda: False)
    db = tmp_path / "shadow.db"
    proposal = make_proposal()
    async with await ShadowDelivery.open(db) as shadow:
        await shadow.deliver(proposal)
    async with await MailboxDelivery.open(db, config=LiveConfig(mode="mirror")) as mailbox:
        await mailbox.deliver(proposal)
        await mailbox.deliver(proposal)
        assert len(await mailbox.recent()) == 1


async def test_mailbox_skips_when_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.watcher.live.is_killed", lambda: True)
    db = tmp_path / "shadow.db"
    proposal = make_proposal()
    async with await ShadowDelivery.open(db) as shadow:
        await shadow.deliver(proposal)
    async with await MailboxDelivery.open(db, config=LiveConfig(mode="live_all")) as mailbox:
        await mailbox.deliver(proposal)
        assert await mailbox.recent() == []


async def test_mailbox_skips_an_abstention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.watcher.live.is_killed", lambda: False)
    db = tmp_path / "shadow.db"
    proposal = make_proposal(steer=None)
    async with await ShadowDelivery.open(db) as shadow:
        await shadow.deliver(proposal)
    async with await MailboxDelivery.open(db, config=LiveConfig(mode="mirror")) as mailbox:
        await mailbox.deliver(proposal)
        assert await mailbox.recent() == []


async def test_tee_fans_shadow_then_mailbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_steer.watcher.live.is_killed", lambda: False)
    db = tmp_path / "shadow.db"
    proposal = make_proposal(project="/w")
    async with await ShadowDelivery.open(db) as shadow, await MailboxDelivery.open(
        db, config=LiveConfig(mode="mirror")
    ) as mailbox:
        await TeeDelivery([shadow, mailbox]).deliver(proposal)
        assert len(await shadow.proposals()) == 1
        [row] = await mailbox.recent()
    assert row["project"] == "/w"
