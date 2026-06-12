from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from cc_transcript.domains.mining import ContextSnapshot, ContextTurn, resolved_model
from pydantic import ValidationError

from cc_pushback.detectors import detect
from cc_pushback.refine import PROMPT_VERSION, RefinedPair, Refinement, build_refine_prompt, refine
from cc_pushback.triage import Verdict, triage
from tests.builders import assistant_text, parse, user_text

if TYPE_CHECKING:
    from cc_pushback.store import FeedbackStore

pytestmark = pytest.mark.anyio

FILE = "/repo/projects/session.jsonl"
MODEL = resolved_model("medium")


def verdict() -> Verdict:
    return Verdict(category="wrong_approach", what_claude_did="produced a diff", confidence=0.9, rationale="r")


def refinement(*complaints: str) -> Refinement:
    return Refinement(
        pairs=[
            RefinedPair(action="produced a diff", complaint_verbatim=text, complaint=f"distilled: {text}")
            for text in (complaints or ("no, stop",))
        ]
    )


async def seed_accepted(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> int:
    events = parse(
        [
            assistant_text("here is the diff"),
            user_text("no, use a generator here, this is wrong"),
            assistant_text("switched to a generator"),
            user_text("also stop hardcoding the path"),
        ]
    )
    assert await store.record_file_scan(FILE, 1.0, detect(Path(FILE), events)) >= 2

    async def fake_judge(prompt: str) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_pushback.triage.structured_judge", lambda *_, **__: fake_judge)
    await triage(store)
    return len(await store.unrefined(prompt_version=PROMPT_VERSION, model=MODEL))


@pytest.mark.unit
def test_build_refine_prompt_includes_action_hint_and_context() -> None:
    snapshot = ContextSnapshot(
        before=(ContextTurn(role="user", text="please clean the build dir"),),
        trigger=ContextTurn(role="assistant", text="cleaning now", tool_calls=("Bash",), tool_inputs=("rm -rf build",)),
        after=(),
    )
    row = {
        "source_kind": "interrupt_rejection",
        "context_json": snapshot.to_json(),
        "text": "no, stop",
        "what_claude_did": "force-pushed to main",
    }
    prompt = build_refine_prompt(row)
    assert "Bash(rm -rf build)" in prompt
    assert "cleaning now" in prompt
    assert "no, stop" in prompt
    assert "please clean the build dir" in prompt
    assert "[source: interrupt_rejection]" in prompt
    assert "force-pushed to main" in prompt


@pytest.mark.unit
def test_refinement_requires_at_least_one_pair() -> None:
    with pytest.raises(ValidationError):
        Refinement(pairs=[])


@pytest.mark.integration
async def test_refine_refines_all_accepted_then_noop(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    accepted = await seed_accepted(store, monkeypatch)
    assert accepted >= 2
    calls: list[str] = []

    async def fake(prompt: str) -> Refinement:
        calls.append(prompt)
        return refinement()

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: fake)
    report = await refine(store)
    assert (report.refined, report.pairs, report.failed, report.pending) == (accepted, accepted, 0, 0)
    assert len(calls) == accepted
    again = await refine(store)
    assert (again.refined, again.pairs, again.failed, again.pending) == (0, 0, 0, 0)
    assert len(calls) == accepted


@pytest.mark.integration
async def test_refine_only_touches_accepted_pushback(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    events = parse(
        [
            assistant_text("here is the diff"),
            user_text("no, use a generator here, this is wrong"),
            assistant_text("switched to a generator"),
            user_text("also stop hardcoding the path"),
        ]
    )
    await store.record_file_scan(FILE, 1.0, detect(Path(FILE), events))

    async def picky_judge(prompt: str) -> Verdict:
        noise = "USER MESSAGE TO CLASSIFY ===\nalso stop hardcoding" in prompt
        return Verdict(
            category="status_update" if noise else "wrong_approach",
            what_claude_did="x",
            confidence=0.9,
            rationale="r",
        )

    monkeypatch.setattr("cc_pushback.triage.structured_judge", lambda *_, **__: picky_judge)
    await triage(store)

    async def fake(prompt: str) -> Refinement:
        return refinement()

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: fake)
    report = await refine(store)
    assert report.refined == 1
    rows = await store.pairs()
    assert len(rows) == 1
    assert "use a generator" in str(rows[0]["original_message"])


@pytest.mark.integration
async def test_multi_complaint_message_splits_into_atomic_pairs(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_accepted(store, monkeypatch)

    async def splitter(prompt: str) -> Refinement:
        if "USER PUSHBACK TO REFINE ===\nno, use a generator" in prompt:
            return refinement("use a generator here", "this is wrong")
        return refinement()

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: splitter)
    report = await refine(store)
    assert report.pairs == report.refined + 1  # one event yields two pairs

    pairs = await store.pairs()
    split = [row for row in pairs if "use a generator" in str(row["original_message"])]
    assert [int(row["pair_index"]) for row in split] == [0, 1]
    assert {str(row["complaint_verbatim"]) for row in split} == {"use a generator here", "this is wrong"}


@pytest.mark.integration
async def test_refine_version_bump_re_refines_and_deliverable_shows_new(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_accepted(store, monkeypatch)

    async def first(prompt: str) -> Refinement:
        return refinement("old complaint")

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: first)
    await refine(store)

    async def second(prompt: str) -> Refinement:
        return refinement("new complaint")

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: second)
    monkeypatch.setattr("cc_pushback.refine.PROMPT_VERSION", PROMPT_VERSION + 1)
    report = await refine(store)
    assert report.refined >= 1
    assert all(str(row["complaint_verbatim"]) == "new complaint" for row in await store.pairs())


@pytest.mark.integration
async def test_one_failing_event_does_not_abort_then_heals(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = await seed_accepted(store, monkeypatch)
    poison = "USER PUSHBACK TO REFINE ===\nalso stop hardcoding"

    async def flaky(prompt: str) -> Refinement:
        if poison in prompt:
            raise subprocess.CalledProcessError(1, ["claude"])
        return refinement()

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: flaky)
    report = await refine(store)
    assert (report.refined, report.failed, report.pending) == (accepted - 1, 1, 1)

    async def healed(prompt: str) -> Refinement:
        return refinement()

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: healed)
    retry = await refine(store)
    assert (retry.refined, retry.failed, retry.pending) == (1, 0, 0)


@pytest.mark.integration
async def test_refine_leaves_triage_untouched(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    from cc_pushback.triage import JUDGE
    from cc_pushback.triage import PROMPT_VERSION as JUDGE_VERSION

    await seed_accepted(store, monkeypatch)
    before = await store.judged(role=JUDGE, prompt_version=JUDGE_VERSION)

    async def fake(prompt: str) -> Refinement:
        return refinement()

    monkeypatch.setattr("cc_pushback.refine.structured_judge", lambda *_, **__: fake)
    await refine(store)
    after = await store.judged(role=JUDGE, prompt_version=JUDGE_VERSION)
    assert {str(row["dedup_key"]) for row in after} == {str(row["dedup_key"]) for row in before}
    assert len(after) == len(before)
