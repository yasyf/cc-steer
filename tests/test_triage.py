from __future__ import annotations

import subprocess
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

import pytest
from cc_transcript.domains.mining import ContextSnapshot, ContextTurn

from cc_pushback.detectors import detect
from cc_pushback.triage import (
    AUDIT_PROMPT,
    AUDIT_VERSION,
    AUDITOR,
    JUDGE,
    JUDGE_PROMPT,
    Verdict,
    audit,
    build_prompt,
    sample_audit,
    stratified,
    triage,
)
from tests.builders import assistant_text, parse, user_text

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cc_pushback.store import FeedbackStore
    from cc_pushback.triage import Category

pytestmark = pytest.mark.anyio

FILE = "/repo/projects/session.jsonl"


async def seed(store: FeedbackStore) -> int:
    events = parse(
        [
            assistant_text("here is the diff"),
            user_text("no, use a generator here, this is wrong"),
            assistant_text("switched to a generator"),
            user_text("also stop hardcoding the path"),
        ]
    )
    inserted = await store.record_file_scan(FILE, 1.0, detect(Path(FILE), events))
    assert inserted >= 2
    return inserted


def verdict(category: Category = "wrong_approach", confidence: float = 0.9) -> Verdict:
    return Verdict(category=category, what_claude_did="produced a diff", confidence=confidence, rationale="r")


def judged_row(key: str, kind: str = "transcript_message", *, confidence: float = 0.5) -> dict[str, object]:
    return {"dedup_key": key, "source_kind": kind, "confidence": confidence, "is_pushback": 1}


@pytest.mark.unit
def test_build_prompt_renders_trigger_inputs_and_text() -> None:
    snapshot = ContextSnapshot(
        before=(ContextTurn(role="user", text="please clean the build dir"),),
        trigger=ContextTurn(
            role="assistant",
            text="cleaning now",
            tool_calls=("Bash",),
            tool_inputs=("rm -rf build",),
        ),
        after=(),
    )
    row = {"source_kind": "interrupt_rejection", "context_json": snapshot.to_json(), "text": "no, stop"}
    prompt = build_prompt(JUDGE_PROMPT, row)
    assert "Bash(rm -rf build)" in prompt
    assert "cleaning now" in prompt
    assert "no, stop" in prompt
    assert "please clean the build dir" in prompt
    assert "[source: interrupt_rejection]" in prompt


@pytest.mark.unit
def test_build_prompt_tolerates_legacy_context_without_tool_inputs() -> None:
    snapshot = ContextSnapshot(
        before=(),
        trigger=ContextTurn(role="assistant", text="ran it", tool_calls=("Bash",)),
        after=(),
    )
    row = {"source_kind": "transcript_message", "context_json": snapshot.to_json(), "text": "stop"}
    assert "Bash()" in build_prompt(AUDIT_PROMPT, row)


@pytest.mark.unit
def test_verdict_derives_is_pushback_from_category() -> None:
    assert verdict("wrong_approach").is_pushback is True
    assert verdict("operational_directive").is_pushback is False


@pytest.mark.integration
async def test_triage_judges_all_then_noop(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    total = await seed(store)
    calls: list[str] = []

    async def fake(prompt: str, **_: Any) -> Verdict:
        calls.append(prompt)
        return verdict()

    monkeypatch.setattr("cc_pushback.triage.run_claude_structured", fake)
    report = await triage(store)
    assert (report.judged, report.failed, report.pending) == (total, 0, 0)
    assert len(calls) == total
    again = await triage(store)
    assert (again.judged, again.failed, again.pending) == (0, 0, 0)
    assert len(calls) == total


@pytest.mark.integration
async def test_triage_leaves_feedback_events_untouched(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    await seed(store)
    before = await store.dedup_keys()

    async def fake(prompt: str, **_: Any) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_pushback.triage.run_claude_structured", fake)
    await triage(store)
    assert await store.dedup_keys() == before


@pytest.mark.integration
async def test_prompt_version_bump_rejudges_and_keeps_old_verdicts(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    total = await seed(store)

    async def fake(prompt: str, **_: Any) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_pushback.triage.run_claude_structured", fake)
    await triage(store)
    monkeypatch.setattr("cc_pushback.triage.PROMPT_VERSION", 2)
    report = await triage(store)
    assert report.judged == total
    assert len(await store.judged(role=JUDGE, prompt_version=1)) == total
    assert len(await store.judged(role=JUDGE, prompt_version=2)) == total


@pytest.mark.integration
async def test_one_failing_row_does_not_abort_the_pass(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    total = await seed(store)
    poison = "USER MESSAGE TO CLASSIFY ===\nalso stop hardcoding"

    async def flaky(prompt: str, **_: Any) -> Verdict:
        if poison in prompt:
            raise subprocess.CalledProcessError(1, ["claude"])
        return verdict()

    monkeypatch.setattr("cc_pushback.triage.run_claude_structured", flaky)
    report = await triage(store)
    assert (report.judged, report.failed, report.pending) == (total - 1, 1, 1)

    async def healed(prompt: str, **_: Any) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_pushback.triage.run_claude_structured", healed)
    retry = await triage(store)
    assert (retry.judged, retry.failed, retry.pending) == (1, 0, 0)


@pytest.mark.unit
def test_sample_audit_is_deterministic_under_a_seed() -> None:
    rows = [judged_row(f"k{i}", confidence=i / 100) for i in range(80)] + [
        judged_row(f"r{i}", confidence=i / 100) | {"is_pushback": 0} for i in range(80)
    ]
    first = sample_audit(rows, accepts=20, rejects=20, seed=7)
    second = sample_audit(rows, accepts=20, rejects=20, seed=7)
    assert [r["dedup_key"] for r in first.core] == [r["dedup_key"] for r in second.core]
    assert [r["dedup_key"] for r in first.oversample] == [r["dedup_key"] for r in second.oversample]


@pytest.mark.unit
def test_stratified_honors_kind_quotas_and_oversample_split() -> None:
    rows: list[Mapping[str, object]] = (
        [judged_row(f"i{i}", "interrupt_rejection") for i in range(5)]
        + [judged_row(f"c{i}", "review_comment", confidence=i / 20) for i in range(15)]
        + [judged_row(f"p{i}", "plan_review", confidence=i / 20) for i in range(15)]
        + [judged_row(f"t{i}", confidence=i / 100) for i in range(50)]
    )
    core, oversample = stratified(rows, 40, Random(1))
    by_kind_core = {kind: sum(r["source_kind"] == kind for r in core) for kind in {str(r["source_kind"]) for r in rows}}
    assert by_kind_core["interrupt_rejection"] == 5  # exhaustive
    assert by_kind_core["review_comment"] == 7  # quota 10, 30% oversampled
    assert by_kind_core["plan_review"] == 7
    assert by_kind_core["transcript_message"] == 11  # remainder 15, round(4.5) = 4 oversampled
    assert len(oversample) == 10
    over_confidences = [float(str(r["confidence"])) for r in oversample if r["source_kind"] == "review_comment"]
    assert over_confidences == sorted(over_confidences)
    assert max(over_confidences) <= 0.15  # the lowest-confidence rows


@pytest.mark.integration
async def test_audit_skips_already_audited_rows(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    total = await seed(store)

    async def fake(prompt: str, **_: Any) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_pushback.triage.run_claude_structured", fake)
    await triage(store)
    first = await audit(store, accepts=10, rejects=10, seed=1)
    assert first.judged == total  # the corpus is tiny: every judged row is sampled
    assert len(await store.judged(role=AUDITOR, prompt_version=AUDIT_VERSION)) == total
    second = await audit(store, accepts=10, rejects=10, seed=1)
    assert second.judged == 0
