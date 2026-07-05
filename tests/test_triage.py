from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

import pytest
from cc_transcript.context import SUMMARY_LABEL
from cc_transcript.judge import JudgeError, sample_audit
from cc_transcript.judge.verdicts import stratified

from cc_steer.detectors import detect
from cc_steer.triage import (
    AUDIT_PROMPT,
    AUDIT_VERSION,
    AUDITOR,
    JUDGE,
    JUDGE_PROMPT,
    KIND_QUOTAS,
    REMAINDER_KIND,
    Verdict,
    audit,
    build_prompt,
    triage,
)
from tests.builders import (
    SESSION,
    assistant_text,
    assistant_tool_use,
    denial_result,
    interrupt_result,
    parse,
    user_text,
    write_transcript,
)
from tests.fixture_prompt import FIXTURE_PATH, render

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Any

    from cc_transcript.models import TranscriptEvent

    from cc_steer.store import FeedbackStore
    from cc_steer.triage import Category

pytestmark = pytest.mark.anyio

FILE = "/repo/projects/session.jsonl"


def seed_entries() -> list[dict[str, Any]]:
    return [
        assistant_text("here is the diff"),
        user_text("no, use a generator here, this is wrong"),
        assistant_text("switched to a generator"),
        user_text("also stop hardcoding the path"),
    ]


async def seed(store: FeedbackStore) -> int:
    inserted = await store.record_file_scan(FILE, 1.0, detect(parse(seed_entries())))
    assert inserted >= 2
    return inserted


def verdict(category: Category = "wrong_approach", confidence: float = 0.9) -> Verdict:
    return Verdict(category=category, what_claude_did="produced a diff", confidence=confidence, rationale="r")


def judged_row(key: str, kind: str = "transcript_message", *, confidence: float = 0.5) -> dict[str, object]:
    return {"dedup_key": key, "source_kind": kind, "confidence": confidence, "accepted": 1}


def candidate_row(events: list[TranscriptEvent], *, source_kind: str) -> dict[str, object]:
    candidate = next(c for c in detect(events) if c.source_kind == source_kind)
    return {"source_kind": candidate.source_kind, "context_json": candidate.window.to_json(), "text": candidate.text}


async def judge_fidelities(store: FeedbackStore) -> set[str]:
    cur = await store.store.conn.execute("SELECT DISTINCT fidelity FROM triage WHERE role = 'judge'")
    return {str(row["fidelity"]) async for row in cur}


@pytest.mark.unit
async def test_build_prompt_renders_full_fidelity_while_the_transcript_lives(projects_root: Path) -> None:
    entries = [
        user_text("please clean the build dir"),
        assistant_text("cleaning now"),
        assistant_tool_use("t1", "Bash", {"command": "rm -rf build"}),
        interrupt_result("t1"),
        user_text("no, stop"),
    ]
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    row = candidate_row(parse(entries), source_kind="interrupt_rejection")
    prompt, fidelity = await build_prompt(JUDGE_PROMPT, row)
    assert fidelity == "full"
    assert "rm -rf build" in prompt
    assert "cleaning now" in prompt
    assert "no, stop" in prompt
    assert "please clean the build dir" in prompt
    assert "[source: interrupt_rejection]" in prompt
    assert "=== the turn the message arrived in ===" in prompt
    assert SUMMARY_LABEL not in prompt


@pytest.mark.unit
async def test_build_prompt_renders_an_oversized_edit_unclipped(projects_root: Path) -> None:
    long_new = "\n".join(f"refreshed_line_{i:03d} = compute_refreshed_value({i})" for i in range(40))
    assert len(long_new) > 1500  # would have been bake-truncated pre-2.0
    entries = [
        user_text("wire up the release job"),
        assistant_tool_use(
            "t2", "Edit", {"file_path": "/repo/app.py", "old_string": "old_line()", "new_string": long_new}
        ),
        denial_result("t2", said="no, stop — this rewrites the deploy script"),
        user_text("leave the deploy script alone"),
    ]
    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    row = candidate_row(parse(entries), source_kind="interrupt_rejection")
    prompt, fidelity = await build_prompt(JUDGE_PROMPT, row)
    assert fidelity == "full"
    assert all(f"+ {line}" in prompt for line in long_new.splitlines())
    assert "…(+" not in prompt  # nothing in this window hits a budget


@pytest.mark.unit
async def test_build_prompt_falls_back_to_labeled_previews_once_expired() -> None:
    entries = [
        assistant_text("here is the diff"),
        user_text("no, this clobbers the config"),
    ]
    row = candidate_row(parse(entries), source_kind="transcript_message")  # no transcript on disk
    prompt, fidelity = await build_prompt(AUDIT_PROMPT, row)
    assert fidelity == "summary"
    assert SUMMARY_LABEL in prompt
    assert "here is the diff" in prompt
    assert "no, this clobbers the config" in prompt
    assert "=== HUMAN MESSAGE TO ASSESS ===" in prompt


@pytest.mark.unit
def test_verdict_derives_is_steering_from_category() -> None:
    assert verdict("wrong_approach").is_steering is True
    assert verdict("operational_directive").is_steering is False


@pytest.mark.unit
def test_verdict_aliases_satisfy_verdict_like() -> None:
    accepted = verdict("wrong_approach")
    assert accepted.accepted is accepted.is_steering is True
    assert accepted.summary == accepted.what_claude_did == "produced a diff"


@pytest.mark.unit
async def test_build_prompt_matches_the_byte_fixture(projects_root: Path) -> None:
    assert (await render(projects_root)).encode() == FIXTURE_PATH.read_bytes()


@pytest.mark.integration
async def test_triage_judges_all_then_noop(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    total = await seed(store)
    calls: list[str] = []

    async def fake(prompt: str) -> Verdict:
        calls.append(prompt)
        return verdict()

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: fake)
    report = await triage(store)
    assert (report.judged, report.failed, report.pending) == (total, 0, 0)
    assert len(calls) == total
    again = await triage(store)
    assert (again.judged, again.failed, again.pending) == (0, 0, 0)
    assert len(calls) == total


@pytest.mark.integration
async def test_triage_records_summary_fidelity_for_expired_transcripts(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed(store)  # FILE is never written under projects_root, so windows cannot hydrate

    async def fake(prompt: str) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: fake)
    await triage(store)
    assert await judge_fidelities(store) == {"summary"}


@pytest.mark.integration
async def test_refresh_summary_rejudges_at_full_fidelity_once_hydratable(
    store: FeedbackStore, projects_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = seed_entries()
    total = await store.record_file_scan(FILE, 1.0, detect(parse(entries)))

    async def fake(prompt: str) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: fake)
    await triage(store)
    assert await judge_fidelities(store) == {"summary"}
    assert (await triage(store)).judged == 0  # without the flag, summary rows stay settled

    write_transcript(projects_root / "proj" / f"{SESSION}.jsonl", entries)
    refreshed = await triage(store, refresh_summary=True)
    assert refreshed.judged == total
    assert await judge_fidelities(store) == {"full"}


@pytest.mark.integration
async def test_triage_leaves_feedback_events_untouched(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    await seed(store)
    before = await store.dedup_keys()

    async def fake(prompt: str) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: fake)
    await triage(store)
    assert await store.dedup_keys() == before


@pytest.mark.integration
async def test_prompt_version_bump_rejudges_and_keeps_old_verdicts(
    store: FeedbackStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cc_steer.triage import PROMPT_VERSION

    total = await seed(store)

    async def fake(prompt: str) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: fake)
    await triage(store)
    monkeypatch.setattr("cc_steer.triage.PROMPT_VERSION", PROMPT_VERSION + 1)
    report = await triage(store)
    assert report.judged == total
    assert len(await store.judged(role=JUDGE, prompt_version=PROMPT_VERSION)) == total
    assert len(await store.judged(role=JUDGE, prompt_version=PROMPT_VERSION + 1)) == total


@pytest.mark.integration
async def test_one_failing_row_does_not_abort_the_pass(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    total = await seed(store)
    poison = "USER MESSAGE TO CLASSIFY ===\nalso stop hardcoding"

    async def flaky(prompt: str) -> Verdict:
        if poison in prompt:
            raise JudgeError("claude exited 1")
        return verdict()

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: flaky)
    report = await triage(store)
    assert (report.judged, report.failed, report.pending) == (total - 1, 1, 1)

    async def healed(prompt: str) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: healed)
    retry = await triage(store)
    assert (retry.judged, retry.failed, retry.pending) == (1, 0, 0)


@pytest.mark.unit
def test_sample_audit_is_deterministic_under_a_seed() -> None:
    rows = [judged_row(f"k{i}", confidence=i / 100) for i in range(80)] + [
        judged_row(f"r{i}", confidence=i / 100) | {"accepted": 0} for i in range(80)
    ]
    first = sample_audit(rows, accepts=20, rejects=20, seed=7, quotas=KIND_QUOTAS, remainder_kind=REMAINDER_KIND)
    second = sample_audit(rows, accepts=20, rejects=20, seed=7, quotas=KIND_QUOTAS, remainder_kind=REMAINDER_KIND)
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
    core, oversample = stratified(rows, 40, Random(1), KIND_QUOTAS, REMAINDER_KIND, 0.3)
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

    async def fake(prompt: str) -> Verdict:
        return verdict()

    monkeypatch.setattr("cc_steer.triage.structured_judge", lambda *_, **__: fake)
    await triage(store)
    first = await audit(store, accepts=10, rejects=10, seed=1)
    assert first.judged == total  # the corpus is tiny: every judged row is sampled
    assert len(await store.judged(role=AUDITOR, prompt_version=AUDIT_VERSION)) == total
    second = await audit(store, accepts=10, rejects=10, seed=1)
    assert second.judged == 0
