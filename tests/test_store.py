from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from cc_transcript.domains.mining import DedupKey

from cc_pushback.detectors import detect
from cc_pushback.refine import RefinedPair, Refinement
from cc_pushback.triage import JUDGE, Verdict
from tests.builders import assistant_tool_use, denial_result, interrupt_result, parse, user_text

if TYPE_CHECKING:
    from cc_pushback.models import FeedbackCandidate
    from cc_pushback.store import FeedbackStore

pytestmark = pytest.mark.anyio

FILE = "/repo/projects/session.jsonl"


async def seeded_keys(store: FeedbackStore) -> list[DedupKey]:
    await store.record_file_scan(FILE, 1.0, sample_candidates())
    rows = await store.unjudged(role=JUDGE, prompt_version=1, model="sonnet")
    return [DedupKey(str(row["dedup_key"])) for row in rows]


def verdict(category: str, *, confidence: float = 0.9) -> Verdict:
    return Verdict.model_validate(
        {"category": category, "what_claude_did": "ran a tool", "confidence": confidence, "rationale": "r"}
    )


def refinement(*complaints: str) -> Refinement:
    return Refinement(
        pairs=[
            RefinedPair(action="ran a tool", complaint_verbatim=text, complaint=f"distilled: {text}")
            for text in (complaints or ("stop that",))
        ]
    )


def sample_candidates() -> list[FeedbackCandidate]:
    events = parse(
        [
            assistant_tool_use("t1", "Write", {"file_path": "/a.py"}),
            denial_result("t1", said="don't do that"),
            assistant_tool_use("t2", "Bash", {"command": "ls"}),
            interrupt_result("t2"),
            user_text("run the tests instead, not the build"),
        ]
    )
    return detect(Path(FILE), events)


@pytest.mark.integration
async def test_record_file_scan_is_idempotent(store: FeedbackStore) -> None:
    candidates = sample_candidates()
    assert len(candidates) >= 2
    first = await store.record_file_scan(FILE, 1.0, candidates)
    second = await store.record_file_scan(FILE, 2.0, candidates)
    assert first == len(candidates)
    assert second == 0
    assert (await store.stats()).total == len(candidates)


@pytest.mark.integration
async def test_record_file_scan_records_mtime(store: FeedbackStore) -> None:
    await store.record_file_scan(FILE, 7.0, sample_candidates())
    assert await store.file_mtimes() == {FILE: 7.0}


@pytest.mark.integration
async def test_record_file_scan_is_atomic_on_failure(store: FeedbackStore, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(path: str, mtime: float) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(store.store, "record_file", boom)
    with pytest.raises(RuntimeError):
        await store.record_file_scan(FILE, 1.0, sample_candidates())
    assert (await store.stats()).total == 0
    assert await store.file_mtimes() == {}


@pytest.mark.integration
async def test_stats_counts_by_source_kind(store: FeedbackStore) -> None:
    await store.record_file_scan(FILE, 1.0, sample_candidates())
    by_source = (await store.stats()).by_source
    assert by_source.get("interrupt_rejection", 0) >= 2
    assert by_source.get("transcript_message", 0) >= 1


@pytest.mark.integration
async def test_events_returns_full_rows_newest_first(store: FeedbackStore) -> None:
    candidates = sample_candidates()
    await store.record_file_scan(FILE, 1.0, candidates)
    rows = await store.events()
    assert len(rows) == len(candidates)
    assert set(rows[0]) == {
        "id",
        "source_kind",
        "occurred_at",
        "text",
        "payload_json",
        "context_json",
        "origin_path",
        "session_id",
    }
    assert all(row["context_json"] for row in rows)
    assert [str(row["occurred_at"]) for row in rows] == sorted((str(row["occurred_at"]) for row in rows), reverse=True)


@pytest.mark.integration
async def test_record_verdict_is_idempotent(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet")
    await store.record_verdict(key, verdict("new_task"), role=JUDGE, prompt_version=1, model="sonnet")
    rows = await store.judged(role=JUDGE, prompt_version=1)
    assert [str(row["category"]) for row in rows if row["dedup_key"] == key] == ["wrong_approach"]


@pytest.mark.integration
async def test_unjudged_honors_version_model_and_limit(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    await store.record_verdict(keys[0], verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet")
    remaining = await store.unjudged(role=JUDGE, prompt_version=1, model="sonnet")
    assert keys[0] not in {str(row["dedup_key"]) for row in remaining}
    assert len(remaining) == len(keys) - 1
    assert len(await store.unjudged(role=JUDGE, prompt_version=2, model="sonnet")) == len(keys)
    assert len(await store.unjudged(role=JUDGE, prompt_version=1, model="haiku")) == len(keys)
    assert len(await store.unjudged(role=JUDGE, prompt_version=1, model="sonnet", limit=1)) == 1


@pytest.mark.integration
async def test_accepted_pushback_filters_noise_and_latest_judge_wins(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    await store.record_verdict(keys[0], verdict("unwanted_action"), role=JUDGE, prompt_version=1, model="sonnet")
    await store.record_verdict(keys[1], verdict("status_update"), role=JUDGE, prompt_version=1, model="sonnet")
    accepted = await store.unrefined(prompt_version=1, model="sonnet")
    assert [str(row["dedup_key"]) for row in accepted] == [keys[0]]
    await store.record_verdict(keys[0], verdict("operational_directive"), role=JUDGE, prompt_version=2, model="sonnet")
    await store.record_verdict(keys[1], verdict("incorrect_change"), role=JUDGE, prompt_version=2, model="sonnet")
    flipped = await store.unrefined(prompt_version=1, model="sonnet")
    assert [str(row["dedup_key"]) for row in flipped] == [keys[1]]


@pytest.mark.integration
async def test_auditor_only_event_is_not_accepted_pushback(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(key, verdict("wrong_approach"), role="auditor", prompt_version=1, model="opus")
    assert await store.unrefined(prompt_version=1, model="sonnet") == []
    assert await store.pairs() == []


@pytest.mark.integration
async def test_record_refinement_is_idempotent(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_refinement(
        key, refinement("use a generator", "stop hardcoding"), prompt_version=1, model="sonnet"
    )
    await store.record_refinement(
        key, refinement("use a generator", "stop hardcoding"), prompt_version=1, model="sonnet"
    )
    rows = await store.pairs()
    assert len(rows) == 2
    assert [int(row["pair_index"]) for row in rows] == [0, 1]
    assert {str(row["complaint_verbatim"]) for row in rows} == {"use a generator", "stop hardcoding"}


@pytest.mark.integration
async def test_unrefined_honors_version_model_and_limit(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    for key in keys:
        await store.record_verdict(key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet")
    await store.record_refinement(keys[0], refinement("x"), prompt_version=1, model="sonnet")
    remaining = await store.unrefined(prompt_version=1, model="sonnet")
    assert keys[0] not in {str(row["dedup_key"]) for row in remaining}
    assert len(remaining) == len(keys) - 1
    assert len(await store.unrefined(prompt_version=2, model="sonnet")) == len(keys)
    assert len(await store.unrefined(prompt_version=1, model="haiku")) == len(keys)
    assert len(await store.unrefined(prompt_version=1, model="sonnet", limit=1)) == 1


@pytest.mark.integration
async def test_refined_pairs_latest_generation_wins(store: FeedbackStore) -> None:
    key = (await seeded_keys(store))[0]
    await store.record_verdict(key, verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet")
    await store.record_refinement(key, refinement("a", "b"), prompt_version=1, model="sonnet")
    await store.record_refinement(key, refinement("c"), prompt_version=2, model="sonnet")
    rows = await store.pairs()
    assert [str(row["complaint_verbatim"]) for row in rows] == ["c"]
    assert rows[0]["prompt_version"] == 2
    assert rows[0]["category"] == "wrong_approach"
    assert rows[0]["action"] == "ran a tool"


@pytest.mark.integration
async def test_triage_stats_counts_by_category(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    await store.record_verdict(keys[0], verdict("wrong_approach"), role=JUDGE, prompt_version=1, model="sonnet")
    await store.record_verdict(keys[1], verdict("status_update"), role=JUDGE, prompt_version=1, model="sonnet")
    stats = await store.triage_stats(prompt_version=1)
    assert (stats.total, stats.judged, stats.accepted) == (len(keys), 2, 1)
    assert stats.by_category == {"wrong_approach": 1, "status_update": 1}


@pytest.mark.integration
async def test_dedup_keys_returns_every_event_key(store: FeedbackStore) -> None:
    keys = await seeded_keys(store)
    assert await store.dedup_keys() == set(keys)
