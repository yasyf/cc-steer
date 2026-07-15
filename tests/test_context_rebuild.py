from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from cc_transcript.activity import SessionActivity
from cc_transcript.context import ContextWindow
from cc_transcript.ids import SessionId

from cc_steer.capture import capture_anchored_window
from cc_steer.context_rebuild import ContextRebuildReport, rebuild_contexts
from cc_steer.detectors import detect, plan_reviews
from cc_steer.export import load_traces
from cc_steer.negatives import GateSample
from cc_steer.rendering import messages
from cc_steer.store import TriageStats
from cc_steer.triage import JUDGE, PROMPT_VERSION, Verdict
from cc_steer.watcher.live import scrub_events
from tests.builders import (
    assistant_text,
    assistant_tool_use,
    denial_result,
    mode_entry,
    parse,
    user_text,
    write_transcript,
)

if TYPE_CHECKING:
    from pathlib import Path

    from cc_transcript.mining import FeedbackCandidate

    from cc_steer.store import FeedbackStore

pytestmark = [pytest.mark.anyio, pytest.mark.integration]

SESSION = "context-rebuild-session"
PLAN = "SENTINEL_REBUILT_PLAN add the module and wire it in"
REJECTION = "SENTINEL_REBUILT_REJECTION show the root cause first"
UNRELATED = "SENTINEL_UNRELATED some compacted leftover prefix chatter"
REENTRY = "SENTINEL_REENTRY this approach is wrong, use a generator"
EARLIER = "SENTINEL_EARLIER write the feature"


def plan_rejection_entries(session_id: str) -> list[dict[str, object]]:
    return [
        user_text("session bootstrap", uuid=f"{session_id}-meta", sessionId=session_id, isMeta=True),
        user_text("earlier request", uuid=f"{session_id}-earlier", sessionId=session_id),
        assistant_text("earlier response", uuid=f"{session_id}-earlier-response", sessionId=session_id),
        user_text("implement the feature", uuid=f"{session_id}-prompt", sessionId=session_id),
        assistant_text("Plan written.", uuid=f"{session_id}-plan-intro", sessionId=session_id),
        assistant_tool_use("p1", "ExitPlanMode", {"plan": PLAN}, uuid=f"{session_id}-plan", sessionId=session_id),
        denial_result("p1", said=REJECTION, uuid=f"{session_id}-anchor", sessionId=session_id),
        assistant_text("addressing the feedback", uuid=f"{session_id}-after", sessionId=session_id),
    ]


def old_candidate(session_id: str) -> tuple[FeedbackCandidate, list[dict[str, object]]]:
    entries = plan_rejection_entries(session_id)
    [candidate] = plan_reviews(parse(entries))
    return replace(candidate, window=replace(candidate.window, before=candidate.window.before[:-1])), entries


def plan_reentry_entries(session_id: str) -> list[dict[str, object]]:
    return [
        user_text(EARLIER, uuid=f"{session_id}-earlier", sessionId=session_id),
        assistant_text("starting", uuid=f"{session_id}-start", sessionId=session_id),
        user_text("continue the current pass", uuid=f"{session_id}-continue", sessionId=session_id),
        assistant_tool_use(
            "e1", "Edit", {"file_path": "/a.py", "old_string": "a", "new_string": "b"},
            uuid=f"{session_id}-edit", sessionId=session_id,
        ),
        mode_entry("plan", sessionId=session_id),
        user_text(REENTRY, uuid=f"{session_id}-anchor", sessionId=session_id),
    ]


def compacted_plan_entries(session_id: str) -> list[dict[str, object]]:
    return [
        user_text(UNRELATED, uuid=f"{session_id}-unrelated", sessionId=session_id),
        assistant_tool_use(
            "p1", "ExitPlanMode", {"plan": "compacted stub"}, uuid=f"{session_id}-plan", sessionId=session_id
        ),
        denial_result("p1", said=REJECTION, uuid=f"{session_id}-anchor", sessionId=session_id),
    ]


def detected(entries: list[dict[str, object]], anchor_uuid: str) -> FeedbackCandidate:
    [candidate] = [
        candidate
        for candidate in detect(scrub_events(parse(entries)))
        if str(candidate.ref.event_uuid) == anchor_uuid
    ]
    return candidate


async def test_rebuild_recovers_context_and_second_run_is_a_noop(
    store: FeedbackStore, projects_root: Path, tmp_path: Path
) -> None:
    candidate, entries = old_candidate(SESSION)
    await store.record_file_scan("/obsolete/.cc-pushback/session.jsonl", 1.0, [candidate])
    mirror_root = tmp_path / "mirror"
    complete = write_transcript(mirror_root / "complete" / f"{SESSION}.jsonl", entries)
    incomplete = write_transcript(
        mirror_root / "incomplete" / f"{SESSION}.jsonl",
        [denial_result("p1", said=REJECTION, uuid=f"{SESSION}-anchor", sessionId=SESSION)],
    )
    os.utime(complete, (1, 1))
    os.utime(incomplete, (2, 2))
    write_transcript(
        projects_root / "project" / f"{SESSION}.jsonl",
        [user_text("compacted transcript", uuid=f"{SESSION}-compacted", sessionId=SESSION)],
    )

    first = await rebuild_contexts(store, (mirror_root, projects_root))
    row = await (
        await store.store.conn.execute(
            "SELECT context_json, quarantined_reason FROM feedback_events WHERE dedup_key = ?",
            (candidate.dedup_key,),
        )
    ).fetchone()
    assert row is not None
    content = "\n".join(
        message["content"] for message in messages(ContextWindow.from_json(str(row["context_json"])).before)
    )
    assert first == ContextRebuildReport(found=1, rebuilt=1, quarantined=0)
    assert PLAN in content
    assert REJECTION not in content
    assert row["quarantined_reason"] is None
    assert await rebuild_contexts(store, (mirror_root, projects_root)) == ContextRebuildReport(
        found=1, rebuilt=0, quarantined=0
    )


async def test_rebuild_repairs_existing_gate_sample_windows(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "gate-sample-rebuild-session"
    candidate, entries = old_candidate(session_id)
    await store.record_file_scan("/obsolete/.cc-pushback/gate.jsonl", 1.0, [candidate])
    await store.record_gate_samples(
        [
            GateSample(
                sample_key=sample_key,
                kind=kind,
                dedup_key=str(candidate.dedup_key),
                session_id=str(candidate.session_id),
                anchor_uuid=str(candidate.ref.event_uuid),
                occurred_at=candidate.occurred_at.isoformat(),
                offset_turns=0,
                window_json=candidate.window.to_json(),
                seed=0,
            )
            for sample_key, kind in (
                (f"pos:{candidate.dedup_key}:0", "positive_window"),
                (f"hard:{candidate.dedup_key}", "hard_negative"),
            )
        ]
    )
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", entries)

    assert await rebuild_contexts(store, (projects_root,)) == ContextRebuildReport(
        found=1, rebuilt=1, quarantined=0, gate_repaired=2
    )
    samples = await store.gate_samples()
    assert len(samples) == 2
    assert all(
        PLAN
        in "\n".join(
            message["content"]
            for message in messages(ContextWindow.from_json(str(sample["window_json"])).before)
        )
        for sample in samples
    )


async def test_rebuild_quarantines_unresolvable_context_and_excludes_it_from_datasets(
    store: FeedbackStore, projects_root: Path, tmp_path: Path
) -> None:
    candidate, _ = old_candidate("missing-session")
    await store.record_file_scan("/obsolete/.cc-pushback/missing.jsonl", 1.0, [candidate])
    await store.record_verdict(
        candidate.dedup_key,
        Verdict.model_validate(
            {
                "category": "wrong_approach",
                "what_claude_did": "submitted a plan",
                "confidence": 0.99,
                "rationale": "r",
            }
        ),
        role=JUDGE,
        prompt_version=PROMPT_VERSION,
        model="sonnet",
        fidelity="full",
    )

    report = await rebuild_contexts(store, (tmp_path / "missing-mirror", projects_root))
    row = await (
        await store.store.conn.execute(
            "SELECT quarantined_reason FROM feedback_events WHERE dedup_key = ?", (candidate.dedup_key,)
        )
    ).fetchone()
    assert report == ContextRebuildReport(found=1, rebuilt=0, quarantined=1)
    assert row is not None and row["quarantined_reason"] == "transcript_not_found"
    assert await load_traces(store) == []
    assert await store.unrefined(prompt_version=1, model="sonnet") == []
    assert await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION) == []
    assert await store.judged(role=JUDGE, prompt_version=PROMPT_VERSION) == []
    assert await store.candidates() == []
    assert await store.triage_stats(prompt_version=PROMPT_VERSION) == TriageStats(
        total=0, judged=0, accepted=0, by_category={}
    )
    assert await rebuild_contexts(store, (tmp_path / "missing-mirror", projects_root)) == ContextRebuildReport(
        found=1, rebuilt=0, quarantined=0
    )


async def test_rebuild_retries_a_previous_transcript_failure(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "retry-rebuild-session"
    candidate, entries = old_candidate(session_id)
    await store.record_file_scan("/obsolete/.cc-pushback/retry.jsonl", 1.0, [candidate])

    assert await rebuild_contexts(store, (projects_root,)) == ContextRebuildReport(
        found=1, rebuilt=0, quarantined=1
    )
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", entries)
    assert await rebuild_contexts(store, (projects_root,)) == ContextRebuildReport(
        found=1, rebuilt=1, quarantined=0
    )
    row = await (
        await store.store.conn.execute(
            "SELECT quarantined_reason FROM feedback_events WHERE dedup_key = ?", (candidate.dedup_key,)
        )
    ).fetchone()
    assert row is not None and row["quarantined_reason"] is None


async def test_rebuild_preserves_plan_reentry_clamped_boundary(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "plan-reentry-rebuild-session"
    entries = plan_reentry_entries(session_id)
    events = parse(entries)
    [candidate] = [
        candidate for candidate in detect(scrub_events(events)) if candidate.payload == {"detector": "plan_reentry"}
    ]
    buggy = replace(
        candidate,
        window=capture_anchored_window(SessionActivity.from_events(SessionId(session_id), events), candidate.ref),
    )
    assert any(EARLIER in ref.preview for ref in buggy.window.before)
    await store.record_file_scan("/obsolete/.cc-pushback/reentry.jsonl", 1.0, [buggy])
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", entries)

    assert await rebuild_contexts(store, (projects_root,)) == ContextRebuildReport(
        found=1, rebuilt=1, quarantined=0
    )
    row = await (
        await store.store.conn.execute(
            "SELECT context_json FROM feedback_events WHERE dedup_key = ?", (candidate.dedup_key,)
        )
    ).fetchone()
    assert row is not None
    before = ContextWindow.from_json(str(row["context_json"])).before
    assert len(before) == 1
    assert "continue the current pass" in before[0].preview
    assert EARLIER not in "\n".join(ref.preview for ref in before)


async def test_rebuild_selects_maximal_content_copy_over_newer_compacted(
    store: FeedbackStore, tmp_path: Path
) -> None:
    session_id = "copy-selection-session"
    complete_entries = plan_rejection_entries(session_id)
    compacted_entries = compacted_plan_entries(session_id)
    compacted_candidate = detected(compacted_entries, f"{session_id}-anchor")
    assert PLAN not in "\n".join(ref.preview for ref in compacted_candidate.window.before)
    await store.record_file_scan("/obsolete/.cc-pushback/copies.jsonl", 1.0, [compacted_candidate])
    complete = write_transcript(tmp_path / "complete" / f"{session_id}.jsonl", complete_entries)
    compacted = write_transcript(tmp_path / "compacted" / f"{session_id}.jsonl", compacted_entries)
    os.utime(complete, (1, 1))
    os.utime(compacted, (2, 2))

    assert await rebuild_contexts(store, (tmp_path / "complete", tmp_path / "compacted")) == ContextRebuildReport(
        found=1, rebuilt=1, quarantined=0
    )
    row = await (
        await store.store.conn.execute(
            "SELECT context_json FROM feedback_events WHERE dedup_key = ?", (compacted_candidate.dedup_key,)
        )
    ).fetchone()
    assert row is not None
    content = "\n".join(
        message["content"] for message in messages(ContextWindow.from_json(str(row["context_json"])).before)
    )
    assert PLAN in content
    assert UNRELATED not in content


async def test_rebuild_heals_a_stale_gate_sample_when_feedback_is_already_current(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "gate-heal-session"
    entries = plan_rejection_entries(session_id)
    candidate = detected(entries, f"{session_id}-anchor")
    await store.record_file_scan("/obsolete/.cc-pushback/heal.jsonl", 1.0, [candidate])
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", entries)
    await store.record_gate_samples(
        [
            GateSample(
                sample_key=f"pos:{candidate.dedup_key}:0",
                kind="positive_window",
                dedup_key=str(candidate.dedup_key),
                session_id=str(candidate.session_id),
                anchor_uuid=str(candidate.ref.event_uuid),
                occurred_at=candidate.occurred_at.isoformat(),
                offset_turns=0,
                window_json=replace(candidate.window, before=()).to_json(),
                seed=0,
            )
        ]
    )

    report = await rebuild_contexts(store, (projects_root,))
    assert report.rebuilt == 0
    assert report.gate_repaired == 1
    [sample] = await store.gate_samples()
    assert str(sample["window_json"]) == candidate.window.to_json()


async def test_rebuild_isolates_a_corrupt_copy_and_reports_it(
    store: FeedbackStore, tmp_path: Path
) -> None:
    session_id = "corrupt-copy-session"
    candidate, entries = old_candidate(session_id)
    await store.record_file_scan("/obsolete/.cc-pushback/corrupt.jsonl", 1.0, [candidate])
    write_transcript(tmp_path / "valid" / f"{session_id}.jsonl", entries)
    corrupt = tmp_path / "corrupt" / f"{session_id}.jsonl"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text(
        json.dumps({"type": "user", "uuid": "x", "sessionId": session_id, "timestamp": "2026-06-01T12:00:00+00:00"})
        + "\n"
    )

    report = await rebuild_contexts(store, (tmp_path / "valid", tmp_path / "corrupt"))
    assert report.rebuilt == 1
    assert len(report.parse_failures) == 1
    assert report.parse_failures[0].path == corrupt.resolve()
    assert "message" in report.parse_failures[0].error
    row = await (
        await store.store.conn.execute(
            "SELECT quarantined_reason FROM feedback_events WHERE dedup_key = ?", (candidate.dedup_key,)
        )
    ).fetchone()
    assert row is not None and row["quarantined_reason"] is None
