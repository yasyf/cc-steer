from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from cc_transcript.activity import SessionActivity
from cc_transcript.context import ContextWindow
from cc_transcript.ids import SessionId
from cc_transcript.mining import DedupKey

import cc_steer.context_rebuild as context_rebuild
from cc_steer.capture import capture_anchored_window
from cc_steer.context_rebuild import (
    LOCK_FILENAME,
    ContextRebuildReport,
    CopyCandidates,
    DetectorDrift,
    parse_copy,
    rebuild_contexts,
)
from cc_steer.detectors import detect, plan_reviews
from cc_steer.export import load_traces
from cc_steer.negatives import GateSample, truncated
from cc_steer.rendering import messages
from cc_steer.store import FeedbackStore, TriageStats
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
    from cc_transcript.mining import FeedbackCandidate

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


def rebuild_report(
    *,
    found: int = 1,
    rebuilt: int = 0,
    quarantined: int = 0,
    gate_repaired: int = 0,
    family_mismatches: int = 0,
) -> ContextRebuildReport:
    return ContextRebuildReport(
        found=found,
        rebuilt=rebuilt,
        quarantined=quarantined,
        rows_at_start=found,
        rows_at_end=found,
        gate_repaired=gate_repaired,
        family_mismatches=family_mismatches,
    )


def gate_sample(
    candidate: FeedbackCandidate,
    sample_key: str,
    kind: str,
    *,
    window_json: str | None = None,
    offset_turns: int = 0,
) -> GateSample:
    return GateSample(
        sample_key=sample_key,
        kind=kind,
        dedup_key=str(candidate.dedup_key),
        session_id=str(candidate.session_id),
        anchor_uuid=str(candidate.ref.event_uuid),
        occurred_at=candidate.occurred_at.isoformat(),
        offset_turns=offset_turns,
        window_json=window_json or candidate.window.to_json(),
        seed=0,
    )


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
    assert first == rebuild_report(rebuilt=1)
    assert PLAN in content
    assert REJECTION not in content
    assert row["quarantined_reason"] is None
    assert await rebuild_contexts(store, (mirror_root, projects_root)) == rebuild_report()


async def test_rebuild_repairs_existing_gate_sample_windows(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "gate-sample-rebuild-session"
    candidate, entries = old_candidate(session_id)
    current = detected(entries, f"{session_id}-anchor")
    await store.record_file_scan("/obsolete/.cc-pushback/gate.jsonl", 1.0, [candidate])
    await store.record_gate_samples(
        [
            gate_sample(candidate, sample_key, kind)
            for sample_key, kind in (
                (f"pos:{candidate.dedup_key}:0", "positive_window"),
                (f"hard:{candidate.dedup_key}", "hard_negative"),
            )
        ]
    )
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", entries)

    assert await rebuild_contexts(store, (projects_root,)) == rebuild_report(rebuilt=1, gate_repaired=4)
    samples = await store.gate_samples()
    assert {str(sample["sample_key"]) for sample in samples} == {
        f"hard:{candidate.dedup_key}",
        *(f"pos:{candidate.dedup_key}:{offset}" for offset in range(3)),
    }
    by_key = {str(sample["sample_key"]): str(sample["window_json"]) for sample in samples}
    assert by_key[f"hard:{candidate.dedup_key}"] == current.window.to_json()
    assert all(
        by_key[f"pos:{candidate.dedup_key}:{offset}"] == truncated(current.window, offset).to_json()
        for offset in range(3)
    )


async def test_rebuild_shortens_an_existing_positive_gate_family(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "gate-family-shorten-session"
    entries = plan_rejection_entries(session_id)
    candidate = detected(entries, f"{session_id}-anchor")
    shortened_entries = entries[1:]
    shortened = detected(shortened_entries, f"{session_id}-anchor")
    assert candidate.dedup_key == shortened.dedup_key
    assert sum(truncated(shortened.window, offset) is not None for offset in range(6)) == 2
    await store.record_file_scan("/obsolete/.cc-pushback/shorten.jsonl", 1.0, [candidate])
    await store.record_gate_samples(
        [
            gate_sample(
                candidate,
                f"pos:{candidate.dedup_key}:{offset}",
                "positive_window",
                window_json=rewound.to_json(),
                offset_turns=-offset,
            )
            for offset in range(3)
            if (rewound := truncated(candidate.window, offset)) is not None
        ]
    )
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", shortened_entries)

    assert await rebuild_contexts(store, (projects_root,)) == rebuild_report(rebuilt=1, gate_repaired=3)
    samples = {str(sample["sample_key"]): sample for sample in await store.gate_samples()}
    assert set(samples) == {f"pos:{candidate.dedup_key}:0", f"pos:{candidate.dedup_key}:1"}
    assert all(
        samples[f"pos:{candidate.dedup_key}:{offset}"]["window_json"]
        == truncated(shortened.window, offset).to_json()
        for offset in range(2)
    )


async def test_rebuild_lengthens_an_existing_positive_gate_family(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "gate-family-lengthen-session"
    candidate, entries = old_candidate(session_id)
    await store.record_file_scan("/obsolete/.cc-pushback/lengthen.jsonl", 1.0, [candidate])
    await store.record_gate_samples(
        [gate_sample(candidate, f"pos:{candidate.dedup_key}:0", "positive_window")]
    )
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", entries)

    assert await rebuild_contexts(store, (projects_root,)) == rebuild_report(rebuilt=1, gate_repaired=3)
    assert {str(sample["sample_key"]) for sample in await store.gate_samples()} == {
        f"pos:{candidate.dedup_key}:0",
        f"pos:{candidate.dedup_key}:1",
        f"pos:{candidate.dedup_key}:2",
    }


async def test_rebuild_keeps_a_hard_only_parent_hard_only_and_reports_its_verdict_mismatch(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "gate-family-hard-session"
    candidate, entries = old_candidate(session_id)
    await store.record_file_scan("/obsolete/.cc-pushback/hard.jsonl", 1.0, [candidate])
    await store.record_gate_samples(
        [gate_sample(candidate, f"hard:{candidate.dedup_key}", "hard_negative")]
    )
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
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", entries)

    assert await rebuild_contexts(store, (projects_root,)) == rebuild_report(
        rebuilt=1,
        gate_repaired=1,
        family_mismatches=1,
    )
    [sample] = await store.gate_samples()
    assert sample["sample_key"] == f"hard:{candidate.dedup_key}"
    assert PLAN in str(sample["window_json"])


async def test_rebuild_leaves_a_quarantined_parents_gate_samples_untouched(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "gate-family-quarantined-session"
    candidate, entries = old_candidate(session_id)
    current = detected(entries, f"{session_id}-anchor")
    await store.record_file_scan("/obsolete/.cc-pushback/quarantined.jsonl", 1.0, [candidate])
    await store.record_gate_samples(
        [gate_sample(candidate, f"pos:{candidate.dedup_key}:0", "positive_window")]
    )
    await store.store.conn.execute(
        "UPDATE feedback_events SET context_json = ?, quarantined_reason = 'anchor_not_found' WHERE dedup_key = ?",
        (current.window.to_json(), candidate.dedup_key),
    )
    [before] = await store.gate_samples()

    assert await rebuild_contexts(store, (projects_root,)) == rebuild_report(quarantined=1)
    [after] = await store.gate_samples()
    assert after == before


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
    assert report == rebuild_report(quarantined=1)
    assert row is not None and row["quarantined_reason"] == "transcript_not_found"
    assert await load_traces(store) == []
    assert await store.unrefined(prompt_version=1, model="sonnet") == []
    assert await store.unjudged(role=JUDGE, prompt_version=PROMPT_VERSION) == []
    assert await store.judged(role=JUDGE, prompt_version=PROMPT_VERSION) == []
    assert await store.candidates() == []
    assert await store.triage_stats(prompt_version=PROMPT_VERSION) == TriageStats(
        total=0, judged=0, accepted=0, by_category={}
    )
    assert await rebuild_contexts(store, (tmp_path / "missing-mirror", projects_root)) == rebuild_report()


async def test_rebuild_retries_a_previous_transcript_failure(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "retry-rebuild-session"
    candidate, entries = old_candidate(session_id)
    await store.record_file_scan("/obsolete/.cc-pushback/retry.jsonl", 1.0, [candidate])

    assert await rebuild_contexts(store, (projects_root,)) == rebuild_report(quarantined=1)
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", entries)
    assert await rebuild_contexts(store, (projects_root,)) == rebuild_report(rebuilt=1)
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

    assert await rebuild_contexts(store, (projects_root,)) == rebuild_report(rebuilt=1)
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

    assert await rebuild_contexts(store, (tmp_path / "complete", tmp_path / "compacted")) == rebuild_report(rebuilt=1)
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


async def test_parse_copy_keeps_the_first_candidate_for_a_colliding_dedup_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "dedup-collision-session"
    entries = plan_rejection_entries(session_id)
    first = detected(entries, f"{session_id}-anchor")
    second = replace(first, window=replace(first.window, before=()))
    monkeypatch.setattr(context_rebuild, "detect", lambda _: [first, second])
    path = write_transcript(tmp_path / f"{session_id}.jsonl", entries)

    parsed = await parse_copy(SessionId(session_id), path, asyncio.Semaphore(1))
    assert isinstance(parsed, CopyCandidates)
    assert parsed.by_dedup[str(first.dedup_key)].window == first.window


async def test_rebuild_reports_detector_drift_without_changing_the_stored_row(
    store: FeedbackStore, projects_root: Path
) -> None:
    session_id = "detector-drift-session"
    entries = plan_rejection_entries(session_id)
    current = detected(entries, f"{session_id}-anchor")
    stored = replace(current, dedup_key=DedupKey("legacy-detector-dedup-key"))
    await store.record_file_scan("/obsolete/.cc-pushback/drift.jsonl", 1.0, [stored])
    write_transcript(projects_root / "project" / f"{session_id}.jsonl", entries)
    before = dict(
        await (
            await store.store.conn.execute(
                "SELECT * FROM feedback_events WHERE dedup_key = ?", (stored.dedup_key,)
            )
        ).fetchone()
    )

    report = await rebuild_contexts(store, (projects_root,))
    after = dict(
        await (
            await store.store.conn.execute(
                "SELECT * FROM feedback_events WHERE dedup_key = ?", (stored.dedup_key,)
            )
        ).fetchone()
    )
    assert report == replace(
        rebuild_report(),
        drifted=(DetectorDrift(stored.dedup_key, current.dedup_key),),
    )
    assert after == before


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
            gate_sample(
                candidate,
                f"pos:{candidate.dedup_key}:0",
                "positive_window",
                window_json=replace(candidate.window, before=()).to_json(),
            )
        ]
    )

    report = await rebuild_contexts(store, (projects_root,))
    assert report.rebuilt == 0
    assert report.gate_repaired == 3
    samples = {str(sample["sample_key"]): sample for sample in await store.gate_samples()}
    assert str(samples[f"pos:{candidate.dedup_key}:0"]["window_json"]) == candidate.window.to_json()


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


async def test_rebuild_isolates_copy_io_failure_across_the_full_copy_lifecycle(
    store: FeedbackStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "io-failure-copy-session"
    candidate, entries = old_candidate(session_id)
    await store.record_file_scan("/obsolete/.cc-pushback/io.jsonl", 1.0, [candidate])
    valid = write_transcript(tmp_path / "valid" / f"{session_id}.jsonl", entries)
    inaccessible = write_transcript(tmp_path / "inaccessible" / f"{session_id}.jsonl", entries)
    transcript_mtime = context_rebuild.TranscriptDiscovery.transcript_mtime

    async def permission_error(path: Path) -> float:
        if path == inaccessible.resolve():
            raise PermissionError("fixture denied")
        return await transcript_mtime(path)

    monkeypatch.setattr(context_rebuild.TranscriptDiscovery, "transcript_mtime", permission_error)

    report = await rebuild_contexts(store, (valid.parent, inaccessible.parent))
    assert report.rebuilt == 1
    assert report.quarantined == 0
    assert report.parse_failures == (
        context_rebuild.CopyParseFailure(
            SessionId(session_id),
            inaccessible.resolve(),
            "PermissionError: fixture denied",
        ),
    )


async def test_rebuild_dry_run_reports_real_changes_without_writing(
    store: FeedbackStore, projects_root: Path
) -> None:
    rebuild_candidate, entries = old_candidate("dry-run-rebuild-session")
    missing_candidate, _ = old_candidate("dry-run-missing-session")
    await store.record_file_scan("/obsolete/.cc-pushback/dry-rebuild.jsonl", 1.0, [rebuild_candidate])
    await store.record_file_scan("/obsolete/.cc-pushback/dry-missing.jsonl", 1.0, [missing_candidate])
    await store.record_gate_samples(
        [
            gate_sample(
                rebuild_candidate,
                f"pos:{rebuild_candidate.dedup_key}:0",
                "positive_window",
            )
        ]
    )
    write_transcript(
        projects_root / "project" / "dry-run-rebuild-session.jsonl",
        entries,
    )

    async def snapshot() -> dict[str, list[dict[str, object]]]:
        return {
            table: [dict(row) async for row in await store.store.conn.execute(f"SELECT * FROM {table} ORDER BY id")]
            for table in ("feedback_events", "gate_sample", "triage")
        }

    before = await snapshot()
    dry_run = await rebuild_contexts(store, (projects_root,), dry_run=True)
    assert dry_run == ContextRebuildReport(
        found=2,
        rebuilt=1,
        quarantined=1,
        rows_at_start=2,
        rows_at_end=2,
        gate_repaired=3,
    )
    assert await snapshot() == before
    assert await rebuild_contexts(store, (projects_root,)) == dry_run


async def test_rebuild_recovers_a_stale_lock_from_a_dead_process(
    store: FeedbackStore, projects_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dead_pid = 2**30
    with pytest.raises(ProcessLookupError):
        os.kill(dead_pid, 0)
    lock = tmp_path / LOCK_FILENAME
    lock.write_text(str(dead_pid))

    assert await rebuild_contexts(store, (projects_root,)) == rebuild_report(found=0)
    assert not lock.exists()
    assert f"dead pid {dead_pid}" in caplog.text


async def test_rebuild_refuses_a_lock_held_by_a_live_process(
    store: FeedbackStore, projects_root: Path, tmp_path: Path
) -> None:
    lock = tmp_path / LOCK_FILENAME
    lock.write_text(str(os.getpid()))

    with pytest.raises(RuntimeError, match=rf"pid {os.getpid()} at {lock}"):
        await rebuild_contexts(store, (projects_root,))


async def test_rebuild_skips_locking_for_an_in_memory_database(projects_root: Path) -> None:
    async with await FeedbackStore.open(Path(":memory:")) as memory_store:
        assert await rebuild_contexts(memory_store, (projects_root,)) == rebuild_report(found=0)
