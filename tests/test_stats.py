from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from cc_transcript.mining import DedupKey

from cc_steer.detectors import detect
from cc_steer.stats import collect_stats
from cc_steer.store import FeedbackStore
from cc_steer.triage import JUDGE, PROMPT_VERSION, Verdict
from tests.builders import assistant_tool_use, denial_result, interrupt_result, parse, user_text

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from cc_steer.models import FeedbackCandidate

pytestmark = pytest.mark.anyio

FILE_A = "/root/projects/-Users-yasyf-Code-projectalpha/session.jsonl"
FILE_B = "/root/projects/-Users-yasyf-Code-projectbeta/session.jsonl"


def verdict(category: str) -> Verdict:
    return Verdict.model_validate(
        {"category": category, "what_claude_did": "ran a tool", "confidence": 0.9, "rationale": "r"}
    )


def make_candidates(*, session: str, tag: str) -> list[FeedbackCandidate]:
    return detect(
        parse(
            [
                assistant_tool_use("t1", "Write", {"file_path": "/a.py", "content": "x = 1"}, sessionId=session),
                denial_result("t1", said=f"don't {tag}", sessionId=session),
                assistant_tool_use("t2", "Bash", {"command": "ls"}, sessionId=session),
                interrupt_result("t2", sessionId=session),
                user_text(f"run the {tag} tests instead, not the build", sessionId=session),
            ]
        )
    )


async def seed(path: Path) -> None:
    async with await FeedbackStore.open(path) as store:
        await store.record_file_scan(FILE_A, 1.0, make_candidates(session="sA", tag="alpha"))
        await store.record_file_scan(FILE_B, 1.0, make_candidates(session="sB", tag="beta"))
        for row in await store.candidates():
            if row["origin_path"] == FILE_A:
                category = "wrong_approach"
            elif row["source_kind"] == "interrupt_rejection":
                category = "unwanted_action"
            else:
                category = "status_update"
            await store.record_verdict(
                DedupKey(str(row["dedup_key"])),
                verdict(category),
                role=JUDGE,
                prompt_version=PROMPT_VERSION,
                model="sonnet",
                fidelity="full",
            )


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Path]:
    path = tmp_path / "feedback.db"
    await seed(path)
    yield path


async def test_collect_stats_reports_ingestion_and_triage(db: Path) -> None:
    report = await collect_stats(db)
    assert (report.ingestion.total, report.ingestion.files) == (6, 2)
    assert report.ingestion.by_source == {"interrupt_rejection": 4, "transcript_message": 2}
    assert (report.triage.total, report.triage.judged, report.triage.accepted) == (6, 6, 5)
    assert report.triage.by_category == {"wrong_approach": 3, "unwanted_action": 2, "status_update": 1}
    assert report.prompt_version == PROMPT_VERSION
    assert report.by is None
    assert report.groups == {}


async def test_render_matches_the_legacy_default_output(db: Path) -> None:
    report = await collect_stats(db)
    assert report.render() == (
        "total: 6  files: 2\n"
        "  interrupt_rejection: 4\n"
        "  transcript_message: 2\n"
        f"triaged: 6/6 (v{PROMPT_VERSION})  accepted: 5 (83%)\n"
        "  wrong_approach: 3\n"
        "  unwanted_action: 2\n"
        "  status_update: 1"
    )


async def test_group_by_project_counts_accepted_events_per_project(db: Path) -> None:
    report = await collect_stats(db, by="project")
    assert report.by == "project"
    assert report.groups == {"projectalpha": 3, "projectbeta": 2}
    assert report.render().endswith("by project:\n  projectalpha: 3\n  projectbeta: 2")


async def test_group_by_category_counts_accepted_events_per_category(db: Path) -> None:
    report = await collect_stats(db, by="category")
    assert report.by == "category"
    assert report.groups == {"wrong_approach": 3, "unwanted_action": 2}
    assert report.render().endswith("by category:\n  wrong_approach: 3\n  unwanted_action: 2")


async def test_to_dict_is_json_serializable_and_carries_the_grouping(db: Path) -> None:
    report = await collect_stats(db, by="project")
    assert json.loads(json.dumps(report.to_dict())) == {
        "ingestion": {"total": 6, "files": 2, "by_source": {"interrupt_rejection": 4, "transcript_message": 2}},
        "triage": {
            "total": 6,
            "judged": 6,
            "accepted": 5,
            "by_category": {"wrong_approach": 3, "unwanted_action": 2, "status_update": 1},
        },
        "prompt_version": PROMPT_VERSION,
        "by": "project",
        "groups": {"projectalpha": 3, "projectbeta": 2},
    }


async def test_grouping_pins_to_prompt_version_and_matches_accepted(db: Path) -> None:
    async with await FeedbackStore.open(db) as store:
        await store.execute(
            "INSERT INTO feedback_events "
            "(dedup_key, source_kind, session_id, event_uuid, occurred_at, text, "
            "payload_json, context_json, cc_version, ingested_at, origin_path) "
            "VALUES (?, 'transcript_message', 'sC', 'evt-stale', ?, 'stale steer', '{}', '{}', '1.0', ?, ?)",
            ("k-stale", "2026-06-01T00:00:00", "2026-06-01T00:00:00", FILE_A),
        )
        await store.record_verdict(
            DedupKey("k-stale"),
            verdict("wrong_approach"),
            role=JUDGE,
            prompt_version=PROMPT_VERSION - 1,
            model="sonnet",
            fidelity="full",
        )
    by_project = await collect_stats(db, by="project")
    by_category = await collect_stats(db, by="category")
    assert sum(by_project.groups.values()) == by_project.triage.accepted
    assert sum(by_category.groups.values()) == by_category.triage.accepted
    # The steer judged only at PROMPT_VERSION-1 never leaks into the accepted-at-PROMPT_VERSION grouping.
    assert by_project.groups == {"projectalpha": 3, "projectbeta": 2}
    assert by_category.groups == {"wrong_approach": 3, "unwanted_action": 2}


async def test_empty_corpus_renders_without_a_share_or_breakdown(tmp_path: Path) -> None:
    report = await collect_stats(tmp_path / "empty.db")
    assert (report.ingestion.total, report.triage.judged) == (0, 0)
    assert report.render() == f"total: 0  files: 0\ntriaged: 0/0 (v{PROMPT_VERSION})  accepted: 0"
