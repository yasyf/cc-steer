from __future__ import annotations

from pathlib import Path

from cc_pushback.models import FeedbackCandidate
from cc_pushback.sources.plan_reviews import PlanReviews
from tests.builders import (
    assistant_text,
    assistant_tool_use,
    denial_result,
    mode_entry,
    parse,
    user_text,
)


def candidates(entries: list[dict[str, object]]) -> list[FeedbackCandidate]:
    return list(PlanReviews().candidates_for_file(Path("/t.jsonl"), parse(entries)))


def test_exit_plan_rejection_extracts_embedded_text() -> None:
    cands = candidates(
        [
            assistant_tool_use("plan1", "ExitPlanMode", {"plan": "do the thing"}),
            denial_result("plan1", "no, split it into two steps first"),
        ]
    )

    assert len(cands) == 1
    assert cands[0].text == "no, split it into two steps first"
    assert cands[0].payload == {"detector": "exit_plan_rejection"}


def test_non_exit_plan_denial_is_not_a_plan_review() -> None:
    cands = candidates(
        [
            assistant_tool_use("e1", "Edit", {"file_path": "/a.py"}),
            denial_result("e1", "use a comprehension"),
        ]
    )

    assert cands == []


def test_plan_reentry_fires_after_edit_cycle() -> None:
    cands = candidates(
        [
            assistant_tool_use("e1", "Edit", {"file_path": "/a.py"}),
            user_text("intermediate"),
            mode_entry("plan"),
            user_text("rethink the data model before editing"),
        ]
    )

    assert len(cands) == 1
    assert cands[0].text == "rethink the data model before editing"
    assert cands[0].payload == {"detector": "plan_reentry"}


def test_plan_reentry_does_not_fire_without_edit_cycle() -> None:
    cands = candidates(
        [
            assistant_text("just discussing, no edits"),
            mode_entry("plan"),
            user_text("rethink the data model"),
        ]
    )

    assert cands == []
