from __future__ import annotations

from pathlib import Path

import pytest
from cc_transcript import keep

from cc_pushback.detectors import (
    detect,
    interrupt_rejections,
    plan_reviews,
    transcript_messages,
)
from cc_pushback.spec import PUSHBACK_SPEC
from tests.builders import (
    assistant_text,
    assistant_tool_use,
    denial_result,
    interrupt_result,
    mode_entry,
    parse,
    tool_result,
    user_text,
)

FILE = Path("session.jsonl")


@pytest.mark.unit
def test_exit_plan_rejection_extracts_embedded_user_text() -> None:
    events = parse(
        [
            assistant_tool_use("t1", "ExitPlanMode", {"plan": "do x"}),
            denial_result("t1", said="actually do Y instead"),
        ]
    )
    [candidate] = list(plan_reviews(FILE, events))
    assert candidate.source_kind == "plan_review"
    assert candidate.text == "actually do Y instead"
    assert candidate.payload == {"detector": "exit_plan_rejection"}


@pytest.mark.unit
def test_plan_reentry_fires_only_after_an_edit() -> None:
    after_edit = parse(
        [
            assistant_tool_use("e1", "Edit", {"file_path": "/a.py"}),
            mode_entry("plan"),
            user_text("this approach is wrong, use a generator"),
        ]
    )
    [candidate] = list(plan_reviews(FILE, after_edit))
    assert candidate.payload == {"detector": "plan_reentry"}
    assert candidate.text == "this approach is wrong, use a generator"

    without_edit = parse(
        [
            assistant_text("here is my plan"),
            mode_entry("plan"),
            user_text("this approach is wrong, use a generator"),
        ]
    )
    assert list(plan_reviews(FILE, without_edit)) == []


@pytest.mark.unit
def test_interrupt_marker_pairs_the_following_correction() -> None:
    events = parse(
        [
            assistant_tool_use("t2", "Bash", {"command": "ls"}),
            interrupt_result("t2"),
            user_text("no, run the tests instead please"),
        ]
    )
    [candidate] = list(interrupt_rejections(FILE, events))
    assert candidate.source_kind == "interrupt_rejection"
    assert candidate.text == "no, run the tests instead please"
    assert candidate.payload == {"detector": "interrupt"}


@pytest.mark.unit
def test_interrupt_marker_without_correction_uses_full_marker() -> None:
    events = parse(
        [
            assistant_tool_use("t2", "Bash", {"command": "ls"}),
            interrupt_result("t2"),
        ]
    )
    [candidate] = list(interrupt_rejections(FILE, events))
    assert candidate.text == "[Request interrupted by user]"


@pytest.mark.unit
def test_interrupt_marker_ignores_marker_buried_in_tool_output() -> None:
    events = parse(
        [
            assistant_tool_use("t9", "Read", {"file_path": "/regexes.py"}),
            tool_result("t9", '1\tPATTERN = r"\\[Request interrupted by user"  # just source code\n2\tx = 1'),
        ]
    )
    assert list(interrupt_rejections(FILE, events)) == []


@pytest.mark.unit
def test_permission_denial_captures_tool_and_path() -> None:
    events = parse(
        [
            assistant_tool_use("t3", "Write", {"file_path": "/a/b.py"}),
            denial_result("t3", said="don't touch that file"),
        ]
    )
    [candidate] = list(interrupt_rejections(FILE, events))
    assert candidate.text == "don't touch that file"
    assert candidate.payload == {"tool": "Write", "file_path": "/a/b.py"}


@pytest.mark.unit
def test_exit_plan_denial_is_not_an_interrupt_rejection() -> None:
    events = parse(
        [
            assistant_tool_use("t1", "ExitPlanMode", {"plan": "do x"}),
            denial_result("t1", said="actually do Y instead"),
        ]
    )
    assert list(interrupt_rejections(FILE, events)) == []


@pytest.mark.unit
def test_transcript_message_keeps_substance_drops_ack() -> None:
    events = parse([user_text("please refactor this to be functional"), user_text("ok")])
    assert [c.text for c in transcript_messages(FILE, events)] == ["please refactor this to be functional"]


@pytest.mark.unit
def test_review_comment_explodes_one_row_per_inline_cite() -> None:
    body = "Two issues:\nIn src/foo.py:L10-12: this is wrong\nIn src/bar.py:L5: fix this too"
    candidates = [c for c in detect(FILE, parse([user_text(body)])) if c.source_kind == "review_comment"]
    assert [c.text for c in candidates] == ["this is wrong", "fix this too"]
    assert [c.payload["file"] for c in candidates if c.payload] == ["src/foo.py", "src/bar.py"]
    assert [c.payload["line_start"] for c in candidates if c.payload] == [10, 5]


@pytest.mark.unit
def test_pushback_spec_keeps_interrupt_marker() -> None:
    [event] = parse([user_text("[Request interrupted by user] run the tests instead, not the build")])
    assert keep(event, PUSHBACK_SPEC) is True


@pytest.mark.unit
def test_pushback_spec_drops_structural_noise_and_acks() -> None:
    [noise] = parse([user_text("<system-reminder>be good</system-reminder>")])
    [ack] = parse([user_text("ok")])
    assert keep(noise, PUSHBACK_SPEC) is False
    assert keep(ack, PUSHBACK_SPEC) is False


@pytest.mark.unit
def test_dedup_keys_are_stable_and_path_independent() -> None:
    events = parse(
        [
            assistant_tool_use("t1", "ExitPlanMode", {"plan": "do x"}),
            denial_result("t1", said="do Y"),
        ]
    )
    [from_a] = list(plan_reviews(Path("/home/a/session.jsonl"), events))
    [from_b] = list(plan_reviews(Path("/elsewhere/session.jsonl"), events))
    assert from_a.dedup_key == from_b.dedup_key
