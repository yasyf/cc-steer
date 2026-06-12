from __future__ import annotations

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


@pytest.mark.unit
def test_exit_plan_rejection_extracts_embedded_user_text() -> None:
    events = parse(
        [
            assistant_tool_use("t1", "ExitPlanMode", {"plan": "do x"}),
            denial_result("t1", said="actually do Y instead"),
        ]
    )
    [candidate] = plan_reviews(events)
    assert candidate.source_kind == "plan_review"
    assert candidate.text == "actually do Y instead"
    assert candidate.payload == {"detector": "exit_plan_rejection"}


@pytest.mark.unit
def test_plan_reentry_fires_only_after_an_edit() -> None:
    after_edit = parse(
        [
            assistant_tool_use("e1", "Edit", {"file_path": "/a.py", "old_string": "a", "new_string": "b"}),
            mode_entry("plan"),
            user_text("this approach is wrong, use a generator"),
        ]
    )
    [candidate] = plan_reviews(after_edit)
    assert candidate.payload == {"detector": "plan_reentry"}
    assert candidate.text == "this approach is wrong, use a generator"

    without_edit = parse(
        [
            assistant_text("here is my plan"),
            mode_entry("plan"),
            user_text("this approach is wrong, use a generator"),
        ]
    )
    assert plan_reviews(without_edit) == []


@pytest.mark.unit
def test_plan_reentry_window_clamps_before_to_the_edit_turn() -> None:
    events = parse(
        [
            user_text("write the feature"),
            assistant_text("starting"),
            user_text("continue"),
            assistant_tool_use("e1", "Edit", {"file_path": "/a.py", "old_string": "a", "new_string": "b"}),
            mode_entry("plan"),
            user_text("this approach is wrong, use a generator"),
        ]
    )
    [candidate] = plan_reviews(events)
    assert len(candidate.window.before) == 1
    assert "continue" in candidate.window.before[0].preview
    assert "write the feature" not in "".join(ref.preview for ref in candidate.window.before)


@pytest.mark.unit
def test_interrupt_marker_pairs_the_following_correction() -> None:
    events = parse(
        [
            assistant_tool_use("t2", "Bash", {"command": "ls"}),
            interrupt_result("t2"),
            user_text("no, run the tests instead please"),
        ]
    )
    [candidate] = interrupt_rejections(events)
    assert candidate.source_kind == "interrupt_rejection"
    assert candidate.text == "no, run the tests instead please"
    assert candidate.payload == {"detector": "interrupt"}


@pytest.mark.unit
def test_candidate_window_anchors_the_signal_event() -> None:
    events = parse(
        [
            assistant_tool_use("t2", "Bash", {"command": "ls"}),
            interrupt_result("t2"),
            user_text("no, run the tests instead please"),
        ]
    )
    [candidate] = interrupt_rejections(events)
    window = candidate.window
    assert candidate.ref is not None
    assert window.anchor == candidate.ref
    assert (window.fidelity, window.origin) == ("full", "live")
    assert window.trigger is not None
    assert "ls" in window.trigger.preview
    assert any("no, run the tests instead please" in ref.preview for ref in window.after)


@pytest.mark.unit
def test_interrupt_marker_without_correction_drops_the_row() -> None:
    events = parse(
        [
            assistant_tool_use("t2", "Bash", {"command": "ls"}),
            interrupt_result("t2"),
        ]
    )
    assert interrupt_rejections(events) == []


@pytest.mark.unit
def test_permission_denial_pairs_the_following_correction() -> None:
    events = parse(
        [
            assistant_tool_use("t5", "Bash", {"command": "rm -rf build"}),
            denial_result("t5"),
            user_text("no, just clean the cache, don't delete build"),
        ]
    )
    [candidate] = interrupt_rejections(events)
    assert candidate.text == "no, just clean the cache, don't delete build"
    assert candidate.payload == {"tool": "Bash", "file_path": None}


@pytest.mark.unit
def test_reasonless_denial_without_correction_drops_the_row() -> None:
    events = parse([assistant_tool_use("t6", "Agent", {}), denial_result("t6")])
    assert interrupt_rejections(events) == []


@pytest.mark.unit
def test_ask_user_question_denial_is_not_pushback() -> None:
    events = parse(
        [
            assistant_tool_use("t7", "AskUserQuestion", {"questions": []}),
            denial_result("t7", said="The user wants to clarify these questions."),
        ]
    )
    assert interrupt_rejections(events) == []


@pytest.mark.unit
def test_interrupt_marker_ignores_marker_buried_in_tool_output() -> None:
    events = parse(
        [
            assistant_tool_use("t9", "Read", {"file_path": "/regexes.py"}),
            tool_result("t9", '1\tPATTERN = r"\\[Request interrupted by user"  # just source code\n2\tx = 1'),
        ]
    )
    assert interrupt_rejections(events) == []


@pytest.mark.unit
def test_permission_denial_captures_tool_and_path() -> None:
    events = parse(
        [
            assistant_tool_use("t3", "Write", {"file_path": "/a/b.py", "content": "x = 1"}),
            denial_result("t3", said="don't touch that file"),
        ]
    )
    [candidate] = interrupt_rejections(events)
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
    assert interrupt_rejections(events) == []


@pytest.mark.unit
def test_transcript_message_keeps_substance_drops_ack() -> None:
    events = parse(
        [
            assistant_text("here is the diff"),
            user_text("please refactor this to be functional"),
            user_text("ok"),
        ]
    )
    assert [c.text for c in transcript_messages(events)] == ["please refactor this to be functional"]


@pytest.mark.unit
def test_transcript_message_drops_bare_interrupt_marker_keeps_marker_with_correction() -> None:
    events = parse(
        [
            assistant_text("running the build"),
            user_text("[Request interrupted by user for tool use]"),
            user_text("[Request interrupted by user] run the tests instead, not the build"),
        ]
    )
    assert [c.text for c in transcript_messages(events)] == [
        "[Request interrupted by user] run the tests instead, not the build"
    ]


@pytest.mark.unit
def test_transcript_message_requires_a_preceding_assistant_turn() -> None:
    events = parse(
        [
            user_text("set up a new project with passkey auth"),
            assistant_text("done — scaffolded it"),
            user_text("no, use JWT, sessions are wrong here"),
        ]
    )
    assert [c.text for c in transcript_messages(events)] == ["no, use JWT, sessions are wrong here"]


@pytest.mark.unit
def test_review_comment_explodes_one_row_per_inline_cite() -> None:
    body = "Two issues:\nIn src/foo.py:L10-12: this is wrong\nIn src/bar.py:L5: fix this too"
    candidates = [c for c in detect(parse([user_text(body)])) if c.source_kind == "review_comment"]
    assert [c.text for c in candidates] == ["this is wrong", "fix this too"]
    assert [c.payload["file"] for c in candidates if c.payload] == ["src/foo.py", "src/bar.py"]
    assert [c.payload["line_start"] for c in candidates if c.payload] == [10, 5]


@pytest.mark.unit
def test_duplicate_review_entries_share_a_dedup_key() -> None:
    body = "In src/foo.py:L10: use a dataclass here"
    events = parse([user_text(body, uuid="uuid-A"), user_text(body, uuid="uuid-B")])
    cands = [c for c in detect(events) if c.source_kind == "review_comment"]
    assert len(cands) == 2
    assert cands[0].dedup_key == cands[1].dedup_key  # same comment, two entries -> one row on insert


@pytest.mark.unit
def test_repeated_interrupt_markers_collapse_on_the_shared_correction() -> None:
    events = parse(
        [
            assistant_tool_use("t1", "Bash", {"command": "a"}),
            interrupt_result("t1"),
            assistant_tool_use("t2", "Bash", {"command": "b"}),
            interrupt_result("t2"),
            user_text("stop, do it the other way"),
        ]
    )
    cands = interrupt_rejections(events)
    assert len(cands) == 2
    assert len({c.dedup_key for c in cands}) == 1  # both markers pair the same correction
    assert all(c.text == "stop, do it the other way" for c in cands)


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
def test_dedup_keys_derive_from_content_not_entry_identity() -> None:
    def entries() -> list[dict[str, object]]:
        return [
            assistant_tool_use("t1", "ExitPlanMode", {"plan": "do x"}),
            denial_result("t1", said="do Y"),
        ]

    [first] = plan_reviews(parse(entries()))
    [second] = plan_reviews(parse(entries()))  # fresh entries: new uuids, same content
    assert first.ref != second.ref
    assert first.dedup_key == second.dedup_key
