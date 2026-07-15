from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
from cc_transcript import keep
from cc_transcript.ids import EventUuid, SessionId
from cc_transcript.mining import CandidateSignal, Confidence, MiningSignal, SourceKind, mine

from cc_steer.detectors import (
    STEERING_MINING_SPEC,
    detect,
    interrupt_rejections,
    parts,
    payload_of,
    plan_reviews,
    survives,
    transcript_messages,
)
from cc_steer.spec import STEERING_SPEC
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

QUESTION_EVIDENCE: dict[str, Any] = {
    "question": "Which auth scheme should the API use?",
    "header": "Auth",
    "multi_select": False,
    "option_pick": True,
    "picked_labels": ["JWT"],
    "recommended_pick": False,
}


def question_signal(evidence: dict[str, Any]) -> MiningSignal:
    return MiningSignal(
        kind=SourceKind("question_answer"),
        detector="ask_user_question",
        session_id=SessionId("sess-1"),
        event_index=0,
        event_uuid=EventUuid("uuid-q1"),
        occurred_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        text=evidence.get("notes", "JWT"),
        cc_version=None,
        trigger_index=None,
        signal=CandidateSignal(confidence=Confidence(0.9)),
        evidence=evidence,
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
    assert window.fidelity == "full"
    assert window.trigger is not None
    assert "ls" in "".join(ref.preview for ref in window.before)
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
def test_ask_user_question_denial_is_not_steering() -> None:
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
    assert all(c.payload["provenance"] == "typed" for c in candidates if c.payload)


@pytest.mark.unit
def test_surfaced_workflow_result_yields_review_comments_past_the_gate() -> None:
    finding = {"file": "src/a.py", "line": "24-51", "description": "guard against None", "suggested_fix": "return"}
    payload = json.dumps({"findings": [finding]})
    events = parse(
        [
            assistant_text("ran the verifier"),
            assistant_tool_use("w1", "Bash", {"command": "verify.sh"}),
            tool_result("w1", payload),
        ]
    )
    [candidate] = [c for c in detect(events) if c.source_kind == "review_comment"]
    assert candidate.text == "guard against None return"
    assert candidate.payload == {
        "format": "workflow-finding",
        "file": "src/a.py",
        "line_start": 24,
        "line_end": 51,
        "provenance": "surfaced",
    }


@pytest.mark.unit
def test_surfaced_carrier_with_empty_text_survives_the_spec() -> None:
    payload = json.dumps({"bugs": [{"location": "x.py", "line": 9, "problem": "off by one"}]})
    events = parse(
        [
            assistant_tool_use("w2", "Bash", {"command": "audit.sh"}),
            tool_result("w2", payload),
        ]
    )
    [carrier] = [e for e in events if type(e).__name__ == "UserEvent"]
    assert keep(carrier, STEERING_SPEC) is False  # the carrier alone would be dropped
    [candidate] = [c for c in detect(events) if c.source_kind == "review_comment"]
    assert candidate.text == "off by one"
    assert candidate.payload and candidate.payload["provenance"] == "surfaced"


@pytest.mark.unit
def test_typed_inline_cites_still_captured() -> None:
    body = "In src/foo.py:L10-12: this is wrong"
    [candidate] = [c for c in detect(parse([user_text(body)])) if c.source_kind == "review_comment"]
    assert candidate.text == "this is wrong"
    assert candidate.payload and candidate.payload["provenance"] == "typed"


@pytest.mark.unit
def test_subagent_review_output_never_surfaces_as_claude() -> None:
    payload = json.dumps({"findings": [{"file": "z.py", "line": 1, "comment": "rename"}]})
    events = parse(
        [
            assistant_tool_use("s1", "Agent", {"prompt": "review the diff"}),
            tool_result("s1", payload),
        ]
    )
    candidates = detect(events)
    assert all(
        (c.payload or {}).get("provenance") != "claude" for c in candidates
    )
    assert [c for c in candidates if c.source_kind == "review_comment"] == []


@pytest.mark.unit
def test_claude_review_provenance_crashes_the_survival_gate() -> None:
    payload = json.dumps({"findings": [{"file": "z.py", "line": 1, "comment": "rename"}]})
    events = parse(
        [
            assistant_tool_use("s2", "Agent", {"prompt": "review the diff"}),
            tool_result("s2", payload),
        ]
    )
    spec = replace(STEERING_MINING_SPEC, review=replace(STEERING_MINING_SPEC.review, surfaces=frozenset({"claude"})))
    [sig] = [s for s in mine(events, spec) if s.detector == "review_comment"]
    assert sig.evidence["provenance"] == "claude"
    with pytest.raises(AssertionError, match="claude"):
        survives(events, sig)


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
def test_steering_spec_keeps_interrupt_marker() -> None:
    [event] = parse([user_text("[Request interrupted by user] run the tests instead, not the build")])
    assert keep(event, STEERING_SPEC) is True


@pytest.mark.unit
def test_steering_spec_drops_structural_noise_and_acks() -> None:
    [noise] = parse([user_text("<system-reminder>be good</system-reminder>")])
    [ack] = parse([user_text("ok")])
    assert keep(noise, STEERING_SPEC) is False
    assert keep(ack, STEERING_SPEC) is False


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


@pytest.mark.unit
def test_ask_user_question_parts_key_on_question_and_answer() -> None:
    assert parts(question_signal(QUESTION_EVIDENCE)) == (
        "sess-1",
        "question_answer",
        "Which auth scheme should the API use?",
        "JWT",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "extras",
    [
        pytest.param({}, id="required-keys-only"),
        pytest.param({"preview": "def auth(): ...", "notes": "sessions need sticky LB"}, id="with-preview-and-notes"),
    ],
)
def test_ask_user_question_payload_is_the_full_evidence(extras: dict[str, Any]) -> None:
    evidence = QUESTION_EVIDENCE | extras
    assert payload_of(question_signal(evidence)) == evidence


@pytest.mark.unit
def test_ask_user_question_survives_unconditionally() -> None:
    assert survives([], question_signal(QUESTION_EVIDENCE)) is True
