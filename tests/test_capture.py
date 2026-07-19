from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from cc_transcript.activity import SessionActivity
from cc_transcript.ids import SessionId

from cc_steer.capture import capture_anchored_window, capture_window
from cc_steer.detectors import detect, plan_reviews
from cc_steer.rendering import context_turns, messages
from tests.builders import (
    assistant_text,
    assistant_tool_use,
    denial_result,
    parse,
    tool_result,
    user_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cc_transcript.ids import EventRef
    from cc_transcript.mining import FeedbackCandidate

PLAN = "SENTINEL_PLAN step 1 add the module; step 2 wire it in"
REJECTION = "SENTINEL_REJECT I still dont see the root cause of this"
POST_ANCHOR = "SENTINEL_POSTANCHOR addressing your feedback now"
REVIEWED_PROSE = "SENTINEL_PROSE I implemented the streaming parser as asked"
FINDING = "SENTINEL_FINDING guard against None"


def flatten(candidate: FeedbackCandidate) -> str:
    return "\n".join(message["content"] for message in messages(context_turns(candidate.window)))


def plan_rejection_case() -> tuple[FeedbackCandidate, list[str], list[str], EventRef]:
    events = parse(
        [
            user_text("session bootstrap", isMeta=True),
            user_text("implement the feature in the repo"),
            assistant_text("Plan written. Let me present it for approval."),
            assistant_tool_use("p1", "ExitPlanMode", {"plan": PLAN}),
            denial_result("p1", said=REJECTION),
            assistant_text(POST_ANCHOR),
        ]
    )
    [candidate] = plan_reviews(events)
    assert candidate.text == REJECTION
    assert candidate.ref.event_uuid == events[4].meta.uuid
    return candidate, [PLAN, "Plan written", "ExitPlanMode"], [REJECTION, POST_ANCHOR], candidate.ref


def review_comment_case() -> tuple[FeedbackCandidate, list[str], list[str], EventRef]:
    finding = {"file": "src/a.py", "line": "24-51", "description": FINDING, "suggested_fix": "return"}
    events = parse(
        [
            user_text("session bootstrap", isMeta=True),
            user_text("review the diff and verify it"),
            assistant_text(REVIEWED_PROSE),
            assistant_tool_use("w1", "Bash", {"command": "verify.sh"}),
            tool_result("w1", json.dumps({"findings": [finding]})),
            assistant_text(POST_ANCHOR),
        ]
    )
    [candidate] = [c for c in detect(events) if c.source_kind == "review_comment"]
    assert FINDING in candidate.text
    assert candidate.ref.event_uuid == events[4].meta.uuid
    return candidate, [REVIEWED_PROSE, "verify.sh"], [FINDING, POST_ANCHOR], candidate.ref


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(plan_rejection_case, id="plan_rejection"),
        pytest.param(review_comment_case, id="review_comment"),
    ],
)
def test_tool_result_anchor_context_holds_prefix_and_excludes_the_correction(
    case: Callable[[], tuple[FeedbackCandidate, list[str], list[str], EventRef]],
) -> None:
    candidate, present, absent, anchor_ref = case()
    context = flatten(candidate)
    assert context.strip(), "the fix must not leave an empty-prompt positive"
    for marker in present:
        assert marker in context
    for marker in absent:
        assert marker not in context
    assert candidate.window.trigger is not None
    assert candidate.window.trigger.role == "user"
    assert all(anchor_ref not in turn.refs for turn in candidate.window.before)
    assert anchor_ref in candidate.window.trigger.refs
    assert candidate.text not in context


@pytest.mark.unit
def test_typed_steer_anchor_is_not_split() -> None:
    events = parse(
        [
            assistant_text("done — scaffolded it"),
            user_text("no, use JWT, sessions are wrong here"),
        ]
    )
    activity = SessionActivity.from_events(SessionId("sess-1"), events)
    [candidate] = [c for c in detect(events) if c.source_kind == "transcript_message"]
    plain = capture_window(activity, candidate.ref, before=6)
    assert capture_anchored_window(activity, candidate.ref, before=6) == plain
