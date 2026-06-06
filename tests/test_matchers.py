from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from cc_pushback.classify import FeedbackEvent
from cc_pushback.matchers import LlmMatcher, matches
from cc_pushback.models import ContextSnapshot
from cc_pushback.patterns import by_name

if TYPE_CHECKING:
    from collections.abc import Mapping

EMPTY = ContextSnapshot(before=(), trigger=None, after=())


def event(text: str, *, payload: Mapping[str, Any] | None = None) -> FeedbackEvent:
    return FeedbackEvent(id=1, source_kind="transcript_message", text=text, context=EMPTY, payload=payload)


REGEX_CASES = [
    ("don't add a fallback, crash instead", "no-defensive-coding", True),
    ("remove the try and let it crash", "no-defensive-coding", True),
    ("we still want backwards-compat for v1", "no-defensive-coding", True),
    ("please add another test case for the parser", "no-defensive-coding", False),
    ("ask me first before touching the schema", "ask-before-assuming", True),
    ("why didn't you ask instead of guessing?", "ask-before-assuming", True),
    ("you should have asked a clarifying question", "ask-before-assuming", True),
    ("the build is green, ship it", "ask-before-assuming", False),
    ("that change is out of scope", "minimal-scope", True),
    ("only change what I asked for", "minimal-scope", True),
    ("don't refactor the whole module", "minimal-scope", True),
    ("add a docstring to the public method", "minimal-scope", False),
    ("match the surrounding code", "match-surrounding-code", True),
    ("follow the existing convention", "match-surrounding-code", True),
    ("do it like the other functions", "match-surrounding-code", True),
    ("rename this variable to total", "match-surrounding-code", False),
    ("send the digest like the other routine", "match-surrounding-code", False),
    ("use a subagent for this", "delegate-dont-bulk-read", True),
    ("don't bulk-read the files yourself", "delegate-dont-bulk-read", True),
    ("spawn an explore agent to gather context", "delegate-dont-bulk-read", True),
    ("the function signature looks wrong", "delegate-dont-bulk-read", False),
    ("quote me verbatim in the plan", "verbatim-feedback", True),
    ("don't paraphrase my comment", "verbatim-feedback", True),
    ("reproduce it word-for-word", "verbatim-feedback", True),
    ("the severity should be major here", "verbatim-feedback", False),
    ("authorized the click with verbatim quote and a carve-out", "verbatim-feedback", False),
    ("run these in parallel", "parallelize-work", True),
    ("do them at the same time", "parallelize-work", True),
    ("dispatch the agents concurrently", "parallelize-work", True),
    ("this loop is off by one", "parallelize-work", False),
    ("don't infer, read the actual file", "observe-dont-infer", True),
    ("actually run the code first", "observe-dont-infer", True),
    ("stop assuming and check the fixture", "observe-dont-infer", True),
    ("the test name is unclear", "observe-dont-infer", False),
    ("use semble for this lookup", "right-search-tool", True),
    ("don't just grep for it", "right-search-tool", True),
    ("do a semantic search instead", "right-search-tool", True),
    ("this regex needs an anchor", "right-search-tool", False),
]


@pytest.mark.parametrize(
    ("text", "pattern_name", "should_match"),
    REGEX_CASES,
    ids=[f"{name}-{'pos' if expected else 'neg'}-{i}" for i, (_, name, expected) in enumerate(REGEX_CASES)],
)
def test_regex_matchers(text: str, pattern_name: str, should_match: bool) -> None:
    assert matches(by_name(pattern_name).matcher, event(text)) is should_match


@pytest.mark.parametrize(
    ("payload", "should_match"),
    [
        pytest.param({"tool": "Edit", "file_path": "/a.py"}, True, id="edit-denied"),
        pytest.param({"tool": "Write", "file_path": "/b.py"}, True, id="write-denied"),
        pytest.param({"tool": "Bash", "file_path": None}, False, id="bash-denied"),
        pytest.param({"detector": "interrupt"}, False, id="interrupt-payload"),
        pytest.param(None, False, id="no-payload"),
    ],
)
def test_structural_denied_edit(payload: Mapping[str, Any] | None, should_match: bool) -> None:
    assert matches(by_name("denied-edit").matcher, event("rejected", payload=payload)) is should_match


def test_llm_matcher_never_fires_in_cheap_pass() -> None:
    assert matches(LlmMatcher("hint"), event("anything at all")) is False
    assert matches(by_name("novel").matcher, event("some totally unmatched pushback")) is False
