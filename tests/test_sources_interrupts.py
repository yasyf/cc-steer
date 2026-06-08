from __future__ import annotations

from pathlib import Path

from cc_pushback.models import FeedbackCandidate
from cc_pushback.sources.base import DENIAL_PREFIX
from cc_pushback.sources.interrupts import Interrupts
from tests.builders import (
    assistant_tool_use,
    denial_result,
    interrupt_result,
    parse,
    user_text,
)


def candidates(entries: list[dict[str, object]]) -> list[FeedbackCandidate]:
    return list(Interrupts().candidates_for_file(Path("/t.jsonl"), parse(entries)))


def test_denial_with_embedded_text_uses_user_correction() -> None:
    cands = candidates(
        [
            assistant_tool_use("e1", "Edit", {"file_path": "/a.py"}),
            denial_result("e1", "rename it to feedback_db instead"),
        ]
    )

    assert len(cands) == 1
    assert cands[0].text == "rename it to feedback_db instead"
    assert cands[0].payload == {"tool": "Edit", "file_path": "/a.py"}


def test_denial_without_embedded_text_uses_full_content() -> None:
    cands = candidates(
        [assistant_tool_use("e1", "Bash", {"command": "rm -rf /"}), denial_result("e1", None)]
    )

    assert len(cands) == 1
    assert cands[0].text.startswith(DENIAL_PREFIX)
    assert cands[0].payload == {"tool": "Bash", "file_path": None}


def test_interrupt_captures_following_user_correction() -> None:
    cands = candidates(
        [
            assistant_tool_use("b1", "Bash", {"command": "sleep 100"}),
            interrupt_result("b1"),
            user_text("stop, run the fast variant instead"),
        ]
    )

    assert len(cands) == 1
    assert cands[0].text == "stop, run the fast variant instead"
    assert cands[0].payload == {"detector": "interrupt"}


def test_interrupt_without_correction_falls_back_to_marker() -> None:
    cands = candidates([assistant_tool_use("b1", "Bash", {"command": "x"}), interrupt_result("b1")])

    assert len(cands) == 1
    assert cands[0].text == "[Request interrupted by user]"


def test_interrupt_with_junk_following_message_falls_back_to_marker() -> None:
    cands = candidates(
        [
            assistant_tool_use("b1", "Bash", {"command": "x"}),
            interrupt_result("b1"),
            user_text("Base directory for this skill: /tmp/skill\n# Bootstrap"),
        ]
    )

    assert len(cands) == 1
    assert cands[0].text == "[Request interrupted by user]"


def test_interrupt_keeps_terse_correction() -> None:
    cands = candidates(
        [
            assistant_tool_use("b1", "Bash", {"command": "x"}),
            interrupt_result("b1"),
            user_text("wrong file"),
        ]
    )

    assert len(cands) == 1
    assert cands[0].text == "wrong file"


def test_exit_plan_denial_not_claimed_here() -> None:
    cands = candidates(
        [
            assistant_tool_use("plan1", "ExitPlanMode", {"plan": "x"}),
            denial_result("plan1", "no"),
        ]
    )

    assert cands == []
