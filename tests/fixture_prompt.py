"""Deterministic synthetic candidate rows for the build_prompt byte fixture.

Running this module as a script regenerates ``tests/fixtures/build_prompt_pre0c.txt``
from the current :func:`cc_pushback.triage.build_prompt`; the regression test in
``tests/test_triage.py`` asserts the rendering reproduces the file byte-for-byte.
The fixture was frozen against the pre-refactor build_prompt, so it proves the
lift onto ``cc_transcript.domains.mining`` preserved every rendering branch:
multi-turn context with tool inputs (Bash + Edit), a missing tool input, a long
trigger and long turns that need clipping, an empty window, and a trigger-less row.
"""

from __future__ import annotations

from pathlib import Path

from cc_transcript.domains.mining import ContextSnapshot, ContextTurn

from cc_pushback.triage import AUDIT_PROMPT, JUDGE_PROMPT, build_prompt

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "build_prompt_pre0c.txt"


def long_text(stem: str, words: int) -> str:
    return " ".join(f"{stem}{i:03d}" for i in range(words))


def full_row() -> dict[str, object]:
    snapshot = ContextSnapshot(
        before=(
            ContextTurn(role="user", text="please clean up the build pipeline"),
            ContextTurn(
                role="assistant",
                text=long_text("before", 150),
                tool_calls=("Bash", "Edit", "Read"),
                tool_inputs=(
                    "rm -rf build && make all",
                    "/repo/app.py\n- old_line()\n+ new_line()",
                ),
            ),
        ),
        trigger=ContextTurn(
            role="assistant",
            text=long_text("trigger", 400),
            tool_calls=("Bash", "Edit"),
            tool_inputs=("git push --force origin main", long_text("editinput", 400)),
        ),
        after=(
            ContextTurn(role="user", text=long_text("after", 150)),
            ContextTurn(role="tool", text="exit status 1"),
        ),
    )
    return {
        "source_kind": "transcript_message",
        "context_json": snapshot.to_json(),
        "text": "no, stop — " + long_text("complaint", 40),
    }


def bare_row() -> dict[str, object]:
    snapshot = ContextSnapshot(before=(), trigger=None, after=())
    return {"source_kind": "interrupt_rejection", "context_json": snapshot.to_json(), "text": "no"}


def render() -> str:
    return "\n\n########\n\n".join(
        build_prompt(template, row) for row in (full_row(), bare_row()) for template in (JUDGE_PROMPT, AUDIT_PROMPT)
    )


if __name__ == "__main__":
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_bytes(render().encode())
