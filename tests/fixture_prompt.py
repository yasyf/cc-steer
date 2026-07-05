"""Deterministic synthetic candidate rows for the build_prompt byte fixture.

Running this module as a script regenerates ``tests/fixtures/build_prompt_2_0.txt``
from the current :func:`cc_steer.triage.build_prompt`; the regression test in
``tests/test_triage.py`` asserts the rendering reproduces the file byte-for-byte.
The fixture covers every rendering branch of the window pipeline: a denial
candidate whose trigger turn carries a >1500-char Edit rendered unclipped under the
generous trigger budget, a transcript-message candidate whose long surrounding
turns clip under the moderate budget, and a candidate whose transcript is gone (the
labeled summary-preview fallback).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import cc_transcript.discovery

from cc_steer.detectors import detect
from cc_steer.triage import AUDIT_PROMPT, JUDGE_PROMPT, build_prompt
from tests.builders import (
    SESSION,
    assistant_text,
    assistant_tool_use,
    denial_result,
    parse,
    user_text,
    write_transcript,
)

if TYPE_CHECKING:
    from typing import Any

    from cc_transcript.mining import FeedbackCandidate
    from cc_transcript.models import TranscriptEvent

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "build_prompt_2_0.txt"

LONG_EDIT = "\n".join(f"    refreshed_line_{i:03d} = compute_refreshed_value({i})" for i in range(40))


def long_text(stem: str, words: int) -> str:
    return " ".join(f"{stem}{i:03d}" for i in range(words))


def session_entries() -> list[dict[str, Any]]:
    return [
        user_text("please clean up the build pipeline"),
        assistant_text(long_text("before", 150)),
        assistant_tool_use("t1", "Bash", {"command": "rm -rf build && make all"}),
        user_text("now wire up the release job"),
        assistant_text(long_text("trigger", 200)),
        assistant_tool_use(
            "t2", "Edit", {"file_path": "/repo/app.py", "old_string": "old_line()", "new_string": LONG_EDIT}
        ),
        denial_result("t2", said="no, stop — " + long_text("direction", 40)),
        user_text(long_text("after", 150)),
    ]


def row_of(candidate: FeedbackCandidate) -> dict[str, object]:
    return {"source_kind": candidate.source_kind, "context_json": candidate.window.to_json(), "text": candidate.text}


def detected_rows(events: list[TranscriptEvent]) -> list[dict[str, object]]:
    candidates = detect(events)
    denial = next(c for c in candidates if c.source_kind == "interrupt_rejection")
    message = next(c for c in candidates if c.source_kind == "transcript_message" and c.text.startswith("after"))
    return [row_of(denial), row_of(message)]


def expired_row() -> dict[str, object]:
    events = parse(
        [
            assistant_text("here is the diff", sessionId="sess-gone"),
            user_text("no, this clobbers the config", sessionId="sess-gone"),
        ]
    )
    [candidate] = [c for c in detect(events) if c.source_kind == "transcript_message"]
    return row_of(candidate)


async def render(root: Path) -> str:
    entries = session_entries()
    write_transcript(root / "proj" / f"{SESSION}.jsonl", entries)
    rows = [*detected_rows(parse(entries)), expired_row()]
    return "\n\n########\n\n".join(
        [(await build_prompt(template, row))[0] for row in rows for template in (JUDGE_PROMPT, AUDIT_PROMPT)]
    )


async def regenerate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cc_transcript.discovery.CLAUDE_PROJECTS_DIR = Path(tmp)
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_bytes((await render(Path(tmp))).encode())


if __name__ == "__main__":
    anyio.run(regenerate)
