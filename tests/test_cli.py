from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from cc_pushback.cli import main


def test_help_exits_cleanly() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage: main")


def test_extract_keeps_only_human_typed_messages(tmp_path: Path) -> None:
    transcripts = tmp_path / "projects"
    transcripts.mkdir()
    entries = [
        {"type": "user", "message": {"role": "user", "content": "don't add a fallback here, crash instead"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "passed"}]}},
        {"type": "user", "message": {"role": "user", "content": "<system-reminder>injected noise</system-reminder>"}},
    ]
    (transcripts / "abc123.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    output = tmp_path / "messages.jsonl"

    result = CliRunner().invoke(
        main, ["extract", "--transcripts", str(transcripts), "--output", str(output)]
    )

    assert result.exit_code == 0
    assert result.output == f"Extracted 1 messages to {output}\n"
    assert json.loads(output.read_text()) == {
        "session": "abc123",
        "text": "don't add a fallback here, crash instead",
    }
