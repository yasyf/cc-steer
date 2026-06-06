from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING, Any

from cc_transcript.parser import parse_events_from_bytes

from cc_pushback.sources.base import DENIAL_PREFIX, USER_SAID_MARKER

if TYPE_CHECKING:
    from pathlib import Path

    from cc_transcript.models import TranscriptEvent

BASE_TS = "2026-06-01T12:00:00+00:00"
SESSION = "sess-1"

uuids = itertools.count()


def next_uuid() -> str:
    return f"uuid-{next(uuids)}"


def envelope(entry_type: str, **overrides: Any) -> dict[str, Any]:
    return {
        "type": entry_type,
        "uuid": overrides.pop("uuid", next_uuid()),
        "parentUuid": overrides.pop("parentUuid", None),
        "sessionId": overrides.pop("sessionId", SESSION),
        "timestamp": overrides.pop("timestamp", BASE_TS),
        "cwd": "/repo",
        "gitBranch": "main",
        "version": "1.2.3",
        "isSidechain": overrides.pop("isSidechain", False),
        "isMeta": overrides.pop("isMeta", False),
        "entrypoint": "cli",
        **overrides,
    }


def user_text(text: str, **overrides: Any) -> dict[str, Any]:
    return envelope("user", message={"role": "user", "content": text}, **overrides)


def assistant_text(text: str, **overrides: Any) -> dict[str, Any]:
    return envelope(
        "assistant",
        message={"role": "assistant", "model": "claude", "content": [{"type": "text", "text": text}]},
        **overrides,
    )


def assistant_tool_use(tool_id: str, name: str, tool_input: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return envelope(
        "assistant",
        message={
            "role": "assistant",
            "model": "claude",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}],
        },
        **overrides,
    )


def tool_result(tool_id: str, content: str, *, is_error: bool = False, **overrides: Any) -> dict[str, Any]:
    return envelope(
        "user",
        message={
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content, "is_error": is_error}],
        },
        **overrides,
    )


def denial_result(tool_id: str, said: str | None, **overrides: Any) -> dict[str, Any]:
    body = DENIAL_PREFIX + "."
    if said is not None:
        body += f"\n\n{USER_SAID_MARKER}{said}"
    return tool_result(tool_id, body, is_error=True, **overrides)


def mode_entry(value: str, **overrides: Any) -> dict[str, Any]:
    return {"type": "mode", "mode": value, "sessionId": overrides.pop("sessionId", SESSION)}


def interrupt_result(tool_id: str, **overrides: Any) -> dict[str, Any]:
    return tool_result(tool_id, "[Request interrupted by user]", is_error=True, **overrides)


def parse(entries: list[dict[str, Any]]) -> list[TranscriptEvent]:
    return parse_events_from_bytes("".join(json.dumps(entry) + "\n" for entry in entries).encode())


def write_transcript(path: Path, entries: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path
