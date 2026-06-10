from __future__ import annotations

import json
import subprocess
from typing import Any

import anyio
import pytest
from pydantic import BaseModel

from cc_pushback.claude import claude_available, run_claude, run_claude_structured


class Pick(BaseModel):
    choice: str
    score: float


def completed(stdout: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(args=["claude"], returncode=0, stdout=stdout, stderr=b"")


def test_claude_available_reflects_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_pushback.claude.shutil.which", lambda _: "/usr/bin/claude")
    assert claude_available() is True
    monkeypatch.setattr("cc_pushback.claude.shutil.which", lambda _: None)
    assert claude_available() is False


@pytest.mark.anyio
async def test_run_claude_builds_argv_and_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    async def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = command
        return completed(b'{"is_error": false, "result": "hello"}')

    monkeypatch.setattr(anyio, "run_process", fake_run)
    assert await run_claude("PROMPT", system="SYS", model="claude-x") == "hello"
    argv = captured["argv"]
    assert argv[:3] == ["claude", "-p", "PROMPT"]
    assert argv[argv.index("--model") + 1] == "claude-x"
    assert argv[argv.index("--system-prompt") + 1] == "SYS"
    assert argv[argv.index("--output-format") + 1] == "json"


@pytest.mark.anyio
async def test_run_claude_raises_on_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(*_: Any, **__: Any) -> subprocess.CompletedProcess[bytes]:
        return completed(b'{"is_error": true, "result": ""}')

    monkeypatch.setattr(anyio, "run_process", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        await run_claude("p", system="s", model="m")


@pytest.mark.anyio
async def test_run_claude_structured_builds_argv_and_parses_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = command
        captured["input"] = kwargs["input"]
        envelope = [{"type": "result", "structured_output": {"choice": "a", "score": 0.9}}]
        return completed(json.dumps(envelope).encode())

    monkeypatch.setattr(anyio, "run_process", fake_run)
    pick = await run_claude_structured("JUDGE THIS", response_model=Pick)
    assert pick == Pick(choice="a", score=0.9)
    argv = captured["argv"]
    assert captured["input"] == b"JUDGE THIS"
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--output-format") + 1] == "json"
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert set(schema["properties"]) == {"choice", "score"}
    assert schema["additionalProperties"] is False


@pytest.mark.anyio
async def test_run_claude_structured_resolves_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["argv"] = command
        return completed(b'{"choice": "b", "score": 0.1}')

    monkeypatch.setattr(anyio, "run_process", fake_run)
    pick = await run_claude_structured("p", response_model=Pick, tier="large")
    assert pick == Pick(choice="b", score=0.1)
    assert captured["argv"][captured["argv"].index("--model") + 1] == "opus"


@pytest.mark.anyio
async def test_run_claude_structured_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_pushback.claude.CLAUDE_TIMEOUT", 0.01)

    async def fake_run(*_: Any, **__: Any) -> subprocess.CompletedProcess[bytes]:
        await anyio.sleep(1)
        return completed(b"{}")

    monkeypatch.setattr(anyio, "run_process", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        await run_claude_structured("p", response_model=Pick)
