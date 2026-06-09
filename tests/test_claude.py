from __future__ import annotations

import subprocess
from typing import Any

import anyio
import pytest

from cc_pushback.claude import claude_available, run_claude


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
